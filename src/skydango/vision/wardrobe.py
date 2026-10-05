"""描述器：把一个人的裁图交给一次性的 Haiku，写成一句话装扮描述（后台线程，一次一张）。

request 由感知线程调、tick 由本线程调，队列和状态用锁护着；describe / on_done 在锁外调。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from ..models.errors import ModelError, ModelUnavailable
from ..brain.images import fit, image_block
from ..config import AppearanceConfig

log = logging.getLogger(__name__)

ME, FRIEND, STRANGER = 0, 1, 2  # 优先级：小的先描述
MAX_FAILS = 3
WINDOW = 3600.0
DESC_MAX = 25

WARDROBE_SYSTEM = """你是《光遇》玩家的眼睛：看一张玩家的裁图，用一句短中文描述他的装扮，给另一个 AI 认人用。
- 只写看得见的颜色和样子：发型 / 头饰、面具、斗篷 / 衣服的颜色和长短、手里拿的东西。
- 不写季节名、物品名、套装名，不猜，不客套。
- 只回一个 JSON：{"desc": "...", "clear": true 或 false}。"""

WARDROBE_REQUEST = """描述这个人的装扮：发型 / 头饰、面具、斗篷 / 衣服（颜色和长短）、手里拿的。
desc 不超过 25 个字，只写颜色和样子。背影、模糊、被挡住、看不出装扮时 clear 写 false（desc 可以留空）。
只回 JSON：{"desc": "...", "clear": true}"""

_JSON = re.compile(r"\{.*?\}", re.DOTALL)


def parse_outfit(text: str) -> tuple[str, bool] | None:
    """取文字里第一个 {…} 解析成 (描述, 看清了没有)；解析不了返回 None。"""
    m = _JSON.search(text or "")
    if m is None:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    desc = re.sub(r"\s+", "", str(data.get("desc") or ""))[:DESC_MAX]
    clear = bool(data.get("clear", True))
    if not desc:
        clear = False
    return desc, clear


class Wardrobe:
    def __init__(
        self,
        cfg: AppearanceConfig,
        describe: Callable[[list[dict]], str],
        on_done: Callable[[str, str, str, np.ndarray], None],
        clock: Callable[[], float] = time.monotonic,
        available: Callable[[], bool] | None = None,  # Claude 总闸开着吗（gate.ok）；为假时不排、不描述（spec 2026-10-04-claude-gate §3.3）
    ) -> None:
        self.cfg = cfg
        self.describe = describe
        self.on_done = on_done
        self.clock = clock
        self.available = available
        self.calls = 0
        self._lock = threading.Lock()
        self._queue: list[tuple[int, int, str, str, np.ndarray, np.ndarray]] = []  # (优先级, 序号, kind, who, 裁图, 特征)
        self._busy: set[tuple[str, str]] = set()  # 在队列里或正在描述
        self._fails: dict[tuple[str, str], int] = {}
        self._retry_at: dict[tuple[str, str], float] = {}
        self._calls: deque[float] = deque()  # 最近调 describe 的时间
        self._limit_until = float("-inf")
        self._seq = 0

    def request(self, kind: str, who: str, priority: int, crop: np.ndarray, feat: np.ndarray, now: float) -> bool:
        key = (kind, who)
        if self._closed():
            return False
        with self._lock:
            if key in self._busy or now < self._retry_at.get(key, float("-inf")) or self._fails.get(key, 0) >= MAX_FAILS:
                return False
            self._seq += 1
            self._queue.append((priority, self._seq, kind, who, crop, feat))
            self._busy.add(key)
            return True

    def reset(self, kind: str, who: str) -> None:
        """换装后清掉失败次数。"""
        with self._lock:
            self._fails.pop((kind, who), None)
            self._retry_at.pop((kind, who), None)

    def _closed(self) -> bool:
        """Claude 总闸关了：等多久都没用，不再排描述。"""
        return self.available is not None and not self.available()

    def tick(self, now: float) -> bool:
        if self._closed():
            return False
        with self._lock:
            while self._calls and now - self._calls[0] >= WINDOW:
                self._calls.popleft()
            if not self._queue or len(self._calls) >= self.cfg.describe_max or now < self._limit_until:
                return False
            item = min(self._queue, key=lambda q: (q[0], q[1]))
            self._queue.remove(item)
            self._calls.append(now)
            self.calls += 1
        _, _, kind, who, crop, feat = item
        key = (kind, who)
        content = [image_block(fit(crop, (512, 512)), 85), {"type": "text", "text": WARDROBE_REQUEST}]
        try:
            outfit = parse_outfit(self.describe(content))
        except ModelError as exc:
            if exc.down == "limit" or isinstance(exc, ModelUnavailable):
                log.warning("描述装扮：模型用不了（%s），%.0f 秒后再试", exc, self.cfg.quota_wait)
                with self._lock:
                    self._limit_until = now + self.cfg.quota_wait
                    self._queue.append(item)
                return True
            log.warning("描述装扮没成（%s %s）：%s", kind, who, exc)
            outfit = None
        except Exception:
            log.exception("描述装扮出错（%s %s）", kind, who)
            outfit = None
        if outfit is None or not outfit[1]:
            with self._lock:
                self._busy.discard(key)
                self._fails[key] = self._fails.get(key, 0) + 1
                self._retry_at[key] = now + self.cfg.retry_after
            return True
        with self._lock:
            self._busy.discard(key)
            self._fails.pop(key, None)
            self._retry_at.pop(key, None)
        try:
            self.on_done(kind, who, outfit[0], feat)
        except Exception:
            log.exception("装扮描述没交出去")
        return True

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(0.5):
            try:
                self.tick(self.clock())
            except Exception:
                log.exception("装扮描述线程出错")
