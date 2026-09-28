"""眼睛：把截图写成文字描述（一次性的 Haiku），后台按时机自动看，缓存最新一份。

不碰设备：只读身体最近一帧（Body.last_frame）和 EnvWatcher 认出的名字位置。
大脑平时只收这份文字；要原图才调 look(image=true)。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from ..config import BrainConfig
from .images import fit, image_block, label_note

log = logging.getLogger(__name__)

AUTO_LOOK_KINDS = {"arrive", "leave", "stranger", "scene_change"}  # 这些事件发生后（隔够 auto_look_min）自动看一眼

EYES_SYSTEM = """你是一个《光遇》玩家的眼睛：看游戏截图，写成简短的中文文字，给另一个 AI 看。
- 只写看到的，不猜；看不清就说看不清。
- 人名只用给出的“名字位置”里的名字；没给名字的人都叫“陌生人”。
- 不客套，不给建议，不提问。"""

LOOK_REQUEST = """按这四项写，每项一两句：
地点和环境：像哪张图 / 什么地方、建筑、天气、白天还是晚上
好友：上面列出的每个人穿什么（斗篷、发型、面具、颜色）、在干什么；没有就写“没看到”
陌生人：大概几个、在干什么
画面状态：有没有弹窗、黑屏、看不懂的图标；正常就写“正常”"""

AROUND_REQUEST = """这是原地转一圈拍的四张图（前、右、后、左）。每个方向一两句：有什么地形 / 建筑、有没有人（认不出名字，只写几个人、穿什么、在干什么）。
最后一句总结现在在什么地方。"""


def eyes_command(base: list[str], cfg: BrainConfig) -> list[str]:
    return [
        *base, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--model", cfg.eyes_model, "--effort", "low", "--tools", "", "--strict-mcp-config",
        "--permission-mode", "dontAsk", "--disable-slash-commands", "--system-prompt", EYES_SYSTEM,
    ]


class Eyes:
    def __init__(
        self,
        cfg: BrainConfig,
        describe: Callable[[list[dict]], str],  # 内容块 → 文字（一次性 claude -p；测试里换成假的）
        frame: Callable[[], np.ndarray | None],  # 身体最近一帧
        labels: Callable[[], dict],  # 名字 → (x, y, w, h, 看到的时间)
        blackout: Callable[[], bool],
        label_keep: float = 7.0,  # 名字多久内看到过才算在这张图里
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.describe = describe
        self.frame = frame
        self.labels = labels
        self.blackout = blackout
        self.label_keep = label_keep
        self.clock = clock
        self.latest: tuple[str, float] | None = None
        self.last_look = float("-inf")
        self.look_request = LOOK_REQUEST
        self._poked = False
        self._lock = threading.Lock()  # 自动看和大脑要看可能撞上：同一时间只看一次

    def notice(self, kind: str) -> None:
        if kind in AUTO_LOOK_KINDS:
            self._poked = True

    def due(self, now: float) -> bool:
        if self.blackout():
            return False
        since = now - self.last_look
        return since >= self.cfg.auto_look_max or (self._poked and since >= self.cfg.auto_look_min)

    def tick(self, now: float) -> bool:
        """后台线程每秒调一次：到了时机就看一眼。返回这次有没有去看。"""
        if not self.due(now):
            return False
        frame = self.frame()
        if frame is None:
            return False
        try:
            self.describe_frame(frame, now)
        except Exception as exc:
            log.warning("眼睛这次没看成：%s", exc)
            self.last_look = now  # 别每秒都重试，等下一个时机
        return True

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(1.0):
            try:
                self.tick(self.clock())
            except Exception:
                log.exception("眼睛出错")

    def describe_frame(self, frame: np.ndarray, now: float) -> str:
        view = fit(frame, tuple(self.cfg.image_size))
        recent = {n: v for n, v in self.labels().items() if now - v[4] <= self.label_keep}
        note = label_note(recent, view.shape[1] / frame.shape[1])
        content = [image_block(view, self.cfg.jpeg_quality), {"type": "text", "text": note + "\n\n" + self.look_request}]
        with self._lock:
            text = self.describe(content).strip()
        self._keep(text, now)
        return text

    def describe_around(self, frames: list[np.ndarray], now: float) -> str:
        content: list[dict] = []
        for side, frame in zip(("前", "右", "后", "左"), frames):
            content += [{"type": "text", "text": f"【{side}】"}, image_block(fit(frame, tuple(self.cfg.image_size)), self.cfg.jpeg_quality)]
        content.append({"type": "text", "text": AROUND_REQUEST if len(frames) > 1 else self.look_request})
        with self._lock:
            text = self.describe(content).strip()
        self._keep(text, now)
        return text

    def summary(self, now: float) -> str:
        if self.latest is None:
            return "场景：还没有场景描述"
        text, t = self.latest
        return f"场景（{now - t:.0f} 秒前看的）：\n{text}"

    def _keep(self, text: str, now: float) -> None:
        self.latest = (text, now)
        self.last_look = now
        self._poked = False
