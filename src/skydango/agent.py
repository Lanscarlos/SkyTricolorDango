"""主循环：截屏 → 读新消息 → 攒一小会儿 → 生成回复 → 限速 → 发送。"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .chat.reader import ChatReader, Message
from .chat.responder import Responder
from .chat.sender import ChatSender
from .chat.tracker import SelfFilter
from .config import Config
from .device.base import Device
from .imageio import imwrite
from .vision.bubbles import annotate

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, min_interval: float, max_per_minute: int) -> None:
        self.min_interval = min_interval
        self.max_per_minute = max_per_minute
        self._sent: deque[float] = deque()

    def allow(self, now: float) -> bool:
        while self._sent and now - self._sent[0] > 60:
            self._sent.popleft()
        if self._sent and now - self._sent[-1] < self.min_interval:
            return False
        return len(self._sent) < self.max_per_minute

    def record(self, now: float) -> None:
        self._sent.append(now)


class Agent:
    def __init__(
        self,
        cfg: Config,
        device: Device,
        reader: ChatReader,
        responder: Responder,
        sender: ChatSender,
        self_filter: SelfFilter,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.reader = reader
        self.responder = responder
        self.sender = sender
        self.self_filter = self_filter
        self.clock = clock
        self.sleep = sleep
        self.limiter = RateLimiter(cfg.reply.min_interval, cfg.reply.max_per_minute)
        self.pending: list[Message] = []
        self.last_new_at = 0.0
        self.sent: list[str] = []  # 记录（含 dry-run），方便测试和日志

    def _save_debug(self, frame: np.ndarray, messages: list[Message]) -> None:
        if not self.cfg.vision.debug_dir:
            return
        out = Path(self.cfg.vision.debug_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}.png"
        imwrite(out / name, annotate(frame, [m.box for m in messages]))

    def step(self) -> str | None:
        """跑一轮；如果这一轮发出（或 dry-run 模拟发出）了回复，返回那句话。"""
        now = self.clock()
        frame = self.device.screenshot()
        fresh = self.reader.read(frame, now)
        if fresh:
            for m in fresh:
                log.info("读到: %s", m.text)
            self._save_debug(frame, fresh)
            self.pending.extend(fresh)
            self.pending = self.pending[-self.cfg.chat.max_pending :]
            self.last_new_at = now

        if not self.pending or now - self.last_new_at < self.cfg.chat.debounce:
            return None
        if not self.limiter.allow(now):
            return None

        batch, self.pending = self.pending, []
        reply = self.responder.reply(batch)
        if reply is None:
            log.info("模型选择不回复")
            return None

        text = self.cfg.reply.disclosure_prefix + reply
        self.limiter.record(now)
        self.sent.append(text)
        if self.cfg.reply.dry_run:
            log.info("[dry-run] 将会发送: %s", text)
        else:
            self.sender.send(text)
            self.self_filter.remember(text, self.clock())
        return text

    def ensure_log_open(self) -> None:
        """log 模式：面板里一行都读不到、也没在打字时，按一下打开面板的键（默认 C）。"""
        key = self.cfg.vision.log_open_key
        if self.cfg.vision.mode != "log" or not key:
            return
        if self.reader.log_rows(self.device.screenshot()):
            return
        if self.device.ime_shown():
            log.warning("输入框开着，没法用按键打开聊天记录面板；请手动打开（光遇里按 C）")
            return
        log.info("聊天记录面板没打开，按键 %d 打开", key)
        self.device.hw_key(key)
        self.sleep(1.0)

    def run(self) -> None:
        mode = "dry-run（只打印不发送）" if self.cfg.reply.dry_run else "LIVE（会真的发送）"
        log.info("Agent 启动，模式: %s；Ctrl+C 退出", mode)
        self.ensure_log_open()
        while True:
            started = self.clock()
            try:
                self.step()
            except Exception:
                log.exception("本轮出错，继续")
            elapsed = self.clock() - started
            self.sleep(max(0.0, self.cfg.vision.poll_interval - elapsed))
