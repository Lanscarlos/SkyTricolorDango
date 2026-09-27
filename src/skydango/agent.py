"""主循环：截屏 → 读新消息 → 攒一小会儿 → 生成回复 → 限速 → 发送。"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

from .chat.reader import ChatReader, Message
from .chat.responder import Responder
from .chat.sender import ChatSender
from .chat.tracker import SelfFilter
from .config import Config
from .device.base import Device
from .runlog import RunDir

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
        run: RunDir | None = None,
        env=None,  # EnvWatcher：识别身边有谁、在哪（vision/env.py）
        social=None,  # SocialHandler：接受好友的牵手 / 拥抱 / 击掌（game/social.py），请求由 env 发现
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.social = social
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
        self._last_reopen = float("-inf")
        self.run_dir = run

    def step(self) -> str | None:
        """跑一轮；如果这一轮发出（或 dry-run 模拟发出）了回复，返回那句话。"""
        now = self.clock()
        fresh: list[Message] = []
        frame = None
        try:
            frame = self.device.screenshot()
        except Exception as exc:  # 实测 adb 截图偶尔会连续失败几秒；只跳过“读”，攒着的消息照样回复
            log.warning("截图失败，这一轮不读新消息: %s", str(exc).splitlines()[0])
        if frame is not None:
            fresh = self.reader.read(frame, now)
            if self.cfg.vision.mode == "log":
                self._maybe_reopen_log(now)
            if self.env is not None:  # 到了间隔会在后台线程里扫一次画面
                self.env.observe(frame, now, panel_visible=self.reader.panel_closed_since is None)
            if self.social is not None and self.env is not None and self.env.requests:
                try:
                    self.social.handle(self.env.requests, now)
                except Exception:
                    log.exception("处理互动请求出错")
        if fresh:
            for m in fresh:
                log.info("读到: %s", f"{m.speaker}：{m.text}" if m.speaker else m.text)
            if self.run_dir:
                self.run_dir.save_frame(frame, [m.box for m in fresh])
            self.pending.extend(fresh)
            self.pending = self.pending[-self.cfg.chat.max_pending :]
            self.last_new_at = now

        if not self.pending or now - self.last_new_at < self.cfg.chat.debounce:
            return None
        if not self.limiter.allow(now):
            return None

        batch, self.pending = self.pending, []
        # 等模型的时候先把输入框打开：头顶显示“正在输入”，像真人在打字；发送时也省掉打开这一步
        typing = not self.cfg.reply.dry_run and self.cfg.sender.type_ahead
        if typing:
            try:
                self.sender.open()
            except Exception:
                log.warning("提前打开输入框失败，发送时再打开", exc_info=True)
        reply = self.responder.reply(batch)
        if reply is None:
            log.info("模型选择不回复")
            if typing:
                self.sender.cancel()
            if self.run_dir:
                self.run_dir.record_reply(batch, None, sent=False)
            return None

        text = self.cfg.reply.disclosure_prefix + reply
        self.limiter.record(now)
        self.sent.append(text)
        if self.cfg.reply.dry_run:
            log.info("[dry-run] 将会发送: %s", text)
        else:
            self.sender.send(text)
            self.self_filter.remember(text, self.clock())
        if self.run_dir:
            self.run_dir.record_reply(batch, text, sent=not self.cfg.reply.dry_run)
        return text

    def ensure_log_open(self) -> bool:
        """log 模式：看不到聊天记录面板、也没在打字时，按一下打开面板的键（默认 C）。返回面板现在开没开。"""
        key = self.cfg.vision.log_open_key
        if self.cfg.vision.mode != "log" or not key:
            return True
        if self.reader.panel_visible(self.device.screenshot()):
            return True
        if self.device.ime_shown():
            log.warning("输入框开着，没法用按键打开聊天记录面板；请手动打开（光遇里按 C）")
            return False
        log.info("聊天记录面板没打开，按键 %d 打开", key)
        self.device.hw_key(key)
        self.sleep(1.0)
        if self.reader.panel_visible(self.device.screenshot()):
            return True
        log.warning("按了键聊天记录面板还是没出现，可能被别的界面挡住了，请看一下游戏画面")
        return False

    def _maybe_reopen_log(self, now: float) -> None:
        since = self.reader.panel_closed_since
        vision = self.cfg.vision
        if since is None or not vision.log_reopen_after or now - since < vision.log_reopen_after:
            return
        if now - self._last_reopen < vision.log_reopen_cooldown:
            return
        self._last_reopen = now
        self.ensure_log_open()

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
