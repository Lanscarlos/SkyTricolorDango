"""主循环：截屏 → 读新消息 → 攒一小会儿 → 生成回复 → 限速 → 发送。"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

from .chat.panel import PanelKeeper
from .chat.reader import ChatReader, Message
from .chat.responder import Responder
from .chat.sender import ChatSender
from .chat.tracker import SelfFilter
from .config import Config
from .device.base import Device
from .game.emotes import EmotePlayer
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
        emotes: EmotePlayer | None = None,
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
        self.run_dir = run
        self.emotes = emotes
        self.emoted: list[str] = []  # 做过（含 dry-run）的动作，方便测试和日志
        # sleep 包一层：测试会在构造之后替换 agent.sleep
        self.panel = PanelKeeper(cfg.vision, device, reader, lambda s: self.sleep(s))

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
                self.panel.maybe_reopen(now)
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
        dry = self.cfg.reply.dry_run
        if reply is None:
            log.info("模型选择不回复")
            if typing:
                self.sender.cancel()
            if self.run_dir:
                self.run_dir.record_reply(batch, None, sent=False)
            return None
        if reply.emote:
            if typing:  # 输入框开着时数字键会变成打字：先关掉提前打开的输入框，发文字时再打开
                self.sender.cancel()
            self._emote(reply.emote)
        if reply.text is None:  # 只做动作不说话：不占发送限速
            if self.run_dir:
                self.run_dir.record_reply(batch, reply.render(), sent=not dry)
            return None

        text = self.cfg.reply.disclosure_prefix + reply.text
        self.limiter.record(now)
        self.sent.append(text)
        if dry:
            log.info("[dry-run] 将会发送: %s", text)
        else:
            self.sender.send(text)
            self.self_filter.remember(text, self.clock())
        if self.run_dir:
            self.run_dir.record_reply(batch, reply.render(self.cfg.reply.disclosure_prefix), sent=not dry)
        return text

    def _emote(self, name: str) -> None:
        self.emoted.append(name)
        if self.emotes is None:
            return
        if self.cfg.reply.dry_run:
            log.info("[dry-run] 将会做动作: %s", name)
            self.emotes.pretend(name)
            return
        try:
            self.emotes.perform(name)
        except Exception:  # 换轮盘没找到图标、adb 出错……动作做不成，话照样说
            log.exception("做动作「%s」失败，跳过", name)

    def ensure_log_open(self) -> bool:
        """log 模式：看不到聊天记录面板、也没在打字时，按一下打开面板的键（默认 C）。返回面板现在开没开。"""
        return self.panel.ensure_open()

    def run(self, duration: float = 0.0) -> None:
        """一直跑；duration > 0 时跑这么多秒后自己退出（别在外面套 timeout：Windows 上停掉外层后 Python 会变成孤儿进程）。"""
        mode = "dry-run（只打印不发送）" if self.cfg.reply.dry_run else "LIVE（会真的发送）"
        log.info("Agent 启动，模式: %s；Ctrl+C 退出%s", mode, f"；{duration:.0f} 秒后自动结束" if duration > 0 else "")
        self.ensure_log_open()
        deadline = self.clock() + duration if duration > 0 else float("inf")
        while self.clock() < deadline:
            started = self.clock()
            try:
                self.step()
            except Exception:
                log.exception("本轮出错，继续")
            elapsed = self.clock() - started
            self.sleep(max(0.0, self.cfg.vision.poll_interval - elapsed))
        log.info("到时间了，Agent 结束")
