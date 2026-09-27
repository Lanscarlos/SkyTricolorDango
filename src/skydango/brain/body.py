"""大脑的身体：截图、读聊天、认人、按规则秒接互动请求，把发生的事变成事件交给大脑。

大脑的工具通过命令队列在这个线程里执行 —— 只有身体线程碰设备，说话、做动作、转视角不会和接受请求撞在一起。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from ..chat.panel import PanelKeeper
from ..chat.reader import Message
from ..config import Config
from ..game.social import IDLE, KIND_NAMES
from .events import EventQueue
from .images import difference, is_black, thumb

log = logging.getLogger(__name__)

REQUEST_KINDS = ("hand", "hug", "highfive", "piggyback", "candle", "*")
PANEL_LOST_AFTER = 30.0  # 面板关了这么久（自动重开也没成功）就告诉大脑
SCENE_EVENT_COOLDOWN = 10.0


class ToolError(Exception):
    """工具做不成：原因原样告诉大脑，让它换个办法。"""


class Body:
    def __init__(
        self,
        cfg: Config,
        device,
        reader,
        sender,
        self_filter,
        events: EventQueue,
        env=None,  # vision.env.EnvWatcher：身边有谁、互动请求、名字位置、圆圈状态
        social=None,  # game.social.SocialHandler：按规则秒接请求
        emotes=None,  # game.emotes.EmotePlayer
        camera=None,  # brain.camera.Camera
        fallback=None,  # chat.responder.Responder：大脑离线时的备用回复
        store=None,  # chat.memory.MemoryStore：live 时记聊天记录
        notes=None,  # chat.memory.NotesKeeper
        run=None,  # runlog.RunDir
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.reader = reader
        self.sender = sender
        self.self_filter = self_filter
        self.events = events
        self.env = env
        self.social = social
        self.emotes = emotes
        self.camera = camera
        self.fallback = fallback
        self.store = store
        self.notes = notes
        self.run_dir = run
        self.clock = clock
        self.sleep = sleep
        self.wall = wall
        self.panel = PanelKeeper(cfg.vision, device, reader, lambda s: self.sleep(s))
        self.brain_offline: Callable[[float], bool] = lambda now: False
        self.chat: deque[tuple[float, str, str]] = deque(maxlen=50)  # (时间, 说话人, 内容)；自己说的说话人是“我”
        self.heard: list[Message] = []  # 上次说话以后听到的，记聊天记录用
        self.said: list[str] = []  # 说过（含 dry-run）的话
        self.emoted: list[str] = []
        self.last_frame = None
        self.last_look = float("-inf")
        self.blackout = False
        self.holding: str | None = None  # 推测正牵着谁的手
        self._holding_since = 0.0
        self._accepted_hand: tuple[str, float] | None = None
        self._nearby: set[str] = set()
        self._requests: set[tuple[str, str]] = set()
        self._ref_thumb = None
        self._ref_at = float("-inf")
        self._scene_event_at = float("-inf")
        self._panel_lost = False
        self._commands: queue.Queue = queue.Queue()
        self._thread = threading.get_ident()  # 身体线程；run() 里会更新成实际跑循环的线程

    # ---- 主循环 ----
    def step(self) -> None:
        now = self.clock()
        frame = None
        try:
            frame = self.device.screenshot()
        except Exception as exc:  # 实测 adb 截图偶尔会连续失败几秒
            self.events.put("error", f"截图失败：{str(exc).splitlines()[0]}")
        if frame is not None:
            self.last_frame = frame
            fresh = self.reader.read(frame, now)
            if self.cfg.vision.mode == "log":
                self.panel.maybe_reopen(now)
                self._watch_panel(now)
            self._watch_screen(frame, now)
            if self.env is not None:
                self.env.observe(frame, now, panel_visible=self.reader.panel_closed_since is None)
                self._watch_people(now)
            self._heard(fresh, frame, now)
        self._run_commands()

    def run(self, duration: float = 0.0, stop: threading.Event | None = None) -> None:
        """一直跑；duration > 0 时跑这么多秒后自己退出（别在外面套 timeout）。stop 被设置时也退出。"""
        self._thread = threading.get_ident()
        mode = "dry-run（只打印不执行）" if self.cfg.reply.dry_run else "LIVE（会真的说话、做动作）"
        log.info("身体启动，模式：%s；Ctrl+C 退出%s", mode, f"；{duration:.0f} 秒后自动结束" if duration > 0 else "")
        self.panel.ensure_open()
        deadline = self.clock() + duration if duration > 0 else float("inf")
        while self.clock() < deadline and not (stop is not None and stop.is_set()):
            started = self.clock()
            try:
                self.step()
            except Exception:
                log.exception("本轮出错，继续")
                self.events.put("error", "身体这一圈出错了（详见日志）")
            self.sleep(max(0.0, self.cfg.vision.poll_interval - (self.clock() - started)))

    def shutdown(self) -> None:
        """退出时：镜头转回原位、轮盘换回去。"""
        if self.camera is not None and not self.cfg.reply.dry_run:
            try:
                log.info(self.camera.reset())
            except Exception:
                log.exception("镜头没转回原位")
        if self.emotes is not None:
            try:
                self.emotes.restore()
            except Exception:
                log.exception("恢复轮盘失败，请用 emotes wheel 检查")

    # ---- 命令队列：大脑的工具在身体线程里执行 ----
    def call(self, fn: Callable[[], Any], timeout: float | None = None) -> Any:
        if threading.get_ident() == self._thread:
            return fn()
        fut: Future = Future()
        self._commands.put((fn, fut))
        try:
            return fut.result(timeout=timeout if timeout is not None else self.cfg.brain.command_timeout)
        except TimeoutError:
            fut.cancel()  # 身体还没开始做就不做了
            raise ToolError(f"身体忙不过来，这个命令超时了（{self.cfg.brain.command_timeout:.0f} 秒）") from None

    def _run_commands(self) -> None:
        while True:
            try:
                fn, fut = self._commands.get_nowait()
            except queue.Empty:
                return
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:
                fut.set_exception(exc)

    # ---- 看到了什么 → 事件 ----
    def _heard(self, fresh: list[Message], frame, now: float) -> None:
        if not fresh:
            return
        for m in fresh:
            log.info("读到: %s", f"{m.speaker}：{m.text}" if m.speaker else m.text)
            self.chat.append((self.wall(), m.speaker, m.text))
        self.heard = (self.heard + fresh)[-20:]
        if self.run_dir is not None:
            self.run_dir.save_frame(frame, [m.box for m in fresh])
        for m in fresh:
            self.events.put("chat", f"聊天  {m.speaker or '（看不出是谁）'}：{m.text}")

    def _watch_panel(self, now: float) -> None:
        since = self.reader.panel_closed_since
        lost = since is not None and now - since >= PANEL_LOST_AFTER
        if lost and not self._panel_lost:
            self.events.put("panel", "聊天记录面板关了半分钟，自动重开没成功，现在读不到聊天")
        elif self._panel_lost and since is None:
            self.events.put("panel", "聊天记录面板又开了")
        self._panel_lost = lost

    def _watch_screen(self, frame, now: float) -> None:
        black = is_black(frame)
        if black != self.blackout:
            self.blackout = black
            self.events.put("scene_change", "画面整屏黑了（可能在切场景）" if black else "画面恢复了")
            self._ref_thumb = None
        if black or now - self._ref_at < 1.0:
            return
        t = thumb(frame)
        if (
            self._ref_thumb is not None
            and difference(self._ref_thumb, t) > self.cfg.brain.scene_change
            and now - self._scene_event_at >= SCENE_EVENT_COOLDOWN
        ):
            self.events.put("scene_change", "画面变化很大（换了地方或者镜头动了）")
            self._scene_event_at = now
        self._ref_thumb, self._ref_at = t, now

    def _watch_people(self, now: float) -> None:
        near = set(self.env.nearby(now))
        for name in sorted(near - self._nearby):
            self.events.put("arrive", f"{name} 来到身边")
        for name in sorted(self._nearby - near):
            self.events.put("leave", f"{name} 走开了（{self.cfg.env.keep:.0f} 秒没看到名字）")
        self._nearby = near

        requests = dict(self.env.requests)
        current = {(r.name, r.kind) for r in requests.values()}
        for name, kind in sorted(current - self._requests):
            self.events.put("request", f"{name} 发起了{KIND_NAMES.get(kind, kind)}")
        self._requests = current
        if self.social is not None and requests:
            try:
                handled = self.social.handle(requests, now)
            except Exception:
                log.exception("处理互动请求出错")
                handled = []
            for item in handled:
                name, kind = item.split(":", 1)
                self.events.put("accepted", f"身体按规则接受了 {name} 的{KIND_NAMES.get(kind, kind)}")
                if kind == "hand":
                    self._accepted_hand = (name, now)
        self._watch_holding(now)

    def _watch_holding(self, now: float) -> None:
        """牵手状态靠猜：接受牵手后对方头顶圆圈消失 → 牵上了；✦ 又出现 → 松开了（game-ops §6）。"""
        circles = self.env.circles
        if self.holding is None and self._accepted_hand is not None:
            name, t = self._accepted_hand
            kind, seen = circles.get(name, (IDLE, float("-inf")))
            if seen > t and kind is None:
                self.holding, self._holding_since, self._accepted_hand = name, now, None
                self.events.put("holding", f"（推测）牵上了 {name} 的手")
            elif now - t > 30.0:
                self._accepted_hand = None
        elif self.holding is not None:
            kind, seen = circles.get(self.holding, (None, float("-inf")))
            if seen > self._holding_since and kind == IDLE:
                self.events.put("released", f"（推测）和 {self.holding} 松手了")
                self.holding = None
