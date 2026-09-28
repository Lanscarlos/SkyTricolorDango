"""主循环：截屏 → 读新消息 → 攒一小会儿 → 生成回复 → 限速 → 发送。"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

from .chat.commands import CommandRouter, is_command
from .chat.memory import MemoryStore
from .chat.panel import PanelKeeper
from .chat.reader import ChatReader, Message, with_speaker_hint
from .chat.responder import Responder
from .chat.sender import ChatSender
from .chat.tracker import SelfFilter
from .config import Config
from .device.base import Device
from .brain.images import is_black
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

    def remaining(self, now: float) -> int:
        """这一分钟还能发几条（`#status` 用；只读，不改变限速状态）。"""
        while self._sent and now - self._sent[0] > 60:
            self._sent.popleft()
        return max(0, self.max_per_minute - len(self._sent))


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
        store: MemoryStore | None = None,  # 主人命令（#friend/#remember）用；跟 dry_run 无关，配了就写
        viewer=None,  # vision.viewer.Viewer：网页上实时显示识别框（run --view）
        camera=None,  # brain.camera.Camera：主人的 #spin 用
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
        self.viewer = viewer
        self.emoted: list[str] = []  # 做过（含 dry-run）的动作，方便测试和日志
        self.paused = False  # #pause/#resume 切换：暂停时忽略别人的消息，不进 pending
        self.camera = camera
        self._last_spin = float("-inf")
        self.commands = CommandRouter(
            store, on_pause=self._set_paused, status=self._status_line,
            spin=self._spin if camera is not None else None, max_turns=cfg.spin.max_turns,
        )
        # sleep 包一层：测试会在构造之后替换 agent.sleep
        self.panel = PanelKeeper(cfg.vision, device, reader, lambda s: self.sleep(s))

    def _show(self, frame, now: float, fresh: list[Message]) -> None:
        from .vision.viewer import panel_box

        info = {"模式": "dry-run" if self.cfg.reply.dry_run else "LIVE", "待回复": f"{len(self.pending)} 条",
                "刚说过": self.sent[-3:][::-1] or "还没说话"}
        try:
            self.viewer.update(frame, now, env=self.env, panel=panel_box(self.cfg.vision, self.reader, frame), messages=fresh, info=info)
        except Exception:
            log.debug("可视化更新出错", exc_info=True)

    def _held(self, reason: str) -> AbstractContextManager:
        """换轮盘、接互动时暂停感知计时（见 brain/body.py 同名方法）。"""
        held = getattr(self.env, "held", None)
        return held(reason) if held is not None else nullcontext()

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
            if self.viewer is not None:
                self._show(frame, now, fresh)
            if self.social is not None and self.env is not None and self.env.requests:
                try:
                    with self._held("social"):
                        self.social.handle(self.env.requests, now)
                except Exception:
                    log.exception("处理互动请求出错")
        commands: list[Message] = []
        owner = self.cfg.reply.owner_name
        if fresh and owner:
            commands = [m for m in fresh if m.speaker == owner and is_command(m.text)]
            if commands:
                skip = {id(m) for m in commands}
                fresh = [m for m in fresh if id(m) not in skip]

        if fresh:
            fresh = with_speaker_hint(self.env, fresh, now)
            for m in fresh:
                log.info("读到: %s", f"{m.speaker}：{m.text}" if m.speaker else m.text)
            if self.run_dir:
                self.run_dir.save_frame(frame, [m.box for m in fresh])
            if self.paused:
                for m in fresh:
                    log.info("已暂停，忽略: %s", f"{m.speaker}：{m.text}" if m.speaker else m.text)
            else:
                self.pending.extend(fresh)
                self.pending = self.pending[-self.cfg.chat.max_pending :]
                self.last_new_at = now

        for m in commands:
            self._handle_command(m)

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
            with self._held("wheel"):
                self.emotes.perform(name)
        except Exception:  # 换轮盘没找到图标、adb 出错……动作做不成，话照样说
            log.exception("做动作「%s」失败，跳过", name)

    def _handle_command(self, m: Message) -> None:
        """主人的 # 命令：独立处理，不占用 pending / RateLimiter，永远回一句固定确认语。"""
        confirm = self.commands.handle(m.text)
        text = self.cfg.reply.disclosure_prefix + confirm
        dry = self.cfg.reply.dry_run
        self.sent.append(text)
        if dry:
            log.info("[dry-run] 命令确认: %s", text)
        else:
            self.sender.send(text)
            self.self_filter.remember(text, self.clock())
        if self.run_dir:
            self.run_dir.record_reply([m], text, sent=not dry)

    def _spin(self, turns: int) -> str:
        """#spin：转 turns 圈、边转边截图存进运行目录（dry-run 也转：镜头只在自己屏幕上转），回一句转了几秒几张。"""
        spin = self.cfg.spin
        now = self.clock()
        if now - self._last_spin < spin.min_interval:
            return f"刚转过，等 {math.ceil(spin.min_interval - (now - self._last_spin))} 秒"
        try:
            if is_black(self.device.screenshot()):
                return "画面黑着，转不了"
            self._last_spin = now
            swept = None
            with self._held("camera"):
                result = self.camera.spin(self.device.screenshot, turns, spin.seconds_per_turn, spin.fps)
                if hasattr(self.env, "sweep"):  # 打开了感知层：顺便汇总一圈里哪个方向有谁
                    swept = self.env.sweep([(0.0, result.before), *result.frames], spin)
        except Exception as exc:
            log.exception("#spin 没转成")
            return "没转成：" + (str(exc).splitlines() or [type(exc).__name__])[0]
        if self.run_dir is not None:
            try:
                folder = self.run_dir.save_spin(result, turns, time.strftime("%H%M%S"))
                log.info("转圈截图存在 %s", folder)
            except OSError:
                log.exception("转圈截图没存成")
        text = f"转完了，{result.seconds:.1f} 秒 {len(result.frames)} 张"
        if result.blackout:
            text += "（中途画面黑了）"
        if not result.panel_reopened:
            text += "（聊天面板没打开，要手动按 C）"
        if swept is not None:
            text += "；" + swept.text()
        return text[: self.cfg.reply.max_chars]

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused

    def _status_line(self) -> str:
        mode = "dry-run" if self.cfg.reply.dry_run else "live"
        state = "已暂停" if self.paused else "运行中"
        remaining = self.limiter.remaining(self.clock())
        return f"{mode}｜{state}｜待处理{len(self.pending)}｜限速{remaining}/{self.limiter.max_per_minute}"

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
