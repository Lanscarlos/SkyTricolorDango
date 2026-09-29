"""大脑的身体：截图、读聊天、认人、按规则秒接互动请求，把发生的事变成事件交给大脑。

大脑的工具通过命令队列在这个线程里执行 —— 只有身体线程碰设备，说话、做动作、转视角不会和接受请求撞在一起。
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from typing import Any

from ..agent import RateLimiter
from ..chat.memory import Turn
from ..chat.panel import PanelManager
from ..chat.reader import Message, with_speaker_hint
from ..chat.responder import clean_reply, format_incoming
from ..chat.tracker import similar
from ..config import Config
from ..game.social import IDLE, KIND_NAMES, PASSIVE
from ..imageio import imwrite
from ..vision.bubbles import Rect, roi_rect
from ..vision.panels import UNKNOWN, Button, PanelReading, describe_reading
from ..vision.people import describe_people
from .camera import MAX_STEPS as CAMERA_MAX_STEPS
from .events import EventQueue
from .occasion import LEVEL_NAMES, Occasion, Spoken, assess, is_friend_fn
from .images import crop_view, difference, fit, image_block, is_black, label_note, scene_note, thumb
from .locomotion import KEYS as MOVE_KEYS, MAX_STEPS as MOVE_MAX_STEPS
from .skills import SkillRunner

log = logging.getLogger(__name__)

REQUEST_KINDS = ("hand", "hug", "highfive", "piggyback", "candle", "*")
PANEL_LOST_AFTER = 30.0  # 面板关了这么久（自动重开也没成功）就告诉大脑
SCENE_EVENT_COOLDOWN = 10.0
BUTTON_NOTES = {"retreat": "可以按", "allow": "可以按", "other": "要主人放行", "never": "不能按"}
OWNER_NOTE = "（主人命令模式）"  # 真用到了主人命令窗口的放宽，结果后面标上（brain.jsonl 里看得出来）
PANEL_FOR_HOLD = {"wheel": "wheel_editor", "friend_tree": "friend_tree"}  # 身体自己打开面板的操作：期间这个面板不算遮挡


def _first_line(exc: BaseException) -> str:
    return (str(exc).splitlines() or [type(exc).__name__])[0]


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
        locomotion=None,  # brain.locomotion.Locomotion：W/A/S/D 小步走
        friend_checker=None,  # game.friendtree.FriendChecker：点人物看好友树
        panels=None,  # vision.panels.PanelWatcher：画面上开着哪些面板
        panel_ops=None,  # game.panels.PanelOps：关面板、按按钮
        viewer=None,  # vision.viewer.Viewer：网页上实时显示识别框
        fallback=None,  # chat.responder.Responder：大脑离线时的备用回复
        store=None,  # chat.memory.MemoryStore：live 时记聊天记录
        notes=None,  # chat.memory.NotesKeeper
        run=None,  # runlog.RunDir
        panel=None,  # chat.panel.PanelManager：cli 建一个、和镜头 / 轮盘 / 互动共用；None = 自己建
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
        self.locomotion = locomotion
        self.friend_checker = friend_checker
        self.panels = panels
        self.panel_ops = panel_ops
        self._panels_held = False
        self._permits: list[tuple[str, float]] = []  # 主人 #允许 的按钮：(文字, 到期时间)
        self._buttons: list[tuple[PanelReading, Button]] = []  # 上次 panel_read 编了号的按钮
        self._read_at = float("-inf")
        self.viewer = viewer
        self._last_friend_check = float("-inf")
        self.fallback = fallback
        self.store = store
        self.notes = notes
        self.run_dir = run
        self.clock = clock
        self.sleep = sleep
        self.wall = wall
        self.panel = panel if panel is not None else PanelManager(cfg.vision, cfg.panel, device, reader, lambda s: self.sleep(s), clock)
        self.brain_offline: Callable[[float], bool] = lambda now: False
        self.brain_busy: Callable[[], bool] = lambda: False  # 大脑正在回聊天（取走了聊天的那一轮还没结束）：聊天面板别因为安静关掉
        self.chat: deque[tuple[float, str, str]] = deque(maxlen=50)  # (时间, 说话人, 内容)；自己说的说话人是“我”
        self.heard: list[Message] = []  # 上次说话以后听到的，记聊天记录用
        self.said: list[str] = []  # 说过（含 dry-run）的话
        self.spoken: deque[Spoken] = deque(maxlen=50)  # 大脑说过的话（含 dry-run）和是不是主动开口；手动控制说的不记
        self.friend_names: Callable[[], list[str]] = lambda: []  # 好友名单（场合里认聊天的说话人），cli 设
        self._news: queue.Queue[str] = queue.Queue()  # 眼睛线程交来的新鲜事，身体线程里过滤后变成 notice
        self._place_seen = ""  # 上次认出的（非空）地名：换了地方就发 notice
        self._notice_at = float("-inf")
        self._notice_last = ""
        self.emoted: list[str] = []
        self.last_frame = None
        self.last_look = float("-inf")
        self.look_frame = None  # 最近一次 look（原图）看的那张，look_at 裁它
        self.blackout = False
        self.holding: str | None = None  # 推测正牵着谁的手
        self._holding_since = 0.0
        self._accepted_hand: tuple[str, float] | None = None
        self._owner_window_until = float("-inf")  # now < 这个值 = 卡洛的 # 命令还在生效
        self._last_move = float("-inf")
        self._nearby: set[str] = set()
        self._strangers = 0  # 上次看到几个陌生人（YOLO 感知层才有）
        self._requests: set[tuple[str, str]] = set()
        self._ref_thumb = None
        self._ref_at = float("-inf")
        self._scene_event_at = float("-inf")
        self._panel_lost = False
        self._commands: queue.Queue = queue.Queue()
        self._thread = threading.get_ident()  # 身体线程；run() 里会更新成实际跑循环的线程
        self.limiter = RateLimiter(cfg.reply.min_interval, cfg.reply.max_per_minute)
        self._fallback_pending: list[Message] = []
        self._fallback_last_new = 0.0
        self.stopped = False  # shutdown 之后不再接大脑的命令
        self.skills = SkillRunner(events, clock, panel=self.panel)  # 大脑交代的事（盯人、走过去……），跟着主循环一圈圈做

    # ---- 主循环 ----
    @contextmanager
    def _held(self, reason: str):
        """会挡住画面的操作期间暂停感知计时（YOLO 感知层 keep 只有 5 s，不暂停会误报"走开了"）；
        身体自己要打开面板的操作（换轮盘、看好友树），期间这个面板不算遮挡。"""
        with ExitStack() as stack:
            held = getattr(self.env, "held", None)
            stack.enter_context(held(reason) if held is not None else nullcontext())
            if self.panels is not None:
                stack.enter_context(self.panels.expect(PANEL_FOR_HOLD.get(reason)))
            yield

    def step(self) -> None:
        now = self.clock()
        if self.events.has("chat") or self.events.has("owner_command") or self.brain_busy():
            self.panel.busy(now)  # 还有没回的话 / 大脑在想：聊天面板不算安静
        frame = None
        try:
            frame = self.device.screenshot()
        except Exception as exc:  # 实测 adb 截图偶尔会连续失败几秒
            self.events.put("error", f"截图失败：{_first_line(exc)}")
        if frame is not None:
            self.last_frame = frame
            fresh: list[Message] = []
            try:
                fresh = self.reader.read(frame, now)
            except Exception:
                log.exception("读聊天出错")
            try:
                self._sense(frame, now, fresh)
            except Exception:
                log.exception("感知出错，这一圈跳过")
                self.events.put("error", "身体感知出错了（详见日志）")
            try:
                self.skills.tick(self, frame, now)
            except Exception:
                log.exception("技能这一圈出错")
            self._heard(fresh, frame, now)
            if self.viewer is not None:
                self._show(frame, now, fresh)
        self._run_commands()
        self._fallback(now)

    def _show(self, frame, now: float, fresh: list[Message]) -> None:
        from ..vision.viewer import panel_box

        info: dict = {"模式": "dry-run" if self.cfg.reply.dry_run else "LIVE", "牵着手": f"{self.holding}（推测）" if self.holding else "没有"}
        if self.blackout:
            info["画面"] = "黑着（切场景？）"
        panel = self.panel.describe(now)
        if panel:
            info["聊天面板"] = panel
        info["正在做"] = self.skills.describe(now).removeprefix("正在做：")
        info["刚说过"] = self.said[-3:][::-1] or "还没说话"
        if self.cfg.proactive.enabled:
            o = self.occasion()
            info["场合"] = f"{LEVEL_NAMES[o.level]} · " + ("不主动" if o.blocked else f"还能主动说 {o.left} 句")
        info["最近事件"] = [e.line() for e in self.events.recent(6)][::-1] or "还没有"
        try:
            self.viewer.update(
                frame, now, env=self.env, panel=panel_box(self.cfg.vision, self.reader, frame), messages=fresh, info=info,
                panels=self.panels,
            )
        except Exception:
            log.debug("可视化更新出错", exc_info=True)

    def _sense(self, frame, now: float, fresh: list[Message]) -> None:
        if self.cfg.vision.mode == "log":
            self.panel.tick(now, fresh, visible=self.reader.panel_closed_since is None, blackout=self.blackout)
            self._watch_panel(now)
        self._watch_screen(frame, now)
        if self.panels is not None:
            self._watch_panels(frame, now)
        if self.env is not None:
            self.env.observe(frame, now, panel_visible=self.reader.panel_closed_since is None)
            self._watch_people(now)
        if self.cfg.proactive.enabled:
            self._watch_news(now)

    def run(self, duration: float = 0.0, stop: threading.Event | None = None) -> None:
        """一直跑；duration > 0 时跑这么多秒后自己退出（别在外面套 timeout）。stop 被设置时也退出。"""
        self._thread = threading.get_ident()
        mode = "dry-run（只打印不执行）" if self.cfg.reply.dry_run else "LIVE（会真的说话、做动作）"
        log.info("身体启动，模式：%s；Ctrl+C 退出%s", mode, f"；{duration:.0f} 秒后自动结束" if duration > 0 else "")
        self.panel.start(self.clock())
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
        """退出时（不等大脑）：不再接命令、排队的命令全部失败、镜头转回原位、轮盘换回去。"""
        self.stopped = True
        try:  # 先停技能（松开按着的键），再复原镜头
            self.skills.cancel(self, "身体停了")
        except Exception:
            log.exception("停技能出错")
        while True:
            try:
                _, fut = self._commands.get_nowait()
            except queue.Empty:
                break
            if fut.set_running_or_notify_cancel():
                fut.set_exception(ToolError("身体已经停了"))
        if self.camera is not None:  # dry-run 下手动（live）也可能转过：总是复原，没动过就什么都不做
            try:
                log.info(self.camera.reset())
            except Exception:
                log.exception("镜头没转回原位")
        if self.emotes is not None:
            try:
                self.emotes.restore()
            except Exception:
                log.exception("恢复轮盘失败，请用 emotes wheel 检查")
        try:
            self.panel.shutdown()  # 按需模式：退出时把聊天面板恢复成开着
        except Exception:
            log.exception("聊天面板没恢复")

    # ---- 命令队列：大脑的工具在身体线程里执行 ----
    def call(self, fn: Callable[[], Any], timeout: float | None = None) -> Any:
        if self.stopped:
            raise ToolError("身体已经停了")
        if threading.get_ident() == self._thread:
            return fn()
        fut: Future = Future()
        self._commands.put((fn, fut))
        used = timeout if timeout is not None else self.cfg.brain.command_timeout
        try:
            return fut.result(timeout=used)
        except TimeoutError:
            if not fut.cancel():  # cancel() 返回 False：身体已经开始做了，取消不掉
                raise ToolError("身体还在做这件事（已经开始了），别重试，等下次醒来看结果") from None
            raise ToolError(f"身体忙不过来，这个命令超时了（{used:.0f} 秒）") from None

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
            except Exception as exc:
                fut.set_exception(exc)
            except BaseException as exc:  # KeyboardInterrupt 等：Future 记下异常，再往外抛，别吞掉 Ctrl+C
                fut.set_exception(exc)
                raise

    # ---- 看到了什么 → 事件 ----
    def _heard(self, fresh: list[Message], frame, now: float) -> None:
        if not fresh:
            return
        fresh = with_speaker_hint(self.env, fresh, now)
        for m in fresh:
            log.info("读到: %s", f"{m.speaker}：{m.text}" if m.speaker else m.text)
            self.chat.append((self.wall(), m.speaker, m.text))
        self.heard = (self.heard + fresh)[-20:]
        if self.run_dir is not None:
            self.run_dir.save_frame(frame, [m.box for m in fresh])
        if self.brain_offline(now):  # 大脑离线：交给备用回复，不排进大脑的事件
            self._fallback_pending.extend(fresh)
            self._fallback_last_new = now
            return
        owner = self.cfg.brain.owner_name
        for m in fresh:
            if owner and m.speaker == owner and m.text.startswith("#"):
                self._owner_window_until = now + self.cfg.brain.owner_window
                if m.text.startswith("#允许") and m.text[3:].strip():  # 放行一个面板按钮（panel_press 用）
                    self._permits.append((m.text[3:].strip(), now + self.cfg.panels.permit_window))
                self.events.put("owner_command", f"卡洛的命令：{m.text}")
                log.info("识别到卡洛的命令：%s（授权窗口延长到 %.0f 秒后）", m.text, self.cfg.brain.owner_window)
            else:
                self.events.put("chat", f"聊天  {m.speaker or '（看不出是谁）'}：「{m.text}」")

    def news(self, text: str) -> None:
        """眼睛自动看时挑出的新鲜事（眼睛线程调，只入队）。"""
        self._news.put(text)

    def _watch_news(self, now: float) -> None:
        """新鲜事 / 换了地图 → notice 事件；没熟人、主动额度用完、太勤、和上一条一样都不发（省得为此叫醒大脑）。"""
        found = []
        while True:
            try:
                found.append("眼睛注意到：" + self._news.get_nowait())
            except queue.Empty:
                break
        place = getattr(self.env, "place", "") or ""
        if place:
            if self._place_seen and place != self._place_seen:
                found.append(f"看起来到了{place}")
            self._place_seen = place
        if not found:
            return
        o = self.occasion()
        for text in found:
            if o.level == "alone" or o.left == 0:
                log.debug("新鲜事没发（%s）：%s", o.blocked or "主动额度用完了", text)
            elif now - self._notice_at < self.cfg.proactive.notice_min or similar(text, self._notice_last, 0.9):
                log.debug("新鲜事没发（太勤或重复）：%s", text)
            else:
                self.events.put("notice", text)
                self._notice_at, self._notice_last = now, text

    def _watch_panel(self, now: float) -> None:
        """只在面板该开着（常开模式 / 聊天中）却没开时告诉大脑；按需模式闲着时关着是正常的。"""
        lost = self.panel.missing_for(now) >= PANEL_LOST_AFTER
        if lost and not self._panel_lost:
            self.events.put("panel", "聊天记录面板关了半分钟，自动重开没成功，现在读不到聊天")
        elif self._panel_lost and self.reader.panel_closed_since is None:
            self.events.put("panel", "聊天记录面板又开了")
        self._panel_lost = lost

    def _watch_panels(self, frame, now: float) -> None:
        """画面上开着哪些面板：开 / 关告诉大脑（聊天记录面板不算，沿用 _watch_panel）；开着时暂停感知计时。"""
        state = self.panels.observe(frame, now)
        for change in self.panels.pop_changes():
            p = change.panel
            if p.name == "chat_log":
                continue
            if change.kind == "close":
                self.events.put("panel", f"关了：{p.label}")
            elif p.name == UNKNOWN:
                self.events.put("panel", "出现不认识的面板：" + (describe_reading(change.reading) if change.reading else "（还没读）"))
            elif change.reading is not None:
                self.events.put("panel", f"开了：{p.describe()}：{describe_reading(change.reading)}")
            else:
                self.events.put("panel", f"开了：{p.describe()}")
        covered = bool(state.others())
        if covered != self._panels_held and self.env is not None and hasattr(self.env, "hold"):
            (self.env.hold if covered else self.env.release)("panel")
        self._panels_held = covered

    def clear_view(self, action: str, live: bool = False) -> str:
        """操作前确认画面没被面板挡着：已核对、卡片允许自动关的顺手关掉；别的拒绝（ToolError），交给大脑。
        返回 dry-run 时要附加的说明（"真执行时会先关掉……"）。"""
        if self.panels is None:
            return ""
        notes = []
        for panel in self.panels.blocking(action):
            card = self.panels.cards.get(panel.name) if panel.name != UNKNOWN else None
            if not (panel.verified and card is not None and card.close_auto and self.panel_ops is not None):
                raise ToolError(f"被「{panel.describe()}」挡着：可以 panel_read 看看，或者 panel_close 关掉")
            if self._dry(live):
                notes.append(f"（真执行时会先关掉{panel.label}）")
                continue
            if not self.panel_ops.close(panel, self.panels.readings.get(panel.name)):
                self.events.put("error", f"{panel.label}挡着，想关没关上")
                raise ToolError(f"被「{panel.describe()}」挡着，想关没关上：可以 panel_read 看看")
            self.events.put("panel", f"顺手关掉了{panel.label}")
        return "".join(notes)

    def _watch_screen(self, frame, now: float) -> None:
        black = is_black(frame)
        if black != self.blackout:
            self.blackout = black
            if self.env is not None and hasattr(self.env, "hold"):  # 黑屏期间看不到人，别算成"走开了"
                (self.env.hold if black else self.env.release)("blackout")
            self.events.put("scene_change", "画面整屏黑了（可能在切场景）" if black else "画面恢复了")
            self._ref_thumb = None
            if not black:  # 黑屏时没看聊天：恢复后补看一眼
                self.panel.trigger("scene", now)
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
        if near - self._nearby:  # 来人常常会打招呼：看一眼聊天
            self.panel.trigger("arrive", now)
        for name in sorted(near - self._nearby):
            self.events.put("arrive", f"{name} 来到身边")
        for name in sorted(self._nearby - near):
            keep = getattr(self.env, "keep", self.cfg.env.keep)
            self.events.put("leave", f"{name} 走开了（{keep:.0f} 秒没看到名字）")
            if name == self.holding:  # 人都走开了，肯定没牵着了
                self.events.put("released", f"（推测）和 {name} 分开了")
                self.holding = None
        self._nearby = near
        if hasattr(self.env, "strangers"):  # 只有 YOLO 感知层认得出陌生人
            n = self.env.strangers(now)
            if n and not self._strangers:
                dark = self.env.unlit(now) if hasattr(self.env, "unlit") else 0
                note = f"，其中 {dark} 个还没点火" if dark else ""
                self.events.put("stranger", f"身边来了陌生人（{n} 个，头顶没有名字{note}）")
            elif not n and self._strangers:
                self.events.put("stranger", "陌生人都走开了")
            self._strangers = n
        if hasattr(self.env, "typing_seen") and self.env.typing_seen(now, strangers=self.cfg.panel.bubble_strangers):
            self.panel.bubble_seen(now)  # 好友头顶冒出"正在输入"：开着面板等他发出来
        if hasattr(self.env, "pop_approaches"):  # 有人朝团子走过来（眼睛不因此自动看，省额度）
            for who in self.env.pop_approaches():
                self.panel.trigger("approach", now)
                if who == self.holding:
                    continue
                self.events.put("approach", "有个陌生人朝你走过来了" if who == "陌生人" else f"{who} 朝你走过来了")
        if hasattr(self.env, "pop_gestures"):  # 好友对团子做了动作（三期 §3）：大脑决定回不回礼
            names = self.cfg.gesture.names
            for who, label in self.env.pop_gestures():
                if who == self.holding:
                    continue
                self.events.put("gesture", f"{who}对你{names.get(label, label)}")

        requests = dict(self.env.requests)
        current = {(r.name, r.kind) for r in requests.values()}
        for name, kind in sorted(current - self._requests):
            self.events.put("request", f"{name} 发起了{KIND_NAMES.get(kind, kind)}")
        self._requests = current
        if self.social is not None and requests and self._social_view_clear():
            try:
                with self._held("social"):
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

    def _social_view_clear(self) -> bool:
        try:
            self.clear_view("social", live=not self.cfg.reply.dry_run)
        except ToolError as exc:
            log.debug("互动请求这次先不接：%s", exc)
            return False
        return True

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
            if seen > self._holding_since and kind in PASSIVE:  # ✦ 或眼睛之类的状态图标又出现了
                self.events.put("released", f"（推测）和 {self.holding} 松手了")
                self.holding = None

    # ---- 给大脑用的：都在身体线程里执行（经 call()） ----
    def look(self) -> list[dict]:
        now = self.clock()
        brain = self.cfg.brain
        if now - self.last_look < brain.look_min_interval:
            raise ToolError(f"{brain.look_min_interval:.0f} 秒内刚看过，等一下再看")
        frame = self.device.screenshot()
        self.last_frame, self.last_look, self.look_frame = frame, now, frame
        view = fit(frame, tuple(brain.image_size))
        scale = view.shape[1] / frame.shape[1]
        if hasattr(self.env, "strangers"):  # YOLO 感知层：好友、陌生人、团子的位置都给
            note = scene_note(self.env, now, scale)
        elif self.env is not None:  # 最近两次扫描内看到的名字才算在画面里；先拍快照，env 后台线程会改这个 dict
            recent = {n: v for n, v in dict(self.env.labels).items() if now - v[4] <= self.cfg.env.interval * 2 + 1}
            note = label_note(recent, scale)
        else:
            note = "（没开环境识别，认不出名字）"
        if self.blackout:
            note += "\n画面现在是黑的（可能在切场景）"
        return [image_block(view, brain.jpeg_quality), {"type": "text", "text": note}]

    def look_at(self, x: int, y: int, w: int, h: int) -> list[dict]:
        brain = self.cfg.brain
        # 大脑给的坐标是按上次 look 那张图算的；人会走动，裁新截的图会对不上
        frame = self.look_frame if self.look_frame is not None else self.device.screenshot()
        try:
            crop, (ax, ay, aw, ah) = crop_view(frame, x, y, w, h, brain.image_size[0], brain.look_at_max)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        text = f"原图 {frame.shape[1]}×{frame.shape[0]} 上的 ({ax}, {ay}) 起 {aw}×{ah}"
        return [image_block(crop, brain.jpeg_quality), {"type": "text", "text": text}]

    def _locate(self, name: str, now: float) -> tuple[Rect, bool] | None:
        """(整张截图上的人物框, 是不是按名字标签估的)；感知层框出的人优先。

        name 常常是聊天面板上 OCR 读出来的，可能差一个字：先找一模一样的，再按相似度找（同感知层认名字）。
        """
        if self.env is None:
            return None
        people = [p for p in self.env.people(now) if p.name] if hasattr(self.env, "people") else []
        labels = {n: v for n, v in dict(self.env.labels).items()  # env 后台线程会改这个 dict：先拍快照
                  if now - v[4] <= self.cfg.env.interval * 2 + 1}
        for same in (lambda n: n == name, lambda n: similar(name, n, 0.75)):
            for p in people:
                if same(p.name):
                    return p.box, False
            for n, label in labels.items():
                if same(n):
                    return self._below_tag(label[:4]), True
        return None

    def _below_tag(self, tag: tuple[int, int, int, int]) -> Rect:
        """人在名字标签正下方：宽 3 倍标签宽、高 6 倍标签高（估计值，没在真机核对）。"""
        x, y, w, h = tag
        fh, fw = self.last_frame.shape[:2] if self.last_frame is not None else (1080, 1920)
        x1, y1 = max(0, round(x + w / 2 - 1.5 * w)), min(fh, y + h)
        x2, y2 = min(fw, round(x + w / 2 + 1.5 * w)), min(fh, y + h + 6 * h)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def _recognized(self, now: float) -> list[str]:
        """现在画面里认得出名字的人（找不到某人时告诉大脑）。"""
        names = [p.name for p in self.env.people(now) if p.name] if hasattr(self.env, "people") else []
        names += [n for n, v in dict(self.env.labels).items() if now - v[4] <= self.cfg.env.interval * 2 + 1]
        return list(dict.fromkeys(names))

    def find_person(self, name: str, now: float) -> Rect | None:
        found = self._locate(name, now)
        return found[0] if found else None

    def look_person(self, name: str) -> list[dict]:
        """按名字找到这个人，把他裁出来给大脑看（和 look 共用频率限制）。"""
        now = self.clock()
        brain = self.cfg.brain
        if now - self.last_look < brain.look_min_interval:
            raise ToolError(f"{brain.look_min_interval:.0f} 秒内刚看过，等一下再看")
        if self.env is None:
            raise ToolError("没开环境识别，认不出名字；用 look(image=true) 自己看")
        found = self._locate(name, now)
        if found is None or found[0].w < 8 or found[0].h < 8:
            known = self._recognized(now)
            where = f"画面里现在认得出：{'、'.join(known)}" if known else "画面里现在一个名字都没认出来"
            raise ToolError(f"没找到 {name}（{where}）；不在画面里的话可以先 look_around 看看在哪个方向")
        box, guessed = found
        bx, by, bw, bh = box.x, box.y, box.w, box.h
        frame = self.device.screenshot()
        self.last_frame, self.last_look = frame, now
        fh, fw = frame.shape[:2]
        mx, my = round(bw * 0.2), round(bh * 0.2)  # 四周各放宽 20%：人会动、框也不一定贴身
        x1, y1, x2, y2 = max(0, bx - mx), max(0, by - my), min(fw, bx + bw + mx), min(fh, by + bh + my)
        crop = fit(frame[y1:y2, x1:x2], tuple(brain.image_size))
        text = f"这是 {name}（原图 ({x1}, {y1}) 起 {x2 - x1}×{y2 - y1}）"
        if guessed:
            text += "；按名字标签估的位置，可能没框全"
        return [image_block(crop, brain.jpeg_quality), {"type": "text", "text": text}]

    def fresh_frame(self):
        """眼睛马上要看：在身体线程里截一张新的。"""
        frame = self.device.screenshot()
        self.last_frame = frame
        return frame

    def capture_around(self, live: bool = False) -> list:
        """环顾四周：转一圈，每 90° 截一张（dry-run 不转，只截当前画面；live = 手动控制，dry-run 下也真转）。"""
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在看不了")
        self.clear_view("camera", live)
        if self.camera is None or self._dry(live):
            return [self.fresh_frame()]
        try:
            with self._held("camera"):
                frames = self.camera.around(self.device.screenshot)
        finally:
            self._ref_thumb = None  # 自己转的镜头，不算画面大变
        self.last_frame = frames[0]
        return frames

    def sweep_around(self, live: bool = False) -> str:
        """打开感知层时的环顾：连续转一圈，YOLO 汇总每个方向有谁（dry-run 不转，只看当前画面；live 见 capture_around）。"""
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在看不了")
        spin = self.cfg.spin
        note = self.clear_view("camera", live)
        if self.camera is None or self._dry(live):
            result = self.env.sweep([(0.0, self.fresh_frame())], spin)
            return "dry-run：没真的转，只看了前面。" + result.text() + note
        try:
            with self._held("camera"):
                shot = self.camera.spin(self.device.screenshot, 1, spin.seconds_per_turn, spin.fps)
                result = self.env.sweep([(0.0, shot.before), *shot.frames], spin)
        finally:
            self._ref_thumb = None  # 自己转的镜头，不算画面大变
        self.last_frame = shot.after
        if not shot.panel_reopened:
            self.events.put("panel", "转完一圈，聊天记录面板没重新打开")
        return result.text() + ("（中途画面黑了，可能在切场景，这一圈不准）" if shot.blackout else "")

    def status(self) -> str:
        now = self.clock()
        parts = ["聊天记录面板" + ("开" if self.reader.panel_closed_since is None else "关")]
        try:
            parts.append("输入框" + ("开" if self.device.ime_shown() else "关"))
        except Exception:
            parts.append("输入框状态读不到")
        others = self.panels.state.others() if self.panels is not None else ()
        if others:
            parts.append("开着的面板：" + "、".join(p.describe() for p in others))
        near = self.env.nearby(now) if self.env is not None else []
        parts.append("身边的好友：" + ("、".join(near) if near else "没看到"))
        if hasattr(self.env, "strangers"):
            parts.append(f"身边的陌生人：{self.env.strangers(now)} 个")
        closest = self.env.nearest(now) if hasattr(self.env, "nearest") else None
        if closest:
            parts.append(f"离你最近的：{closest[0]}（{closest[1]}）")
        people = describe_people(self.env.people(now)) if hasattr(self.env, "people") else ""
        if people:
            parts.append("画面里：" + people)
        if self.holding:
            parts.append(f"牵着手：{self.holding}（推测）")
        if self.blackout:
            parts.append("画面黑着")
        if self.camera is not None:
            parts.append("镜头：" + self.camera.describe())
        parts.append("上次看图：" + (f"{now - self.last_look:.0f} 秒前" if self.last_look > float("-inf") else "还没看过"))
        if self.emotes is not None:
            parts.append("能做的动作：" + ("、".join(self.emotes.available(self._owner(now))) or "暂时没有（刚做过，要等一会儿）"))
        if self.social is not None:
            parts.append("互动规则：" + self.social.describe_policy())
        if self.said:
            parts.append("刚说过：" + " | ".join(self.said[-3:]))
        if self.cfg.proactive.enabled:
            parts.append("场合：" + self.occasion().line(self.wall()))
        parts.append(self.skills.describe(now))
        if self.cfg.reply.dry_run:
            parts.append("dry-run（说话、动作、转视角、走动都不会真的执行）")
        return " / ".join(parts)

    def occasion(self) -> Occasion:
        """现在是什么场合（热闹 / 安静 / 没熟人）、上次主动开口有没有人接、还能不能主动开口。"""
        now = self.clock()
        friends = self.env.nearby(now) if self.env is not None else []
        strangers = self.env.strangers(now) if hasattr(self.env, "strangers") else 0
        return assess(
            self.cfg.proactive, self.wall(), friends, strangers, list(self.chat), list(self.spoken),
            is_friend_fn(self.friend_names()),
        )

    def chat_log(self, n: int = 20) -> str:
        if self.panel.auto and self.panel.state == "idle":
            self._peek_now()
        rows = list(self.chat)[-max(1, min(n, 50)) :]
        if not rows:
            return "还没有聊天"
        return "\n".join(
            f"{time.strftime('%H:%M:%S', time.localtime(t))} {who or '（看不出是谁）'}：{text}" for t, who, text in rows
        )

    def _peek_now(self) -> None:
        """聊天面板关着（闲着）时大脑要看聊天：马上打开看一眼，读到的照常变成事件。冷却中 / 黑屏打不开就算了。"""
        self.panel.trigger("chat_log", self.clock())
        wait = self.panel.cfg.open_timeout + 1.0
        for _ in range(math.ceil(wait / max(self.cfg.vision.poll_interval, 0.05)) + 1):
            now = self.clock()
            frame = self.device.screenshot()
            fresh = self.reader.read(frame, now)
            self.panel.tick(now, fresh, visible=self.reader.panel_closed_since is None, blackout=self.blackout)
            self._heard(fresh, frame, now)
            if self.panel.state != "peek":  # 看完了（关上 / 读到新消息 / 等气泡），或者这会儿开不了
                return
            self.sleep(self.cfg.vision.poll_interval)

    def say(self, text: str, live: bool = False, reply: bool = False) -> str:
        """live = 手动控制：dry-run 下也真的发（护栏照旧）。reply = 明确是接话（大脑离线时的备用回复），不算主动开口。"""
        now = self.clock()
        body = clean_reply(text, self.cfg.reply.max_chars)  # 含“不能声称自己是真人”的硬过滤
        if body is None:
            raise ToolError("这句没发：是空的、<skip>，或者说了自己是真人（不能这么说）。换个说法")
        if not self.limiter.allow(now):
            raise ToolError("说得太快了，等几秒再说")
        proactive = self.cfg.proactive.enabled and not live and not reply and not self.brain_busy()
        if proactive:
            blocked = self.occasion().blocked
            if blocked:
                raise ToolError(blocked)
        note = self.clear_view("say", live)
        full = self.cfg.reply.disclosure_prefix + body
        self.limiter.record(now)
        self.said.append(full)
        self.said[:] = self.said[-50:]  # 只留最近 50 条，别无限长
        self.chat.append((self.wall(), "我", full))
        if not live:
            self.spoken.append(Spoken(self.wall(), full, proactive))
        if self._dry(live):
            log.info("[dry-run] 将会发送: %s", full)
            self._remember(body, full, sent=False)
            return f"dry-run：没真的发，“{full}”{note}"
        self.panel.before_speak(now)  # 先开面板再按 Enter：说完对方的回复马上读得到
        self.sender.send(full)
        self.self_filter.remember(full, self.clock())
        self._remember(body, full, sent=True, manual=live)
        return f"已发送：{full}"

    def _remember(self, body: str, full: str, sent: bool, manual: bool = False) -> None:
        if manual:  # 主人手动让团子说的：不是 AI 的回复——运行记录里标出来，不写聊天历史 / 记忆（模型会模仿 history），不拿走待回复的消息
            if self.run_dir is not None:
                self.run_dir.record_reply([], full, sent=sent, manual=True)
            return
        heard, self.heard = self.heard, []
        if self.run_dir is not None:
            self.run_dir.record_reply(heard, full, sent=sent)
        if not sent or self.store is None:  # dry-run 的话没真的说出去，不记
            return
        user = format_incoming(heard) if heard else "（没人说话，你主动开口）"
        now = self.wall()
        try:
            self.store.history.append(user, body, now)
        except OSError:
            log.exception("写聊天记录失败")
        if self.notes is not None:
            self.notes.turn_added(Turn(now, user, body))

    def emote(self, name: str, force: bool = False, live: bool = False) -> str:
        if self.emotes is None:
            raise ToolError("这次没开动作（--no-emotes 或者图标库是空的）")
        owner = self._owner(self.clock())
        available = self.emotes.available(owner)
        if name not in available:
            raise ToolError(f"「{name}」现在做不了；能做的：{'、'.join(available) or '暂时没有（刚做过动作，要等一会儿）'}")
        relaxed = owner and name not in self.emotes.available()  # 平时这会儿还在动作限速里
        if self.holding and not force:
            if not owner:
                raise ToolError(f"正牵着 {self.holding} 的手，做动作会松手；确定要松手再做就传 force=true")
            relaxed = True
        note = self.clear_view("emote", live)
        self.emoted.append(name)
        self.emoted[:] = self.emoted[-50:]  # 只留最近 50 条，别无限长
        if self._dry(live):
            self.emotes.pretend(name)
            return f"dry-run：没真的做「{name}」{note}"
        try:
            with self._held("wheel"):
                self.emotes.perform(name)
        except Exception as exc:
            raise ToolError(f"「{name}」没做成：{exc}") from None
        return f"做了「{name}」" + (OWNER_NOTE if relaxed else "")

    def set_policy(self, who: str, kind: str, accept: bool) -> str:
        if self.social is None:
            raise ToolError("这次没开互动请求处理（[social] 关了或者没有图标模板）")
        if kind not in REQUEST_KINDS:
            raise ToolError(f"不认识的请求类型 {kind}，可以用：{'、'.join(REQUEST_KINDS)}")
        who = who.strip() or "*"
        if who == "stranger" and accept and kind != "candle":
            raise ToolError("陌生人只能接点火，牵手 / 拥抱 / 击掌 / 背背都不接陌生人的")
        self.social.set_policy(who, kind, accept)
        return "现在的规则：" + self.social.describe_policy()

    def check_friend(self, x: int, y: int) -> list[dict] | str:
        """点一下人物（坐标按上次 look(image=true) 那张图）打开好友树面板，截图给大脑看，再关掉。"""
        fc = self.cfg.friend_check
        if self.friend_checker is None or not fc.enabled:
            raise ToolError("这次没开好友树确认（[friend_check] enabled = false：面板还没在真机上核对过）")
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在点不了")
        if self.holding:
            raise ToolError(f"正牵着 {self.holding} 的手，先别点人")
        now = self.clock()
        if self.look_frame is None or now - self.last_look > fc.max_look_age:
            raise ToolError(f"先 look(image=true) 看一眼现在的画面（{fc.max_look_age:.0f} 秒内），坐标按那张图给；人会走动，旧图对不上")
        fh, fw = self.look_frame.shape[:2]
        scale = fw / self.cfg.brain.image_size[0]
        sx, sy = round(x * scale), round(y * scale)
        if not (0 <= sx < fw and 0 <= sy < fh):
            raise ToolError("坐标在画面外（按 look(image=true) 那张 1280×720 的图给）")
        return self.check_friend_at(sx, sy)

    def check_friend_at(self, sx: int, sy: int, live: bool = False) -> list[dict] | str:
        """点原图 (sx, sy) 上的人打开好友树面板、截图、关掉（check_friend 换算好坐标后调；手动控制直接给原图坐标）。"""
        fc = self.cfg.friend_check
        if self.friend_checker is None or not fc.enabled:
            raise ToolError("这次没开好友树确认（[friend_check] enabled = false：面板还没在真机上核对过）")
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在点不了")
        if self.holding:
            raise ToolError(f"正牵着 {self.holding} 的手，先别点人")
        now = self.clock()
        if now - self._last_friend_check < fc.min_interval:
            raise ToolError(f"刚确认过，{fc.min_interval - (now - self._last_friend_check):.0f} 秒后再点")
        frame = self.look_frame if self.look_frame is not None else self.last_frame
        if frame is None:
            frame = self.fresh_frame()
        fh, fw = frame.shape[:2]
        if not (0 <= sx < fw and 0 <= sy < fh):
            raise ToolError("坐标在画面外")
        area = roi_rect(self.cfg.env.roi, fw, fh)
        if not (area.x <= sx < area.x2 and area.y <= sy < area.y2):
            raise ToolError("那里是底部的按钮栏，不是人")
        panel = roi_rect(self.cfg.vision.log_roi, fw, fh)
        if self.reader.panel_closed_since is None and panel.x <= sx < panel.x2 and panel.y <= sy < panel.y2:
            raise ToolError("那里被聊天记录面板挡着，点不到人")
        note = self.clear_view("check_friend", live)
        if self._dry(live):
            return f"dry-run：没真的点（会点原图上的 ({sx}, {sy})，打开好友树看完再关掉）{note}"
        self._last_friend_check = now
        try:
            with self._held("friend_tree"):
                result = self.friend_checker.check(sx, sy)
        except Exception as exc:
            raise ToolError(f"没点成：{_first_line(exc)}") from None
        finally:
            self._ref_thumb = None  # 自己点开的面板，不算画面大变
        self.last_frame = result.after
        if self.run_dir is not None:
            folder = self.run_dir.path / "friend-check"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H%M%S", time.localtime(self.wall()))
            for tag, img in (("before", result.before), ("opened", result.opened), ("after", result.after)):
                imwrite(folder / f"{stamp}-{tag}.jpg", img)
        brain = self.cfg.brain
        if not self.friend_checker.looks_open(result.changed):
            note = (f"点了原图上的 ({sx}, {sy})，右边没什么变化：可能没点中人（人走开了？），或者面板没出来。"
                    "图是点完之后的样子；要再试先重新 look(image=true)。")
            return [image_block(fit(result.opened, tuple(brain.image_size)), brain.jpeg_quality), {"type": "text", "text": note}]
        left = round(result.opened.shape[1] * self.friend_checker.cfg.panel_left)
        side = fit(result.opened[:, left:], (brain.look_at_max, brain.look_at_max))
        note = (f"点了原图上的 ({sx}, {sy})，右边打开了面板（第一张是整个画面，第二张是右边放大）。"
                "自己看图判断这个人是不是你的好友；看不出来就老实说不确定，别猜。")
        if result.closed:
            note += " 面板已经关上了。"
        else:
            note += " 注意：面板没关上，画面上还开着（试过 " + "、".join(self.friend_checker.cfg.close) + "）。"
            self.events.put("error", "好友树面板没关上，画面上还开着")
        return [
            image_block(fit(result.opened, tuple(brain.image_size)), brain.jpeg_quality),
            image_block(side, brain.jpeg_quality),
            {"type": "text", "text": note},
        ]

    # ---- 面板 ----
    def _need_panels(self) -> None:
        if self.panels is None or self.panel_ops is None:
            raise ToolError("这次没开面板识别（[panels] enabled = false）")

    def panel_read(self, image: bool = False) -> str | list[dict]:
        """读开着的面板（不含聊天记录面板）：名字、标题、正文、编了号的按钮和能不能按。"""
        self._need_panels()
        now = self.clock()
        others = self.panels.state.others()
        self._buttons, self._read_at = [], now
        if not others:
            return "没有开着的面板"
        frame = self.fresh_frame()
        blocks, crops = [], []
        for panel in others:
            reading = self.panels.read(frame, panel, now)
            numbered = []
            for button in reading.buttons:
                self._buttons.append((reading, button))
                numbered.append(f"[{len(self._buttons)}] {button.text}（{BUTTON_NOTES[button.kind]}）")
            blocks.append(
                f"{panel.describe()}\n  标题：{reading.title or '（没读到）'}\n  正文：{reading.text.replace(chr(10), ' ') or '（没有）'}\n"
                f"  按钮：{'  '.join(numbered) or '没认出按钮'}"
            )
            crops.append(panel.box.crop(frame))
        text = "\n\n".join(blocks)
        if not image:
            return text
        brain = self.cfg.brain
        return [{"type": "text", "text": text}] + [
            image_block(fit(crop, (brain.look_at_max, brain.look_at_max)), brain.jpeg_quality) for crop in crops
        ]

    def panel_press(self, button: str, live: bool = False) -> str:
        """按 panel_read 列出的按钮（编号或文字）；过安全规则：never 不按，其他（非撤退类）要主人 #允许。"""
        self._need_panels()
        now = self.clock()
        ttl = self.cfg.panels.read_ttl
        if now - self._read_at > ttl:
            raise ToolError(f"先 panel_read（{ttl:.0f} 秒内）再按：面板可能变了")
        picked = self._pick_button(button.strip())
        if picked is None:
            listed = "、".join(f"[{i}] {b.text}" for i, (_, b) in enumerate(self._buttons, 1)) or "没有"
            raise ToolError(f"没有这个按钮；现在的按钮：{listed}")
        reading, target = picked
        current = next((p for p in self.panels.state.panels if p.name == reading.panel.name), None)
        if current is None:
            raise ToolError(f"{reading.panel.label}已经关了，重新 panel_read 看看")
        # 按之前再读一次（画面没变时用缓存）：不认识的面板都叫 unknown，读完之后可能已经换了一个弹框
        reading = self.panels.read(self.fresh_frame(), current, now)
        fresh = next((b for b in reading.buttons if b.text == target.text), None)
        if fresh is None or fresh.kind != target.kind:
            raise ToolError("面板变了，重新 panel_read 看看")
        target = fresh
        if target.kind == "never":
            raise ToolError(f"「{target.text}」不能按（花钱、删好友、退出这类按钮，主人放行也不按）")
        permit = None
        if target.kind == "other":
            permit = self._find_permit(target.text, now)
            if permit is None:
                raise ToolError(
                    f"「{target.text}」要卡洛放行才能按：卡洛在聊天里发「#允许 {target.text}」后 "
                    f"{self.cfg.panels.permit_window:.0f} 秒内可以按一次"
                )
        if self._dry(live):
            return f"dry-run：没真的点（会点「{target.text}」）"
        if permit is not None:
            self._permits.remove(permit)  # 放行用一次就作废
        state, changed = self.panel_ops.press(reading, target)
        self._read_at = float("-inf")  # 按完面板变了：要按下一个先重新读
        self._ref_thumb = None
        still = "、".join(p.describe() for p in state.others()) or "没有了"
        return f"按了「{target.text}」" + ("" if changed else "，画面没变（可能没点中）") + f"。现在开着的面板：{still}"

    def panel_close(self, live: bool = False) -> str:
        """关最上面的面板（卡片关法 → 撤退类按钮 → ×）。"""
        self._need_panels()
        top = self.panels.state.top()
        if top is None:
            return "没有开着的面板"
        if self._dry(live):
            return f"dry-run：没真的关（会关掉{top.label}）"
        if not self.panel_ops.close(top, self.panels.readings.get(top.name)):
            raise ToolError(f"没关上{top.label}：找不到能用的关法，或者关了没反应")
        self._ref_thumb = None
        return f"关掉了{top.label}"

    def _pick_button(self, key: str) -> tuple[PanelReading, Button] | None:
        if key.isdigit():
            i = int(key)
            return self._buttons[i - 1] if 1 <= i <= len(self._buttons) else None
        for match in (lambda t: t == key, lambda t: key in t, lambda t: similar(t, key, 0.8)):
            for item in self._buttons:
                if key and match(item[1].text):
                    return item
        return None

    def _find_permit(self, text: str, now: float) -> tuple[str, float] | None:
        self._permits = [p for p in self._permits if p[1] >= now]
        return next((p for p in self._permits if p[0] in text or text in p[0] or similar(p[0], text, 0.8)), None)

    def camera_move(self, action: str, steps: int = 1, live: bool = False) -> str:
        if self.camera is None:
            raise ToolError("没有视角控制")
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在转不了")
        owner = self._owner(self.clock())
        note = self.clear_view("camera", live)
        if self._dry(live):
            return f"dry-run：没真的转（{action} ×{steps}）{note}"
        try:
            with self._held("camera"):
                result = self.camera.move(action, steps, CAMERA_MAX_STEPS * 2 if owner else CAMERA_MAX_STEPS)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        self._ref_thumb = None  # 自己转的镜头，不算画面大变
        self._forget_self()
        return "镜头现在：" + result + (OWNER_NOTE if owner and steps > CAMERA_MAX_STEPS else "")

    def move(self, direction: str, steps: int = 1, force: bool = False, live: bool = False) -> str:
        """小步走（W/A/S/D），走出去回不去、没有复位。卡洛的 # 命令生效期间：一次最多走两倍步数、不用等间隔、牵着手也不用 force，
        真用到了放宽就在结果后面标“（主人命令模式）”。"""
        if self.locomotion is None:
            raise ToolError("没有移动能力")
        if direction not in MOVE_KEYS:
            raise ToolError(f"不认识的移动方向：{direction}（可以用 {'、'.join(MOVE_KEYS)}）")
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在走不了")
        now = self.clock()
        owner = self._owner(now)
        relaxed = owner and steps > MOVE_MAX_STEPS
        if self.holding and not force:
            if not owner:
                raise ToolError(f"正牵着 {self.holding} 的手，走动会松手；确定要松手再走就传 force=true")
            relaxed = True
        wait = self.cfg.brain.move_min_interval - (now - self._last_move)
        if wait > 0:
            if not owner:
                raise ToolError(f"刚走过，等 {math.ceil(wait)} 秒再走（先看看走到哪了）")
            relaxed = True
        max_steps = MOVE_MAX_STEPS * 2 if owner else MOVE_MAX_STEPS
        note = self.clear_view("move", live)
        self._last_move = now
        if self._dry(live):
            return f"dry-run：没真的走（{direction} ×{max(1, min(steps, max_steps))}）{note}"
        result = self.locomotion.move(direction, steps, max_steps)
        self._ref_thumb = None  # 自己走的，不算画面大变
        return result + (OWNER_NOTE if relaxed else "")

    def stop_task(self) -> str:
        return self.skills.cancel(self, "大脑叫停")

    def camera_reset(self, live: bool = False) -> str:
        if self.camera is None:
            raise ToolError("没有视角控制")
        note = self.clear_view("camera", live)
        if self._dry(live):
            return "dry-run：没真的转" + note
        with self._held("camera"):
            result = self.camera.reset()
        self._ref_thumb = None
        self._forget_self()
        return result

    def _owner(self, now: float) -> bool:
        """卡洛的 # 命令还在生效：move / emote / camera 放宽限制（设计见 2026-09-27-brain-move-design.md）。"""
        return now < self._owner_window_until

    def _dry(self, live: bool) -> bool:
        """只打印不执行：dry-run 且不是手动控制（live）。"""
        return self.cfg.reply.dry_run and not live

    def _forget_self(self) -> None:
        """镜头拉近拉远、俯仰变了：转圈认出的团子框（位置、大小）不准了，下次转圈再认。"""
        if hasattr(self.env, "self_box"):
            self.env.self_box = None

    # ---- 大脑离线时的备用回复 ----
    def _fallback(self, now: float) -> None:
        if not self._fallback_pending:
            return
        if self.fallback is None:
            self._fallback_pending.clear()
            return
        if now - self._fallback_last_new < self.cfg.chat.debounce or not self.limiter.allow(now):
            return
        batch, self._fallback_pending = self._fallback_pending, []
        reply = self.fallback.reply(batch)
        if reply is None or reply.text is None:
            return
        try:
            self.say(reply.text, reply=True)
        except ToolError as exc:
            log.warning("备用回复没发出去：%s", exc)
            return
        self.events.put("fallback", f"大脑离线时，备用回复说了：{reply.text}")
