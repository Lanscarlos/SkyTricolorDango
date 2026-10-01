"""大脑的身体：截图、读聊天、认人、按规则秒接互动请求，把发生的事变成事件交给大脑。

大脑的工具通过命令队列在这个线程里执行 —— 只有身体线程碰设备，说话、做动作、转视角不会和接受请求撞在一起。
"""

from __future__ import annotations

import bisect
import copy
import logging
import math
import queue
import random
import re
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
from ..game.social import IDLE, KIND_NAMES, LIGHT, LIGHT_KEY, PASSIVE
from ..imageio import imwrite
from ..device.base import LINUX_KEY_Q
from ..vision.bubbles import Rect, roi_rect
from ..vision.halo import HaloWatch
from ..vision.track import iou
from ..vision.panels import DISCONNECT, UNKNOWN, Button, PanelReading, describe_reading
from ..vision.people import describe_people, describe_things
from .calling import CallResult, call_available, event_text as call_event_text, status_text as call_status_text
from .camera import KEYS as CAMERA_KEYS, MAX_STEPS as CAMERA_MAX_STEPS
from .events import EventQueue
from .occasion import LEVEL_NAMES, Occasion, Spoken, assess, is_friend_fn
from .images import crop_view, difference, fit, image_block, is_black, label_note, scene_note, thumb
from .locomotion import KEYS as MOVE_KEYS, MAX_STEPS as MOVE_MAX_STEPS
from .reflex import Reflexes, addressed
from ..inner.effects import NEUTRAL, Effects, effects as inner_effects
from ..inner.energy import Energy, awake_minutes, energy as inner_energy, energy_parts, format_parts
from ..inner.ledger import card_line, match_friend
from ..inner.ledger import ago
from ..inner.log import ENERGY_EVERY, diff as diff_inner
from ..inner.lull import Lull, LullTracker, parse_musing
from ..inner.mind import sounds_upset
from ..inner.reflect import materials as reflect_materials_text
from .attention import Attention, Target as AttnTarget
from .peek import Done as PeekDone, Obs as PeekObs, PeekPlanner, Turn as PeekTurn, occluded, pick_self
from .skills import SkillRunner
from .track import TrackSkill

log = logging.getLogger(__name__)

REQUEST_KINDS = ("hand", "hug", "highfive", "piggyback", "candle", "light", "*")
PANEL_LOST_AFTER = 30.0  # 面板关了这么久（自动重开也没成功）就告诉大脑
SCENE_EVENT_COOLDOWN = 10.0
SHUTDOWN_REFINE = 8.0  # 退出时镜头闭环复位的细调最多花几秒（粗转照做）；控制台 stop_timeout 60 秒
BUTTON_NOTES = {"retreat": "可以按", "allow": "可以按", "other": "要主人放行", "never": "不能按"}
OWNER_NOTE = "（主人命令模式）"  # 真用到了主人命令窗口的放宽，结果后面标上（brain.jsonl 里看得出来）
CALL_TRACK_STALE = 1.0  # 喊一声时取感知层最近一帧的人物框：比这旧的不要（同 perception.PEOPLE_STALE）
PANEL_FOR_HOLD = {"wheel": "wheel_editor", "friend_tree": "friend_tree"}  # 身体自己打开面板的操作：期间这个面板不算遮挡


def _first_line(exc: BaseException) -> str:
    return (str(exc).splitlines() or [type(exc).__name__])[0]


FORGET_KINDS = ("catchphrase", "joke", "opinion")  # 网页上能删的性格条目类别


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
        ledger=None,  # inner.ledger.Ledger：内心账本（好友关系卡、这次上线）；None = 不记
        mind=None,  # inner.mind.Mind：心情、别扭、心愿（内心层第 2 期）；None = 没有
        reflector=None,  # inner.reflect.Reflector：反思；None = 不反思
        persona=None,  # inner.persona.Persona：性格档案（内心层第 3 期）；None = 性格关着（不沉淀、不记收着点）
        mind_log=None,  # inner.log.MindLog：内心流水账（反思改了什么、精力曲线、删性格条目）；None = 不记
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall: Callable[[], float] = time.time,
        rng: random.Random | None = None,  # 反射抽概率用；测试传固定的
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
        self.ledger = ledger
        self.mind = mind
        self.reflector = reflector
        self.persona = persona
        self.mind_log = mind_log
        self._energy_logged = float("-inf")  # 上次往流水账记精力的墙上时间
        self._soft_until: dict[str, tuple[float, str]] = {}  # 说了难过的好友 → (收着点到什么时候, 他的原话)
        self.profile_text: Callable[[], str] = lambda: cfg.reply.persona  # 反思用的人设（cli 设成 profile.md）
        self.memory_notes: Callable[[], str] = lambda: ""  # 反思用的笔记（cli 设成 notes.md + inbox.md）
        self._energy: Energy | None = None  # 这一圈的精力（每圈算一次）
        self._born_wall = wall()
        self._cheered_at = float("-inf")  # 最近一次有好友跟团子说话（墙上时间）
        self._busy_log: deque[tuple[float, float]] = deque()  # (墙上时间, 秒)：热闹的时段，只留一小时
        self._last_tick_wall: float | None = None
        self._reflect_chat: list[tuple[float, str, str]] = []  # 上次反思以来听到的 / 说出的
        self._reflect_comings: list[str] = []
        self._session_chat: deque[tuple[float, str, str]] = deque(maxlen=80)  # 这次上线的聊天（下线写日记用）
        self._session_comings: deque[str] = deque(maxlen=40)
        self._arrive_notes: dict[str, str] = {}  # 算作新见面、还没发 arrive 的好友 → 交情说明（跟踪中攒着，发 arrive / leave 时取走）
        self.clock = clock
        self.sleep = sleep
        self.wall = wall
        self.panel = panel if panel is not None else PanelManager(cfg.vision, cfg.panel, device, reader, lambda s: self.sleep(s), clock)
        self.brain_offline: Callable[[float], bool] = lambda now: False
        self.brain_turn: Callable[[], tuple[float, float]] = lambda: (float("-inf"), float("-inf"))  # 大脑最近一轮的 (开始, 结束)，cli 设
        self.on_blocked: Callable[[str, str], None] | None = None  # say 被过滤 / 主动护栏拦下时调 (原话, 原因)；沙盒记进聊天记录
        self.on_line: Callable[[str, str, str, str], None] | None = None  # 真机聊天记录：(kind, text, who, why)；沙盒不接（沙盒在 world / control 里记）
        self._bubble_at: float | None = None  # 身体替大脑开输入框（冒“正在输入”）的时间；None = 没开（spec 2026-09-30-body-reflex §2）
        self.rng = rng or random.Random()
        self.reflexes = Reflexes(cfg.reflex, self.rng, clock())
        # 空闲注意力（东张西望，spec 2026-09-30-idle-attention）
        self.attention = Attention(cfg.attention, cfg.track, self.rng, clock())
        self._attention_pressed_at = float("-inf")  # 注意力上次按键：之后 settle 秒内冒出的"走近"是自己转出来的
        self._attention_why = ""  # 这一圈为什么不动（status / 网页）
        self._gestures_seen: deque[tuple[str, str, float]] = deque(maxlen=20)  # (谁, 动作, 时间)：回礼反射取走了也留一份
        self._held_pending: str | None = None  # 先看一眼：推迟了哪个面板等待
        self._hold_until = float("-inf")
        self._camera_moved_at = float("-inf")  # 别人（大脑工具、技能、换角度、环顾）最近一次动镜头
        self._attn_seen_move = float("-inf")  # 注意力已经知道的那次
        self._said_at = float("-inf")  # 团子上次说话（clock）：“刚说完好友就接话”算在跟团子说
        self.brain_busy: Callable[[], bool] = lambda: False  # 大脑正在回聊天（取走了聊天的那一轮还没结束）：聊天面板别因为安静关掉
        self.chat: deque[tuple[float, str, str]] = deque(maxlen=50)  # (时间, 说话人, 内容)；自己说的说话人是“我”
        self.heard: list[Message] = []  # 上次说话以后听到的，记聊天记录用
        self.said: list[str] = []  # 说过（含 dry-run）的话
        self.spoken: deque[Spoken] = deque(maxlen=50)  # 大脑说过的话（含 dry-run）和是不是主动开口；手动控制说的不记
        self.friend_names: Callable[[], list[str]] = lambda: []  # 好友名单（场合里认聊天的说话人），cli 设
        # 冷场时的心理活动（spec 2026-10-01-lull-musing）：好友不说话了 / 聊着聊着走了，叫醒大脑、记它心里想的
        self.lulls: LullTracker | None = (
            LullTracker(cfg.lull, lambda who: is_friend_fn(self.friend_names())(who)) if cfg.lull.enabled else None
        )
        self.on_musing: Callable[[str], None] | None = None  # 记下心里想的一句时调；沙盒记进聊天记录
        self._lull_near: list[str] | None = None  # 这一圈人来人走用的身边名单：冷场追踪用同一份（别在圈尾再取，名字刚好过期会对不上）
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
        self._left_at: dict[str, float] = {}  # 好友上次走开的时间：brain.rejoin 秒内又出现算"回来了"
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
        self._raised: tuple[int, tuple[int, int], float] | None = None  # 举着蜡烛等他亮起来：(轨迹 id, 圆盘位置, 举起时间)
        self._bow: tuple[float, float, float | None] | None = None  # 点火后要鞠躬：(到点, 放弃, 兜底放下的蜡烛几时举的)
        self._dark_at = float("-inf")  # 最近一次黑屏是几时开始的：举蜡烛之后黑过屏（切场景），蜡烛还举没举着说不准
        self._lit_at = float("-inf")
        # 按 Q 喊一声（spec 2026-10-01-q-call）
        self.last_call: CallResult | None = None  # 上次喊（status 用；窗口结束后 seen 补上）
        self._call_at = float("-inf")  # 上次按 Q（dry-run 也记）：min_gap 从这里算
        self._call_times: deque[float] = deque()  # 身体自动喊的时间（额度）
        self._auto_called: dict[str, float] = {}  # 好友 → 为他哪一次走开（_left_at 的时间）自动喊过
        self._pending_auto: CallResult | None = None  # 自动喊了、还在等窗口结果

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
        self._watch_bubble(self.clock())
        self._watch_idle(self.clock())
        self._watch_light(self.clock())
        self._watch_bow(self.clock())
        self._watch_attention(self.clock())
        self._watch_lulls(self.clock())
        self._watch_call(self.clock())
        self._fallback(now)
        self._inner_tick()
        self._ledger_call("save", self.wall())

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
        if self.cfg.reflex.enabled:
            info["反射"] = self._recent_reflex(now) or "还没有"
        if self._attention_on():
            info["注意力"] = self.attention.describe() + (f"（先不动：{self._attention_why}）" if self._attention_why else "")
        if self.cfg.proactive.enabled:
            try:
                o = self.occasion()
                info["场合"] = f"{LEVEL_NAMES[o.level]} · " + ("不主动" if o.blocked else f"还能主动说 {o.left} 句")
            except Exception:
                log.debug("算场合出错", exc_info=True)
                info["场合"] = "算不出来"
        if self.mind is not None:
            info["心情"] = self.mind.mood.text or self.mind.mood.level
            info["精力"] = self._energy.note if self._energy is not None else "算不出来"
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
        # --duration 按真实时间：沙盒快进拨的是模拟时钟，不该让它提前下线
        deadline = time.monotonic() + duration if duration > 0 else float("inf")
        while time.monotonic() < deadline and not (stop is not None and stop.is_set()):
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
        if self._raised is not None:  # 退出时还举着蜡烛：放下
            self._lower_candle(self._raised[2])
            self._raised = None
        if self._bow is not None and self._bow[2] is not None:  # 点亮了、鞠躬还没做：蜡烛还举着，放下
            self._lower_candle(self._bow[2])
        self._bow = None
        if self.emotes is not None:  # 轮盘先恢复：镜头闭环复位可能要好几秒，被强杀时轮盘更要紧
            try:
                self.emotes.restore()
            except Exception:
                log.exception("恢复轮盘失败，请用 emotes wheel 检查")
        if self.camera is not None:  # dry-run 下手动（live）也可能转过：总是复原，没动过就什么都不做
            try:
                log.info(self.camera.reset(refine_seconds=SHUTDOWN_REFINE))
            except Exception:
                log.exception("镜头没转回原位")
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
        if self.ledger is not None:
            for m in fresh:
                try:
                    to_me = self._addressed(m, now)
                except Exception:  # 好友名单读不了之类：账少记一点，这批聊天照样变成事件
                    log.exception("判断是不是在跟团子说话出错")
                    to_me = False
                self._ledger_call("heard", m.speaker, m.text, to_me, self.wall())
                if to_me:
                    self._cheered_at = self.wall()
        if self.mind is not None:
            friends = self._safe_friends()
            for m in fresh:
                self._reflect_chat.append((self.wall(), m.speaker, m.text))
                self._session_chat.append((self.wall(), m.speaker, m.text))
                who = match_friend(m.speaker, friends)
                if who is not None and sounds_upset(m.text) and self.mind.forgive(who, self.wall()):
                    log.info("%s 说「%s」：别扭立刻作废", who, m.text)  # 代码兜底，不等反思
                    self._inner_call(self._save_mind)
                if self.reflector is not None:
                    self.reflector.heard(who is not None, now)
        if self.persona is not None:
            self._inner_call(lambda: self._watch_upset(fresh))
        self.heard = (self.heard + fresh)[-20:]
        for m in fresh:
            self._line("heard", m.text, m.speaker or "（看不出是谁）")
        if self.run_dir is not None:
            self.run_dir.save_frame(frame, [m.box for m in fresh])
        notes = [self._lull_call(lambda m=m: self.lulls.heard(self.wall(), m.speaker, m.text), "") or "" for m in fresh]
        if self.brain_offline(now):  # 大脑离线：交给备用回复，不排进大脑的事件
            self._fallback_pending.extend(fresh)
            self._fallback_last_new = now
            return
        owner = self.cfg.brain.owner_name
        for m, note in zip(fresh, notes):  # note：这句结束了冷场时附上的“你刚才在想”
            if owner and m.speaker == owner and m.text.startswith("#"):
                self._owner_window_until = now + self.cfg.brain.owner_window
                if m.text.startswith("#允许") and m.text[3:].strip():  # 放行一个面板按钮（panel_press 用）
                    self._permits.append((m.text[3:].strip(), now + self.cfg.panels.permit_window))
                self.events.put("owner_command", f"卡洛的命令：{m.text}{note}")
                log.info("识别到卡洛的命令：%s（授权窗口延长到 %.0f 秒后）", m.text, self.cfg.brain.owner_window)
            else:
                self.events.put("chat", f"聊天  {m.speaker or '（看不出是谁）'}：「{m.text}」{note}")
        if self.cfg.reflex.enabled:
            self._on_heard(fresh, now)

    # ---- 身体反射（spec 2026-09-30-body-reflex）----
    def _addressed(self, m: Message, now: float) -> bool:
        """这句是不是在跟团子说（叫名字 / 团子刚说完 / 身边只有他一个好友）。"""
        nearby = self.env.nearby(now) if self.env is not None else []
        since = now - self._said_at if self._said_at > float("-inf") else None
        return addressed(
            m.speaker, m.text, is_friend=is_friend_fn(self.friend_names()), self_names=self.cfg.proactive.self_names,
            nearby=nearby, since_said=since, followup_window=self.cfg.reflex.followup_window, owner=self.cfg.brain.owner_name,
        )

    def _on_heard(self, fresh: list[Message], now: float) -> None:
        """有人在跟团子说话：偶尔先做个小动作，再马上冒输入气泡，大脑想好了用这个框发。一批最多一次。"""
        self.reflexes.stir(now, scale=self.effects().idle)
        if not any(self._addressed(m, now) for m in fresh):
            return
        if self._all_from_grudge(fresh):  # 在跟他闹别扭：故意晚点接（不冒气泡、不做小动作），大脑照常收到消息
            log.debug("在跟 %s 闹别扭，不冒输入气泡", self.mind.grudge.who)
            return
        if self.skills.active is not None:  # 技能在按方向键：框开着按键会变成打字，动作也会打断它
            log.debug("正在%s，不冒输入气泡", self.skills.active.goal)
            return
        busy = self._bubble_blocked()
        if busy:  # 开了也会马上被关掉（气泡来回闪），或者会在别的面板上按 Enter
            log.debug("不冒输入气泡：%s", busy)
            return
        name = self.reflexes.pick_addressed(now, self._wheel(), scale=self.effects().addressed)
        if name:
            self._reflex_emote(name, f"有人叫你，你下意识{name}", now)
        if not self.cfg.reflex.bubble:
            return
        if self._dry(False):
            log.info("[dry-run] 会冒输入气泡（有人在跟团子说话）")
            return
        self._open_bubble(now)

    def _bubble_blocked(self) -> str:
        """这会儿不该开框 / 做反射动作的原因；空 = 可以。"""
        if self.blackout:
            return "画面黑着"
        if self._requests:  # social 要点圆圈会先关框；接不了的请求挂着时开了又关会来回闪
            return "有互动请求挂着"
        if self.panels is not None and self.panels.blocking("say"):
            return "有面板挡着"
        return ""

    def _wheel(self) -> list[str]:
        return self.emotes.on_wheel() if self.emotes is not None else []

    def _reflex_emote(self, name: str, why: str, now: float) -> bool:
        """反射做一个轮盘上的动作：做不了（被挡、在忙、刚做过）就算了，不抛。不占大脑的动作冷却。"""
        if self.emotes is None or self.blackout or self.skills.active is not None or self._requests or self._raised is not None:
            return False
        if now - self.emotes.last_any < self.cfg.reflex.min_gap:
            return False
        try:
            self.clear_view("emote")  # 顺带关掉替大脑开的框、面板挡着就不做
        except ToolError as exc:
            log.debug("反射「%s」没做：%s", name, exc)
            return False
        try:
            if self._dry(False):
                self.emotes.pretend(name, reflex=True)
                log.info("[dry-run] 反射：%s", why)
            else:
                with self._held("wheel"):
                    self.emotes.perform(name, reflex=True)
                log.info("反射：%s", why)
        except Exception:
            log.warning("反射「%s」没做成", name, exc_info=True)
            return False
        self._line("act", f"（团子下意识地 {name}）", "团子")
        self.emoted.append(name)
        self.emoted[:] = self.emoted[-50:]
        self.reflexes.done(now, why, scale=self.effects().idle)
        self.events.put("reflex", why)
        return True

    def _return_gesture(self, who: str, label: str, text: str, now: float) -> bool:
        """别人对团子挥手 / 鞠躬：按概率回同一个动作。框开着就回完再开回来。"""
        name = self.reflexes.pick_return(now, label, self._wheel())
        if name is None:
            return False
        bubble = self._bubble_at
        if not self._reflex_emote(name, f"{text}，你回了个{name}", now):
            return False
        if bubble is not None and self._open_bubble(now):
            self._bubble_at = bubble  # 还是那一次：大脑那一轮结束 / 超时照原来的算
        return True

    def _watch_idle(self, now: float) -> None:
        """闲着很久：从清单里挑个小动作（框开着 = 正在打字，不做）。"""
        if not self.cfg.reflex.enabled or self._bubble_at is not None or self.emotes is None:
            return
        name = self.reflexes.pick_idle(now, self._wheel())
        if name:
            self._reflex_emote(name, f"闲着，你{name}", now)

    def _recent_reflex(self, now: float) -> str:
        if not self.reflexes.recent:
            return ""
        t, text = self.reflexes.recent[-1]
        return f"{text}（{now - t:.0f} 秒前）" if now - t <= 300 else ""

    def _open_bubble(self, now: float) -> bool:
        if self.sender.opened:  # 上一句还没想完，框还开着：按这一句重新计时（回上一句的那一轮结束时别关）
            if self._bubble_at is not None:
                self._bubble_at = self.clock()
            return True
        try:
            self.panel.before_speak(now)  # 先开面板再按 Enter：说完对方的回复马上读得到
            opened = self.sender.open()
        except Exception:
            log.warning("冒输入气泡失败", exc_info=True)
            return False
        if opened:
            self._bubble_at = self.clock()  # 按下 Enter 之后（晚于这批消息进事件队列）：取事件更早的那一轮结束时不会关它
            log.info("有人在跟团子说话：先冒输入气泡")
            self._line("act", "（团子头顶冒出输入气泡）", "团子")
        return opened

    def _close_bubble(self, why: str) -> None:
        """关掉身体替大脑开的框（只关自己开的；say 用掉了就什么都不做）。"""
        if self._bubble_at is None:
            return
        self._bubble_at = None
        if self.sender.opened:
            try:
                self.sender.cancel()
            except Exception:
                log.warning("关输入框失败", exc_info=True)
            log.debug("关掉替大脑开的输入框：%s", why)

    # ---- 空闲注意力（东张西望，spec 2026-09-30-idle-attention） ----
    def set_attention(self, mode: str, focus: str | None = None) -> str:
        """大脑的 attention 工具：定东张西望的模式和关注谁（本次运行有效）。"""
        try:
            self.attention.set_mode(mode, focus)
        except ValueError:
            raise ToolError("模式只能是：随意 / 好奇 / 专心 / 别动") from None
        log.info("注意力模式：%s%s", mode, f"，关注{self.attention.focus}" if self.attention.focus else "")
        line = self.attention.describe()
        return line if line.startswith("注意力：") else f"注意力：{mode}（{line}）"

    def _self_motion_window(self) -> float:
        """注意力按键后多久内冒出的"走近"算自己转出来的：走近判断看 approach_window 内的框高变化。"""
        return max(self.cfg.track.settle, self.cfg.perception.approach_window)

    def _attention_on(self) -> bool:
        """注意力有没有在算：开着、聊天面板是 auto 模式（always 模式下它从不动，也不占随机数）。"""
        return self.cfg.attention.enabled and self.panel.auto

    def _attention_targets(self, now: float) -> list[AttnTarget]:
        """画面里值得看的：说话的、走近的、对团子做动作的、站着的好友。好友按名字认（几个来源是同一个人），陌生人按轨迹 / 位置。"""
        env, out = self.env, []
        if env is None:
            return out
        if hasattr(env, "talkers"):
            for t in env.talkers(now):
                key = f"n:{t.name}" if t.name else f"t:{t.track_id}"
                out.append(AttnTarget(key, "talk_friend" if t.friend else "talk_stranger", t.x, t.name, t.start))
        if hasattr(env, "recent_approaches"):
            for who, cx, t in env.recent_approaches(now):
                if 0 <= t - self._attention_pressed_at < self._self_motion_window():
                    continue  # 自己转镜头转出来的
                stranger = who == "陌生人"
                key = f"a:{round(cx / 50) * 50}" if stranger else f"n:{who}"
                out.append(AttnTarget(key, "approach", cx, None if stranger else who, t))
        for who, _label, t in list(self._gestures_seen):
            found = self.target_x(who, now) if now - t <= 5.0 else None
            if found is not None:
                out.append(AttnTarget(f"n:{who}", "act_on_me", found[0], who, t))
        if hasattr(env, "people"):
            for p in env.people(now):
                if p.kind == "friend" and p.name:
                    out.append(AttnTarget(f"n:{p.name}", "friend_present", p.box.x + p.box.w / 2, p.name))
        return out

    def _attention_blocked(self, now: float, ignore_quiet: bool = False, check_ime: bool = True) -> str:
        """注意力这一圈为什么不能按键（空串 = 能按），顺序同 spec §1。"""
        if not self.cfg.attention.enabled:
            return "没开"
        if self.attention.mode == "别动":
            return "别动"
        if self.camera is None or not hasattr(self.env, "people"):
            return "没有镜头 / 感知层"
        if not self.panel.auto:
            return "聊天面板不是 auto 模式"
        if self.reader.panel_closed_since is None:
            return "聊天面板开着"
        if now - self._camera_moved_at < self.cfg.track.settle:
            return "镜头刚被转过"
        start, end = self.brain_turn()
        if end < start and self._camera_moved_at >= start:  # 大脑这一轮动过镜头（先 camera 再 look）：这一轮别转回去
            return "大脑在用镜头"
        closed = self.reader.panel_closed_since
        if closed is not None and 0 <= now - closed < self.cfg.track.settle:  # 刚关面板：画面整体横移过
            return "聊天面板刚关上"
        holding_window = self._held_pending is not None and now < self._hold_until
        if not ignore_quiet and not holding_window and not self.panel.quiet(now, self.cfg.track.settle + 1):
            return "聊天面板马上要开"
        if self._bubble_at is not None or self.sender.opened:
            return "输入框开着"
        if self.skills.active is not None:
            return "在做事"
        if getattr(self.env, "requests", None):
            return "有互动请求"
        if self.events.has("chat") or self.events.has("owner_command") or self.brain_busy():
            return "在聊天"
        if self.blackout:
            return "画面黑着"
        if self.holding:
            return "牵着手"
        if self.panels is not None and self.panels.state.others():
            return "有面板挡着"
        if self.emotes is not None and now - self.emotes.last_any < self.cfg.reflex.min_gap:
            return "刚做完动作"
        if self._dry(False):
            return "dry-run"
        if check_ime and self.device.ime_shown():  # 最后才问（一次 adb）
            return "输入框开着"
        return ""

    def _attention_look_first(self, th, now: float) -> None:
        """冒气泡 / 有人走近时先看一眼再让聊天面板开：推迟面板，转到中间 / 到时间 / 目标没了就放手。"""
        panel = self.panel
        if self._held_pending is not None:
            if th.centered or th.current is None or now >= self._hold_until or panel.pending != self._held_pending:
                panel.hold_off(None, now)
                log.debug("先看一眼结束（%s），聊天面板照常开", "转到了" if th.centered else "没转到")
                self._held_pending = None
            return
        if not th.look_first or panel.pending not in ("bubble", "approach"):
            return
        if self._attention_blocked(now, ignore_quiet=True, check_ime=False):
            return
        until = now + self.cfg.attention.look_first
        if panel.hold_off(until, now):
            self._held_pending, self._hold_until = panel.pending, until

    def _watch_attention(self, now: float) -> None:
        """每圈最后：想看什么（dry-run 也算，进 status）；闲着就小步转过去 / 随意看一眼。不借面板、不算"有动静"。"""
        if not self._attention_on():
            return
        a = self.attention
        a.width = self.frame_width
        if self.skills.active is not None:  # 技能（track）在动镜头
            self._camera_moved_at = now
        if self._camera_moved_at > self._attn_seen_move:
            self._attn_seen_move = self._camera_moved_at
            a.external_move(self._camera_moved_at)
        th = a.think(self._attention_targets(now), now, self.effects().wander)
        self._attention_look_first(th, now)
        self._attention_why = self._attention_blocked(now, check_ime=th.action is not None)
        if th.action is None or self._attention_why:
            return
        try:
            self.camera.nudge(th.action.direction, th.action.seconds, record=False)  # 原位挪到这里：不进复位账
        except Exception:
            log.exception("注意力转镜头出错")
            return
        a.pressed(th.action, now)
        self._attention_pressed_at = now
        self._ref_thumb = None  # 自己转的，不算画面大变
        log.debug("注意力按%s %.2f s（%s）", "右" if th.action.direction == "right" else "左", th.action.seconds, a.describe())

    def _watch_bubble(self, now: float) -> None:
        if self._bubble_at is None:
            return
        if not self.sender.opened:  # say 用这个框发出去了
            self._bubble_at = None
            return
        start, end = self.brain_turn()
        if start >= self._bubble_at and end >= start:  # 开框之后才开始的那一轮结束了，没说话
            self._close_bubble("大脑这一轮没说话")
        elif now - self._bubble_at >= self.cfg.reflex.bubble_max:
            self._close_bubble("开太久了")

    def news(self, text: str) -> None:
        """眼睛自动看时挑出的新鲜事（眼睛线程调，只入队）。"""
        self._news.put(text)

    def _watch_news(self, now: float) -> list[tuple[str, str]]:
        """新鲜事 / 换了地图 → notice 事件；没熟人、主动额度用完、太勤、和上一条一样都不发（省得为此叫醒大脑）。

        返回这次被拦下的 [(新鲜事, 原因)]（沙盒页上显示）。"""
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
            return []
        o = self.occasion()
        blocked: list[tuple[str, str]] = []
        for text in found:
            if o.level == "alone" or o.left == 0:
                why = o.blocked or "主动额度用完了"
            elif now - self._notice_at < self.cfg.proactive.notice_min:
                why = f"{self.cfg.proactive.notice_min:.0f} 秒内刚有过一条新鲜事"
            elif similar(text, self._notice_last, 0.9):
                why = "和上一条新鲜事差不多"
            else:
                self.events.put("notice", text)
                self._notice_at, self._notice_last = now, text
                continue
            log.debug("新鲜事没发（%s）：%s", why, text)
            blocked.append((text, why))
        return blocked

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
            elif p.name == DISCONNECT:  # 掉线了：按钮都不按（卡上 never），只报告，等卡洛自己点重试
                what = describe_reading(change.reading) if change.reading else "还没读"
                log.warning("游戏掉线了（%s），要手动点「重试」", what)
                self.events.put("panel", f"开了：{p.describe()}：{what}。游戏掉线了：弹框上的按钮都别按，等卡洛自己点重试")
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
        if action != "say":  # 框开着时按键会变成打字、点屏幕会先收起键盘：身体替大脑开的框先关掉
            self._close_bubble(action)
        self.reflexes.stir(self.clock(), scale=self.effects().idle)  # 有动静：闲着的小动作重新计时
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
            if black:
                self._dark_at = now
            if self.env is not None and hasattr(self.env, "hold"):  # 黑屏期间看不到人，别算成"走开了"
                (self.env.hold if black else self.env.release)("blackout")
            self.events.put("scene_change", "画面整屏黑了（可能在切场景）" if black else "画面恢复了")
            self._ref_thumb = None
            if black:  # 可能换了场景：镜头的参照图不作数了
                self._forget_camera_reference()
            if not black:  # 黑屏时没看聊天：恢复后补看一眼
                self.panel.trigger("scene", now)
        if getattr(self.skills.active, "needs_camera", False):  # 技能在转镜头：画面大变是自己转的，结束后重新拿参照
            self._ref_thumb = None
            return
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
        near = self.env.nearby(now)  # 只取一次：账本和人来人走看的是同一份名单
        self._lull_near = list(near)
        self._arrive_notes.update(self._ledger_call("present", near, self.wall(), default={}) or {})  # 跟踪中也照记
        if getattr(self.skills.active, "quiet_people", False):
            # 跟踪中转镜头：人进出画面是自己转的，不是人来了 / 走了。不发人来人走的事件、不更新比较基准，技能结束后下一圈照常比较。
            # 转镜头时框变大变小会被当成"走过来"、动作也认不准：攒着的丢掉，免得跟踪结束后冒出过时的事件。
            # 互动请求、自动接受、牵手状态照常（下面）
            for pop in ("pop_approaches", "pop_gestures", "pop_stranger_backs"):
                if hasattr(self.env, pop):
                    getattr(self.env, pop)()
            if hasattr(self.env, "typing_seen") and self.env.typing_seen(now, strangers=self.cfg.panel.bubble_strangers):
                self.panel.bubble_seen(now)
            self._watch_outfits()  # 装扮照记（账本）
        else:
            self._watch_comings(now, near)
        self._watch_requests(now)

    # ---- 冷场（spec 2026-10-01-lull-musing） ----
    def _lull_call(self, fn: Callable[[], Any], default: Any = None) -> Any:
        """冷场相关的调用：没开（lulls 为 None）返回 default；出错只记日志，这一圈照常。"""
        if self.lulls is None:
            return default
        try:
            return fn()
        except Exception:
            log.exception("冷场追踪出错")
            return default

    def _watch_lulls(self, now: float) -> None:
        if self.lulls is None:
            return
        wall = self.wall()
        near = self._lull_near if self._lull_near is not None else (self.env.nearby(now) if self.env is not None else [])
        cues = self._lull_call(lambda: self.lulls.tick(wall, near, list(self.chat), paused=self.blackout), []) or []
        for cue in cues:
            self.events.put("lull", cue.text)
            if cue.final and self.reflector is not None:  # 冷到最后一个节点：算一次动静，反思到点照常跑
                self.reflector.stirred(now)
        for lull in self._lull_call(self.lulls.pop_finished, []) or []:
            self._lull_note(lull)

    def _lull_note(self, lull: Lull) -> None:
        """结束的冷场写成一行（说话人“（冷场）”），按冷场开始的时间插进反思材料。"""
        if self.mind is None:
            return
        row = (lull.t0, "（冷场）", self.lulls.summary(lull, self.wall()))
        bisect.insort(self._reflect_chat, row, key=lambda r: r[0])
        session = list(self._session_chat)
        bisect.insort(session, row, key=lambda r: r[0])
        self._session_chat.clear()
        self._session_chat.extend(session)

    def mused(self, text: str, began: float | None = None) -> None:
        """大脑一轮最后的文字（身体线程里调）：有“心里：”且正在冷场，就记下来。
        began：这一轮开始的时间（clock）；这之后才开始的冷场不挂（大脑想的不是它）。"""
        if self.lulls is None:
            return
        thought = parse_musing(text, self.cfg.lull.musing_max)
        wall = self.wall()
        since = None if began is None else wall - (self.clock() - began)
        if not thought or not self._lull_call(lambda: self.lulls.muse(thought, wall, since), False):
            return
        log.info("心里：%s", thought)
        self._line("event", f"── 心里：{thought} ──")
        if self.mind_log is not None:
            for lull in self.lulls.active():
                if lull.musings and lull.musings[-1][2] == thought and lull.musings[-1][0] == wall:  # 这次挂上的
                    self._inner_call(lambda lull=lull: self.mind_log.musing(wall, list(lull.who), lull.kind, thought))
        if self.on_musing is not None:
            try:
                self.on_musing(thought)
            except Exception:
                log.exception("on_musing 出错")

    def _watch_comings(self, now: float, near_list: list[str] | None = None) -> None:
        """人来人走：身边有谁、陌生人、正在输入的气泡、有人走过来、对团子做动作。"""
        near = set(self.env.nearby(now) if near_list is None else near_list)
        if near - self._nearby:  # 来人常常会打招呼：看一眼聊天
            self.panel.trigger("arrive", now)
        for name in sorted(near - self._nearby):
            back = self._lull_call(lambda: self.lulls.returned(name, self.wall()))  # 聊着聊着走开的回来了：立刻叫醒
            if now - self._left_at.get(name, float("-inf")) <= self.cfg.brain.rejoin:  # 走出画面又回来：不用再打招呼
                if back:
                    self.events.put("lull", f"{name} 回来了{back}", who=name)
                else:
                    self.events.put("return", f"{name} 回来了", who=name)
            else:
                want = self._inner_call(lambda: self.mind.want_note(name), default="") if self.mind is not None else ""
                joke = ""
                if self.persona is not None and name not in self.soft_names(self.wall()):  # 收着点的人：不提老梗
                    joke = self._inner_call(lambda: self.persona.joke_note(name), default="")
                text = f"{name} 来到身边{self._arrive_notes.pop(name, '')}{want or ''}{joke or ''}{back or ''}"
                self.events.put("arrive", text, who=name)
                self._reflect_note(f"{name} 来到身边")
        for name in sorted(self._nearby - near):
            keep = getattr(self.env, "keep", self.cfg.env.keep)
            self._lull_call(lambda: self.lulls.left(name, self.wall(), list(self.chat)))  # 聊着聊着走了：过 leave_grace 秒还没回来才叫醒
            self.events.put("leave", f"{name} 走开了（{keep:.0f} 秒没看到名字）", who=name)
            self._arrive_notes.pop(name, None)
            self._reflect_note(f"{name} 走开了")
            self._left_at[name] = now
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
                if now - self._attention_pressed_at < self._self_motion_window():
                    continue  # 注意力刚转过镜头：框变大是自己转出来的，不算走近（也别把面板叫开）
                self.panel.trigger("approach", now)
                if who == self.holding:
                    continue
                self.events.put("approach", "有个陌生人朝你走过来了" if who == "陌生人" else f"{who} 朝你走过来了")
        if hasattr(self.env, "pop_gestures"):  # 好友对团子做了动作（三期 §3）：大脑决定回不回礼
            names = self.cfg.gesture.names
            for who, label in self.env.pop_gestures():
                self._gestures_seen.append((who, label, now))  # 空闲注意力也想看他（回礼反射会取走原队列）
                if who == self.holding:
                    continue
                text = f"{who}对你{names.get(label, label)}"
                if self.cfg.reflex.enabled and self._return_gesture(who, label, text, now):
                    continue  # 身体已经回礼了（reflex 事件），不再叫大脑决定
                self.events.put("gesture", text)
        if hasattr(self.env, "pop_stranger_backs"):  # 认装扮（spec 2026-10-01-appearance §6）：陌生人走开又被认回来
            book = getattr(self.env, "appearance", None)
            for sid in self.env.pop_stranger_backs():
                look = book.look("stranger", sid) if book is not None else ""
                self.events.put("stranger_back", f"刚才那个{sid}（{look}）又回来了" if look else f"刚才那个{sid}又回来了")
        self._watch_outfits()

    def _watch_outfits(self) -> None:
        """好友的装扮写进关系卡；换装后新描述回来了 → outfit 背景事件。没开内心层（没账本）只取走、不记。"""
        if not hasattr(self.env, "pop_outfits"):
            return
        notes = self.env.pop_outfits()
        if self.ledger is None:
            return
        wall = self.wall()
        for o in notes:
            if o.state == "described":
                prev = self._ledger_call("describe_outfit", o.name, o.desc, wall)
                if prev:
                    self.events.put("outfit", f"{o.name}换了装扮：上次是「{prev}」，现在「{o.desc}」", who=o.name)
            else:
                # 只有 changed 才另起一套；new（卡里没有可比的特征，比如换了特征模型）接着用最后一套、换上新特征
                self._ledger_call("wear", o.name, o.feat, o.key, o.state == "changed", wall)

    def _watch_requests(self, now: float) -> None:
        """互动请求、按规则自动接受、牵手状态（跟踪中也照常）。light（团子举蜡烛点亮陌生人）在 _watch_light 里做。"""
        requests = {k: r for k, r in dict(self.env.requests).items() if r.kind != LIGHT}
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
            if handled and (self._raised is not None or (self._bow is not None and self._bow[2] is not None)):
                # 点圆圈接受互动会放下手里的蜡烛：之后再按 3 就是把它举起来了
                log.info("举着蜡烛时接受了%s，蜡烛当作已经放下，不再等那个陌生人亮起来、也不再按 3 放下", "、".join(handled))
                self._raised = None
                if self._bow is not None:
                    self._bow = (self._bow[0], self._bow[1], None)  # 鞠躬照做，只是不用兜底放下了
            for item in handled:
                name, kind = item.split(":", 1)
                self.events.put("accepted", f"身体按规则接受了 {name} 的{KIND_NAMES.get(kind, kind)}")
                if kind == "hand":
                    self._accepted_hand = (name, now)
                if kind == "candle":
                    self._schedule_bow(now, None)
        self._watch_holding(now)

    # ---- 点亮没点火的陌生人（spec 2026-10-01-light-unlit-stranger） ----
    def _watch_light(self, now: float) -> None:
        """黑影在身边站够了：按 3 号键举蜡烛（绝不点他身上的火焰圆盘：点了会跟着他走），等 YOLO 看到他亮起来。"""
        if self._raised is not None:
            self._check_lit(now)
            return
        if self._bow is not None:  # 鞠躬还在排队：蜡烛还举着，再按 3 会把它放下，之后兜底放下又举起来
            return
        if self.env is None:
            return
        req = dict(self.env.requests).get(LIGHT_KEY)
        if req is None or req.track is None or self.social is None or self.emotes is None:
            return
        if not self.social.allowed(req):
            return
        if self.skills.active is not None or self.holding or self._bubble_blocked():
            return
        if self._bubble_at is not None:  # 身体替大脑开着输入框（有人在跟团子说话）：按键会把框关掉，先不举
            return
        if now - self.emotes.last_any < self.cfg.reflex.min_gap:  # 刚做完动作就按 3 可能被动画吞掉
            return
        try:
            self.clear_view("emote")
        except ToolError as exc:
            log.debug("先不举蜡烛：%s", exc)
            return
        self.env.mark_tried(req.track)  # 不管成没成，这个人只举一次
        if self._dry(False):
            log.info("[dry-run] 会举蜡烛点亮身边的陌生人")
            return
        try:
            with self._held("wheel"):
                self.emotes.press_slot(self.cfg.social.candle_slot)
        except Exception:
            log.warning("举蜡烛没成功", exc_info=True)
            return
        self._raised = (req.track, req.pos, now)
        log.info("举起蜡烛给身边没点火的陌生人点火")

    def _check_lit(self, now: float) -> None:
        track, pos, raised_at = self._raised
        if self.emotes is not None and self.emotes.last_any > raised_at:  # 做动作已经把蜡烛放下了
            self._raised = None
            log.info("举蜡烛时做了别的动作，蜡烛已经放下，不再等他亮起来")
            return
        if self.env.lit(track, pos, raised_at):
            self._raised = None
            self._lit_at = now
            log.info("陌生人亮起来了（举蜡烛 %.1f 秒）", now - raised_at)
            self.events.put("accepted", "你举起蜡烛给身边一个没点火的陌生人点了火（他亮起来了）")
            self._schedule_bow(now, raised_at)
        elif now - raised_at >= self.cfg.social.light_timeout:
            self._raised = None
            log.warning("举了蜡烛 %.0f 秒他还是黑的（或者走了），不再点他", self.cfg.social.light_timeout)
            self._lower_candle(raised_at)

    def _schedule_bow(self, now: float, raised_at: float | None) -> None:
        """点亮了别人（raised_at = 举蜡烛的时间）/ 接受了别人点火（None）：过 bow_delay 鞠躬；做不了鞠躬时把自己举的蜡烛放下。"""
        if self._bow is not None:  # 已经排了一个鞠躬，一个就够：保留时间，只把还缺的 raised_at 补上（先于下面的"立刻放下"，鞠躬排着说明能做）
            due, give_up, old = self._bow
            self._bow = (due, give_up, old if old is not None else raised_at)
            return
        name = self.cfg.social.after_light
        if not name or name not in self._wheel() or self.holding:
            if raised_at is not None:
                self._lower_candle(raised_at)
            return
        due = now + self.cfg.social.bow_delay
        self._bow = (due, due + 5.0, raised_at)

    def _watch_bow(self, now: float) -> None:
        if self._bow is None or now < self._bow[0]:
            return
        _, give_up, raised_at = self._bow
        if self._reflex_emote(self.cfg.social.after_light, f"点完火你{self.cfg.social.after_light}了一下", now):
            self._bow = None
        elif now >= give_up:
            self._bow = None
            log.info("点火后的%s一直做不了，算了", self.cfg.social.after_light)
            if raised_at is not None:
                self._lower_candle(raised_at)

    def _lower_candle(self, raised_at: float) -> None:
        """按 3 放下自己举的蜡烛——只在举起之后没做过任何动作时（做动作会放下蜡烛，再按就又举起来了）。"""
        if self.emotes is None or self.emotes.last_any > raised_at or self._dry(False):
            return
        if self._dark_at >= raised_at:
            log.info("举蜡烛之后黑过屏（可能切了场景），蜡烛还举没举着说不准，不按 3")
            return
        self._close_bubble("放下蜡烛")  # 按数字键前会关掉输入框：替大脑开的框要走这里关，sender 才知道
        try:
            with self._held("wheel"):
                self.emotes.press_slot(self.cfg.social.candle_slot)
            log.info("放下蜡烛")
        except Exception:
            log.warning("放下蜡烛没成功", exc_info=True)

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
        people = [p for p in self.env.people(now) if p.name and p.sure] if hasattr(self.env, "people") else []  # "像小明"见 _locate_by_look
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

    def _locate_by_look(self, name: str, now: float) -> tuple[Rect, str] | None:
        """_locate 找不到时按外观找：先找编号对得上的陌生人（"陌生人A"），再找没看到名字、像这个好友的人。
        返回 (框, 附注)：像谁的附注"（没看到名字，按外观认的）"，陌生人编号的附注是空的。"""
        if self.env is None or not hasattr(self.env, "people"):
            return None
        people = list(self.env.people(now))
        for p in people:
            if p.sid and p.sid == name:
                return p.box, ""
        maybe = [p for p in people if p.kind == "friend" and p.name and not p.sure]
        for same in (lambda n: n == name, lambda n: similar(name, n, 0.75)):
            for p in maybe:
                if same(p.name):
                    return p.box, "（没看到名字，按外观认的）"
        return None

    def target_x(self, name: str, now: float) -> tuple[float, str] | None:
        """技能 track 用：(这个好友在画面上的中心 x 像素, 来源 "body" / "tag")；看不到返回 None。

        人物框优先（感知层 people() 本来就只给 ≤ 1 s 的）；框时有时无（人挨着团子时 v4 只框住 67%）就用
        名字标签的 x 接上，但标签要 track.max_age 秒内的。名字先找一模一样的，再容忍 OCR 错一两个字（同 _locate）。
        """
        if self.env is None:
            return None
        people = [p for p in self.env.people(now) if p.kind == "friend" and p.name] if hasattr(self.env, "people") else []
        people.sort(key=lambda p: not p.sure)  # 看到名字的排前面；只有"像他"的也照样能盯
        max_age = self.cfg.track.max_age
        labels = {n: v for n, v in dict(self.env.labels).items() if now - v[4] <= max_age}  # 后台线程会改：先拍快照
        for same in (lambda n: n == name, lambda n: similar(name, n, 0.75)):
            for p in people:
                if same(p.name):
                    return p.box.x + p.box.w / 2, "body"
            for n, (x, _y, w, _h, _t) in labels.items():
                if same(n):
                    return x + w / 2, "tag"
        return None

    @property
    def frame_width(self) -> int:
        return int(self.last_frame.shape[1]) if self.last_frame is not None else 1920

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

    def look_person(self, name: str, live: bool = False) -> list[dict]:
        """按名字找到这个人，把他裁出来给大脑看（和 look 共用频率限制）。

        他躲在团子身后（只有名字标签、压在团子框上）时先换角度：边转边看 YOLO 让他露出来、太小就拉近（_peek），
        看完镜头不复位，结果里告诉大脑镜头现在在哪。"""
        now = self.clock()
        brain = self.cfg.brain
        if now - self.last_look < brain.look_min_interval:
            raise ToolError(f"{brain.look_min_interval:.0f} 秒内刚看过，等一下再看")
        if self.env is None:
            raise ToolError("没开环境识别，认不出名字；用 look(image=true) 自己看")
        found, by_look = self._locate(name, now), ""
        if found is None:  # 没看到名字：陌生人编号 / "像他"的人
            alt = self._locate_by_look(name, now)
            if alt is not None:
                found, by_look = (alt[0], False), alt[1]
        if found is None or found[0].w < 8 or found[0].h < 8:
            known = self._recognized(now)
            where = f"画面里现在认得出：{'、'.join(known)}" if known else "画面里现在一个名字都没认出来"
            raise ToolError(f"没找到 {name}（{where}）；不在画面里的话可以先 look_around 看看在哪个方向")
        box, guessed = found
        note, frame = "", None
        tag, me = self._fresh_tag(name, now), self._self_box(now)
        log.debug("look_person %s：%s %s；名字标签 %s，团子框 %s，压在团子上 %s", name, "按标签估的框" if guessed else "身体框",
                  box, tag, me, occluded(tag, me) if tag is not None and me is not None else "-")
        if guessed and self._hidden_tag(name, now) is not None:
            box, guessed, note, frame = self._peek(name, box, live)
        bx, by, bw, bh = box.x, box.y, box.w, box.h
        if frame is None:  # 换过角度就用量框那一帧（面板开 / 关画面会横移），没换才现截
            frame = self.device.screenshot()
        self.last_frame, self.last_look = frame, now
        fh, fw = frame.shape[:2]
        mx, my = round(bw * 0.2), round(bh * 0.2)  # 四周各放宽 20%：人会动、框也不一定贴身
        x1, y1, x2, y2 = max(0, bx - mx), max(0, by - my), min(fw, bx + bw + mx), min(fh, by + bh + my)
        crop = fit(frame[y1:y2, x1:x2], tuple(brain.image_size))
        text = f"这是 {name}（原图 ({x1}, {y1}) 起 {x2 - x1}×{y2 - y1}）"
        if guessed:
            text += "；按名字标签估的位置，可能没框全"
        return [image_block(crop, brain.jpeg_quality), {"type": "text", "text": text + note + by_look}]

    # ---- look_person 换角度（peek）：好友躲在团子身后 ----
    def _self_box(self, now: float, since: float | None = None) -> Rect | None:
        """团子框：最近一帧 YOLO 的 self，只信在画面水平中间附近的（团子是镜头支点）。

        since = None（判断挡住、status）时也算上转圈认出的 self_box；换角度的循环里只要 since 之后的 YOLO 结果
        （缩放过之后转圈那个框大小不对）。"""
        if self.env is None:
            return None
        cfg = self.cfg.peek
        boxes = [t.box for t in list(getattr(self.env, "last_tracks", ()))
                 if t.cls == "self" and now - t.last <= cfg.tag_age and (since is None or t.last >= since)]
        if since is None and getattr(self.env, "self_box", None) is not None:
            boxes.append(self.env.self_box)
        return pick_self(boxes, self.frame_width, cfg.self_center)

    def _fresh_tag(self, name: str, now: float, since: float = float("-inf")) -> Rect | None:
        """这个人 peek.tag_age 秒内（且不早于 since）的名字标签；名字先找一模一样的，再容忍 OCR 错一两个字。"""
        if self.env is None:
            return None
        labels = {n: v for n, v in dict(self.env.labels).items()  # 后台线程会改：先拍快照
                  if now - v[4] <= self.cfg.peek.tag_age and v[4] >= since}
        for same in (lambda n: n == name, lambda n: similar(name, n, 0.75)):
            for n, (x, y, w, h, _t) in labels.items():
                if same(n):
                    return Rect(x, y, w, h)
        return None

    def _body_of(self, name: str, now: float, since: float | None = None) -> Rect | None:
        """这个人的身体框；since：只要感知层在 since 之后又看到过的（people() 会给 1 秒内的旧框）。"""
        people = [p for p in self.env.people(now) if p.name] if hasattr(self.env, "people") else []
        if since is not None:
            seen = {t.id for t in list(getattr(self.env, "last_tracks", ())) if t.last >= since}
            people = [p for p in people if p.track_id in seen]
        for same in (lambda n: n == name, lambda n: similar(name, n, 0.75)):
            for p in people:
                if same(p.name):
                    return p.box
        return None

    def _hidden_tag(self, name: str, now: float) -> Rect | None:
        """这个好友被团子挡住了就返回他的名字标签：标签新鲜、没有身体框、标签压在团子框上。"""
        tag = self._fresh_tag(name, now)
        me = self._self_box(now)
        if tag is None or me is None or self._body_of(name, now) is not None:
            return None
        return tag if occluded(tag, me) else None

    def hidden_friends(self, now: float) -> list[str]:
        """status 用：被团子挡住的好友。"""
        if self.env is None or not hasattr(self.env, "people"):
            return []
        names = [n for n, v in dict(self.env.labels).items() if now - v[4] <= self.cfg.peek.tag_age]
        return [n for n in names if self._hidden_tag(n, now) is not None]

    def _peek_blocked(self, live: bool) -> str:
        """换不了角度的原因（空串 = 可以换）。"""
        skill = self.skills.active
        if not self.cfg.peek.enabled:
            return "没开换角度（[peek] enabled）"
        if self._dry(live):
            return "dry-run 不转镜头"
        if self.camera is None:
            return "没有视角控制"
        if self.blackout:
            return "画面黑着"
        if skill is not None and getattr(skill, "needs_camera", False):
            return f"正在{skill.goal}，没打断"
        return ""

    def _peek_obs(self, name: str, since: float) -> PeekObs | None:
        """since 之后感知层的结果（标签、身体框、团子框都要是按键 + 停稳之后看到的）；这个人一样都没有返回 None。"""
        now = self.clock()
        tag, body = self._fresh_tag(name, now, since), self._body_of(name, now, since)
        if tag is None and body is None:
            return None
        return PeekObs(tag, body, self._self_box(now, since))

    def _peek_look(self, name: str, since: float) -> tuple[PeekObs | None, Any]:
        """画面停稳之后：每隔 poll 秒截一张喂感知层（capture = "body" 时感知层靠身体给帧），看到 since 之后的结果就返回
        (结果, 刚截的那一帧)；lost_after 秒内一直看不到这个人返回 (None, None)。"""
        cfg, track = self.cfg.peek, self.cfg.track
        deadline = self.clock() + track.lost_after
        for _ in range(max(1, math.ceil(track.lost_after / cfg.poll)) + 1):  # 时钟不走（测试）时也有上限
            frame = self.device.screenshot()
            self.last_frame = frame
            self.env.observe(frame, self.clock(), False)
            obs = self._peek_obs(name, since)
            if obs is not None:
                return obs, frame
            if self.clock() >= deadline:
                break
            self.sleep(cfg.poll)
        return None, None

    def _peek(self, name: str, box: Rect, live: bool) -> tuple[Rect, bool, str, Any]:
        """好友被团子挡住：闭环换角度。返回 (裁图用的框, 是不是按标签估的, 附加说明, 量框的那一帧 / None)。镜头不复位。"""
        blocked = self._peek_blocked(live)
        if not blocked:
            try:
                self.clear_view("camera", live)
            except ToolError as exc:
                blocked = str(exc)
        if blocked:
            return box, True, f"；{name} 被你挡住了（{blocked}，没换角度）", None
        cfg, track = self.cfg.peek, self.cfg.track
        width = self.frame_width
        height = int(self.last_frame.shape[0]) if self.last_frame is not None else 1080
        planner = PeekPlanner(cfg, track, width, height)
        start, pressed, reason = self.clock(), 0, "budget"
        last, shot = None, None
        try:
            with self.panel.borrow("look_person") as was_open:
                if was_open:  # 关面板时画面整体横移：等停稳再重新量
                    self.sleep(track.settle)
                obs, frame = self._peek_look(name, self.clock())
                last, shot = (obs, frame) if obs is not None else (None, None)
                while True:
                    action = planner.next(obs)
                    if isinstance(action, PeekDone):
                        reason = action.reason
                        break
                    if pressed >= cfg.max_presses or self.clock() - start >= cfg.max_seconds:
                        break
                    if isinstance(action, PeekTurn):
                        self.camera.nudge(action.direction, action.seconds)
                    else:
                        self.camera.zoom_once(action.direction)
                    pressed += 1
                    self.sleep(track.settle)
                    obs, frame = self._peek_look(name, self.clock())
                    if obs is not None:
                        last, shot = obs, frame
        finally:
            if pressed:
                self._ref_thumb = None  # 自己转的，不算画面大变
                self._camera_moved_at = self.clock()
                self._forget_self()
        if reason in ("budget", "lost") and planner.enlarging:  # 已经露出来过（调大小时用完预算 / 拉近把人推没了）：用露出来那一帧
            reason = "revealed"
        where = f"镜头：{self.camera.describe()}，要转回去用 camera_reset" if pressed else ""
        if reason == "revealed" and last is not None:
            done = f"；刚才被你挡住了，转了一下镜头才看清（{where}）" if pressed else ""
            if last.body is not None:
                return last.body, False, done, shot
            return self._below_tag((last.tag.x, last.tag.y, last.tag.w, last.tag.h)), True, done, shot
        why = {"lost": "换角度时看不到他了", "stuck": "离得太近转不开"}.get(reason, "转了几下还是没露出来")
        note = f"；{name} 被你挡住了，换了角度也没看清（{why}{'；' + where if where else ''}）"
        if last is None:  # 一次都没看到：只能用转之前估的位置、现截
            return box, True, note, None
        if last.tag is not None:
            return self._below_tag((last.tag.x, last.tag.y, last.tag.w, last.tag.h)), True, note, shot
        return last.body, True, note, shot

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
        stopped = self._stop_camera_task("要环顾四周")
        if stopped:
            log.info("环顾四周%s", stopped)
        try:
            with self._held("camera"):
                frames = self.camera.around(self.device.screenshot)
        finally:
            self._ref_thumb = None  # 自己转的镜头，不算画面大变
            self._camera_moved_at = self.clock()
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
        stopped = self._stop_camera_task("要环顾四周")
        try:
            with self._held("camera"):
                shot = self.camera.spin(self.device.screenshot, 1, spin.seconds_per_turn, spin.fps)
                result = self.env.sweep([(0.0, shot.before), *shot.frames], spin)
        finally:
            self._ref_thumb = None  # 自己转的镜头，不算画面大变
            self._camera_moved_at = self.clock()
        self.last_frame = shot.after
        if not shot.panel_reopened:
            self.events.put("panel", "转完一圈，聊天记录面板没重新打开")
        return result.text() + ("（中途画面黑了，可能在切场景，这一圈不准）" if shot.blackout else "") + stopped

    def status(self) -> str:
        now = self.clock()
        parts = ["聊天记录面板" + ("开" if self.reader.panel_closed_since is None else "关")]
        if self._bubble_at is not None:
            parts.append("输入框：开着（身体替你开的，想好就 say）")
        else:
            try:
                parts.append("输入框" + ("开" if self.device.ime_shown() else "关"))
            except Exception:
                parts.append("输入框状态读不到")
        others = self.panels.state.others() if self.panels is not None else ()
        if others:
            parts.append("开着的面板：" + "、".join(p.describe() for p in others))
        near = self.env.nearby(now) if self.env is not None else []
        me = self.env.my_look() if hasattr(self.env, "my_look") else ""
        if me:
            parts.append("你自己：" + me)
        looks = self.env.looks(list(near)) if near and hasattr(self.env, "looks") else {}
        friends = "、".join(f"{n}（{looks[n]}）" if looks.get(n) else n for n in near)
        if near and self.ledger is not None:
            args = (near, self.wall(), looks) if looks else (near, self.wall())
            friends = self._ledger_call("status_line", *args, default="") or friends
        parts.append("身边的好友：" + (friends or "没看到"))
        if hasattr(self.env, "strangers"):
            parts.append(f"身边的陌生人：{self.env.strangers(now)} 个")
        closest = self.env.nearest(now) if hasattr(self.env, "nearest") else None
        if closest:
            parts.append(f"离你最近的：{closest[0]}（{closest[1]}）")
        people = describe_people(self.env.people(now)) if hasattr(self.env, "people") else ""
        if people:
            parts.append("画面里：" + people)
        hidden = self.hidden_friends(now)
        if hidden:
            parts.append("被你挡住：" + "、".join(hidden) + ("" if self._peek_blocked(False) else "（look_person 会自己换角度看）"))
        things = describe_things(self.env.objects(now)) if hasattr(self.env, "objects") else ""
        if things:
            parts.append("画面里的东西：" + things)
        if self._raised is not None:
            parts.append("正在举蜡烛给陌生人点火")
        elif now - self._lit_at <= self.cfg.social.remember:
            parts.append(f"{now - self._lit_at:.0f} 秒前你给一个陌生人点了火")
        if self.holding:
            parts.append(f"牵着手：{self.holding}（推测）")
        if self.blackout:
            parts.append("画面黑着")
        reflex = self._recent_reflex(now)
        if reflex:
            parts.append("刚才下意识：" + reflex)
        if self._attention_on():
            parts.append(self.attention.describe())
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
            try:
                parts.append("场合：" + self.occasion().line(self.wall()))
            except Exception:
                log.exception("算场合出错")
                parts.append("场合：算不出来（详见日志）")
        lull = self._lull_call(lambda: self.lulls.status(self.wall()), "")
        if lull:
            parts.append(lull)
        if self.mind is not None:
            heart = self._inner_call(lambda: self.mind.line(self.wall(), self._energy))
            if heart:
                parts.append("心里：" + heart)
        soft = self._inner_call(lambda: self._soft_line(self.wall()), default="") if self.persona is not None else ""
        if soft:
            parts.append(soft)
        if self.last_call is not None and self.last_call.seen is not None:
            parts.append(call_status_text(self.last_call, now))
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
            is_friend_fn(self.friend_names()), quota_scale=self.effects().quota,
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
        if self._held_pending is not None:  # 空闲注意力在"先看一眼"：大脑要看聊天，别推迟了
            self.panel.hold_off(None, self.clock())
            self._held_pending = None
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

    def _pending_reply(self, wall: float) -> bool:
        """有还没接的好友聊天（reply_window 内、团子在那之后还没说过）：这时说的话是接话，不算主动开口。
        被别的事件（来人、走开……）叫醒的一轮里接上一句也算（spec 2026-10-01-lull-musing §4 修复 1）。"""
        is_friend = is_friend_fn(self.friend_names())
        for t, who, _ in reversed(self.chat):
            if who == "我":
                return False
            if is_friend(who):
                return wall - t <= self.cfg.proactive.reply_window
        return False

    def say(self, text: str, live: bool = False, reply: bool = False) -> str:
        """live = 手动控制：dry-run 下也真的发（护栏照旧）。reply = 明确是接话（大脑离线时的备用回复），不算主动开口。"""
        now = self.clock()
        body = clean_reply(text, self.cfg.reply.max_chars)  # 含“不能声称自己是真人”的硬过滤
        if body is None:
            self._blocked(text, "过滤掉了（空的、<skip>，或者说了自己是真人）")
            raise ToolError("这句没发：是空的、<skip>，或者说了自己是真人（不能这么说）。换个说法")
        if not self.limiter.allow(now):
            self._blocked(text, "说得太快了")
            raise ToolError("说得太快了，等几秒再说")
        proactive = (
            self.cfg.proactive.enabled and not live and not reply and not self.brain_busy() and not self._pending_reply(self.wall())
        )
        if proactive:
            blocked = self.occasion().blocked
            if blocked:
                self._blocked(text, blocked)
                raise ToolError(blocked)
        note = self.clear_view("say", live)
        full = self.cfg.reply.disclosure_prefix + body
        self.limiter.record(now)
        self._said_at = now
        self.reflexes.stir(now, scale=self.effects().idle)
        self.said.append(full)
        self._ledger_call("said", self.wall())
        if self.mind is not None:
            self._reflect_chat.append((self.wall(), "我", full))
            self._session_chat.append((self.wall(), "我", full))
            if self.reflector is not None:
                self.reflector.stirred(now)
        self.said[:] = self.said[-50:]  # 只留最近 50 条，别无限长
        self.chat.append((self.wall(), "我", full))
        if not live:
            self.spoken.append(Spoken(self.wall(), full, proactive))
        if self._dry(live):
            log.info("[dry-run] 将会发送: %s", full)
            self._remember(body, full, sent=False)
            self._line("said", full + "（dry-run，没真的发）", "团子")
            return f"dry-run：没真的发，“{full}”{note}"
        self.panel.before_speak(now)  # 先开面板再按 Enter：说完对方的回复马上读得到
        self.sender.send(full)
        self._line("said", full, "团子")
        self.self_filter.remember(full, self.clock())
        self._remember(body, full, sent=True, manual=live)
        return f"已发送：{full}"

    def _blocked(self, text: str, why: str) -> None:
        self._line("blocked", text, "团子", why)
        if self.on_blocked is None:
            return
        try:
            self.on_blocked(text, why)
        except Exception:
            log.exception("on_blocked 出错")

    def _line(self, kind: str, text: str, who: str = "", why: str = "") -> None:
        """交给真机聊天记录；出错只记日志。"""
        if self.on_line is None:
            return
        try:
            self.on_line(kind, text, who, why)
        except Exception:
            log.exception("on_line 出错")

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

    def emote(self, name: str, live: bool = False) -> str:
        """做动作不会松开牵手（用户实测；以前记的“会松手”是对方自己断开的），牵着手也照做。"""
        if self.emotes is None:
            raise ToolError("这次没开动作（--no-emotes 或者图标库是空的）")
        owner = self._owner(self.clock())
        available = self.emotes.available(owner)
        if name not in available:
            raise ToolError(f"「{name}」现在做不了；能做的：{'、'.join(available) or '暂时没有（刚做过动作，要等一会儿）'}")
        relaxed = owner and name not in self.emotes.available()  # 平时这会儿还在动作限速里
        gap = self.cfg.reflex.min_gap - (self.clock() - self.emotes.last_any) if self.cfg.reflex.enabled else 0.0
        if gap > 0:  # 上一个动作（多半是反射）的动画还没做完
            raise ToolError(f"刚做完一个动作，{math.ceil(gap)} 秒后再做")
        note = self.clear_view("emote", live)
        self.emoted.append(name)
        self.emoted[:] = self.emoted[-50:]  # 只留最近 50 条，别无限长
        if self._dry(live):
            self.emotes.pretend(name)
            self._line("act", f"（团子做了 {name}）", "团子")
            return f"dry-run：没真的做「{name}」{note}"
        try:
            with self._held("wheel"):
                self.emotes.perform(name)
        except Exception as exc:
            raise ToolError(f"「{name}」没做成：{exc}") from None
        self._line("act", f"（团子做了 {name}）", "团子")
        return f"做了「{name}」" + (OWNER_NOTE if relaxed else "")

    def set_policy(self, who: str, kind: str, accept: bool) -> str:
        if self.social is None:
            raise ToolError("这次没开互动请求处理（[social] 关了或者没有图标模板）")
        if kind not in REQUEST_KINDS:
            raise ToolError(f"不认识的请求类型 {kind}，可以用：{'、'.join(REQUEST_KINDS)}")
        who = who.strip() or "*"
        if who == "stranger" and accept and kind not in ("candle", "light"):
            raise ToolError("陌生人只能接点火（candle）、点亮他（light），牵手 / 拥抱 / 击掌 / 背背都不接陌生人的")
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
        stopped = self._stop_camera_task("要点人看好友树")
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
            return [image_block(fit(result.opened, tuple(brain.image_size)), brain.jpeg_quality), {"type": "text", "text": note + stopped}]
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
            {"type": "text", "text": note + stopped},
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
        self._close_bubble("panel_press")
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
        self._close_bubble("panel_close")
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
        if action not in CAMERA_KEYS:  # 先认参数：参数不对别把正在盯的人停了
            raise ToolError(f"不认识的视角操作：{action}（可以用 {'、'.join(CAMERA_KEYS)}）")
        stopped = self._stop_camera_task("要自己转镜头")
        try:
            with self._held("camera"):
                result = self.camera.move(action, steps, CAMERA_MAX_STEPS * 2 if owner else CAMERA_MAX_STEPS)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        self._ref_thumb = None  # 自己转的镜头，不算画面大变
        self._camera_moved_at = self.clock()
        self._forget_self()
        return "镜头现在：" + result + (OWNER_NOTE if owner and steps > CAMERA_MAX_STEPS else "") + stopped

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
        self._forget_camera_reference()  # 走过之后转之前那张参照图对不上了：复位时只粗转
        return result + (OWNER_NOTE if relaxed else "")

    # ---- 按 Q 喊一声（spec 2026-10-01-q-call §2） ----
    def call_out(self, reason: str, *, live: bool = False) -> CallResult:
        """短按 Q 喊一声：同步按键 + 连拍 burst 秒看光圈，开感知层的呼喊窗口（结果约 window 秒后 env.call_result(at) 才有）。
        reason："brain" / "auto" / "manual"。不能喊时不抛异常，CallResult.refused 写原因。约 1~1.3 秒。"""
        cfg = self.cfg.call
        now = self.clock()
        if not cfg.enabled:
            return CallResult(now, reason, refused="没开")
        if not call_available(self.cfg, self.env):  # 整图 OCR 也有 called()，但收不到呼喊窗口
            return CallResult(now, reason, refused="没开感知层，喊了也收不到名字")
        if self.blackout:
            return CallResult(now, reason, refused="画面黑着（在切场景），现在喊不了")
        gap = now - self._call_at
        if gap < cfg.min_gap:
            if reason == "brain" and self._owner(now):
                reason = "brain-owner"
            else:
                return CallResult(now, reason, refused=f"{gap:.0f} 秒前刚喊过，等一会儿再喊")
        if self._dry(live):
            self._call_at = now
            log.info("dry-run：会喊一声（%s）", reason)
            return CallResult(now, reason, dry=True)
        try:
            self.clear_view("call", live)
        except ToolError as exc:
            return CallResult(now, reason, refused=str(exc))
        if self.device.ime_shown():  # 身体开的框上面关过了：还开着就是别人开的
            return CallResult(now, reason, refused="输入框开着（按 Q 会变成打字）")
        moved = max(self._camera_moved_at, self._attention_pressed_at)
        with self.panel.borrow("call") as was_open:  # 聊天面板开着时按 Q 没反应
            if was_open:
                self.sleep(self.cfg.track.settle)  # 面板关掉那一下画面横移：等停稳再拍基准
            base = self.device.screenshot()
            tracks = [t for t in list(getattr(self.env, "last_tracks", ())) if self.clock() - t.last <= CALL_TRACK_STALE]
            selfs = [t for t in tracks if t.cls == "self"]
            heads = {t.id: t.box for t in selfs}
            # 和 self 框重叠的 player 框是团子本人（同感知层 _is_self）：别让一个光圈点亮两个区域、判成"别人也在喊"
            heads.update({t.id: t.box for t in tracks if t.cls == "player" and not any(iou(t.box, s.box) >= 0.5 for s in selfs)})
            at = self.clock()  # 按键命令发出之前：adb 往返之后才记会把最亮那一下算到窗口之前
            watch = HaloWatch(base, heads, at, cfg)
            self.device.hw_key(LINUX_KEY_Q)
            if at - moved < self.cfg.track.settle:
                watch.skipped = True  # 镜头刚动过：头顶区域对不上
            if was_open:
                watch.skipped = True  # 面板开着：关它时画面横移，感知层这段时间也没新帧，框对不上
            for _ in range(int(cfg.burst * 100) + 1):  # 张数上限：时钟不走（测试、沙盒）也会停
                if self.clock() - at >= cfg.burst or watch.skipped:
                    break
                try:
                    frame = self.device.screenshot()
                except Exception:
                    log.debug("喊一声连拍截图失败", exc_info=True)
                    watch.skipped = True
                    break
                if is_black(frame):
                    watch.skipped = True
                    break
                watch.add(frame, self.clock())
            self.env.called(at)
        self._call_at = at
        self._ref_thumb = None  # 光圈、面板开关不算画面大变
        result = CallResult(at, reason)
        if cfg.halo:
            result.halo, tid = watch.result(int(base.shape[1]))
            if result.halo == "self" and tid is not None:
                result.self_box = watch.boxes[tid]
                yolo = [t for t in tracks if t.cls == "self"]
                if len(yolo) != 1:  # YOLO 已经稳稳认出团子就不覆盖
                    self.env.self_box = result.self_box
                    log.info("光圈认出团子：%s", result.self_box)
        self.last_call = result
        log.info("喊了一声（%s），光圈：%s", reason, result.halo)
        return result

    def _watch_call(self, now: float) -> None:
        """身体自动兜底（spec §2.3）：好友刚"走开"、画面里还有没挂名字的人 → 多半只是走远了标签淡掉，喊一声确认。
        喊完不等：窗口结束后放背景事件 call；认回来的好友照常 return（抵消那条 leave）。"""
        cfg = self.cfg.call
        if not (cfg.enabled and cfg.auto) or self.env is None or not hasattr(self.env, "unnamed"):
            return
        try:
            self._collect_auto_call(now)
            if self._pending_auto is not None:
                return
            who = [n for n, t in self._left_at.items()
                   if now - t <= cfg.auto_after_leave and n not in self._nearby and self._auto_called.get(n) != t]
            if not who or now - self._call_at < cfg.min_gap:
                return
            while self._call_times and now - self._call_times[0] > cfg.auto_window:
                self._call_times.popleft()
            if len(self._call_times) >= cfg.auto_quota or self.env.unnamed(now) <= 0 or self._auto_call_blocked(now):
                return
            for n in who:
                self._auto_called[n] = self._left_at[n]
            self._call_times.append(now)
            log.info("%s 刚走开、画面里还有没挂名字的人：自动喊一声找找", "、".join(who))
            r = self.call_out("auto")
            if r.refused:
                log.info("自动喊一声没喊成：%s", r.refused)
            elif not r.dry:
                self._pending_auto = r
        except Exception:
            log.exception("自动喊一声出错")

    def _auto_call_blocked(self, now: float) -> bool:
        if self._bubble_at is not None or self.sender.opened or self.skills.active is not None:
            return True
        if self._raised is not None or self.brain_busy() or self.blackout:
            return True
        max_age = self.cfg.social.max_age  # 带着请求圈走开的好友：请求留在 env.requests 里不会被清，过时的不算
        if any(now - getattr(r, "seen_at", now) <= max_age for r in list(dict(getattr(self.env, "requests", None) or {}).values())):
            return True
        if self.panels is not None and self.panels.state.others():
            return True
        if self.emotes is not None and now - self.emotes.last_any < self.cfg.reflex.min_gap:
            return True
        return bool(self.device.ime_shown())  # 最后才问（一次 adb）

    def _collect_auto_call(self, now: float) -> None:
        r = self._pending_auto
        if r is None:
            return
        seen = self.env.call_result(r.at)
        if seen is not None:
            r.seen = seen
            self.events.put("call", call_event_text(seen))
            self._pending_auto = None
        elif now - r.at > self.cfg.call.window + 10:
            log.info("自动喊一声的结果一直没等到（感知层暂停了？），不等了")
            self._pending_auto = None

    def stop_task(self) -> str:
        return self.skills.cancel(self, "大脑叫停")

    def _stop_camera_task(self, why: str) -> str:
        """要自己转镜头 / 点人之前：正在跑要转镜头的技能（track）就先停下，返回给结果用的说明（没停返回空串）。"""
        skill = self.skills.active
        if skill is None or not getattr(skill, "needs_camera", False):
            return ""
        goal = skill.goal
        self.skills.cancel(self, why)
        return f"（先停下了{goal}）"

    def track(self, name: str, seconds: int = 30, live: bool = False) -> str:
        """开始技能 track：小步转镜头把这个好友保持在画面中间，做完 / 跟丢发 task_done / task_failed。

        跟踪期间不调 env.held()（要靠感知层一直看着他）；镜头不自动复原，交给 camera_reset。"""
        if self._dry(live):
            raise ToolError("dry-run 不转镜头盯人")
        if self.camera is None:
            raise ToolError("没有视角控制，盯不了人")
        if self.env is None or not hasattr(self.env, "people"):
            raise ToolError("没开感知层（[perception]），认不准人在画面哪里，盯不了")
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在盯不了")
        seconds = max(1, min(int(seconds), self.cfg.track.max_seconds))
        now = self.clock()
        if self.target_x(name, now) is None:
            known = self._recognized(now)
            where = f"现在认得出：{'、'.join(known)}" if known else "现在一个名字都没认出来"
            raise ToolError(f"画面里没看到 {name}（{where}）；先 look_around 找找")
        note = self.clear_view("camera", live)
        return self.skills.start(self, TrackSkill(name, seconds)) + note

    def camera_reset(self, live: bool = False) -> str:
        if self.camera is None:
            raise ToolError("没有视角控制")
        note = self.clear_view("camera", live)
        if self._dry(live):
            return "dry-run：没真的转" + note
        stopped = self._stop_camera_task("要把镜头复位")
        with self._held("camera"):
            result = self.camera.reset()
        self._ref_thumb = None
        self._camera_moved_at = self.clock()
        self._forget_self()
        return result + stopped

    # ---- 内心层第 2 期：心情、精力、反思（spec 2026-09-30-inner-phase2）----
    def _inner_call(self, fn: Callable[[], Any], default=None):
        """内心层出错只记日志，不影响身体。"""
        try:
            return fn()
        except Exception:
            log.exception("内心层出错")
            return default

    def effects(self) -> Effects:
        """心情和精力的倍数（主动额度、反射、心跳）；没有内心层或出错时是中性的。"""
        if self.mind is None:
            return NEUTRAL
        try:
            return inner_effects(self.mind.mood.level, self._energy.level if self._energy is not None else "精神")
        except Exception:
            log.exception("算心情的后果出错")
            return NEUTRAL

    def energy_now(self) -> Energy:
        return inner_energy(*self._energy_args())

    def _energy_args(self) -> tuple[float, float, bool, float]:
        """(几点, 连着挂了几分钟, 最近有没有好友跟团子说话, 最近一小时热闹了几分钟)：精力和 introspect 共用。"""
        wall = self.wall()
        t = time.localtime(wall)
        if self.ledger is not None:
            awake = awake_minutes(wall, self.ledger.session.start, self.ledger.history, self.cfg.inner.rest_gap)
        else:
            awake = max(0.0, wall - self._born_wall) / 60
        busy = sum(sec for at, sec in self._busy_log if wall - at <= 3600) / 60
        return t.tm_hour + t.tm_min / 60, awake, wall - self._cheered_at < 600, busy

    # ---- 幕后：introspect（spec 2026-10-01-backstage §2；只读、身体线程）----
    def introspect(self, topic: str) -> str:
        owner = self.cfg.brain.owner_name or "卡洛"
        off = f"{topic}没开（{owner}没打开这个）"
        wall = self.wall()
        if topic == "精力":
            args = self._energy_args()
            return "精力：" + format_parts(energy_parts(*args), inner_energy(*args))
        if topic == "反思":
            if self.reflector is None:
                return off
            rows = [r for r in (self.mind_log.recent() if self.mind_log is not None else []) if r.get("kind") == "reflect"][-3:]
            lines = [self._reflect_line(r, wall) for r in rows] or ["这次上线还没反思过"]
            nxt = self.reflector.next_in(self.clock())
            if nxt is not None:
                lines.append(f"下次大概 {max(1, round(nxt / 60))} 分钟后（有动静才反思）")
            return "\n".join(lines)
        if topic == "性格":
            if self.persona is None:
                return off
            return self._persona_lines(wall)
        if topic == "日记":
            if self.ledger is None or not self.cfg.inner.reflect:
                return off
            diaries = self.ledger.store.last_diaries(1)
            if not diaries:
                return "还没写过日记"
            m = re.match(r"(\d+月\d+日：)?(.*)", diaries[-1], re.S)  # 日期不算进 600 字
            return "最近一篇日记：" + (m.group(1) or "") + m.group(2)[:600]
        raise ToolError("topic 只能是 精力 / 反思 / 性格 / 日记 / 眼睛")

    @staticmethod
    def _reflect_line(r: dict, wall: float) -> str:
        mood = r.get("mood") or {}
        text = f"{ago(wall - r['t'])}前{'（下线那次）' if r.get('final') else ''}：心情{mood.get('level', '')}（{mood.get('text', '')}）"
        if r.get("changes"):
            text += "；改了：" + "、".join(r["changes"])
        if r.get("dropped"):
            text += "；没收下：" + "、".join(r["dropped"])
        return text

    def _persona_lines(self, wall: float) -> str:
        fade = self.cfg.inner.fade_days

        def line(t, label: str) -> str:
            left = max(0, int(fade - (wall - max(t.last_used, t.since)) / 86400))
            used = f"{ago(wall - t.last_used)}前用过" if t.last_used else "还没用过"
            return f"- {label}：用过 {t.hits} 次，{used}，再过 {left} 天不用就淡出"

        p = self.persona
        lines = [line(t, f"口头禅「{t.text}」") for t in p.catchphrases]
        lines += [line(t, f"和{t.who}的老梗「{t.text}」") for t in p.jokes]
        lines += [line(t, f"对{t.topic}的看法「{t.text}」") for t in p.opinions]
        return "\n".join(lines) if lines else "还没攒下什么口头禅、老梗和看法"

    def _safe_friends(self) -> list[str]:
        try:
            return list(self.friend_names())
        except Exception:
            log.exception("读好友名单出错")
            return []

    def _all_from_grudge(self, fresh: list[Message]) -> bool:
        if self.mind is None or self.mind.grudge is None:
            return False
        wall, friends = self.wall(), self._safe_friends()
        return all(self.mind.grudge_on(match_friend(m.speaker, friends) or m.speaker, wall) for m in fresh)

    def _reflect_note(self, text: str) -> None:
        if self.mind is None:
            return
        self._reflect_comings.append(text)
        self._session_comings.append(text)
        if self.reflector is not None:
            self.reflector.stirred(self.clock())

    def _inner_tick(self) -> None:
        """每圈：别扭 / 心愿到期、算精力、记热闹、取回反思结果、该反思了就开始。"""
        if self.mind is None:
            return
        wall, now = self.wall(), self.clock()
        self._inner_call(lambda: self.mind.expire(wall))
        self._energy = self._inner_call(self.energy_now)
        if self.mind_log is not None and wall - self._energy_logged >= ENERGY_EVERY:
            self._energy_logged = wall
            self._inner_call(lambda: self.mind_log.energy(wall, self._energy))
        self._inner_call(lambda: self._track_busy(wall))
        if self.reflector is None:
            return
        result = self._inner_call(self.reflector.poll)
        if result is not None:
            self._inner_call(lambda: self.apply_reflection(result))
        if self._inner_call(lambda: self.reflector.due(now), default=False):
            content = self._inner_call(lambda: self.reflect_materials(False))
            if content is not None:
                self._reflect_chat, self._reflect_comings = [], []
                self._inner_call(lambda: self.reflector.start(content, now))

    def _track_busy(self, wall: float) -> None:
        last, self._last_tick_wall = self._last_tick_wall, wall
        if last is not None and self.occasion().level == "busy":
            self._busy_log.append((wall, min(max(0.0, wall - last), self.cfg.inner.max_step)))
        while self._busy_log and wall - self._busy_log[0][0] > 3600:
            self._busy_log.popleft()

    def reflect_materials(self, final: bool) -> str:
        """拼给反思的材料：上次反思以来的聊天和来去、相关好友的关系卡、笔记里提到他们的行。"""
        wall = self.wall()
        if final:  # 下线前还没结束的冷场也写进去
            for lull in self._lull_call(lambda: self.lulls.flush(wall), []) or []:
                self._lull_note(lull)
        friends = self._safe_friends()
        names = list(dict.fromkeys(
            [n for n in (match_friend(who, friends) for _, who, _ in self._reflect_chat) if n]
            + [n for n in (match_friend(x, friends) for x in (self.env.nearby(self.clock()) if self.env is not None else [])) if n]
        ))
        cards = []
        for n in names:
            card = self.ledger.card(n) if self.ledger is not None else None
            if card is not None:
                cards.append(card_line(n, card, wall))
        notes = [line.strip() for line in (self.memory_notes() or "").splitlines()
                 if line.strip() and not line.startswith("#") and any(n in line for n in names)][:10]
        mind_line = self.mind.line(wall, self._energy) if self.mind is not None else ""
        if self.mind is not None and self.mind.updated is not None:
            mind_line += f"（{ago(wall - self.mind.updated)}前想的）"
        chat, comings = (self._session_chat, self._session_comings) if final else (self._reflect_chat, self._reflect_comings)
        return reflect_materials_text(
            wall, self._energy.note if self._energy is not None else "", mind_line, list(chat), list(comings), cards, notes,
            self.profile_text() or "", final, self._traits(),
        )

    def _traits(self) -> str | None:
        """反思材料里的「你攒下的性格」：性格关着 / 拼出错 → None（不写这一段）。"""
        if self.persona is None:
            return None
        return self._inner_call(self.persona.section)

    def apply_reflection(self, result: dict) -> None:
        """反思结果套进 Mind（身体线程里）；live 时写 mind.json。"""
        if self.mind is None:
            return
        cards = self.ledger.cards if self.ledger is not None else {}
        before_mind, before_persona = copy.deepcopy(self.mind), copy.deepcopy(self.persona)  # 流水账记“改了什么”用
        dropped = self.mind.apply(result, cards, self._safe_friends(), self.wall(), self.cfg.inner)
        for why in dropped:
            log.info("反思：%s", why)
        log.info("反思完了，心里：%s", self.mind.line(self.wall(), self._energy))
        self._save_mind()
        if self.persona is not None:
            dropped = dropped + (self._inner_call(lambda: self._apply_persona(result, cards), default=[]) or [])
        if self.mind_log is not None:
            self._inner_call(lambda: self.mind_log.reflect(
                self.wall(), False, self.mind, self._energy, diff_inner(before_mind, self.mind, before_persona, self.persona), dropped))

    # ---- 内心层第 3 期：性格、收着点（spec 2026-09-30-inner-phase3）----
    def _apply_persona(self, result: dict, cards) -> list[str]:
        """反思结果套进性格档案；live 时写 persona.json。返回丢掉了哪些。"""
        wall = self.wall()
        dropped = self.persona.apply(result, cards, self._safe_friends(), wall, self.cfg.inner, soft=self.soft_names(wall))
        for why in dropped:
            log.info("性格：%s", why)
        self._save_persona()
        return dropped

    def _save_persona(self) -> None:
        if self.persona is not None and self.ledger is not None and self.ledger.persist and self.ledger.store is not None:
            self.ledger.store.write_persona(self.persona)

    # ---- 内心页（spec 2026-09-30-inner-viewer §4）：只在身体线程里调（viewer 经 body.call）----
    def forget(self, kind: str, text: str, who: str = "", topic: str = "") -> str:
        """网页上删一条性格条目。空字符串 = 删掉了；否则是原因。live 时写 persona.json；都记一条 forget。"""
        if self.persona is None:
            return "性格档案没开"
        if kind not in FORGET_KINDS:
            return "不认识的类别"
        if not self.persona.remove(kind, text, who, topic):
            return "找不到这条（可能已经淡出了）"
        log.info("网页上删了性格条目：%s %s%s%s", kind, who + "——" if who else "", topic + "——" if topic else "", text)
        self._inner_call(self._save_persona)
        if self.mind_log is not None:
            self._inner_call(lambda: self.mind_log.forget(self.wall(), kind, text, who, topic))
        return ""

    def inner_snapshot(self) -> dict:
        """内心页的「现在」：心情、精力、别扭、心愿、收着点、性格档案、内存里最近的流水账。没开反思时这几项是 None / 空。"""
        wall = self.wall()
        m, e, g = self.mind, self._energy, self.mind.grudge if self.mind is not None else None
        return {
            "running": True,
            "at": wall,
            "mood": {"level": m.mood.level, "text": m.mood.text, "since": m.mood.since} if m is not None else None,
            "energy": {"level": e.level, "score": e.score, "note": e.note} if m is not None and e is not None else None,
            "grudge": {"who": g.who, "why": g.why, "until": g.until} if g is not None and wall < g.until else None,
            "wants": [{"kind": w.kind, "text": w.text, "who": w.who, "until": w.until} for w in m.wants] if m is not None else [],
            "soft": [{"who": who, "text": text, "until": until} for who, (until, text) in self._soft_until.items() if wall < until],
            "persona": self.persona.to_dict() if self.persona is not None else None,
            "log": self.mind_log.recent() if self.mind_log is not None else [],
            "musing": self._lull_call(lambda: self.lulls.snapshot(wall), []) or [],
        }

    def _watch_upset(self, fresh: list[Message]) -> None:
        """好友说难过 / 不舒服……：接下来 soft_minutes 分钟对他收着点（不损、不唱反调、不拒绝）。"""
        friends, wall = self._safe_friends(), self.wall()
        for m in fresh:
            who = match_friend(m.speaker, friends)
            if who is not None and sounds_upset(m.text):
                self._soft_until[who] = (wall + self.cfg.inner.soft_minutes * 60, m.text[:20])
                log.info("%s 说「%s」：接下来对他收着点", who, m.text)

    def soft_names_this_session(self) -> set[str]:
        """这次上线里说过难过的好友（下线反思看的是整次上线：这些人都不记新老梗）。"""
        return set(self._soft_until)

    def soft_names(self, wall: float) -> set[str]:
        """现在要收着点的好友。"""
        return {who for who, (until, _) in self._soft_until.items() if wall < until}

    def _soft_line(self, wall: float) -> str:
        live = [(who, text) for who, (until, text) in self._soft_until.items() if wall < until]
        if not live:
            return ""
        if len(live) == 1:
            return f"对{live[0][0]}收着点（他刚说「{live[0][1]}」）"
        return "对" + "、".join(who for who, _ in live) + "收着点（他们刚说难过）"

    def _save_mind(self) -> None:
        """live 时写 mind.json。"""
        if self.mind is not None and self.ledger is not None and self.ledger.persist and self.ledger.store is not None:
            self.ledger.store.write_mind(self.mind)

    def _ledger_call(self, method: str, *args, default=None):
        """内心账本（spec 2026-09-30-inner-phase1 §5）：出错只记日志，不影响身体。"""
        if self.ledger is None:
            return default
        try:
            return getattr(self.ledger, method)(*args)
        except Exception:
            log.exception("内心账本出错（%s）", method)
            return default

    def _owner(self, now: float) -> bool:
        """卡洛的 # 命令还在生效：move / emote / camera 放宽限制（设计见 2026-09-27-brain-move-design.md）。"""
        return now < self._owner_window_until

    def _dry(self, live: bool) -> bool:
        """只打印不执行：dry-run 且不是手动控制（live）。"""
        return self.cfg.reply.dry_run and not live

    def _forget_camera_reference(self) -> None:
        forget = getattr(self.camera, "forget_reference", None)
        if forget is not None:
            forget()

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
