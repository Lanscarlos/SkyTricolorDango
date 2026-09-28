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

from ..agent import RateLimiter
from ..chat.memory import Turn
from ..chat.panel import PanelKeeper
from ..chat.reader import Message
from ..chat.responder import clean_reply, format_incoming
from ..config import Config
from ..game.social import IDLE, KIND_NAMES
from .events import EventQueue
from ..imageio import imwrite
from ..vision.bubbles import roi_rect
from .images import crop_view, difference, fit, image_block, is_black, label_note, thumb

log = logging.getLogger(__name__)

REQUEST_KINDS = ("hand", "hug", "highfive", "piggyback", "candle", "*")
PANEL_LOST_AFTER = 30.0  # 面板关了这么久（自动重开也没成功）就告诉大脑
SCENE_EVENT_COOLDOWN = 10.0


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
        friend_checker=None,  # game.friendtree.FriendChecker：点人物看好友树
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
        self.friend_checker = friend_checker
        self._last_friend_check = float("-inf")
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
        self.look_frame = None  # 最近一次 look（原图）看的那张，look_at 裁它
        self.blackout = False
        self.holding: str | None = None  # 推测正牵着谁的手
        self._holding_since = 0.0
        self._accepted_hand: tuple[str, float] | None = None
        self._owner_window_until = float("-inf")  # now < 这个值 = 卡洛的 # 命令还在生效
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

    # ---- 主循环 ----
    def step(self) -> None:
        now = self.clock()
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
                self._sense(frame, now)
            except Exception:
                log.exception("感知出错，这一圈跳过")
                self.events.put("error", "身体感知出错了（详见日志）")
            self._heard(fresh, frame, now)
        self._run_commands()
        self._fallback(now)

    def _sense(self, frame, now: float) -> None:
        if self.cfg.vision.mode == "log":
            self.panel.maybe_reopen(now)
            self._watch_panel(now)
        self._watch_screen(frame, now)
        if self.env is not None:
            self.env.observe(frame, now, panel_visible=self.reader.panel_closed_since is None)
            self._watch_people(now)

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
        """退出时（不等大脑）：不再接命令、排队的命令全部失败、镜头转回原位、轮盘换回去。"""
        self.stopped = True
        while True:
            try:
                _, fut = self._commands.get_nowait()
            except queue.Empty:
                break
            if fut.set_running_or_notify_cancel():
                fut.set_exception(ToolError("身体已经停了"))
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
                self.events.put("owner_command", f"卡洛的命令：{m.text}")
                log.info("识别到卡洛的命令：%s（授权窗口延长到 %.0f 秒后）", m.text, self.cfg.brain.owner_window)
            else:
                self.events.put("chat", f"聊天  {m.speaker or '（看不出是谁）'}：「{m.text}」")

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

    # ---- 给大脑用的：都在身体线程里执行（经 call()） ----
    def look(self) -> list[dict]:
        now = self.clock()
        brain = self.cfg.brain
        if now - self.last_look < brain.look_min_interval:
            raise ToolError(f"{brain.look_min_interval:.0f} 秒内刚看过，等一下再看")
        frame = self.device.screenshot()
        self.last_frame, self.last_look, self.look_frame = frame, now, frame
        view = fit(frame, tuple(brain.image_size))
        recent = {}
        if self.env is not None:  # 最近两次扫描内看到的名字才算在画面里；先拍快照，env 后台线程会改这个 dict
            recent = {n: v for n, v in dict(self.env.labels).items() if now - v[4] <= self.cfg.env.interval * 2 + 1}
        note = label_note(recent, view.shape[1] / frame.shape[1]) if self.env is not None else "（没开环境识别，认不出名字）"
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

    def fresh_frame(self):
        """眼睛马上要看：在身体线程里截一张新的。"""
        frame = self.device.screenshot()
        self.last_frame = frame
        return frame

    def capture_around(self) -> list:
        """环顾四周：转一圈，每 90° 截一张（dry-run 不转，只截当前画面）。"""
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在看不了")
        if self.camera is None or self.cfg.reply.dry_run:
            return [self.fresh_frame()]
        try:
            frames = self.camera.around(self.device.screenshot)
        finally:
            self._ref_thumb = None  # 自己转的镜头，不算画面大变
        self.last_frame = frames[0]
        return frames

    def status(self) -> str:
        now = self.clock()
        parts = ["聊天记录面板" + ("开" if self.reader.panel_closed_since is None else "关")]
        try:
            parts.append("输入框" + ("开" if self.device.ime_shown() else "关"))
        except Exception:
            parts.append("输入框状态读不到")
        near = self.env.nearby(now) if self.env is not None else []
        parts.append("身边的好友：" + ("、".join(near) if near else "没看到"))
        if hasattr(self.env, "strangers"):
            parts.append(f"身边的陌生人：{self.env.strangers(now)} 个")
        if self.holding:
            parts.append(f"牵着手：{self.holding}（推测）")
        if self.blackout:
            parts.append("画面黑着")
        if self.camera is not None:
            parts.append("镜头：" + self.camera.describe())
        parts.append("上次看图：" + (f"{now - self.last_look:.0f} 秒前" if self.last_look > float("-inf") else "还没看过"))
        if self.emotes is not None:
            parts.append("能做的动作：" + ("、".join(self.emotes.available()) or "暂时没有（刚做过，要等一会儿）"))
        if self.social is not None:
            parts.append("互动规则：" + self.social.describe_policy())
        if self.said:
            parts.append("刚说过：" + " | ".join(self.said[-3:]))
        if self.cfg.reply.dry_run:
            parts.append("dry-run（说话、动作、转视角都不会真的执行）")
        return " / ".join(parts)

    def chat_log(self, n: int = 20) -> str:
        rows = list(self.chat)[-max(1, min(n, 50)) :]
        if not rows:
            return "还没有聊天"
        return "\n".join(
            f"{time.strftime('%H:%M:%S', time.localtime(t))} {who or '（看不出是谁）'}：{text}" for t, who, text in rows
        )

    def say(self, text: str) -> str:
        now = self.clock()
        body = clean_reply(text, self.cfg.reply.max_chars)  # 含“不能声称自己是真人”的硬过滤
        if body is None:
            raise ToolError("这句没发：是空的、<skip>，或者说了自己是真人（不能这么说）。换个说法")
        if not self.limiter.allow(now):
            raise ToolError("说得太快了，等几秒再说")
        full = self.cfg.reply.disclosure_prefix + body
        self.limiter.record(now)
        self.said.append(full)
        self.said[:] = self.said[-50:]  # 只留最近 50 条，别无限长
        self.chat.append((self.wall(), "我", full))
        if self.cfg.reply.dry_run:
            log.info("[dry-run] 将会发送: %s", full)
            self._remember(body, full, sent=False)
            return f"dry-run：没真的发，“{full}”"
        self.sender.send(full)
        self.self_filter.remember(full, self.clock())
        self._remember(body, full, sent=True)
        return f"已发送：{full}"

    def _remember(self, body: str, full: str, sent: bool) -> None:
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

    def emote(self, name: str, force: bool = False) -> str:
        if self.emotes is None:
            raise ToolError("这次没开动作（--no-emotes 或者图标库是空的）")
        available = self.emotes.available()
        if name not in available:
            raise ToolError(f"「{name}」现在做不了；能做的：{'、'.join(available) or '暂时没有（刚做过动作，要等一会儿）'}")
        if self.holding and not force:
            raise ToolError(f"正牵着 {self.holding} 的手，做动作会松手；确定要松手再做就传 force=true")
        self.emoted.append(name)
        self.emoted[:] = self.emoted[-50:]  # 只留最近 50 条，别无限长
        if self.cfg.reply.dry_run:
            self.emotes.pretend(name)
            return f"dry-run：没真的做「{name}」"
        try:
            self.emotes.perform(name)
        except Exception as exc:
            raise ToolError(f"「{name}」没做成：{exc}") from None
        return f"做了「{name}」"

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
        if now - self._last_friend_check < fc.min_interval:
            raise ToolError(f"刚确认过，{fc.min_interval - (now - self._last_friend_check):.0f} 秒后再点")
        fh, fw = self.look_frame.shape[:2]
        scale = fw / self.cfg.brain.image_size[0]
        sx, sy = round(x * scale), round(y * scale)
        if not (0 <= sx < fw and 0 <= sy < fh):
            raise ToolError("坐标在画面外（按 look(image=true) 那张 1280×720 的图给）")
        area = roi_rect(self.cfg.env.roi, fw, fh)
        if not (area.x <= sx < area.x2 and area.y <= sy < area.y2):
            raise ToolError("那里是底部的按钮栏，不是人")
        panel = roi_rect(self.cfg.vision.log_roi, fw, fh)
        if self.reader.panel_closed_since is None and panel.x <= sx < panel.x2 and panel.y <= sy < panel.y2:
            raise ToolError("那里被聊天记录面板挡着，点不到人")
        if self.cfg.reply.dry_run:
            return f"dry-run：没真的点（会点原图上的 ({sx}, {sy})，打开好友树看完再关掉）"
        self._last_friend_check = now
        try:
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

    def camera_move(self, action: str, steps: int = 1) -> str:
        if self.camera is None:
            raise ToolError("没有视角控制")
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在转不了")
        if self.cfg.reply.dry_run:
            return f"dry-run：没真的转（{action} ×{steps}）"
        try:
            result = self.camera.move(action, steps)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        self._ref_thumb = None  # 自己转的镜头，不算画面大变
        return "镜头现在：" + result

    def camera_reset(self) -> str:
        if self.camera is None:
            raise ToolError("没有视角控制")
        if self.cfg.reply.dry_run:
            return "dry-run：没真的转"
        result = self.camera.reset()
        self._ref_thumb = None
        return result

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
            self.say(reply.text)
        except ToolError as exc:
            log.warning("备用回复没发出去：%s", exc)
            return
        self.events.put("fallback", f"大脑离线时，备用回复说了：{reply.text}")
