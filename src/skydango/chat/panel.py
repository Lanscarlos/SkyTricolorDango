"""聊天记录面板（光遇里按 C）的开关：只有这里按开面板的键，Agent 和大脑的身体共用一个。

- always：一直开着，关久了自动重开（原来的做法）
- auto：平时关着（闲着 idle），定时 / 有人来时按一下看一眼（peek），读到新消息就一直开着（聊天中 chatting），
  安静 quiet_close 秒再关上；好友头顶冒出"正在输入"气泡时开着等（bubble），对方发出来才进
- auto + 无障碍读法（reader.reads_bubbles 为真）：新消息全是从头顶气泡读到的 → 聊着（talking），面板关着，
  每 chat_peek 秒看一眼面板（接住画面外的人），安静 quiet_close 秒回闲着；团子说话时跟谁聊的名字标签都还看得见就不开面板
- 面板管理者自己不读聊天：主循环照常 reader.read()，把结果和"这一帧面板开没开"交给 tick()
- 转镜头、换轮盘、接互动、好友树、技能要面板关着时用 borrow() 借走，用完归还；嵌套时最外层归还才恢复
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from ..config import PanelConfig, VisionConfig
from ..device.base import Device

log = logging.getLogger(__name__)

CLOSE_DELAY = 0.8  # 按键关面板后等它消失
POLL = 0.1  # 等面板出现时多久截一张图看
# 无障碍读法（10-04 晚真机）：关面板的动画约 2 秒，这段时间树里的面板节点还在、还会闪（关了 2.6 秒后还"看得到"），
# 自己关的这么久之内看到面板不算别人打开的；快照每 0.15 秒一份，看一眼至少开这么久，等面板行加载完再关
A11Y_CLOSE_SETTLE = 3.5
A11Y_PEEK_MIN = 1.0
TAG_RECENT = 3.0  # 好友的名字标签这么久之内看到过就算"在画面里"：标签会闪，只看这一帧会让团子一说话就开面板（10-04 晚）


class PanelManager:
    def __init__(
        self,
        vision: VisionConfig,
        cfg: PanelConfig,
        device: Device,
        reader,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.vision = vision
        self.cfg = cfg
        self.device = device
        self.reader = reader  # ChatReader：panel_visible(frame)、panel_closed_since
        self.sleep = sleep
        self.clock = clock
        self._last_reopen = float("-inf")
        self._depth = 0  # borrow 嵌套了几层
        self.lent: str | None = None  # 正借给谁（最外层）
        # auto 模式的状态（always 模式恒为 chatting）
        self.state = "idle" if cfg.mode == "auto" else "chatting"
        self._last_read = 0.0  # 上次读到面板（面板开着的每一帧都算）
        self._last_activity = 0.0  # 聊天中：上次有新消息 / 团子说话 / 在准备回复
        self._last_peek = float("-inf")  # 上次看一眼结束
        self._closed_at = float("-inf")  # 上次自己关上面板
        self._opened_at = 0.0  # 看一眼：什么时候按的键
        self._seen = 0  # 看一眼：打开后连续几帧看到了面板
        self._seen_at: float | None = None  # 看一眼：第一次看到面板的时刻
        self._pending: str | None = None  # 闲着时等着看一眼的原因（冷却中先记着）
        self._missing_since: float | None = None  # 聊天中：面板从什么时候开始不见了
        self._bubble_start = 0.0  # 等气泡：什么时候开始等
        self.on_press: Callable[[float], None] | None = None  # 按 C 时以按键时刻调用（身体用它先告诉感知层画面要横移了）
        self._pressed_at = float("-inf")  # 上次按开面板的键（self.clock()）：之后 open_timeout 秒内面板可能还在动画里
        self._expect_open = True  # 上次按键后面板应该是开还是关
        self._returned_was_open = True  # 最近一次归还的借出：借之前面板开没开
        self._last_bubble = 0.0  # 等气泡：上次看到气泡
        self._hold_until: float | None = None  # 空闲注意力"先看一眼"：闲着时推迟到这个时间再开面板
        self._held_once = False  # 这个 _pending 已经推迟过一次
        self._talk_with: set[str] = set()  # 聊着：跟谁聊（在画面里说过话的人）
        self._tag_seen: dict[str, float] = {}  # 好友名 → 最后一次看到他的名字标签（无障碍读法）
        self._talk_since = 0.0  # 聊着：什么时候开始的（看一眼的计时从这儿和上次读到面板里晚的那个算）
        self._peek_from = "idle"  # 看一眼 / 等气泡从哪个状态发起：没读到新消息时回到它（聊着 / 闲着）
        self.still: str | None = None  # 先别动（谁要的）：tick 不按键、不开不关（hold_still）

    @property
    def auto(self) -> bool:
        return self.active and self.cfg.mode == "auto"

    @property
    def active(self) -> bool:
        """log 模式且配了开面板的键：否则什么键都不按。"""
        return self.vision.mode == "log" and bool(self.vision.log_open_key)

    def visible_now(self) -> bool:
        return self.active and bool(self.reader.panel_visible(self.device.screenshot()))

    def ensure_open(self) -> bool:
        """log 模式：看不到聊天记录面板、也没在打字时，按一下打开面板的键（默认 C）。返回面板现在开没开。"""
        key = self.vision.log_open_key
        if self.vision.mode != "log" or not key:
            return True
        if self.reader.panel_visible(self.device.screenshot()):
            return True
        if self.device.ime_shown():
            log.warning("输入框开着，没法用按键打开聊天记录面板；请手动打开（光遇里按 C）")
            return False
        log.info("聊天记录面板没打开，按键 %d 打开", key)
        self._press(True)
        self.sleep(1.0)
        if self.reader.panel_visible(self.device.screenshot()):
            return True
        log.warning("按了键聊天记录面板还是没出现，可能被别的界面挡住了，请看一下游戏画面")
        return False

    def start(self, now: float) -> bool:
        if not self.auto:
            return self.ensure_open()
        self.state, self._last_read = "idle", now
        log.info("聊天面板按需打开：平时关着，每 %.0f 秒看一眼", self.cfg.idle_peek)
        return True

    def _reads_bubbles(self) -> bool:
        """reader 现在能读头顶气泡（无障碍读法）；OCR 的 ChatReader 没有这个属性。"""
        return bool(getattr(self.reader, "reads_bubbles", False))

    def _user_opened(self, now: float) -> bool:
        """面板开着、不是刚被自己关上还在动画里：算别人（玩家）打开的。"""
        settle = max(self.cfg.open_timeout, A11Y_CLOSE_SETTLE) if self._reads_bubbles() else self.cfg.open_timeout
        return now - self._closed_at > settle

    def should_be_open(self) -> bool:
        if self.lent is not None:
            return False
        return self.state == "chatting" if self.auto else True

    def trigger(self, reason: str, now: float) -> None:
        """闲着 / 聊着时有事（有人来了、走近、画面恢复、大脑要看聊天、挂不上名字的气泡）：尽快看一眼（冷却中先记着）。"""
        if self.auto and self.state in ("idle", "talking") and self._pending is None:
            self._pending = reason

    @property
    def pending(self) -> str | None:
        """闲着时等着看一眼的原因（"bubble" / "approach" / "arrive"…）；没有是 None。"""
        return self._pending

    def _until_peek(self, now: float) -> float:
        return self.cfg.idle_peek - (now - self._last_read)

    def quiet(self, now: float, margin: float) -> bool:
        """auto 模式闲着、没借出、没有等着看的、离定时看一眼还有 margin 秒以上：这段时间面板不会开（空闲注意力用）。"""
        return (self.auto and self.state == "idle" and self.lent is None and self._pending is None
                and self._until_peek(now) >= margin)

    def hold_off(self, until: float | None, now: float) -> bool:
        """空闲注意力"先看一眼再开面板"：闲着时推迟到 until 再开（None = 放手）。同一个 _pending 只推迟一次，
        拒绝时返回 False；只管 idle 状态，读到新消息 / 要说话照常马上进聊天。"""
        if until is None:
            self._hold_until = None
            return True
        if self._held_once:
            return False
        self._hold_until, self._held_once = until, True
        log.debug("聊天面板推迟到 %.1f 秒后再开（先看一眼）", until - now)
        return True

    def hold_still(self, who: str | None) -> None:
        """面板先别动（who = 谁要的，None = 放手）：期间 tick 不按开面板的键——不看一眼、不关、不重开，状态照记，放手后照常。
        开关面板时画面横移 300~400 px，点亮陌生人这种盯着画面里一个位置的事会跟丢（10-04 19:51:45）。
        团子说话（before_speak）、借面板（borrow）是明确要按键的，不受影响。"""
        if who != self.still and self.active:
            log.debug("面板先别动：%s" if who else "面板照常（%s 放手了）", who or self.still)
        self.still = who

    def bubble_seen(self, now: float) -> None:
        """这一圈看到了该触发的"正在输入"气泡（好友的；bubble_strangers 时陌生人的也算）。聊着时什么都不做。"""
        if not self.auto:
            return
        if self.state == "idle":
            self._pending = "bubble"
        elif self.state == "bubble":
            self._last_bubble = now

    def before_speak(self, now: float) -> None:
        """团子要说话（按 Enter 之前）：面板要开着，说完对方的回复才能马上读到；进聊天中。
        聊着、跟谁聊的名字标签都还看得见：不开面板（他的回复从头顶气泡读），只刷新活动时间；
        闲着时主动开口、画面里有好友：不开面板，进聊着、跟画面里的好友聊。"""
        if not self.auto:
            return
        if self.state == "talking" and self._talk_with and self._reads_bubbles() and self._in_view(self._talk_with, now):
            self._last_activity = max(self._last_activity, now)
            return
        if self.state == "idle" and self._reads_bubbles() and not self.visible_now():
            near = {n for n, t in self._tag_seen.items() if now - t <= TAG_RECENT}
            if near:
                self._last_activity = max(self._last_activity, now)
                self._talk_since = now
                self._set("talking", "团子要说话，好友在画面里")
                self._talk_with = near
                return
        self._last_activity = max(self._last_activity, now)
        self._missing_since = None
        if self.state != "chatting":
            self._pending = None
            self._set("chatting", "团子要说话")
        if self.lent is not None:
            return  # 借出中：归还时按聊天中重开
        self._settle()
        if self.visible_now() or self.device.ime_shown():
            return  # 输入框开着按键会打出字母
        if not self._open_and_wait():
            log.warning("说话前没能打开聊天记录面板，照样说")

    def busy(self, now: float) -> None:
        """在准备回复（攒消息、等模型、大脑还没回）：聊天中不算安静。"""
        self._last_activity = max(self._last_activity, now)

    def missing_for(self, now: float) -> float:
        """该开却没开了多少秒（给身体发"面板丢了"的事件用）。"""
        if not self.active or not self.should_be_open():
            return 0.0
        if self.auto:
            return now - self._missing_since if self._missing_since is not None else 0.0
        since = self.reader.panel_closed_since
        return now - since if since is not None else 0.0

    def describe(self, now: float) -> str:
        text = self._describe(now)
        return f"{text}（先别动：{self.still}）" if text and self.still and self.lent is None else text

    def _describe(self, now: float) -> str:
        if not self.active:
            return ""
        if self.lent is not None:
            return f"借出：{self.lent}"
        if not self.auto:
            return "常开"
        if self.state == "idle":
            return f"闲着（{max(0, math.ceil(self._until_peek(now)))} 秒后看一眼）"
        if self.state == "bubble":
            return f"等气泡（还剩 {max(0, math.ceil(self.cfg.bubble_wait - (now - self._bubble_start)))} 秒）"
        if self.state == "talking":
            return f"聊着（面板关着，{max(0, math.ceil(self._until_chat_peek(now)))} 秒后看一眼）"
        return {"peek": "看一眼", "chatting": "聊天中"}.get(self.state, self.state)

    def shutdown(self) -> None:
        """退出：面板恢复成开着（和用户接手时的习惯一致）。"""
        if self.auto:
            self._settle()
            if not self.visible_now():
                self.ensure_open()

    def restored(self) -> bool:
        """面板回到了该有的状态（不该开时关着也算）。"""
        if not self.active or not self.should_be_open():
            return True
        if not self.auto and not self._returned_was_open:  # 常开模式：借之前就关着，和原来一样不算"没恢复"
            return True
        self._settle()
        return self.visible_now()

    def tick(self, now: float, fresh: list, visible: bool, blackout: bool = False) -> None:
        """主循环每圈调一次（fresh = 这一圈读到的新消息，visible = 这一帧面板开没开）。不 sleep 太久。"""
        if not self.active or self.lent is not None:
            return
        if not self.auto:
            if self.still is None:
                self._maybe_reopen(now)
            return
        if self._reads_bubbles():
            tags = getattr(self.reader, "tags_in_view", None)
            for name in tags() if callable(tags) else ():
                self._tag_seen[name] = now
            want = getattr(self.reader, "want_peek", None)
            reason = want() if callable(want) else None
            if reason:
                self.trigger(reason, now)
        if fresh and self._reads_bubbles() and self._all_in_view(fresh, now):
            self._heard_in_view(now, fresh, visible)
            return
        if fresh:
            self._last_read = self._last_activity = now
            self._missing_since = None
            if self.state != "chatting":
                self._set("chatting", "读到新消息")
            return
        if visible:
            self._last_read = now
        if self.state == "idle":
            self._tick_idle(now, visible, blackout)
        elif self.state == "peek":
            self._tick_peek(now, visible)
        elif self.state == "bubble":
            self._tick_bubble(now, visible)
        elif self.state == "chatting":
            self._tick_chatting(now, visible, blackout)
        elif self.state == "talking":
            self._tick_talking(now, visible, blackout)

    def _set(self, state: str, why: str) -> None:
        names = {"idle": "闲着", "peek": "看一眼", "bubble": "等气泡", "chatting": "聊天中", "talking": "聊着"}
        log.info("面板：%s → %s（%s）", names.get(self.state, self.state), names.get(state, state), why)
        if state in ("idle", "chatting"):
            self._talk_with = set()
        self.state = state

    def _in_view(self, names, now: float) -> bool:
        """这些好友最近 `TAG_RECENT` 秒内都看到过名字标签。"""
        return all(now - self._tag_seen.get(n, float("-inf")) <= TAG_RECENT for n in names)

    def _all_in_view(self, fresh: list, now: float) -> bool:
        """这一圈的新消息全是从头顶气泡读到的，或者说话的人都在画面里（面板开着时读到的是面板行）。"""
        if all(getattr(m, "source", "") == "bubble" for m in fresh):
            return True
        speakers = {getattr(m, "speaker", "") for m in fresh}
        return "" not in speakers and self._in_view(speakers, now)

    def _heard_in_view(self, now: float, fresh: list, visible: bool) -> None:
        """说话的人都在画面里（无障碍读法）：不用开面板，进聊着 / 刷新活动时间；面板开着就关上。"""
        self._last_activity = now
        if visible:
            self._last_read = now
        speakers = {m.speaker for m in fresh if getattr(m, "speaker", "")}
        if self.state in ("peek", "bubble"):  # 面板正在开：看完回聊着
            self._talk_with |= speakers
            self._peek_from = "talking"
            return
        if self.state == "talking":
            self._talk_with |= speakers
            return
        self._talk_since = now
        if self.state == "chatting" and visible:
            if self.still is not None:  # 先别动：接着聊天中，放手后安静够了照常关
                return
            self._close(now, "说话的人在画面里，关面板接着聊", "talking")  # 输入框开着时 _close 会留在聊天中
        else:
            self._set("talking", "说话的人在画面里")
        if self.state == "talking":
            self._talk_with = speakers

    def _until_chat_peek(self, now: float) -> float:
        return self.cfg.chat_peek - (now - max(self._last_read, self._talk_since))

    def _after_peek(self, now: float) -> str:
        """看一眼 / 等气泡没读到新消息：回到聊着（从聊着发起、还没安静够、还能读气泡）还是闲着；
        从聊着发起、看的途中退回了 OCR（读不了气泡）→ 聊天中（照「聊着立刻改成聊天中」）。"""
        if self._peek_from == "talking" and now - self._last_activity < self.cfg.quiet_close:
            return "talking" if self._reads_bubbles() else "chatting"
        return "idle"

    def _end_peek(self, now: float, why: str) -> None:
        """看一眼 / 等气泡结束：关面板回闲着 / 聊着；要回聊天中（途中退回了 OCR）就让面板开着。"""
        to = self._after_peek(now)
        if to == "chatting":
            self._last_activity = now
            self._set("chatting", "读聊天退回了 OCR")
        else:
            self._close(now, why, to)
        self._last_peek = now

    def _tick_talking(self, now: float, visible: bool, blackout: bool) -> None:
        if not self._reads_bubbles():  # 中途退回了 OCR：读不了气泡，照原来的做法开着面板聊
            self._last_activity = now
            self._set("chatting", "读聊天退回了 OCR")
            if self.still is None:
                self.ensure_open()
            return
        if visible:
            if self._user_opened(now):  # 刚关上的那一阵还看得到，不算
                self._last_activity = now
                self._set("chatting", "面板开着（不是自己开的）")
            return
        quiet = now - self._last_activity >= self.cfg.quiet_close
        due = self._until_chat_peek(now) <= 0
        wanted = self._pending is not None and now - self._last_peek >= self.cfg.peek_cooldown
        if not quiet and (blackout or not (due or wanted) or self.still is not None):
            return
        # 到点了才查输入框（每圈都查要多一条 adb）：开着说明在说话（反射替大脑开的框），算活动、这次不看 ——
        # 交给 _open_peek 会因为输入框开着升级成聊天中，框一关就把面板打开了
        if self.device.ime_shown():
            self._last_activity = now
            return
        if quiet:
            self._set("idle", f"安静 {self.cfg.quiet_close:.0f} 秒")
            return
        self._open_peek(now, self._pending if wanted else "聊着：看一眼面板")

    def _tick_idle(self, now: float, visible: bool, blackout: bool) -> None:
        if visible:
            if self._user_opened(now):  # 刚关上的那一阵还看得到，不算
                self._last_activity = now
                self._set("chatting", "面板开着（不是自己开的）")
            return
        if self._hold_until is not None and now < self._hold_until:  # 空闲注意力在先看一眼
            return
        if self.still is not None:
            return
        due = now - self._last_read >= self.cfg.idle_peek
        wanted = self._pending is not None and now - self._last_peek >= self.cfg.peek_cooldown
        if blackout or not (due or wanted):
            return
        self._open_peek(now, self._pending if wanted else "定时")

    def _open_peek(self, now: float, reason: str) -> None:
        self._peek_from = self.state
        self._pending = None
        self._hold_until, self._held_once = None, False
        if self.device.ime_shown():  # 按键会打出字母；输入框开着说明在说话，本来就该是聊天中
            self._last_activity = now
            self._set("chatting", "输入框开着")
            return
        self._press(True)
        self._opened_at, self._seen, self._seen_at = now, 0, None
        if reason == "bubble":  # 对方还在打字：开着等他发出来
            self._bubble_start = self._last_bubble = now
            self._set("bubble", "好友头顶冒出了气泡")
        else:
            self._set("peek", reason)

    def _tick_peek(self, now: float, visible: bool) -> None:
        if visible:
            self._seen += 1
            if self._seen_at is None:
                self._seen_at = now
            long_enough = not self._reads_bubbles() or now - self._seen_at >= A11Y_PEEK_MIN
            if self._seen >= 2 and long_enough and not self._settling() and self.still is None:  # 第一帧面板可能还没画完；读聊天的还在等确认也先别关
                self._end_peek(now, "看一眼：没有新消息")
        elif now - self._opened_at >= self.cfg.open_timeout:
            log.warning("按了键 %.1f 秒聊天记录面板还没出现（被别的界面挡住了？），下个周期再看", self.cfg.open_timeout)
            self._last_read = self._last_peek = now
            self._set(self._after_peek(now), "没打开")

    def _tick_bubble(self, now: float, visible: bool) -> None:
        if not visible:
            if now - self._opened_at >= self.cfg.open_timeout and now - self._last_read >= self.cfg.open_timeout:
                log.warning("按了键 %.1f 秒聊天记录面板还没出现（被别的界面挡住了？）", self.cfg.open_timeout)
                self._last_read = self._last_peek = now
                self._set(self._after_peek(now), "没打开")
            return
        if now - self._last_bubble >= self.cfg.bubble_gone:
            why = f"气泡没了 {self.cfg.bubble_gone:.0f} 秒也没等到消息"
        elif now - self._bubble_start >= self.cfg.bubble_wait:
            why = f"等了 {self.cfg.bubble_wait:.0f} 秒没等到消息"
        else:
            return
        if self._settling() or self.still is not None:
            return
        self._end_peek(now, why)

    def _tick_chatting(self, now: float, visible: bool, blackout: bool) -> None:
        if visible:
            self._missing_since = None
        elif self._missing_since is None:
            self._missing_since = now
        elif (not blackout and self.still is None and now - self._missing_since >= self.vision.log_reopen_after
              and now - self._last_reopen >= self.vision.log_reopen_cooldown):
            self._last_reopen = now
            self.ensure_open()
        if now - self._last_activity < self.cfg.quiet_close or self._settling() or self.still is not None:
            return
        if self.device.ime_shown():
            self._last_activity = now
            return
        if visible:
            self._close(now, f"安静 {self.cfg.quiet_close:.0f} 秒")
        else:
            self._closed_at = now
            self._set("idle", f"安静 {self.cfg.quiet_close:.0f} 秒")
        self._last_read, self._missing_since = now, None

    def _close(self, now: float, why: str, to: str = "idle") -> None:
        if self.device.ime_shown():  # 按键会打出字母；输入框开着说明在说话，就当聊天中
            self._last_activity = now
            self._set("chatting", "输入框开着")
            return
        self._press(False)
        # 记真按键的时刻，不是这一圈开始的 now：这一圈读聊天、跑识别可能已经花了快 1 秒，
        # 读聊天确认面板关了又要晚零点几秒，用 now 会超过 open_timeout，被当成"别人打开的"（2026-09-30 实测）
        self._closed_at = self._pressed_at
        self._set(to, why)

    def _settling(self) -> bool:
        """读聊天的看到了新行、在等下一帧确认（ChatReader.settling）：这时关面板会让消息拖到下一次才读到。"""
        return bool(getattr(self.reader, "settling", False))

    def _press(self, expect_open: bool) -> None:
        self.device.hw_key(self.vision.log_open_key)
        self._pressed_at, self._expect_open = self.clock(), expect_open
        if self.on_press is not None:
            try:
                self.on_press(self._pressed_at)
            except Exception:
                log.exception("按面板键的回调出错")

    def _settle(self) -> None:
        """刚按过键（open_timeout 秒内）：面板可能还在开 / 关的动画里，截图看到的不准 —— 等它变成按键后该有的样子。"""
        if self.clock() - self._pressed_at >= self.cfg.open_timeout:
            return
        for _ in range(max(1, math.ceil(self.cfg.open_timeout / POLL))):
            if self.visible_now() == self._expect_open:
                return
            self.sleep(POLL)

    def _maybe_reopen(self, now: float) -> None:
        since = self.reader.panel_closed_since
        vision = self.vision
        if since is None or not vision.log_reopen_after or now - since < vision.log_reopen_after:
            return
        if now - self._last_reopen < vision.log_reopen_cooldown:
            return
        self._last_reopen = now
        self.ensure_open()

    @contextmanager
    def borrow(self, who: str, close: bool = True) -> Iterator[bool]:
        """借走面板（要它关着）：close = 面板开着就按键关掉；False = 调用方自己会关（点屏幕会顺带关面板）。
        yield 借之前面板开没开（嵌套借时是 False）。进门按键出错也会归还（不然 lent 一直留着，面板再也不重开）。"""
        was_open = False
        outer = self._depth == 0
        self._depth += 1
        try:
            if outer:
                self.lent = who
                if self.active:
                    log.info("面板借给 %s", who)
                    if self.state in ("peek", "bubble"):  # 看一眼 / 等气泡被打断：归还后补上
                        self._pending = "bubble" if self.state == "bubble" else (self._pending or "补看")
                        # 从聊着发起的回聊着（安静够了 _tick_talking 自己回闲着）；途中退回了 OCR 就回聊天中（归还时重开面板）
                        if self._peek_from == "talking" and self._reads_bubbles():
                            self.state = "talking"
                        elif self._peek_from == "talking":
                            self.state, self._pending, self._talk_with = "chatting", None, set()
                        else:
                            self.state = "idle"
                    self._settle()
                    was_open = self.visible_now()
                    if close and was_open and not self.device.ime_shown():  # 输入框开着按键会打出字母
                        self._press(False)
                        self.sleep(CLOSE_DELAY)
            yield was_open
        finally:
            self._depth -= 1
            if outer:
                self.lent = None
                self._returned_was_open = was_open
                try:
                    self._give_back(was_open)
                except Exception:  # 别盖掉借的一方的异常；重开交给 tick
                    log.exception("归还聊天面板时出错，主循环稍后会再试")

    def _give_back(self, was_open: bool) -> None:
        if not self.active or not self.should_be_open():
            return
        if self.cfg.mode == "always" and not was_open:  # 借之前就关着：和原来一样不管，交给 tick 过一会儿重开
            return
        self._settle()
        if self.visible_now():
            return
        if self.device.ime_shown():  # 按键会变成打字：交给 tick 以后重开
            return
        if not self._open_and_wait():
            log.warning("聊天记录面板没能重新打开，主循环稍后会再试")

    def _open_and_wait(self) -> bool:
        """按一下开面板的键，等它出现（最多 open_timeout 秒）。"""
        self._press(True)
        for _ in range(max(1, math.ceil(self.cfg.open_timeout / POLL))):
            self.sleep(POLL)
            if self.visible_now():
                return True
        return False
