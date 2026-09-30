"""聊天记录面板（光遇里按 C）的开关：只有这里按开面板的键，Agent 和大脑的身体共用一个。

- always：一直开着，关久了自动重开（原来的做法）
- auto：平时关着（闲着 idle），定时 / 有人来时按一下看一眼（peek），读到新消息就一直开着（聊天中 chatting），
  安静 quiet_close 秒再关上；好友头顶冒出"正在输入"气泡时开着等（bubble），对方发出来才进面板管理者自己不读聊天：主循环照常 reader.read()，把结果和"这一帧面板开没开"交给 tick()
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
        self._pending: str | None = None  # 闲着时等着看一眼的原因（冷却中先记着）
        self._missing_since: float | None = None  # 聊天中：面板从什么时候开始不见了
        self._bubble_start = 0.0  # 等气泡：什么时候开始等
        self._pressed_at = float("-inf")  # 上次按开面板的键（self.clock()）：之后 open_timeout 秒内面板可能还在动画里
        self._expect_open = True  # 上次按键后面板应该是开还是关
        self._returned_was_open = True  # 最近一次归还的借出：借之前面板开没开
        self._last_bubble = 0.0  # 等气泡：上次看到气泡
        self._hold_until: float | None = None  # 空闲注意力"先看一眼"：闲着时推迟到这个时间再开面板
        self._held_once = False  # 这个 _pending 已经推迟过一次

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

    def should_be_open(self) -> bool:
        if self.lent is not None:
            return False
        return self.state == "chatting" if self.auto else True

    def trigger(self, reason: str, now: float) -> None:
        """闲着时有事（有人来了、走近、画面恢复、大脑要看聊天）：尽快看一眼（冷却中先记着）。"""
        if self.auto and self.state == "idle" and self._pending is None:
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

    def bubble_seen(self, now: float) -> None:
        """这一圈看到了该触发的"正在输入"气泡（好友的；bubble_strangers 时陌生人的也算）。"""
        if not self.auto:
            return
        if self.state == "idle":
            self._pending = "bubble"
        elif self.state == "bubble":
            self._last_bubble = now

    def before_speak(self, now: float) -> None:
        """团子要说话（按 Enter 之前）：面板要开着，说完对方的回复才能马上读到；进聊天中。"""
        if not self.auto:
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
            self._maybe_reopen(now)
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

    def _set(self, state: str, why: str) -> None:
        names = {"idle": "闲着", "peek": "看一眼", "bubble": "等气泡", "chatting": "聊天中"}
        log.info("面板：%s → %s（%s）", names.get(self.state, self.state), names.get(state, state), why)
        self.state = state

    def _tick_idle(self, now: float, visible: bool, blackout: bool) -> None:
        if visible:
            if now - self._closed_at > self.cfg.open_timeout:  # 刚关上的那一两帧还看得到，不算
                self._last_activity = now
                self._set("chatting", "面板开着（不是自己开的）")
            return
        if self._hold_until is not None and now < self._hold_until:  # 空闲注意力在先看一眼
            return
        due = now - self._last_read >= self.cfg.idle_peek
        wanted = self._pending is not None and now - self._last_peek >= self.cfg.peek_cooldown
        if blackout or not (due or wanted):
            return
        self._open_peek(now, self._pending if wanted else "定时")

    def _open_peek(self, now: float, reason: str) -> None:
        self._pending = None
        self._hold_until, self._held_once = None, False
        if self.device.ime_shown():  # 按键会打出字母；输入框开着说明在说话，本来就该是聊天中
            self._last_activity = now
            self._set("chatting", "输入框开着")
            return
        self._press(True)
        self._opened_at, self._seen = now, 0
        if reason == "bubble":  # 对方还在打字：开着等他发出来
            self._bubble_start = self._last_bubble = now
            self._set("bubble", "好友头顶冒出了气泡")
        else:
            self._set("peek", reason)

    def _tick_peek(self, now: float, visible: bool) -> None:
        if visible:
            self._seen += 1
            if self._seen >= 2 and not self._settling():  # 第一帧面板可能还没画完；读聊天的还在等确认也先别关
                self._close(now, "看一眼：没有新消息")
                self._last_peek = now
        elif now - self._opened_at >= self.cfg.open_timeout:
            log.warning("按了键 %.1f 秒聊天记录面板还没出现（被别的界面挡住了？），下个周期再看", self.cfg.open_timeout)
            self._last_read = self._last_peek = now
            self._set("idle", "没打开")

    def _tick_bubble(self, now: float, visible: bool) -> None:
        if not visible:
            if now - self._opened_at >= self.cfg.open_timeout and now - self._last_read >= self.cfg.open_timeout:
                log.warning("按了键 %.1f 秒聊天记录面板还没出现（被别的界面挡住了？）", self.cfg.open_timeout)
                self._last_read = self._last_peek = now
                self._set("idle", "没打开")
            return
        if now - self._last_bubble >= self.cfg.bubble_gone:
            why = f"气泡没了 {self.cfg.bubble_gone:.0f} 秒也没等到消息"
        elif now - self._bubble_start >= self.cfg.bubble_wait:
            why = f"等了 {self.cfg.bubble_wait:.0f} 秒没等到消息"
        else:
            return
        if self._settling():
            return
        self._close(now, why)
        self._last_peek = now

    def _tick_chatting(self, now: float, visible: bool, blackout: bool) -> None:
        if visible:
            self._missing_since = None
        elif self._missing_since is None:
            self._missing_since = now
        elif (not blackout and now - self._missing_since >= self.vision.log_reopen_after
              and now - self._last_reopen >= self.vision.log_reopen_cooldown):
            self._last_reopen = now
            self.ensure_open()
        if now - self._last_activity < self.cfg.quiet_close or self._settling():
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

    def _close(self, now: float, why: str) -> None:
        if self.device.ime_shown():  # 按键会打出字母；输入框开着说明在说话，就当聊天中
            self._last_activity = now
            self._set("chatting", "输入框开着")
            return
        self._press(False)
        self._closed_at = now
        self._set("idle", why)

    def _settling(self) -> bool:
        """读聊天的看到了新行、在等下一帧确认（ChatReader.settling）：这时关面板会让消息拖到下一次才读到。"""
        return bool(getattr(self.reader, "settling", False))

    def _press(self, expect_open: bool) -> None:
        self.device.hw_key(self.vision.log_open_key)
        self._pressed_at, self._expect_open = self.clock(), expect_open

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
