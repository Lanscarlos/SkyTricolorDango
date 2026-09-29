"""聊天记录面板（光遇里按 C）的开关：只有这里按开面板的键，Agent 和大脑的身体共用一个。

- always：一直开着，关久了自动重开（原来的做法）
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
        self.device.hw_key(key)
        self.sleep(1.0)
        if self.reader.panel_visible(self.device.screenshot()):
            return True
        log.warning("按了键聊天记录面板还是没出现，可能被别的界面挡住了，请看一下游戏画面")
        return False

    def start(self, now: float) -> bool:
        return self.ensure_open()

    def should_be_open(self) -> bool:
        return self.lent is None

    def restored(self) -> bool:
        """面板回到了该有的状态（不该开时关着也算）。"""
        return not self.active or not self.should_be_open() or self.visible_now()

    def tick(self, now: float, fresh: list, visible: bool, blackout: bool = False) -> None:
        """主循环每圈调一次（fresh = 这一圈读到的新消息，visible = 这一帧面板开没开）。不 sleep 太久。"""
        if not self.active or self.lent is not None:
            return
        self._maybe_reopen(now)

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
        yield 借之前面板开没开（嵌套借时是 False）。"""
        was_open = False
        outer = self._depth == 0
        if outer and self.active:
            was_open = self.visible_now()
            self.lent = who
            log.debug("面板借给 %s", who)
            if close and was_open:
                self.device.hw_key(self.vision.log_open_key)
                self.sleep(CLOSE_DELAY)
        elif outer:
            self.lent = who
        self._depth += 1
        try:
            yield was_open
        finally:
            self._depth -= 1
            if outer:
                self.lent = None
                self._give_back(was_open)

    def _give_back(self, was_open: bool) -> None:
        if not self.active or not self.should_be_open():
            return
        if self.cfg.mode == "always" and not was_open:  # 借之前就关着：和原来一样不管，交给 tick 过一会儿重开
            return
        if self.visible_now():
            return
        if self.device.ime_shown():  # 按键会变成打字：交给 tick 以后重开
            return
        if not self._open_and_wait():
            log.warning("聊天记录面板没能重新打开，主循环稍后会再试")

    def _open_and_wait(self) -> bool:
        """按一下开面板的键，等它出现（最多 open_timeout 秒）。"""
        self.device.hw_key(self.vision.log_open_key)
        for _ in range(max(1, math.ceil(self.cfg.open_timeout / POLL))):
            self.sleep(POLL)
            if self.visible_now():
                return True
        return False
