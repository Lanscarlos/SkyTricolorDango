"""聊天时做表情动作：轮盘上有的直接按数字键；白名单里的先换到指定格子再做。

- 聊天记录面板（C）开着时数字键照样能做动作，但长按 Z 没反应 → 只有换轮盘时才关面板、换完再打开
- 输入框开着时按键会变成打字 → 先按 BACK 关掉
- 启动时记下可换格子原来的动作，退出时换回去
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from ..config import EmoteConfig
from ..device.base import KEYCODE_BACK, Device
from .wheel import Wheel, WheelError

log = logging.getLogger(__name__)


class EmotePlayer:
    def __init__(
        self,
        device: Device,
        wheel: Wheel,
        cfg: EmoteConfig,
        panel_visible: Callable[[], bool],
        panel_key: int,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.device = device
        self.wheel = wheel
        self.cfg = cfg
        self.panel_visible = panel_visible
        self.panel_key = panel_key  # 开关聊天记录面板的键；0 表示没有面板要管
        self.sleep = sleep
        self.clock = clock
        known = wheel.library.templates
        self.extra = [n for n in cfg.extra if n in known]
        missing = [n for n in cfg.extra if n not in known]
        if missing:
            log.warning("emotes.extra 里这些动作图标库里没有，忽略: %s", "、".join(missing))
        self.swap_slots = [s for s in cfg.swap_slots if s not in wheel.cfg.locked_slots]
        if len(self.swap_slots) < len(cfg.swap_slots):
            log.warning("emotes.swap_slots 里有锁定的格子（wheel.locked_slots），不会换它们")
        self.original: dict[int, str] = {}  # 可换格子原来的动作，退出时换回去
        self.last_emote = float("-inf")
        self.last_swap = float("-inf")

    def start(self) -> None:
        """读一次轮盘，记下可换格子原来的动作。"""
        self._close_input()
        with self._panel_closed():
            self.wheel.refresh()
        for slot in list(self.swap_slots):
            name = self.wheel.slots.get(slot)
            if name is None:
                log.warning("格子 %d 里的动作认不出（图标库里没有），换了就换不回去，所以不用它", slot)
                self.swap_slots.remove(slot)
            else:
                self.original[slot] = name

    def on_wheel(self) -> list[str]:
        locked = self.wheel.cfg.locked_slots
        return [name for slot, name in sorted(self.wheel.slots.items()) if name and slot not in locked]

    def available(self) -> list[str]:
        """这一轮能给模型用的动作：动作限速中为空；换轮盘限速中只有轮盘上现有的。"""
        now = self.clock()
        if now - self.last_emote < self.cfg.min_interval:
            return []
        names = self.on_wheel()
        if self.swap_slots and now - self.last_swap >= self.cfg.swap_min_interval:
            names += [n for n in self.extra if n not in names]
        return names

    def perform(self, name: str) -> int:
        self._close_input()
        now = self.clock()
        if name not in self.on_wheel():
            if name not in self.extra or not self.swap_slots:
                raise WheelError(f"「{name}」不在轮盘上，也不在可以换上去的白名单里")
            with self._panel_closed():
                self.wheel.ensure(name, candidates=self.swap_slots)
            self.last_swap = now
        slot = self.wheel.perform(name)
        self.last_emote = now
        return slot

    def pretend(self, name: str) -> None:
        """dry-run：不按键，只记下时间，让限速和真跑时一样。"""
        now = self.clock()
        if name not in self.on_wheel():
            self.last_swap = now
        self.last_emote = now

    def restore(self) -> None:
        """把换过的格子换回启动时的动作，再读一遍核对。"""
        changed = {s: n for s, n in self.original.items() if self.wheel.slots.get(s) != n}
        if not changed:
            return
        self._close_input()
        with self._panel_closed():
            for slot, name in changed.items():
                try:
                    self.wheel.assign(slot, name)
                except WheelError as exc:
                    log.warning("格子 %d 没换回「%s」: %s", slot, name, exc)
            self.wheel.refresh()
        wrong = [f"{s}（应为{n}）" for s, n in changed.items() if self.wheel.slots.get(s) != n]
        if wrong:
            log.warning("这些格子没恢复，请用 emotes wheel 检查: %s", "、".join(wrong))
        else:
            log.info("轮盘已恢复: %s", "、".join(f"{s}={n}" for s, n in changed.items()))

    def _close_input(self) -> None:
        if self.device.ime_shown():
            self.device.key(KEYCODE_BACK)
            self.sleep(0.3)

    @contextmanager
    def _panel_closed(self) -> Iterator[None]:
        was_open = bool(self.panel_key) and self.panel_visible()
        if was_open:
            self.device.hw_key(self.panel_key)
            self.sleep(self.wheel.cfg.ui_delay)
        try:
            yield
        finally:
            if was_open:
                self.device.hw_key(self.panel_key)
                self.sleep(self.wheel.cfg.ui_delay)
