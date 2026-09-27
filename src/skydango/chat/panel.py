"""聊天记录面板（光遇里按 C）的维护：没开就按键打开，关着太久自动重开。Agent 和大脑的身体共用。"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..config import VisionConfig
from ..device.base import Device

log = logging.getLogger(__name__)


class PanelKeeper:
    def __init__(self, vision: VisionConfig, device: Device, reader, sleep: Callable[[float], None]) -> None:
        self.vision = vision
        self.device = device
        self.reader = reader  # ChatReader：panel_visible(frame)、panel_closed_since
        self.sleep = sleep
        self._last_reopen = float("-inf")

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

    def maybe_reopen(self, now: float) -> None:
        since = self.reader.panel_closed_since
        vision = self.vision
        if since is None or not vision.log_reopen_after or now - since < vision.log_reopen_after:
            return
        if now - self._last_reopen < vision.log_reopen_cooldown:
            return
        self._last_reopen = now
        self.ensure_open()
