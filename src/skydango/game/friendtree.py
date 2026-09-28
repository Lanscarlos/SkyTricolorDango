"""点一下人物、看右侧打开的好友树面板，确认这个人是不是好友（game-ops §5，用户告知，**未截图核对**）。

用在名字标签认不出的时候：点过火的陌生人外观和好友一样，只能靠头顶标签分，标签被挡住 / 太远时就不知道。

面板长什么样、怎么看出"是不是好友"还没录到，所以这里不写死判断：点开后把截图原样交给大脑，让它自己看；
身体只负责点、截图、关面板、把聊天记录面板恢复原样。关面板的办法也没验证过：按 `close` 里的顺序一个个试，
每试一次截图看右侧是不是变回点之前的样子。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..brain.images import difference, thumb
from ..config import FriendCheckConfig
from ..device.base import KEYCODE_BACK
from .social import touch_mode

log = logging.getLogger(__name__)

LINUX_KEY_ESC = 1


@dataclass
class CheckResult:
    before: np.ndarray  # 点之前
    opened: np.ndarray  # 点完等了 open_delay 秒
    after: np.ndarray  # 关完
    changed: float  # 右侧和点之前差多少（0~1）：大 = 大概打开了面板
    closed: bool  # 右侧变回原样了没有
    closed_by: str  # 用哪个办法关上的（没关上时为空）


class FriendChecker:
    def __init__(
        self,
        device,
        cfg: FriendCheckConfig,
        panel_visible: Callable[[np.ndarray], bool] | None = None,  # 聊天记录面板开没开（点屏幕会关掉它）
        panel_key: int = 46,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.device = device
        self.cfg = cfg
        self.panel_visible = panel_visible
        self.panel_key = panel_key
        self.sleep = sleep

    def _side(self, frame: np.ndarray) -> np.ndarray:
        """右侧（面板出现的地方）的缩略图。"""
        return thumb(frame[:, round(frame.shape[1] * self.cfg.panel_left) :])

    def change(self, a: np.ndarray, b: np.ndarray) -> float:
        return difference(self._side(a), self._side(b))

    def looks_open(self, changed: float) -> bool:
        return changed >= self.cfg.changed

    def check(self, x: int, y: int) -> CheckResult:
        """点 (x, y)（整张截图坐标）打开好友树，截图，再关掉。"""
        frame = self.device.screenshot()
        panel_was_open = self.panel_visible(frame) if self.panel_visible else False
        try:
            if not touch_mode(frame):  # 键盘模式下第一下触摸只切到触屏模式（game-ops §1）
                self.device.tap(x, y)
                self.sleep(0.6)
                frame = self.device.screenshot()
                if not touch_mode(frame):
                    log.info("点了一下还没切到触屏模式，照样再点")
            before = frame
            self.device.tap(x, y)
            self.sleep(self.cfg.open_delay)
            opened = self.device.screenshot()
            changed = self.change(before, opened)
            closed_by = ""
            after = opened
            if self.looks_open(changed):
                for way in self.cfg.close:
                    self._close(way, after)
                    self.sleep(self.cfg.close_delay)
                    after = self.device.screenshot()
                    if not self.looks_open(self.change(before, after)):
                        closed_by = way
                        break
                if not closed_by:
                    log.warning("好友树面板没关上（试了 %s），请手动关掉", "、".join(self.cfg.close))
            return CheckResult(before, opened, after, changed, not self.looks_open(self.change(before, after)), closed_by)
        finally:
            if panel_was_open:
                self._reopen_panel()

    def _close(self, way: str, frame: np.ndarray) -> None:
        if way == "esc":
            self.device.hw_key(LINUX_KEY_ESC)
        elif way == "back":
            self.device.key(KEYCODE_BACK)
        elif way == "tap":  # 点左边空地（close_tap，归一化坐标）
            if len(self.cfg.close_tap) != 2:
                return
            h, w = frame.shape[:2]
            self.device.tap(round(self.cfg.close_tap[0] * w), round(self.cfg.close_tap[1] * h))
        else:
            log.warning("不认识的关面板办法：%s", way)

    def _reopen_panel(self) -> None:
        """点屏幕会关掉聊天记录面板：按键重新打开（同 SocialHandler）。"""
        if self.panel_visible is None or not self.panel_key:
            return
        for _ in range(2):
            if self.panel_visible(self.device.screenshot()) or self.device.ime_shown():
                return
            self.device.hw_key(self.panel_key)
            self.sleep(1.5)
