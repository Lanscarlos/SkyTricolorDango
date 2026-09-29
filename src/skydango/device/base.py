"""设备抽象。Agent 只依赖这个接口，以后接真机（无障碍桥）或 PC 客户端时再加实现。"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

KEYCODE_BACK = 4
KEYCODE_ENTER = 66
# Linux 输入子系统的键码（sendevent 用），和 Android keycode 不是一套
LINUX_KEY_ENTER = 28


class Device(Protocol):
    def screenshot(self) -> np.ndarray:
        """返回当前画面，BGR，形状 (h, w, 3)。坐标系与 tap 一致。"""

    def tap(self, x: int, y: int) -> None: ...

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None: ...

    def key(self, keycode: int) -> None: ...

    def input_text(self, text: str) -> None:
        """往当前获得焦点的输入框里输入文字（支持中文）。"""

    def editor_action(self, code: int) -> None:
        """触发输入法的编辑器动作（发送 / 完成）。"""

    def hw_key(self, code: int) -> None:
        """模拟实体键盘按一下（Linux 键码）。有的游戏只认实体键盘，不认 `input keyevent`。"""

    def hw_key_down(self, code: int) -> None:
        """按住实体键盘的一个键（长按用），之后要 hw_key_up。"""

    def hw_key_up(self, code: int) -> None: ...

    def hw_key_hold(self, code: int, seconds: float) -> None:
        """按住实体键盘的一个键指定秒数再松开（转视角的短按用）。

        默认实现：按下、在本机 sleep、抬起（sleep 被打断也会抬起）；
        AdbDevice 覆盖成一条 shell 命令，sleep 在模拟器里，时长不受 adb 往返影响。
        """
        self.hw_key_down(code)
        try:
            time.sleep(seconds)
        finally:
            self.hw_key_up(code)

    def ime_shown(self) -> bool:
        """软键盘 / 输入框当前是否打开。"""
