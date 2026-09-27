"""角色移动（光遇的键盘操作，见 game-ops §2）：W/A/S/D 按住一小段时间模拟一小步，没有"复位"这回事。

- 前进 / 后退 / 左右对应 W / A / S / D；聊天记录面板开着也有效，不用像 camera 那样先关面板再开
- 2026-09-27 实测：按住 0.3 s 几乎看不出移动，1 s 幅度很大（从好友群里走到几米外），默认给一个短的单步时长
- 输入框开着时按键会变成打字 → 先按 BACK
- 走出去之后回不去：不维护位移量、不提供 reset()，避免造成"能一键复位"的错误预期
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..device.base import KEYCODE_BACK

KEYS = {"forward": 17, "back": 31, "left": 30, "right": 32}  # Linux 键码：W A S D（A/D 未在真机验证，按对称假设）
NAMES = {"forward": "前进", "back": "后退", "left": "向左", "right": "向右"}
MAX_STEPS = 3


class Locomotion:
    def __init__(
        self,
        device,
        step: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.device = device
        self.step = step
        self.sleep = sleep

    def move(self, direction: str, steps: int = 1, max_steps: int = MAX_STEPS) -> str:
        if direction not in KEYS:
            raise ValueError(f"不认识的移动方向：{direction}（可以用 {'、'.join(KEYS)}）")
        steps = max(1, min(int(steps), max_steps))
        if self.device.ime_shown():
            self.device.key(KEYCODE_BACK)
            self.sleep(0.3)
        code = KEYS[direction]
        for _ in range(steps):
            self.device.hw_key_down(code)
            self.sleep(self.step)
            self.device.hw_key_up(code)
            self.sleep(0.2)
        return f"{NAMES[direction]}走了 {steps} 步"
