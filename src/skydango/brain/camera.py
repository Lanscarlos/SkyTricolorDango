"""转视角 / 缩放（光遇的键盘操作，见 game-ops §2），记下净偏移，能转回原位。

- 方向键 ←→ 水平转、↑ 镜头压低往上看、↓ 抬回来：按住 step 秒算一步（0.5 s 约 90°）
- 减号拉近、加号拉远（和直觉相反）；聊天记录面板开着时缩放没反应 → 操作前先关面板，做完再打开
- 输入框开着时按键会变成打字 → 先按 BACK
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from ..device.base import KEYCODE_BACK

KEYS = {"left": 105, "right": 106, "up": 103, "down": 108, "zoom_in": 12, "zoom_out": 13}  # Linux 键码
AXIS = {
    "left": ("turn", -1), "right": ("turn", 1),
    "up": ("pitch", 1), "down": ("pitch", -1),
    "zoom_in": ("zoom", 1), "zoom_out": ("zoom", -1),
}
UNDO = {"turn": ("right", "left"), "pitch": ("up", "down"), "zoom": ("zoom_in", "zoom_out")}  # (正方向, 反方向)
MAX_STEPS = 4


class Camera:
    def __init__(
        self,
        device,
        step: float,
        panel_visible: Callable[[], bool],
        panel_key: int,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.device = device
        self.step = step
        self.panel_visible = panel_visible
        self.panel_key = panel_key  # 开关聊天记录面板的键；0 表示没有面板要管
        self.sleep = sleep
        self.offset = {"turn": 0, "pitch": 0, "zoom": 0}

    def move(self, action: str, steps: int = 1) -> str:
        if action not in KEYS:
            raise ValueError(f"不认识的视角操作：{action}（可以用 {'、'.join(KEYS)}）")
        steps = max(1, min(int(steps), MAX_STEPS))
        with self._ready():
            for _ in range(steps):
                self._press(action)
        axis, sign = AXIS[action]
        self.offset[axis] += sign * steps
        return self.describe()

    def reset(self) -> str:
        undo = [(UNDO[axis][1] if value > 0 else UNDO[axis][0], abs(value)) for axis, value in self.offset.items() if value]
        if not undo:
            return "镜头已经在原位"
        with self._ready():
            for action, n in undo:
                for _ in range(n):
                    self._press(action)
        self.offset = {axis: 0 for axis in self.offset}
        return "镜头转回原位了（来回转会有一点偏差）"

    def describe(self) -> str:
        turn, pitch, zoom = self.offset["turn"], self.offset["pitch"], self.offset["zoom"]
        parts = []
        if turn:
            parts.append(f"{'右' if turn > 0 else '左'}转了 {abs(turn)} 步")
        if pitch:
            parts.append(f"往{'上' if pitch > 0 else '下'}看了 {abs(pitch)} 步")
        if zoom:
            parts.append(f"拉{'近' if zoom > 0 else '远'}了 {abs(zoom)} 步")
        return "，".join(parts) or "原位"

    def _press(self, action: str) -> None:
        code = KEYS[action]
        if action.startswith("zoom"):
            self.device.hw_key(code)
            self.sleep(0.3)
        else:
            self.device.hw_key_down(code)
            self.sleep(self.step)
            self.device.hw_key_up(code)
            self.sleep(0.2)

    @contextmanager
    def _ready(self) -> Iterator[None]:
        if self.device.ime_shown():
            self.device.key(KEYCODE_BACK)
            self.sleep(0.3)
        was_open = bool(self.panel_key) and self.panel_visible()
        if was_open:
            self.device.hw_key(self.panel_key)
            self.sleep(0.8)
        try:
            yield
        finally:
            if was_open:
                self.device.hw_key(self.panel_key)
                self.sleep(0.8)
