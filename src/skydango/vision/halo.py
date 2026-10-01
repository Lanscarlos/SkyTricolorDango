"""呼唤光圈认团子（spec docs/superpowers/specs/2026-10-01-q-call-design.md §2.2）：纯计算，不碰设备。

短按 Q 后头上冒一圈亮光（0.25~0.45 s，最亮约 0.2 s，game-ops「呼唤特效」）；别人按 Q 的一模一样，只能对时间：
团子按下后一小段时间里，恰好一个人头顶变亮、他又在画面中间附近，才算认出团子。对不上就放弃。

亮度 = 头顶区域灰度均值比按键前（基准图）高出多少，再减掉整帧均值的变化（画面整体变亮变暗抵消掉）。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import CallConfig
from .bubbles import Rect

HALO_FROM = 0.0  # 按键后只看这段时间（秒）；起点是按键命令发出之前（adb 往返要 0.1~0.2 s，最亮约在按下后 0.2 s）
HALO_TO = 0.8


def head_region(box: Rect, width: int, height: int) -> Rect | None:
    """头顶区域：以框顶为中心，宽 2.5 倍框宽；纵向从框顶往上 0.5 倍框高到往下 0.3 倍框高（光圈约 1.3 → 3 倍头宽）。裁到画面内。"""
    cx = box.x + box.w / 2
    x1, x2 = max(0, round(cx - 1.25 * box.w)), min(width, round(cx + 1.25 * box.w))
    y1, y2 = max(0, round(box.y - 0.5 * box.h)), min(height, round(box.y + 0.3 * box.h))
    if x2 <= x1 or y2 <= y1:
        return None
    return Rect(x1, y1, x2 - x1, y2 - y1)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


class HaloWatch:
    """按 Q 之后连拍的每一帧交给 add()；result() 说认没认出团子。"""

    def __init__(self, base: np.ndarray, boxes: dict[int, Rect], pressed_at: float, cfg: CallConfig) -> None:
        g = _gray(base)
        h, w = g.shape[:2]
        self.cfg = cfg
        self.pressed_at = pressed_at
        self.boxes = dict(boxes)
        self.regions = {i: r for i, b in boxes.items() if (r := head_region(b, w, h)) is not None}
        self._base_all = float(g.mean())
        self._base = {i: float(r.crop(g).mean()) for i, r in self.regions.items()}
        self.peaks: dict[int, float] = {i: 0.0 for i in self.regions}
        self.skipped = False  # 身体置位：连拍中截图失败 / 黑屏、按键前镜头刚动过

    def add(self, frame: np.ndarray, t: float) -> None:
        if not self.pressed_at + HALO_FROM <= t <= self.pressed_at + HALO_TO:
            return
        g = _gray(frame)
        drift = float(g.mean()) - self._base_all
        for i, r in self.regions.items():
            rise = float(r.crop(g).mean()) - self._base[i] - drift
            if rise > self.peaks[i]:
                self.peaks[i] = rise

    def result(self, width: int) -> tuple[str, int | None]:
        """("self", 轨迹 id) / ("others", None) / ("none", None) / ("skipped", None)。"""
        if self.skipped or not self.regions:  # 没有能看的人（框都过时了）：说不准，别报"没看到"
            return "skipped", None
        rise = self.cfg.halo_rise
        over = [i for i, v in self.peaks.items() if v >= rise]
        if not over:
            return "none", None
        if len(over) > 1 or any(v >= rise / 2 for i, v in self.peaks.items() if i != over[0]):
            return "others", None  # 别人也在喊（或者分不清）
        b = self.boxes[over[0]]
        if abs(b.x + b.w / 2 - width / 2) > self.cfg.halo_center * width:
            return "others", None  # 冒圈的不在画面中间：多半是别人
        return "self", over[0]
