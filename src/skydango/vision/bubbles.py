"""聊天气泡检测：不用模型，靠颜色阈值 + 形状过滤。

光遇的聊天气泡是浅色、低饱和的圆角框，里面是深色文字。先把“又亮又不鲜艳”的像素
圈出来，再按尺寸、填充率（像不像实心圆角矩形）、墨迹占比（里面有没有字）筛掉云、雪地、
UI 按钮等误检。参数都在 BubbleConfig 里，需要用真实截图调。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import BubbleConfig


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def pad(self, n: int, width: int, height: int) -> "Rect":
        x1, y1 = max(0, self.x - n), max(0, self.y - n)
        x2, y2 = min(width, self.x2 + n), min(height, self.y2 + n)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y : self.y2, self.x : self.x2]


def roi_rect(roi: list[float], width: int, height: int) -> Rect:
    x1, y1, x2, y2 = roi
    px1, py1 = int(round(x1 * width)), int(round(y1 * height))
    px2, py2 = int(round(x2 * width)), int(round(y2 * height))
    return Rect(px1, py1, max(1, px2 - px1), max(1, py2 - py1))


def find_bubbles(img: np.ndarray, params: BubbleConfig, roi: list[float] | None = None) -> list[Rect]:
    """返回气泡在整张图里的外接矩形，按从上到下、从左到右排序。"""
    height, width = img.shape[:2]
    area = roi_rect(roi, width, height) if roi else Rect(0, 0, width, height)
    region = area.crop(img)

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((val >= params.min_value) & (sat <= params.max_saturation)).astype(np.uint8) * 255
    # 轻微开运算去掉零星亮点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w, max_w = params.min_width * width, params.max_width * width
    min_h, max_h = params.min_height * height, params.max_height * height

    found: list[Rect] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        # 外轮廓面积不含内部的文字“洞”，所以实心圆角框的填充率很高
        fill = cv2.contourArea(contour) / float(w * h)
        if fill < params.min_fill:
            continue
        inner = val[y : y + h, x : x + w]
        ink = float(np.count_nonzero(inner < params.ink_value)) / inner.size
        if not (params.min_ink <= ink <= params.max_ink):
            continue
        found.append(Rect(x + area.x, y + area.y, w, h))

    found.sort(key=lambda r: (r.y, r.x))
    return found


def annotate(img: np.ndarray, rects: list[Rect], labels: list[str] | None = None) -> np.ndarray:
    """调试用：画出检测框和序号（cv2 画不了中文，文字内容看终端输出）。"""
    out = img.copy()
    for i, r in enumerate(rects):
        cv2.rectangle(out, (r.x, r.y), (r.x2, r.y2), (0, 0, 255), 2)
        tag = labels[i] if labels else str(i)
        cv2.putText(out, tag, (r.x, max(12, r.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def draw_grid(img: np.ndarray, step: float = 0.1) -> np.ndarray:
    """标定用：叠加归一化坐标网格，方便在配置里填按钮位置。"""
    out = img.copy()
    height, width = out.shape[:2]
    n = int(round(1 / step))
    for i in range(1, n):
        x, y = int(i * step * width), int(i * step * height)
        cv2.line(out, (x, 0), (x, height), (0, 255, 255), 1)
        cv2.line(out, (0, y), (width, y), (0, 255, 255), 1)
        label = f"{i * step:.1f}"
        cv2.putText(out, label, (x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(out, label, (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out
