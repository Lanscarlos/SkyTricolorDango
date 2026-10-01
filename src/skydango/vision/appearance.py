"""认装扮：人物框裁图、颜色特征（认人靠外观，名字标签看不到时用）。"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import AppearanceConfig
from .bubbles import Rect
from .embed import OnnxEmbedder, unit
from .track import iou


def _inter(a: Rect, b: Rect) -> int:
    w = min(a.x2, b.x2) - max(a.x, b.x)
    h = min(a.y2, b.y2) - max(a.y, b.y)
    return w * h if w > 0 and h > 0 else 0


def good_crop(
    frame: np.ndarray, box: Rect, others: list[Rect], blocked: list[Rect], min_height: float, max_overlap: float
) -> np.ndarray | None:
    """好样本才返回裁图（框中间 60% 宽、整高），否则 None：框太小、被别的人 / 团子盖住、压在聊天面板上都不要。"""
    fh, fw = frame.shape[:2]
    if box.w <= 0 or box.h <= 0 or box.h < min_height * fh:
        return None
    area = box.w * box.h
    for o in others:
        if iou(box, o) > max_overlap or _inter(box, o) / area > max_overlap:
            return None
    for b in blocked:
        if _inter(box, b) / area > max_overlap:
            return None
    margin = int(round(box.w * 0.2))
    crop = Rect(box.x + margin, box.y, box.w - 2 * margin, box.h)
    x1, y1, x2, y2 = max(0, crop.x), max(0, crop.y), min(fw, crop.x2), min(fh, crop.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def describe_crop(frame: np.ndarray, box: Rect) -> np.ndarray:
    """送去描述用的裁图：框四周各扩 15%（带上头部），夹到画面内。"""
    fh, fw = frame.shape[:2]
    pad = Rect(
        box.x - int(round(box.w * 0.15)),
        box.y - int(round(box.h * 0.15)),
        box.w + 2 * int(round(box.w * 0.15)),
        box.h + 2 * int(round(box.h * 0.15)),
    )
    x1, y1, x2, y2 = max(0, pad.x), max(0, pad.y), min(fw, pad.x2), min(fh, pad.y2)
    return frame[y1:y2, x1:x2]


class ColorEmbedder:
    """内置的颜色特征：头（上 30%）和身体（下 70%）各一份 HSV 色相 / 饱和度直方图。亮度不进直方图，所以同色深浅分不开（已知限制）。"""

    key = "color-v1"

    def embed(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        cut = max(1, int(round(h * 0.3)))
        parts = [self._hist(img[:cut]), self._hist(img[cut:])]
        return unit(np.concatenate(parts))

    @staticmethod
    def _hist(region: np.ndarray) -> np.ndarray:
        if region.size == 0:
            return np.zeros(16 * 4, np.float32)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        mask = ((v >= 30) & (v <= 245)).astype(np.uint8) * 255  # 去掉过黑 / 过曝的像素
        hist = cv2.calcHist([hsv], [0, 1], mask, [16, 4], [0, 180, 0, 256])
        return unit(hist)


def make_embedder(cfg: AppearanceConfig):
    if cfg.model == "color":
        return ColorEmbedder()
    if cfg.model.lower().endswith(".onnx"):
        return OnnxEmbedder(cfg.model, cfg.size, cfg.norm, cfg.device, what="appearance.model")
    raise ValueError(f"appearance.model 只能是 \"color\" 或 .onnx 路径：{cfg.model}")
