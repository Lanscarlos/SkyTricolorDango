"""动作图标的识别：光遇的动作图标是深色底上的米白色剪影，没有文字。

只比较“剪影”（亮像素的形状），不比较颜色和背景，这样轮盘里（图标更大、背景是 3D 画面）
和动作列表里（深色格子）的同一个图标也能对上。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .bubbles import Rect

SILHOUETTE_VALUE = 150  # 灰度高于这个值算图标的一部分


def silhouette(img: np.ndarray, threshold: int = SILHOUETTE_VALUE) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return (gray > threshold).astype(np.uint8) * 255


def trim(mask: np.ndarray) -> np.ndarray:
    """裁掉四周的空白；全空时原样返回。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


@dataclass(frozen=True)
class Match:
    score: float
    x: int  # 匹配到的图标中心（相对 haystack）
    y: int


def best_match(haystack: np.ndarray, template: np.ndarray, scales: list[float]) -> Match:
    """在 haystack（剪影）里找 template（已 trim 的剪影），返回多尺度下最好的一处。"""
    best = Match(0.0, 0, 0)
    for s in scales:
        t = cv2.resize(template, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        th, tw = t.shape[:2]
        if th < 4 or tw < 4 or th > haystack.shape[0] or tw > haystack.shape[1] or not t.any():
            continue
        result = cv2.matchTemplate(haystack, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if score > best.score:
            best = Match(float(score), loc[0] + tw // 2, loc[1] + th // 2)
    return best


def find_icons(img: np.ndarray, min_size: int, max_size: int, merge: int = 9) -> list[Rect]:
    """在一块区域里找出所有图标的外接框（从上到下、从左到右）。

    一个图标可能由几块分开的剪影组成（头、身体、小点），先膨胀再找连通域把它们并起来。
    """
    mask = silhouette(img)
    merged = cv2.dilate(mask, np.ones((merge, merge), np.uint8))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_size <= w <= max_size and min_size <= h <= max_size:
            rects.append(Rect(x, y, w, h))
    rects.sort(key=lambda r: (r.y // max(1, min_size), r.x))
    return rects


def same_icon(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    """两个已 trim 的剪影是不是同一个图标（允许 ±10% 的大小差异）。"""
    pad = 8
    big = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return best_match(big, b, [0.9, 0.95, 1.0, 1.05, 1.1]).score >= threshold
