"""火焰圆盘：没点火的黑影站到团子身边时，他身上出现深色实心圆 + 白色空心火焰（没有白圈）。

只用来判断"这个黑影站在能点火的距离里"（spec 2026-10-01-light-unlit-stranger）。**绝不点它**：
点了团子会一直跟着那个陌生人走，点火靠按 3 号键举蜡烛。
白圈 + 火焰是陌生人举蜡烛要给团子点火（social 的 candle），外环是亮的，这里不算。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..game.social import cream
from ..imageio import imread
from .bubbles import Rect
from .icons import best_match, trim

FLAME = "assets/candle/flame.png"  # 录像 candle-20260930-d 第 16 秒截的，只有火焰
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5]  # 人远近不同，圆盘大小跟着变
RING_IN, RING_OUT = 1.2, 2.3  # 火焰半高的这么多倍之间是圆盘的外环（录像里白圈在 1.35~2 倍，随火焰大小变）
BLUR = 2.0  # 火焰会抖（大小、形状每帧不同），模板和画面都先糊一下再比，真火焰的分数高一截
SECTORS = 24  # 外环分成这么多扇形；白圈是一整圈亮的，旁边玩家的白衣服只占几个扇形
RING_SECTORS = 0.5  # 亮的扇形占比超过这个 = 有白圈
SECTOR_PIXELS = 3  # 一个扇形里至少这么多米白像素才算亮
# 孤儿圆圈（头顶没名字的）里是火焰时：白圈占一圈扇形的这么多以上 = 陌生人举蜡烛要给团子点火（social 的 candle），
# 不到 = 深色火焰圆盘（点了会跟着他走）。环带取圆圈半径的 0.7~1.1 倍。不看亮度：半透明圆盘会透出后面亮的东西
# （最终审查在录像 candle-20260930-c 上量的：圆盘 0.17~0.33、白圈 1.00；白圈刚冒出来那一帧太淡会是 0，那一帧不点也没关系）
RING_WHITE = 0.5
RING_BAND = (0.7, 1.1)


@dataclass(frozen=True)
class Disk:
    x: int  # 圆心（整图坐标）
    y: int
    r: float  # 火焰半高（像素）
    score: float


def load_flame(path: str | Path = FLAME) -> np.ndarray:
    return trim(cream(imread(path)))


def _band(frame: np.ndarray, cx: int, cy: int, r_in: float, r_out: float):
    """(cx, cy) 周围 r_in~r_out 的环带：(那一小块画面, 环带掩码, 每个像素在第几个扇形)；出了画面 / 空的返回 None。"""
    x1, y1 = max(0, int(cx - r_out)), max(0, int(cy - r_out))
    patch = frame[y1 : int(cy + r_out) + 1, x1 : int(cx + r_out) + 1]
    if patch.size == 0:
        return None
    yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    d = np.hypot(xx - (cx - x1), yy - (cy - y1))
    ring = (d >= r_in) & (d <= r_out)
    if not ring.any():
        return None
    sector = (np.arctan2(yy - (cy - y1), xx - (cx - x1)) + np.pi) / (2 * np.pi) * SECTORS
    return patch, ring, np.minimum(sector.astype(int), SECTORS - 1)


def _lit_fraction(patch: np.ndarray, ring: np.ndarray, sector: np.ndarray) -> float:
    bright = (cream(patch) > 0) & ring
    return float((np.bincount(sector[bright], minlength=SECTORS) >= SECTOR_PIXELS).mean())


def ring_fraction(frame: np.ndarray, cx: int, cy: int, r_in: float, r_out: float) -> float:
    """(cx, cy) 周围 r_in~r_out 的环带里，有米白像素（白圈）的扇形占几成：一整圈白圈 ≈ 1，旁边白衣服只占几个扇形。"""
    band = _band(frame, cx, cy, r_in, r_out)
    return 0.0 if band is None else _lit_fraction(*band)


def white_ring(frame: np.ndarray, cx: int, cy: int, r: float) -> bool:
    """以 (cx, cy) 为圆心、半径约 r 的圆圈有没有白圈：有 = 举蜡烛请求（可以点），没有 = 深色火焰圆盘（绝不点）。"""
    return ring_fraction(frame, cx, cy, RING_BAND[0] * r, RING_BAND[1] * r) >= RING_WHITE


def dark_ring(frame: np.ndarray, cx: int, cy: int, r_in: float, r_out: float, dark: float) -> bool:
    """(cx, cy) 周围 r_in~r_out 的环带够暗、没有白色描边。"""
    band = _band(frame, cx, cy, r_in, r_out)
    if band is None:
        return False
    patch, ring, sector = band
    value = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
    if float(value[ring].mean()) >= dark:
        return False
    return _lit_fraction(patch, ring, sector) <= RING_SECTORS


def _region(box: Rect, width: int, height: int) -> tuple[int, int, int, int] | None:
    """黑影框左右各扩 0.5 倍框宽、往上扩 0.6 倍框高、下到框底（圆盘在胸口或头顶附近）。"""
    x1, x2 = max(0, round(box.x - 0.5 * box.w)), min(width, round(box.x2 + 0.5 * box.w))
    y1, y2 = max(0, round(box.y - 0.6 * box.h)), min(height, box.y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return x1, y1, x2, y2


def find_disk(frame: np.ndarray, box: Rect, flame: np.ndarray, min_score: float = 0.68, dark: float = 90.0) -> Disk | None:
    """在这个黑影身上 / 头顶找火焰圆盘；没有（或者是白圈的举蜡烛请求）返回 None。"""
    area = _region(box, frame.shape[1], frame.shape[0])
    if area is None:
        return None
    x1, y1, x2, y2 = area
    mask = cv2.GaussianBlur(cream(frame[y1:y2, x1:x2]), (0, 0), BLUR)
    flame = cv2.GaussianBlur(flame, (0, 0), BLUR)
    best, scale = None, 1.0
    for s in SCALES:
        m = best_match(mask, flame, [s])
        if best is None or m.score > best.score:
            best, scale = m, s
    if best is None or best.score < min_score:
        return None
    cx, cy, r = x1 + best.x, y1 + best.y, flame.shape[0] * scale / 2
    if not dark_ring(frame, cx, cy, RING_IN * r, RING_OUT * r, dark):
        return None
    return Disk(cx, cy, r, best.score)
