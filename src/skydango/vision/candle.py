"""火焰圆盘：没点火的黑影站到团子身边时，他身上出现深色实心圆 + 白色空心火焰；他举蜡烛时外面多一圈白圈（一闪一闪）。

`find_flame` 在调用方给的范围（团子周围）里找火焰（spec 2026-10-01-light-flame-around-self），只用来判断"身边有个能点火的黑影"——
有没有白圈都算。团子按 3 号键举蜡烛，不碰圆圈。**绝不点圆盘**：点了团子会一直跟着那个陌生人走。
外环的亮度不看：半透明圆盘透出后面亮的东西时并不暗（10-01 晚真机外环亮度 128~150，白圈帧又整圈是亮的，两样都刷掉后凑不满 3 秒）。
误匹配（场景里灯笼上的菱形能到 0.77）靠 `[social] disk_sure` 挡：一段里至少要有一帧匹配得够好。

`black` 判人物有多黑：中间一半宽、上 70% 高里很暗的像素占比。黑影 ≈ 高，点亮后显出衣服颜色 ≈ 低。
孤儿圆圈要不要点（social 的 candle）还是看有没有白圈（`white_ring`）。
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
BLUR = 2.0  # 火焰会抖（大小、形状每帧不同），模板和画面都先糊一下再比，真火焰的分数高一截
SECTORS = 24  # 外环分成这么多扇形；白圈是一整圈亮的，旁边玩家的白衣服只占几个扇形
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


def find_flame(frame: np.ndarray, area: Rect, flame: np.ndarray, min_score: float = 0.68) -> Disk | None:
    """在 area（整图坐标，调用方给：团子周围那一块）里找火焰，只取最好的一处；没有返回 None。
    有没有白圈、外环亮不亮都不看（10-01 晚真机）；分数够不够确定由调用方按 disk_sure 看。"""
    x1, y1 = max(0, area.x), max(0, area.y)
    x2, y2 = min(frame.shape[1], area.x2), min(frame.shape[0], area.y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    mask = cv2.GaussianBlur(cream(frame[y1:y2, x1:x2]), (0, 0), BLUR)
    flame = cv2.GaussianBlur(flame, (0, 0), BLUR)
    best, scale = None, 1.0
    for s in SCALES:
        m = best_match(mask, flame, [s])
        if best is None or m.score > best.score:
            best, scale = m, s
    if best is None or best.score < min_score:
        return None
    return Disk(x1 + best.x, y1 + best.y, flame.shape[0] * scale / 2, best.score)


BLACK_MIN_VISIBLE = 0.4  # 扣掉团子后剩下的像素不到区域的这么多：被团子挡住大半，量不准


def black(frame: np.ndarray, box: Rect, v: int = 50, exclude: Rect | None = None) -> float | None:
    """人物框有多黑：中间一半宽、上 70% 高里很暗（HSV 的 V < v）的像素占比。黑影 ≈ 高，点亮后显出衣服颜色 ≈ 低。
    下面 30% 不看（脚下的影子），两边不看（背景）。框和画面不重叠返回 0。
    exclude（团子框）：落在里面的像素不算（黑影抱着团子站时，团子橙色的身体会把 black 拉低）；
    剩下的不到区域的 BLACK_MIN_VISIBLE 返回 None（看不清）。"""
    x1 = max(0, round(box.x + box.w / 4))
    x2 = min(frame.shape[1], round(box.x + 3 * box.w / 4))
    y1 = max(0, box.y)
    y2 = min(frame.shape[0], round(box.y + 0.7 * box.h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    value = frame[y1:y2, x1:x2].max(axis=2)  # HSV 的 V = BGR 三通道最大值
    dark = value < v
    if exclude is None:
        return float(dark.mean())
    keep = np.ones(dark.shape, bool)
    keep[max(0, exclude.y - y1):max(0, exclude.y2 - y1), max(0, exclude.x - x1):max(0, exclude.x2 - x1)] = False
    if keep.mean() < BLACK_MIN_VISIBLE:
        return None
    return float(dark[keep].mean())
