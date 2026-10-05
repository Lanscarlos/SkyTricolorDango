"""认地图交互图标的纯计算（spec 2026-10-05-icon-detection §3.3 / §3.6）：归属、叫法、投票、裁图。

不碰设备、不跑模型；运行时接线在后面的任务里。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..game.social import KIND_NAMES
from .track import Rect

# 圈的归属（好友在调用方用名字标签先判，不进这里）
MAP_OWNERS: tuple[str, ...] = ("spirit", "bonfire", "bench", "instrument", "map")
UNKNOWN = "unknown"

_OVERRIDES = {("candle", "bonfire"): "篝火点燃", ("candle", "map"): "可以点的蜡烛 / 灯"}
_UNKNOWN_LABEL = "不认识的图标"


@dataclass(frozen=True)
class Icon:
    track_id: int
    kind: str  # 种类（UNKNOWN = 没认出）
    owner: str  # person / spirit / 物件类别名 / map
    label: str
    box: Rect
    side: str  # 左边 / 中间 / 右边……


def under_gap(ring: Rect, box: Rect, under_x: float, under_up: float) -> float | None:
    """圈在框的头顶上方范围内时返回圈心到框顶的竖直距离，否则 None。

    横着：圈心在框左右各放宽 under_x 倍框宽内；竖着：圈心在框顶往上 under_up 倍框高到框中线之间。"""
    cx, cy = ring.x + ring.w / 2, ring.y + ring.h / 2
    if not (box.x - under_x * box.w <= cx <= box.x2 + under_x * box.w):
        return None
    if not (box.y - under_up * box.h <= cy <= box.y + 0.5 * box.h):
        return None
    return abs(cy - box.y)


def _any_under(ring: Rect, boxes: list[Rect], under_x: float, under_up: float) -> bool:
    return any(under_gap(ring, b, under_x, under_up) is not None for b in boxes)


def owner_of(ring: Rect, people: list[Rect], spirits: list[Rect], things: list[tuple[str, Rect]],
             under_x: float, under_up: float) -> str:
    """圈属于谁：person > spirit > 物件（最近的那个类别）> map。"""
    if _any_under(ring, people, under_x, under_up):
        return "person"
    if _any_under(ring, spirits, under_x, under_up):
        return "spirit"
    best: tuple[float, str] | None = None
    for name, box in things:
        gap = under_gap(ring, box, under_x, under_up)
        if gap is not None and (best is None or gap < best[0]):
            best = (gap, name)
    return best[1] if best else "map"


def map_label(kind: str | None, owner: str) -> str:
    if kind is None or kind == UNKNOWN:
        return _UNKNOWN_LABEL
    if (kind, owner) in _OVERRIDES:
        return _OVERRIDES[(kind, owner)]
    return KIND_NAMES.get(kind, kind)


def vote(history: list[str]) -> str | None:
    """多数票；UNKNOWN 不计票，平票取最近一次出现的；全是 UNKNOWN 返回 UNKNOWN，空返回 None。"""
    if not history:
        return None
    counts: dict[str, int] = {}
    last: dict[str, int] = {}
    for i, k in enumerate(history):
        if k == UNKNOWN:
            continue
        counts[k] = counts.get(k, 0) + 1
        last[k] = i
    if not counts:
        return UNKNOWN
    return max(counts, key=lambda k: (counts[k], last[k]))


def icon_crop(frame: np.ndarray, box: Rect, scale: float) -> np.ndarray:
    """以框中心裁 scale × 长边的正方形，超出画面的部分去掉；完全在外面返回空数组。"""
    side = max(box.w, box.h) * scale
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    h, w = frame.shape[:2]
    x1, x2 = max(0, round(cx - side / 2)), min(w, round(cx + side / 2))
    y1, y2 = max(0, round(cy - side / 2)), min(h, round(cy + side / 2))
    if x2 <= x1 or y2 <= y1:
        return frame[0:0, 0:0]
    return frame[y1:y2, x1:x2]


def describe_icons(icons: list[Icon]) -> str:
    """"坐下（右边）、留影（左边）、不认识的图标 1 个"：认得的按顺序，不认识的合成一项放最后。"""
    known = [f"{i.label}（{i.side}）" for i in icons if i.kind != UNKNOWN]
    unknown = sum(1 for i in icons if i.kind == UNKNOWN)
    if unknown:
        known.append(f"{_UNKNOWN_LABEL} {unknown} 个")
    return "、".join(known)
