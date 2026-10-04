"""环绕扫描的纯计算：方位角、8 个方向、同一个人多帧合并、转圈认团子、远近分档（感知层二期 §1、§4）。

转一圈时每帧的检测由 `PerceptionWatcher.sweep()` 跑，这里只管算：
- 方位 = 开始转以来按住的时长折成的角度 + 框中心偏离画面中心多少 × 水平视野角；0° = 转之前的正前方，向右转为正
- 转圈认团子：镜头绕着团子转，团子在画面上几乎不动、别的东西都横着扫过 → 一直在、几乎不动、离中心最近的那条就是团子
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

from .bubbles import Rect
from .track import iou

DIRECTIONS = ["前", "右前", "右", "右后", "后", "左后", "左", "左前"]
BESIDE = "身边"  # 转圈时几乎不动、紧挨着团子的人（牵着手）：方位没意义
SIDE_TEXT = {BESIDE: "身边", "前": "正前方", "右前": "右前方", "右": "右边", "右后": "右后方", "后": "正后方",
             "左后": "左后方", "左": "左边", "左前": "左前方"}
STRANGER_WHO = "陌生人"
UNLIT_WHO = "陌生人（没点火）"
UNKNOWN_WHO = "没认出名字的人"  # 有名字标签，但读不出来或者不在好友名单里
GENERIC = (STRANGER_WHO, UNLIT_WHO, UNKNOWN_WHO)


def bearing(t: float, cx: float, width: int, seconds_per_turn: float, hfov: float) -> float:
    return (t / seconds_per_turn * 360.0 + (cx - width / 2) / width * hfov) % 360.0


def direction(deg: float) -> str:
    return DIRECTIONS[int(((deg % 360.0) + 22.5) // 45) % 8]


def distance(box_h: float, ref_h: float, near: float, far: float) -> str:
    """人物框高 ÷ 参照高度（团子框高）：≥ near 近，≥ far 中，否则远。"""
    ratio = box_h / ref_h if ref_h > 0 else 0.0
    return "近" if ratio >= near else ("中" if ratio >= far else "远")


def _mean_deg(values: list[float]) -> float:
    s = sum(math.sin(math.radians(v)) for v in values)
    c = sum(math.cos(math.radians(v)) for v in values)
    return math.degrees(math.atan2(s, c)) % 360.0


def _gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


@dataclass
class Sighting:
    degrees: float
    who: str  # 好友名 / STRANGER_WHO / UNLIT_WHO / UNKNOWN_WHO
    frame: int
    height: float | None = None  # 人物框高（算远近）；只看到名字标签时 None
    beside: bool = False  # 名字标签下的人在转圈时几乎不动（紧挨团子）


@dataclass
class SweepEntry:
    direction: str  # "右后"；紧挨团子的是 "身边"
    degrees: float
    who: str  # 好友名 / "陌生人" / "陌生人（没点火）" / "没认出名字的人"
    frames: int  # 在多少帧里出现
    distance: str | None  # 近 / 中 / 远


@dataclass
class SweepResult:
    entries: list[SweepEntry]
    self_box: Rect | None  # 转圈认出的团子
    frames: int
    seconds: float
    enrolled: int = 0  # 启动登记（sweep(enroll=True)）钉进团子底库的样本张数

    def text(self) -> str:
        """"正前方：懒洋洋大王；右后方：2 个陌生人（1 个没点火）"。"""
        parts = []
        for side in [BESIDE, *DIRECTIONS]:
            here = [e for e in self.entries if e.direction == side]
            if not here:
                continue
            items = [e.who + (f"（{e.distance}）" if e.distance else "") for e in here if e.who not in GENERIC]
            strangers = sum(e.who in (STRANGER_WHO, UNLIT_WHO) for e in here)
            unlit = sum(e.who == UNLIT_WHO for e in here)
            if strangers:
                items.append(f"{strangers} 个陌生人" + (f"（{unlit} 个没点火）" if unlit else ""))
            unknown = sum(e.who == UNKNOWN_WHO for e in here)
            if unknown:
                items.append(f"{unknown} 个没认出名字的人")
            parts.append(f"{SIDE_TEXT[side]}：{'、'.join(items)}")
        return "；".join(parts) or "转了一圈，身边没看到别人"


def _entry(group: list[Sighting], who: str, ref_h: float, near: float, far: float, beside: bool = False) -> SweepEntry:
    deg = 0.0 if beside else _mean_deg([s.degrees for s in group])
    heights = [s.height for s in group if s.height is not None]
    return SweepEntry(
        BESIDE if beside else direction(deg), round(deg, 1), who, len({s.frame for s in group}),
        distance(max(heights), ref_h, near, far) if heights else None,
    )


def _cluster(sightings: list[Sighting], merge_deg: float) -> list[list[Sighting]]:
    clusters: list[list[Sighting]] = []
    for s in sorted(sightings, key=lambda s: s.degrees):
        best = min(clusters, key=lambda c: _gap(_mean_deg([x.degrees for x in c]), s.degrees), default=None)
        if best is not None and _gap(_mean_deg([x.degrees for x in best]), s.degrees) < merge_deg:
            best.append(s)
        else:
            clusters.append([s])
    return clusters


def merge(sightings: list[Sighting], merge_deg: float, ref_h: float, near: float, far: float) -> list[SweepEntry]:
    """同一个人在多帧里出现合成一条：好友按名字，陌生人 / 黑影按方位聚类（取出现帧数最多的类别），没认出名字的单独聚类。"""
    out: list[SweepEntry] = []
    # 紧挨团子、转圈时几乎不动的人（牵着手）：方位没意义，不管认没认出名字都只报"身边"
    beside = [s for s in sightings if s.beside]
    beside_named: dict[str, list[Sighting]] = {}
    for s in beside:
        if s.who not in GENERIC:
            beside_named.setdefault(s.who, []).append(s)
    for name, group in beside_named.items():
        out.append(_entry(group, name, ref_h, near, far, beside=True))
    if beside and not beside_named:  # 名字一帧都没读出来（不在好友名单里 / 一直糊）
        out.append(_entry(beside, UNKNOWN_WHO, ref_h, near, far, beside=True))

    rest = [s for s in sightings if not s.beside]
    friends: dict[str, list[Sighting]] = {}
    for s in rest:
        if s.who not in GENERIC and s.who not in beside_named:
            friends.setdefault(s.who, []).append(s)
    named = [_entry(group, name, ref_h, near, far) for name, group in friends.items()]
    out += named
    for kinds in ((STRANGER_WHO, UNLIT_WHO), (UNKNOWN_WHO,)):
        for group in _cluster([s for s in rest if s.who in kinds], merge_deg):
            counts = {k: len({s.frame for s in group if s.who == k}) for k in kinds}
            who = max(kinds, key=lambda k: counts[k])
            entry = _entry(group, who, ref_h, near, far)
            # 好友旁边的"陌生人 / 没认出名字的人"多半是同一个好友某几帧标签没检测到或没读出来（转圈时画面糊）
            if who != UNLIT_WHO and any(_gap(entry.degrees, f.degrees) < merge_deg for f in named):
                continue
            out.append(entry)
    return sorted(out, key=lambda e: e.degrees)


@dataclass
class SelfFound:
    box: Rect | None  # 各帧框的中位数；None = 没认出或者分不开（牵手）
    static: set[tuple[int, int]] = field(default_factory=set)  # (帧号, 框号)：所有"几乎不动"的链上的框
    per_frame: dict[int, Rect] = field(default_factory=dict)  # 认出的团子在各帧的框


def find_self(
    boxes: list[list[Rect]], width: int, self_motion: float,
    min_ratio: float = 0.8, link_iou: float = 0.3, min_frames: int = 5,
) -> SelfFound:
    """转圈认团子：boxes[帧][框] 是每帧的人物框（player / self）。"""
    n = len(boxes)
    if n < min_frames:
        return SelfFound(None, set(), {})
    chains: list[list[tuple[int, int, Rect]]] = []
    for fi, frame in enumerate(boxes):
        pairs = sorted(
            ((iou(chain[-1][2], box), ci, bi) for ci, chain in enumerate(chains) if chain[-1][0] < fi
             for bi, box in enumerate(frame)),
            reverse=True,
        )
        used_c: set[int] = set()
        used_b: set[int] = set()
        for overlap, ci, bi in pairs:
            if overlap < link_iou or ci in used_c or bi in used_b:
                continue
            chains[ci].append((fi, bi, frame[bi]))
            used_c.add(ci)
            used_b.add(bi)
        chains += [[(fi, bi, box)] for bi, box in enumerate(frame) if bi not in used_b]

    still = []  # (离中心的距离, 平均中心, 链)
    for chain in chains:
        if len({fi for fi, _, _ in chain}) / n < min_ratio:
            continue
        centers = [(b.x + b.w / 2, b.y + b.h / 2) for _, _, b in chain]
        mx = sum(c[0] for c in centers) / len(centers)
        my = sum(c[1] for c in centers) / len(centers)
        motion = sum(math.hypot(x - mx, y - my) for x, y in centers) / len(centers) / width
        if motion >= self_motion:
            continue
        still.append((abs(mx - width / 2), (mx, my), chain))  # 只比横向：团子在画面中线上，竖直位置随镜头高低变
    if not still:
        return SelfFound(None, set(), {})
    static = {(fi, bi) for _, _, chain in still for fi, bi, _ in chain}
    still.sort(key=lambda s: s[0])
    _, (cx, cy), chain = still[0]
    box_w = median(b.w for _, _, b in chain)
    if any(math.hypot(ox - cx, oy - cy) < box_w for _, (ox, oy), _ in still[1:]):
        return SelfFound(None, static, {})  # 牵着手：两个都不动、紧挨着，分不开
    me = Rect(*(int(round(median(getattr(b, k) for _, _, b in chain))) for k in ("x", "y", "w", "h")))
    return SelfFound(me, static, {fi: b for fi, _, b in chain})
