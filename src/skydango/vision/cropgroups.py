"""外形页的裁图分组（纯函数）：同一个人连着几帧的裁图串成一组，页面上按一下键整组判。

只在同一来源里串（数据集 = 同一录像前缀 + 同一 split；难例 / 录像 = 同一运行）；
相邻两帧时间差超过 GAP 就断开；一组最多 CAP 张；取不到时间 / 行写坏了的裁图各自一组。
分组不写进任何数据文件，每次现算。
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from .attrs_data import frame_group
from .bubbles import Rect
from .detect import Detection
from .track import Tracker

GAP = 3.0  # 相邻两帧隔这么久（秒）就不串了
CAP = 30  # 一组最多这么多张
_TIME = re.compile(r"_(\d+(?:\.\d+)?)s$")


def frame_time(stem: str) -> float | None:
    """帧名末尾 _<秒>s（如 0351_175.50s → 175.5）；没有 = None。"""
    m = _TIME.search(stem)
    return float(m.group(1)) if m else None


def _valid(row: dict) -> tuple[str, float, list[int]] | None:
    """(帧名, 时间, 框)；缺 image / box、框不是 4 个数、帧名没有时间 = None。"""
    image, box = row.get("image"), row.get("box")
    if not isinstance(image, str) or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
        return None
    stem = Path(image).stem
    t = frame_time(stem)
    if t is None:
        return None
    return stem, t, [round(v) for v in box]


def _source_key(row: dict, stem: str) -> tuple:
    if row.get("source") == "dataset":
        return ("dataset", str(row.get("dataset")), str(row.get("split")), frame_group(stem))
    return (str(row.get("source")), str(row.get("group")), str(row.get("split")))


def _tracker(gap: float) -> Tracker:
    return Tracker(buffer=gap, min_iou=0.3, center_gate=1.0)


def groups(rows: Iterable[dict], gap: float = GAP, cap: int = CAP) -> list[list[str]]:
    """裁图名分组；每个带 crop 名的行恰好出现在一组里，组内按时间排。"""
    buckets: dict[tuple, list[tuple[float, str, str, list[int]]]] = {}
    singles: list[str] = []
    seen: set[str] = set()
    for row in rows:
        crop = row.get("crop") if isinstance(row, dict) else None
        if not isinstance(crop, str) or crop in seen:
            continue
        seen.add(crop)
        ok = _valid(row)
        if ok is None:
            singles.append(crop)
            continue
        stem, t, box = ok
        buckets.setdefault(_source_key(row, stem), []).append((t, row["image"], crop, box))

    out: list[tuple[tuple, float, list[str]]] = []
    for key, items in buckets.items():
        items.sort(key=lambda it: (it[0], it[1], it[2]))
        members: dict[tuple[int, int], list[tuple[float, str]]] = {}
        tracker, epoch, last_t = _tracker(gap), 0, None
        i = 0
        while i < len(items):
            j = i
            while j < len(items) and items[j][:2] == items[i][:2]:  # 同一张图 = 同一帧
                j += 1
            frame = items[i:j]
            t = frame[0][0]
            if last_t is not None and t - last_t > gap:
                tracker, epoch = _tracker(gap), epoch + 1
            last_t = t
            tracks = tracker.update([Detection("player", Rect(*it[3]), 1.0) for it in frame], t)
            for it, tr in zip(frame, tracks):
                members.setdefault((epoch, tr.id), []).append((it[0], it[2]))
            i = j
        for mem in members.values():
            mem.sort()
            for k in range(0, len(mem), cap):
                part = mem[k:k + cap]
                out.append((key, part[0][0], [c for _, c in part]))
    out.extend((("~",), -1.0, [crop]) for crop in singles)
    out.sort(key=lambda g: (g[0], g[1], g[2][0]))
    return [g[2] for g in out]
