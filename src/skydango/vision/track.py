"""简单的多目标追踪：同一类别里按框的重叠（IoU）把这一帧的检测接到上一帧的轨迹上。

15fps 下相邻两帧的人物移动不大，IoU 贪心匹配就够用；转视角时轨迹会断，
好友靠名字重新接回（world.py），陌生人断了就当新的一条 —— 数陌生人时用"最近几秒同时出现的最多人数"，不数轨迹条数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bubbles import Rect
from .detect import Detection


def iou(a: Rect, b: Rect) -> float:
    w = min(a.x2, b.x2) - max(a.x, b.x)
    h = min(a.y2, b.y2) - max(a.y, b.y)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a.w * a.h + b.w * b.h - inter)


@dataclass
class Track:
    id: int
    cls: str
    box: Rect
    score: float
    first: float  # 第一次看到的时间
    last: float  # 最近一次看到的时间
    hits: int = 1
    flips: int = 0  # 类别在 cross 组里来回变了几次（player ↔ player_unlit，难例收集用）
    data: dict = field(default_factory=dict)  # 上层挂的东西（名字投票、身份……）


class Tracker:
    def __init__(self, buffer: float = 1.0, min_iou: float = 0.3, cross: frozenset[str] = frozenset(),
                 cross_iou: float = 0.5) -> None:
        self.buffer = buffer  # 轨迹这么久没匹配上就删
        self.min_iou = min_iou
        self.cross = cross  # 这几个类别之间也能接上（同一个人一会儿认成 player、一会儿认成 player_unlit）
        self.cross_iou = cross_iou  # 跨类别要重叠得更多才算同一个
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, dets: list[Detection], now: float) -> list[Track]:
        """接上这一帧的检测，返回这一帧看到的轨迹（顺序同 dets）。"""
        for tid in [t.id for t in self.tracks.values() if now - t.last > self.buffer]:
            del self.tracks[tid]
        pairs = []
        for di, det in enumerate(dets):
            for track in self.tracks.values():
                if track.cls == det.cls:
                    need = self.min_iou
                elif track.cls in self.cross and det.cls in self.cross:
                    need = self.cross_iou
                else:
                    continue
                overlap = iou(track.box, det.box)
                if overlap >= need:
                    pairs.append((overlap, di, track.id))
        pairs.sort(reverse=True)
        matched: dict[int, Track] = {}
        used: set[int] = set()
        for _, di, tid in pairs:
            if di in matched or tid in used:
                continue
            track = self.tracks[tid]
            det = dets[di]
            if track.cls != det.cls:
                track.cls = det.cls
                track.flips += 1
            track.box, track.score, track.last = det.box, det.score, now
            track.hits += 1
            matched[di] = track
            used.add(tid)
        out = []
        for di, det in enumerate(dets):
            track = matched.get(di)
            if track is None:
                track = Track(self._next, det.cls, det.box, det.score, now, now)
                self.tracks[track.id] = track
                self._next += 1
            out.append(track)
        return out

    def shift(self, d: float) -> None:
        """感知暂停了 d 秒：所有轨迹的时间往后挪，恢复后不会因为"太久没看到"而断掉。"""
        for track in self.tracks.values():
            track.first += d
            track.last += d
