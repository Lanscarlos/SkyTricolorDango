"""简单的多目标追踪：同一类别里按框的重叠（IoU）把这一帧的检测接到上一帧的轨迹上。

15fps 下相邻两帧的人物移动不大，IoU 贪心匹配就够用；转视角时轨迹会断，
好友靠名字重新接回（world.py），陌生人断了就当新的一条 —— 数陌生人时用"最近几秒同时出现的最多人数"，不数轨迹条数。

升级（spec 2026-10-01-tracking-relink-motion §2，新参数不传时行为不变）：
- 两段匹配：高分框先配；剩下的轨迹再和低分框（low_conf ~ conf）配，低分框只续旧轨迹、默认不开新轨迹（open_low 里的类别可以开待复核的新轨迹）
- 速度预测（predict）：轨迹记 vx / vy / vh，按速度往前推最多 PREDICT_MAX 秒再算 IoU
- 中心距离兜底（center_gate > 0）：小框挪几个像素 IoU 就掉光，中心离预测框不远、框高差不多也算候选，永远排在 IoU 候选之后
- 画面平移（shift）：转镜头时整幅画面平移，累计到每条轨迹的 pan 上，预测框加上它（带 / 不带各试一次：近处的人和远处背景平移量不一样）
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from .bubbles import Rect
from .detect import Detection

PREDICT_MAX = 0.5  # 速度预测最多往前推这么久（秒）
VELOCITY_GAP = 0.5  # 两次匹配隔得比这久就不更新速度（中间可能被挡、换了人）
VELOCITY_ALPHA = 0.5  # 速度的指数平均
GATE_RATIO = (0.67, 1.5)  # 中心距离候选：框高比要在这之间
PAN_MIN_RESPONSE = 0.1  # 画面平移估计：相位相关的响应低于这个不信（估的）
# 隔着一段暂停（身体转了镜头）估平移要严得多：两幅不相关的噪声图响应也能到 0.3，真平移约 0.7
PAN_RECHECK_RESPONSE = 0.4


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
    vx: float = 0.0  # 速度（像素 / 秒，指数平均；不含画面平移）
    vy: float = 0.0
    vh: float = 0.0  # 框高变化（像素 / 秒）
    weak_hits: int = 0  # 被低分框续上的次数（track-eval 用）
    pan: tuple[float, float] = (0.0, 0.0)  # 上次匹配以来累计的画面平移
    drift: tuple[float, float] = (0.0, 0.0)  # 这条轨迹被画面平移带着走了多少（只算带平移的预测框配上的那几次；运动方向用）
    strong_last: float = float("-inf")  # 最近一次被高分框接上（或创建）的时间：只靠低分框续着的轨迹别无限续命


def _moved(box: Rect, dx: float, dy: float, dh: float = 0.0) -> Rect:
    """框平移 (dx, dy)、高度变化 dh（中心不动，宽按比例跟着变）。"""
    h = max(1.0, box.h + dh)
    w = box.w * h / box.h if box.h else box.w
    cx, cy = box.x + box.w / 2 + dx, box.y + box.h / 2 + dy
    return Rect(round(cx - w / 2), round(cy - h / 2), round(w), round(h))


def _center_dist(a: Rect, b: Rect) -> float:
    return float(np.hypot(a.x + a.w / 2 - b.x - b.w / 2, a.y + a.h / 2 - b.y - b.h / 2))


class Tracker:
    def __init__(self, buffer: float = 1.0, min_iou: float = 0.3, cross: frozenset[str] = frozenset(),
                 cross_iou: float = 0.5, *, center_gate: float = 0.0, predict: bool = False,
                 open_low: frozenset[str] = frozenset()) -> None:
        self.buffer = buffer  # 轨迹这么久没匹配上就删
        self.min_iou = min_iou
        self.cross = cross  # 这几个类别之间也能接上（同一个人一会儿认成 player、一会儿认成 player_unlit）
        self.cross_iou = cross_iou  # 跨类别要重叠得更多才算同一个
        self.center_gate = center_gate  # 中心距离门槛（× 预测框高）；0 = 不用
        self.predict = predict  # 按速度预测位置
        self.open_low = open_low  # 这些类别的低分框没配上旧轨迹时也开新轨迹（待第二层复核；可随时改，空 = 只续旧轨迹）
        self.tracks: dict[int, Track] = {}
        self.dropped: list[Track] = []  # 最近一次（prune 的）update 删掉的轨迹
        self.calm_until = float("-inf")  # 这之前的匹配不更新速度（镜头缩放、走路后画面还没停稳）
        self._next = 1

    def calm(self, until: float) -> None:
        """速度清零，until 之前不再更新速度（镜头缩放 / 走路 / 转圈、暂停恢复时）。"""
        self.calm_until = max(self.calm_until, until)
        for track in list(self.tracks.values()):  # 身体线程会调：感知线程可能同时在增删轨迹
            track.vx = track.vy = track.vh = 0.0

    def predicted(self, track: Track, now: float) -> list[Rect]:
        """预测框：最后的框 + 速度 × 时间（最多 PREDICT_MAX 秒，predict 关着时不动）+ 累计的画面平移。
        平移不为零时再给一个不加平移的（近处的人和远处背景平移量不一样，两个都试）。"""
        return [box for box, _ in self._preds(track, now)]

    def _preds(self, track: Track, now: float) -> list[tuple[Rect, bool]]:
        """(预测框, 是不是加了画面平移的那个)。"""
        dt = min(max(now - track.last, 0.0), PREDICT_MAX) if self.predict else 0.0
        base = (track.vx * dt, track.vy * dt, track.vh * dt)
        px, py = track.pan
        if px or py:
            return [(_moved(track.box, base[0] + px, base[1] + py, base[2]), True), (_moved(track.box, *base), False)]
        return [(_moved(track.box, *base) if any(base) else track.box, False)]

    def _need(self, track: Track, det: Detection) -> float | None:
        if track.cls == det.cls:
            return self.min_iou
        if track.cls in self.cross and det.cls in self.cross:
            return self.cross_iou
        return None

    def _pairs(self, dets: Sequence[Detection], tracks: Iterable[Track], now: float, gate_on: bool = True) -> list:
        """候选 (分数, 检测序号, 轨迹 id, 是不是带平移的预测框配上的)；分数是元组 (1, IoU) / (0, 中心距离分)，
        IoU 候选永远排在前面；分数一样时序号大的先配（同原来的 sort(reverse=True)）。gate_on = False：只认 IoU（低分框）。"""
        pairs = []
        for track in tracks:
            preds = self._preds(track, now)
            for di, det in enumerate(dets):
                need = self._need(track, det)
                if need is None:
                    continue
                overlap, panned = max((iou(p, det.box), panned) for p, panned in preds)
                if overlap >= need:
                    pairs.append(((1, overlap), di, track.id, panned))
                    continue
                if not gate_on or self.center_gate <= 0 or track.cls != det.cls or not track.box.h:
                    continue
                ratio = det.box.h / track.box.h
                if not GATE_RATIO[0] <= ratio <= GATE_RATIO[1]:
                    continue
                dist, gate, panned = min((_center_dist(p, det.box), self.center_gate * p.h, panned) for p, panned in preds)
                if dist <= gate:
                    pairs.append(((0, 0.3 * (1 - dist / gate) if gate else 0.0), di, track.id, panned))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]), reverse=True)
        return pairs

    def _assign(self, dets: Sequence[Detection], free: dict[int, Track], now: float, strong: bool = True) -> dict[int, Track]:
        """贪心配对：返回 {检测序号: 轨迹}，配上的轨迹从 free 里拿掉。strong = False（低分框）：只认 IoU。"""
        matched: dict[int, Track] = {}
        for _, di, tid, panned in self._pairs(dets, list(free.values()), now, gate_on=strong):
            if di in matched or tid not in free:
                continue
            track = free.pop(tid)
            self._hit(track, dets[di], now, panned, strong)
            matched[di] = track
        return matched

    def _hit(self, track: Track, det: Detection, now: float, panned: bool = False, strong: bool = True) -> None:
        """panned：带平移的预测框配上的（人站在背景里，被平移带着走）；不带平移的配上 = 人跟着镜头一起动（身边的人），
        这时背景平移不是他的位移，不能从速度里减掉。"""
        if track.cls != det.cls:
            track.cls = det.cls
            track.flips += 1
        pan = track.pan if panned else (0.0, 0.0)
        dt = now - track.last
        if 0 < dt <= VELOCITY_GAP and now >= self.calm_until:
            old, new = track.box, det.box
            sx = (new.x + new.w / 2 - old.x - old.w / 2 - pan[0]) / dt
            sy = (new.y + new.h / 2 - old.y - old.h / 2 - pan[1]) / dt
            sh = (new.h - old.h) / dt
            a = VELOCITY_ALPHA
            track.vx, track.vy, track.vh = (1 - a) * track.vx + a * sx, (1 - a) * track.vy + a * sy, (1 - a) * track.vh + a * sh
        track.drift = (track.drift[0] + pan[0], track.drift[1] + pan[1])
        track.box, track.score, track.last = det.box, det.score, now
        if strong:
            track.strong_last = now
        track.pan = (0.0, 0.0)
        track.hits += 1

    def update(self, dets: list[Detection], now: float, *, low: Sequence[Detection] = (),
               shift: tuple[float, float] | None = None, prune: bool = True) -> list[Track]:
        """接上这一帧的检测，返回这一帧看到的轨迹（顺序同 dets；被低分框续上的、再是低分框新开的附在后面）。

        low：这一帧 low_conf ~ conf 之间的框，续旧轨迹（类别在 open_low 里的没配上就开待复核的新轨迹）；shift：这一帧相对上一帧的画面平移（像素，None = 不知道）；
        prune = False：不删过期轨迹、不动 dropped（同一帧第二次调用时用）。"""
        if shift is not None and (shift[0] or shift[1]):
            for track in self.tracks.values():
                track.pan = (track.pan[0] + shift[0], track.pan[1] + shift[1])
        if prune:
            self.dropped = [t for t in self.tracks.values() if now - t.last > self.buffer]
            for track in self.dropped:
                del self.tracks[track.id]
        free = dict(self.tracks)
        matched = self._assign(dets, free, now)
        weak = self._assign(low, free, now, strong=False) if low and free else {}
        # 没被用掉的低分框：指定类别的开待复核的新轨迹（strong_last 保持 -inf）
        opened = []
        if self.open_low:
            used = set(weak)
            for di, det in enumerate(low):
                if di in used or det.cls not in self.open_low:
                    continue
                track = Track(self._next, det.cls, det.box, det.score, now, now)
                self.tracks[track.id] = track
                self._next += 1
                opened.append(track)
        for track in weak.values():
            track.weak_hits += 1
        out = []
        for di, det in enumerate(dets):
            track = matched.get(di)
            if track is None:
                track = Track(self._next, det.cls, det.box, det.score, now, now, strong_last=now)
                self.tracks[track.id] = track
                self._next += 1
            out.append(track)
        return out + [weak[i] for i in sorted(weak)] + opened

    def shift(self, d: float) -> None:
        """感知暂停了 d 秒：所有轨迹的时间往后挪，恢复后不会因为"太久没看到"而断掉。"""
        for track in self.tracks.values():
            track.first += d
            track.last += d


def estimate_shift(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray | None,
                   min_response: float = PAN_MIN_RESPONSE) -> tuple[float, float] | None:
    """两张同尺寸灰度缩略图之间的画面平移（缩略图像素，内容往右 / 下挪为正）；估不出返回 None。

    mask：True = 可用（人物框、聊天面板之类会自己动的地方填成均值，不让它们带偏）。
    纯色 / 纹理太少、相位相关的响应太低、或者挪得比宽度的 1/3 还多（多半是切了画面）都不信。"""
    a, b = prev.astype(np.float32), cur.astype(np.float32)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 8:
        return None
    if mask is not None:
        if not mask.any():
            return None
        fill_a, fill_b = float(a[mask].mean()), float(b[mask].mean())
        a, b = np.where(mask, a, fill_a), np.where(mask, b, fill_b)
        used = a[mask], b[mask]
    else:
        used = a, b
    if min(float(used[0].std()), float(used[1].std())) < 1.0:
        return None
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    if not np.isfinite(response) or response < min_response or not (np.isfinite(dx) and np.isfinite(dy)):
        return None
    if abs(dx) > a.shape[1] / 3 or abs(dy) > a.shape[0] / 3:
        return None
    return float(dx), float(dy)
