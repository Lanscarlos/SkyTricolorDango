"""呼唤光圈的离线标定（`perception halo-eval`，spec docs/superpowers/specs/2026-10-01-q-call-design.md §5）。

录像上逐帧跑感知层，每条人物 / 团子轨迹记头顶区域的亮度（区域灰度均值 − 整帧均值，抵消画面整体变亮变暗），
减去这条轨迹最开始 0.3 秒的中位数作基准 → 亮度变化曲线。超过门槛的时间段就是"冒圈"；
团子短按 Q 时他那条的峰值、旁边人的噪声放在一起看，给出建议的 halo_rise。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..imageio import imwrite
from .halo import head_region

BASE_SECONDS = 0.3  # 每条轨迹前这么久的中位数当基准
MIN_PEAK = 10.0  # 最高的峰不到这么多：录像里多半没有冒圈，不给建议
CANDIDATES = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0)  # 报告里列出各门槛下有几段超过


class Curves:
    def __init__(self) -> None:
        self.raw: dict[int, list[tuple[float, float]]] = {}

    def add(self, frame: np.ndarray, t: float, tracks) -> None:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = g.shape[:2]
        whole = float(g.mean())
        for tr in tracks:
            if tr.cls not in ("player", "self"):
                continue
            r = head_region(tr.box, w, h)
            if r is not None:
                self.raw.setdefault(tr.id, []).append((t, float(r.crop(g).mean()) - whole))

    def rises(self) -> dict[int, list[tuple[float, float]]]:
        out = {}
        for tid, pts in self.raw.items():
            t0 = pts[0][0]
            base = float(np.median([v for t, v in pts if t < t0 + BASE_SECONDS]))
            out[tid] = [(t, v - base) for t, v in pts]
        return out


def segments(curve: list[tuple[float, float]], threshold: float) -> list[tuple[float, float, float]]:
    """连续超过门槛的时间段：(开始, 结束, 峰值)。"""
    out, cur = [], None
    for t, v in curve:
        if v >= threshold:
            cur = [t, t, v] if cur is None else [cur[0], t, max(cur[2], v)]
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def suggest(rises: dict[int, list[tuple[float, float]]]) -> dict:
    """噪声 = 低于峰值一半的点里 |变化| 的 99 分位；建议门槛取噪声和峰值的中间。
    峰值不到噪声两倍、不到 MIN_PEAK、或者全是高点（看不出噪声）就分不开（None）。"""
    values = [v for pts in rises.values() for _, v in pts]
    if not values:
        return {"noise": None, "peak": None, "halo_rise": None}
    peak = max(values)
    quiet = [abs(v) for v in values if v < peak / 2]
    noise = float(np.percentile(quiet, 99)) if quiet else None
    ok = noise is not None and peak >= MIN_PEAK and peak >= 2 * max(noise, 1.0)
    return {"noise": None if noise is None else round(noise, 1), "peak": round(peak, 1), "halo_rise": round((noise + peak) / 2, 1) if ok else None}


def report_md(summary: dict) -> str:
    s = summary["suggest"]
    fmt = lambda v: "—" if v is None else f"{v:.1f}"  # noqa: E731
    lines = [
        "# 呼唤光圈标定（perception halo-eval）", "",
        f"- 录像：`{summary['source']}`，{summary['frames']} 帧",
        f"- 噪声（99 分位）：{fmt(s['noise'])}；最高的峰：{fmt(s['peak'])}",
        f"- **建议 halo_rise：{fmt(s['halo_rise'])}**（现在 {summary['current']:g}）"
        + ("" if s["halo_rise"] is not None else "：峰值不到噪声两倍，分不开，别开 halo"),
        "", "曲线见 `curves.png`。团子短按 Q 的那几次，应该只有团子那条冒出一段；别人也在喊时会有好几条。", "",
        "## 每条轨迹", "", "| 轨迹 | 峰值 | 超过现在门槛的时间段（秒） |", "|---|---|---|",
    ]
    for tid, info in summary["tracks"].items():
        segs = "、".join(f"{a:.2f}~{b:.2f}（{p:.0f}）" for a, b, p in info["segments"]) or "—"
        lines.append(f"| {tid} | {info['peak']:.1f} | {segs} |")
    if summary.get("counts"):
        lines += ["", "## 各门槛下冒圈的段数", "", "| halo_rise | 段数 |", "|---|---|"]
        lines += [f"| {k} | {n} |" for k, n in summary["counts"].items()]
    return "\n".join(lines) + "\n"


def plot(rises: dict[int, list[tuple[float, float]]], path: Path, size: tuple[int, int] = (1200, 500)) -> None:
    """每条轨迹一条折线（cv2 画，不依赖 matplotlib）；横轴时间、纵轴亮度变化，灰线是 0。"""
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    pts_all = [p for pts in rises.values() for p in pts]
    if pts_all:
        t0, t1 = min(t for t, _ in pts_all), max(t for t, _ in pts_all)
        lo, hi = min(min(v for _, v in pts_all), 0.0), max(max(v for _, v in pts_all), 1.0)
        sx = lambda t: int(40 + (t - t0) / max(t1 - t0, 1e-6) * (w - 60))  # noqa: E731
        sy = lambda v: int(h - 30 - (v - lo) / (hi - lo) * (h - 60))  # noqa: E731
        cv2.line(img, (40, sy(0.0)), (w - 20, sy(0.0)), (200, 200, 200), 1)
        rng = np.random.default_rng(0)
        for tid, pts in rises.items():
            color = tuple(int(c) for c in rng.integers(0, 200, 3))
            line = np.array([(sx(t), sy(v)) for t, v in pts], np.int32)
            cv2.polylines(img, [line], False, color, 2)
            cv2.putText(img, str(tid), tuple(int(x) for x in line[int(np.argmax([v for _, v in pts]))]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)
        cv2.putText(img, f"{t0:.1f}s", (40, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(img, f"{t1:.1f}s", (w - 70, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(img, f"{hi:.0f}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    imwrite(Path(path), img)
