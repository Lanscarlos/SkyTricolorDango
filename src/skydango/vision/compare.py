"""离线对比（`perception compare`，一期 M2）：同一批录像上跑现有的整图 OCR（EnvWatcher）和 YOLO 感知层，看 YOLO 够不够好。

- 现有方案按 env.interval 模拟定时扫描（按文件名里的时间，距上次扫描 ≥ interval 才扫）
- YOLO 每张都跑
不做自动的"真值"：两边结论不一样的帧存成左右拼图，由人判断谁对。mAP 用 `yolo val` 单独看。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_TIME = re.compile(r"_(\d+(?:\.\d+)?)s$")  # record 录的文件名：0012_006.00s.jpg

COLORS = {  # BGR，和可视化网页的颜色大致一致
    "name": (80, 200, 80), "friend": (80, 200, 80), "tag": (0, 210, 255), "player": (0, 210, 255),
    "stranger": (0, 140, 255), "unlit": (200, 80, 160), "self": (200, 200, 200),
    "ring": (220, 200, 0), "request": (0, 0, 255),
}


def frame_time(path: Path) -> float | None:
    """record 录的文件名里带着"开始录之后第几秒"；别的图没有，返回 None。"""
    m = _TIME.search(Path(path).stem)
    return float(m.group(1)) if m else None


@dataclass
class FrameResult:
    file: str
    t: float
    env: set[str] | None  # None = 现有方案这一帧没扫
    yolo: set[str]
    env_requests: set[tuple[str, str]]
    yolo_requests: set[tuple[str, str]]
    strangers: int
    env_ms: float | None
    yolo_ms: float
    far: int = 0  # 这一帧远处（框高 < far_height）的 player 有几个
    far_named: int = 0  # 其中认出了是哪个好友的


def compare_frames(
    items: Iterable[tuple[float, str, np.ndarray, bool]],
    env,  # vision.env.EnvWatcher（background=False）
    yolo,  # vision.perception.PerceptionWatcher（background=False）
    on_diff: Callable[[FrameResult, np.ndarray], None] | None = None,
) -> list[FrameResult]:
    """items：(时间, 文件名, 帧, 聊天面板开没开)，按时间排好。两边结论不一样（现有方案扫了的帧）时调 on_diff。"""
    now = [0.0]
    yolo.clock = lambda: now[0]  # 录像时间当时钟：暂停计时、集体消失都按录像里的时间算
    results = []
    for t, name, frame, panel in items:
        now[0] = t
        started = time.perf_counter()
        env.observe(frame, t, panel)
        env_ms = (time.perf_counter() - started) * 1000
        scanned = env._last_scan == t
        started = time.perf_counter()
        yolo.process(frame, t, panel)
        yolo_ms = (time.perf_counter() - started) * 1000
        far = [t for t in yolo.last_tracks
               if t.cls == "player" and t.box.h < yolo.cfg.far_height * frame.shape[0]]
        result = FrameResult(
            file=name,
            t=t,
            env={n for n, seen in env.last_seen.items() if seen == t} if scanned else None,
            yolo={n for n, seen in yolo.last_seen.items() if seen == t},
            env_requests={(r.name, r.kind) for r in env.requests.values() if r.seen_at == t} if scanned else set(),
            yolo_requests={(r.name, r.kind) for r in yolo.requests.values() if r.seen_at == t},
            strangers=yolo.strangers(t),
            env_ms=env_ms if scanned else None,
            yolo_ms=yolo_ms,
            far=len(far),
            far_named=sum(bool(tr.data.get("name")) for tr in far),
        )
        results.append(result)
        if on_diff is not None and result.env is not None and result.env != result.yolo:
            on_diff(result, frame)
    return results


def _leaves(sightings: list[tuple[float, set[str]]], times: list[float], keep: float) -> list[dict]:
    """按 keep 模拟身体的"走开了"：名字最后一次看到后 keep 秒都没再看到（在录像时间内）→ 一次 leave。"""
    seen_at = dict(sightings)
    last: dict[str, float] = {}
    out = []
    for t in times:
        for name, at in sorted(last.items()):
            if t - at > keep:
                out.append({"name": name, "t": at + keep})
                del last[name]
        for name in seen_at.get(t, ()):
            last[name] = t
    return out


def summarize(results: list[FrameResult], keep_env: float, keep_yolo: float) -> dict:
    scanned = [r for r in results if r.env is not None]
    names = sorted({n for r in results for n in r.yolo} | {n for r in scanned for n in r.env})
    friends = {
        n: {
            "env": sum(n in r.env for r in scanned),
            "yolo_same_frames": sum(n in r.env and n in r.yolo for r in scanned),
            "yolo_all": sum(n in r.yolo for r in results),
        }
        for n in names
    }
    first: dict[str, dict] = {}
    for r in results:
        for side, reqs in (("env", r.env_requests), ("yolo", r.yolo_requests)):
            for name, kind in reqs:
                entry = first.setdefault(f"{name}:{kind}", {"env": None, "yolo": None, "delay": None})
                if entry[side] is None:
                    entry[side] = r.t
    for entry in first.values():
        if entry["env"] is not None and entry["yolo"] is not None:
            entry["delay"] = round(entry["env"] - entry["yolo"], 3)  # 正数 = YOLO 早发现这么多秒
    strangers, prev = [], 0
    for r in results:
        if bool(r.strangers) != bool(prev):
            strangers.append({"t": r.t, "n": r.strangers, "kind": "arrive" if r.strangers else "leave"})
        prev = r.strangers
    times = [r.t for r in results]
    far_players, far_named = sum(r.far for r in results), sum(r.far_named for r in results)
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    return {
        "frames": len(results),
        "scanned": len(scanned),
        "friends": friends,
        "diff_frames": [r.file for r in scanned if r.env != r.yolo],
        "requests": first,
        "stranger_events": strangers,
        "leave_events": {
            "env": _leaves([(r.t, r.env) for r in scanned], times, keep_env),
            "yolo": _leaves([(r.t, r.yolo) for r in results], times, keep_yolo),
        },
        "far": {"players": far_players, "named": far_named,
                "rate": round(far_named / far_players, 3) if far_players else None},
        "timing": {"env_ms": mean([r.env_ms for r in scanned]), "yolo_ms": mean([r.yolo_ms for r in results])},
    }


def report_md(s: dict) -> str:
    lines = ["# 感知离线对比：整图 OCR（现有）vs YOLO", "",
             f"共 {s['frames']} 帧，现有方案扫了其中 {s['scanned']} 帧（按 env.interval 模拟）。", "",
             "## 好友认出率", "",
             "“现有”和“YOLO（同帧）”只在现有方案扫描的那些帧上比；“YOLO（全部）”是所有帧里 YOLO 认出的次数。", "",
             "| 好友 | 现有 | YOLO（同帧） | YOLO（全部） |", "|---|---|---|---|"]
    lines += [f"| {n} | {v['env']} | {v['yolo_same_frames']} | {v['yolo_all']} |" for n, v in s["friends"].items()]
    if not s["friends"]:
        lines.append("| （两边都没认出好友） | | | |")
    diff = s["diff_frames"]
    lines += ["", f"两边不一致的帧：{len(diff)} 张（左右拼图在 `diff/`，左现有、右 YOLO，由人判断谁对）"]
    lines += [f"- {f}" for f in diff[:50]] + (["- ……"] if len(diff) > 50 else [])
    lines += ["", "## 互动请求", "", "延迟 = 现有方案比 YOLO 晚发现几秒（目标：YOLO ≤ 1 s）。", "",
              "| 谁:哪种 | 现有第一次 | YOLO 第一次 | 延迟 |", "|---|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    lines += [f"| {k} | {fmt(v['env'])} | {fmt(v['yolo'])} | {fmt(v['delay'])} |" for k, v in s["requests"].items()]
    if not s["requests"]:
        lines.append("| （没有互动请求） | | | |")
    lines += ["", "## 陌生人（stranger 事件）", "",
              "YOLO 按身体的规则（陌生人数 0 → 有、有 → 0）会发的事件；对照录像看有没有把好友误判成陌生人（目标：10 分钟 ≤ 1 次误报）。", ""]
    lines += [f"- {e['t']:.2f} s：{'来了 ' + str(e['n']) + ' 个' if e['kind'] == 'arrive' else '都走了'}" for e in s["stranger_events"]]
    if not s["stranger_events"]:
        lines.append("- （没有）")
    lines += ["", "## 走开（leave 事件）", "", "各自按 keep 模拟出的“走开了”（录像里人其实还在的就是误报）。", ""]
    for side, title in (("env", "现有"), ("yolo", "YOLO")):
        events = s["leave_events"][side]
        lines.append(f"- {title}：" + ("、".join(f"{e['name']} @ {e['t']:.2f} s" for e in events) or "（没有）"))
    far = s.get("far", {"players": 0, "named": 0, "rate": None})
    rate = "—" if far["rate"] is None else f"{far['rate']:.0%}"
    lines += ["", "## 远处的人（框高 < perception.far_height）", "",
              f"远处 player 共 {far['players']} 人次，认出是哪个好友的 {far['named']} 人次（{rate}；分母里也有陌生人，只用来对比）。", "",
              "对比二次检测：同一批录像跑两次，`--far-crops 0` 和默认各一次，看这里的比例差多少（目标：提升 ≥ 30 个百分点）；",
              "整帧耗时看 `perception bench --far-crops …` 的 p95（目标 ≤ 66 ms）。"]
    t = s["timing"]
    lines += ["", "## 耗时", "",
              f"- 现有：每次扫描平均 {fmt(t['env_ms'])} ms",
              f"- YOLO：每帧平均 {fmt(t['yolo_ms'])} ms（含检测、追踪、名字 OCR、圆圈匹配）", ""]
    return "\n".join(lines)


def _draw(frame: np.ndarray, items: list[dict], title: str) -> np.ndarray:
    out = frame.copy()
    for it in items:
        color = COLORS.get(it["kind"], (0, 0, 255))
        x, y, w, h = int(it["x"]), int(it["y"]), int(it["w"]), int(it["h"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = str(it.get("label", "")).encode("ascii", "replace").decode()  # cv2 画不了中文
        cv2.putText(out, f"{it['kind']} {label}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def side_by_side(frame: np.ndarray, env_items: list[dict], yolo_items: list[dict]) -> np.ndarray:
    """左：现有方案认出的框；右：YOLO 的框。"""
    return np.hstack([_draw(frame, env_items, "OCR (env)"), _draw(frame, yolo_items, "YOLO")])


def timed_files(files: Iterable[Path]) -> tuple[list[tuple[float, Path]], list[Path]]:
    """按文件名里的时间排好；没时间的挑出来。一张带时间的都没有 → ValueError（不是 record 录的，没法模拟定时扫描）。"""
    timed, skipped = [], []
    for f in files:
        t = frame_time(f)
        (skipped.append(f) if t is None else timed.append((t, Path(f))))
    if not timed:
        raise ValueError("文件名里没有时间：要用 python -m skydango record 录的目录（文件名形如 0012_006.00s.jpg）")
    return sorted(timed), skipped
