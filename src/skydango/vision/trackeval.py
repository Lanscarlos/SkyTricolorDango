"""离线评估追踪和接回（`perception track-eval`，spec 2026-10-01-tracking-relink-motion §6）。

同一批录像跑两遍：基线（追踪升级的开关全关）和当前配置，按录像时间抽帧模拟身体截图的帧率，每个好友统计：
- 轨迹断了几次、为什么断（断前那条轨迹第一次没接上的那一帧看：预测位置附近有低分框 / 有高分框但没过线 / 画面平移比框宽还大 / 什么都没有）
- 假的"走开"（从 nearby 掉出去、keep 秒内又回来）
- 确认冤枉（判成陌生人的轨迹后来挂上了他的名字标签）
- 接回（后来标签证实对 / 错 / 一直没证实）
- 运动方向时间线
录像里没有镜头事件，只靠平移估计。
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterable, Iterator

from ..config import PerceptionConfig
from .track import Track

SWITCHES = ("sticky_names", "track_low", "track_predict", "track_pan", "relink", "motion")
CAUSES = ("低分框", "位移", "画面平移", "漏检")
VERDICTS = ("对", "错", "未证实")
TIMELINE_MAX = 60  # 报告里最多列几条轨迹的运动方向


def baseline(cfg: PerceptionConfig) -> PerceptionConfig:
    """基线：追踪升级的开关全关（= 升级之前的行为），其余照抄。"""
    return dataclasses.replace(cfg, **{k: False for k in SWITCHES})


def subsample(items: Iterable[tuple], fps: float) -> Iterator[tuple]:
    """按录像时间抽帧（item[0] 是秒）：离上一张留下的至少 1 / fps 秒才留。"""
    gap = 1.0 / max(fps, 0.01)
    last = float("-inf")
    for item in items:
        if item[0] - last >= gap - 1e-6:
            last = item[0]
            yield item


def _who(track: Track) -> str | None:
    """这条轨迹算谁：挂过标签的名字，或者按位置接回的"像他"。"""
    d = track.data
    if d.get("tagged") and d.get("name"):
        return d["name"]
    if d.get("maybe_by") == "relink":
        return d.get("maybe")
    return None


def _cause(track: Track, watcher, now: float) -> str:
    """轨迹第一次没接上的那一帧：为什么。"""
    pred = watcher.tracker.predicted(track, now)[0]
    cx, cy = pred.x + pred.w / 2, pred.y + pred.h / 2

    def near(dets, k: float) -> bool:
        return any(
            d.cls in ("player", "player_unlit")
            and abs(d.box.x + d.box.w / 2 - cx) <= k * pred.h and abs(d.box.y + d.box.h / 2 - cy) <= k * pred.h
            for d in dets
        )

    if near(watcher.last_low, 1.0):
        return "低分框"
    if near(watcher.last_dets, 2.0):
        return "位移"
    shift = watcher.last_shift
    if shift is not None and abs(shift[0]) > track.box.w:
        return "画面平移"
    return "漏检"


def _friend() -> dict:
    return {"breaks": Counter(), "false_leaves": 0, "wronged": 0, "relinks": dict.fromkeys(VERDICTS, 0)}


def evaluate(items: Iterable[tuple], watcher, keep: float | None = None) -> dict:
    """items：(时间, 文件名, 帧, 聊天面板开没开)，按时间排好。watcher 是 background=False 的 PerceptionWatcher。"""
    keep = watcher.cfg.keep if keep is None else keep
    now = [0.0]
    watcher.clock = lambda: now[0]  # 录像时间当时钟
    friends: dict[str, dict] = {}
    miss: dict[int, str] = {}  # 有名字的轨迹 id → 第一次没接上时判的原因
    strangers: set[int] = set()  # 判过陌生人的轨迹
    wronged: set[int] = set()
    relinked: dict[int, str] = {}  # 接回过的轨迹 id → 接成谁（还没被标签证实）
    left: dict[str, float] = {}
    near_prev: set[str] = set()
    motions: dict[int, list] = {}  # 轨迹 id → [谁, [(开始, 结束, 方向)]]
    frames = 0
    for t, _name, frame, panel in items:
        frames += 1
        now[0] = t
        watcher.process(frame, t, panel)
        for track in watcher.tracker.dropped:
            who = _who(track)
            if who:
                friends.setdefault(who, _friend())["breaks"][miss.pop(track.id, "漏检")] += 1
        for track in watcher.tracker.tracks.values():
            if track.cls != "player" or not _who(track):
                continue
            if track.last == t:
                miss.pop(track.id, None)
            elif track.id not in miss:
                miss[track.id] = _cause(track, watcher, t)
        for track in watcher.last_tracks:
            if track.cls not in ("player", "player_unlit"):
                continue
            d = track.data
            name = d.get("name")
            if d.get("stranger"):
                strangers.add(track.id)
            if name and track.id in strangers and track.id not in wronged:
                wronged.add(track.id)
                friends.setdefault(name, _friend())["wronged"] += 1
            if d.get("maybe_by") == "relink" and d.get("maybe"):
                relinked[track.id] = d["maybe"]
            elif name and track.id in relinked:
                was = relinked.pop(track.id)
                friends.setdefault(was, _friend())["relinks"]["对" if name == was else "错"] += 1
            who = name or d.get("maybe")
            motion = d.get("motion")
            if who and motion:
                entry = motions.setdefault(track.id, [who, []])
                entry[0] = who
                segs = entry[1]
                if segs and segs[-1][2] == motion and t - segs[-1][1] <= 1.0:
                    segs[-1] = (segs[-1][0], t, motion)
                else:
                    segs.append((t, t, motion))
        near = set(watcher.nearby(t))
        for name in near:
            friends.setdefault(name, _friend())  # 出现过的好友都列出来（没出事的也是结果）
        for name in near_prev - near:
            left[name] = t
        for name in near - near_prev:
            if name in left and t - left.pop(name) <= keep:
                friends.setdefault(name, _friend())["false_leaves"] += 1
        near_prev = near
    for was in relinked.values():
        friends.setdefault(was, _friend())["relinks"]["未证实"] += 1
    for f in friends.values():
        f["breaks"] = dict(f["breaks"])
    timeline = [
        f"{who} #{tid} " + " → ".join(f"{a:.1f}~{b:.1f} {m}" for a, b, m in segs)
        for tid, (who, segs) in sorted(motions.items())
    ]
    return {"frames": frames, "friends": friends, "timeline": timeline}


def _cell(base: dict | None, cur: dict | None, get) -> str:
    fmt = lambda v: "—" if v is None else str(v)  # noqa: E731
    return f"{fmt(get(base) if base else None)} → {fmt(get(cur) if cur else None)}"


def report_md(base: dict, cur: dict, meta: dict) -> str:
    """基线和当前配置并排（每格"基线 → 当前配置"）。"""
    lines = [
        "# 追踪和接回：离线评估", "",
        f"录像：{meta.get('source', '?')}，{meta.get('frames', '?')} 帧，按 {meta.get('fps', '?')} 帧 / 秒抽（模拟身体截图）。",
        "每格是 **基线 → 当前配置**（基线 = 追踪升级的开关全关）。目标：接回证实错 = 0，确认冤枉明显下降。", "",
        "| 好友 | 断开：" + " / ".join(CAUSES) + " | 假走开 | 确认冤枉 | 接回 对 / 错 / 未证实 |",
        "|---|---|---|---|---|",
    ]
    names = sorted(set(base["friends"]) | set(cur["friends"]))
    for name in names:
        b, c = base["friends"].get(name), cur["friends"].get(name)
        breaks = " / ".join(_cell(b, c, lambda f, k=k: f["breaks"].get(k, 0)) for k in CAUSES)
        relinks = " / ".join(_cell(b, c, lambda f, k=k: f["relinks"][k]) for k in VERDICTS)
        lines.append(f"| {name} | {breaks} | {_cell(b, c, lambda f: f['false_leaves'])} "
                     f"| {_cell(b, c, lambda f: f['wronged'])} | {relinks} |")
    if not names:
        lines.append("| （两边都没认出好友） | | | | |")
    lines += ["", "## 运动方向（当前配置）", ""]
    tl = cur["timeline"]
    lines += [f"- {row}" for row in tl[:TIMELINE_MAX]] + (["- ……"] if len(tl) > TIMELINE_MAX else [])
    if not tl:
        lines.append("- （没有）")
    lines += ["", f"基线 {base['frames']} 帧、当前配置 {cur['frames']} 帧。", ""]
    return "\n".join(lines)
