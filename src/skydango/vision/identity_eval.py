"""身份板的标准答案和评估（spec 2026-10-22-identity-board-design.md §7）。

`perception identity-truth`：用旧流程（身份板关，只有追踪和读名字）回放录像，每条轨迹记下框、分到的名字、self 命中数，
起草"这条轨迹是谁"（`truth-draft.toml`）+ 拼图（`sheet.jpg`）给人核对，核对完 `--apply` 展开成 `truth.jsonl`。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..imageio import imread, imwrite
from .bubbles import Rect

RESERVED = ("团子", "陌生人", "不要", "?")  # 草稿里除好友名外还能写的值
NAME_GAP = 1.0  # 同一个名字的两次读到隔得比这久就算断开（秒）


@dataclass
class TrackLog:
    """一条轨迹在回放里的记录。"""

    id: int
    first: float
    last: float
    frames: list[tuple[str, Rect]] = field(default_factory=list)  # (帧文件名, 框)
    names: list[tuple[float, str]] = field(default_factory=list)  # (时间, 这一帧分到的读出名字)
    self_hits: int = 0  # 被当成团子的帧数（self 类别或团子标记）

    def to_json(self) -> dict:
        return {"id": self.id, "first": self.first, "last": self.last,
                "frames": [[f, [r.x, r.y, r.w, r.h]] for f, r in self.frames],
                "names": [[t, n] for t, n in self.names], "self_hits": self.self_hits}

    @classmethod
    def from_json(cls, d: dict) -> "TrackLog":
        return cls(d["id"], d["first"], d["last"], [(f, Rect(*b)) for f, b in d["frames"]],
                   [(t, n) for t, n in d["names"]], d["self_hits"])


def save_logs(path: Path, logs: dict[int, TrackLog]) -> None:
    path.write_text(json.dumps([lg.to_json() for lg in logs.values()], ensure_ascii=False) + "\n", encoding="utf-8")


def load_logs(path: Path) -> dict[int, TrackLog]:
    logs = [TrackLog.from_json(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    return {lg.id: lg for lg in logs}


def collect_logs(items: Iterable[tuple], watcher) -> dict[int, TrackLog]:
    """回放：items 是 (时间, 文件名, 帧, 聊天面板开没开)，watcher 是 background=False 的 PerceptionWatcher（身份板关）。"""
    now = [0.0]
    watcher.clock = lambda: now[0]  # 录像时间当时钟
    logs: dict[int, TrackLog] = {}
    for t, name, frame, panel in items:
        now[0] = t
        watcher.process(frame, t, panel)
        for track in watcher.last_tracks:
            if track.cls not in ("player", "player_unlit", "self"):
                continue
            lg = logs.get(track.id)
            if lg is None:
                lg = logs[track.id] = TrackLog(track.id, t, t)
            lg.last = t
            lg.frames.append((name, track.box))
            d = track.data
            if d.get("name") and d.get("tagged", True):
                lg.names.append((t, d["name"]))
            if track.cls == "self" or d.get("dango"):
                lg.self_hits += 1
    return logs


def draft_identities(tracks_seen: dict[int, TrackLog], tag_hold: float) -> dict[int, str]:
    """起草每条轨迹是谁：连续挂着 ≥ tag_hold 秒的名字里次数最多的；没有名字但 self 命中过半 = 团子；否则 ?。"""
    out: dict[int, str] = {}
    for tid, lg in tracks_seen.items():
        votes: Counter[str] = Counter()
        run: list[tuple[float, str]] = []

        def flush() -> None:
            if run and run[-1][0] - run[0][0] >= tag_hold:
                votes[run[0][1]] += len(run)

        for t, name in sorted(lg.names):
            if run and (name != run[0][1] or t - run[-1][0] > NAME_GAP):
                flush()
                run = []
            run.append((t, name))
        flush()
        if votes:
            out[tid] = votes.most_common(1)[0][0]
        elif lg.frames and lg.self_hits >= len(lg.frames) / 2:
            out[tid] = "团子"
        else:
            out[tid] = "?"
    return out


def write_draft(path: Path, ids: dict[int, str], logs: dict[int, TrackLog]) -> None:
    lines = [
        "# 每行 t<轨迹编号> = \"身份\"，对着 sheet.jpg 改。取值：好友名 / 团子 / 陌生人 / 不要 / ?（? 和 不要 不写进标准答案）",
    ]
    for tid, who in ids.items():
        lg = logs[tid]
        lines.append(f't{tid} = "{who}"  # 出现 {lg.first:.1f}~{lg.last:.1f} s，{len(lg.frames)} 帧')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sheet(path: Path, logs: dict[int, TrackLog], frames_dir: Path, per_track: int = 4) -> None:
    """每条轨迹一行：编号 + 均匀挑 per_track 张裁图。"""
    cell_h, cell_w, label_w, rows_per_col = 160, 100, 90, 12
    row_w = label_w + cell_w * per_track
    cells = []
    for lg in logs.values():
        row = np.zeros((cell_h, row_w, 3), np.uint8)
        cv2.putText(row, f"t{lg.id}", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(row, f"{len(lg.frames)}f", (6, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        n = len(lg.frames)
        picks = sorted({round(i * (n - 1) / max(per_track - 1, 1)) for i in range(per_track)}) if n else []
        for k, idx in enumerate(picks):
            name, box = lg.frames[idx]
            try:
                img = imread(frames_dir / name)
            except Exception:
                continue
            crop = box.pad(0, img.shape[1], img.shape[0]).crop(img)
            if crop.size == 0:
                continue
            s = min(cell_w / crop.shape[1], cell_h / crop.shape[0])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
            x = label_w + k * cell_w
            row[: crop.shape[0], x : x + crop.shape[1]] = crop
        cv2.rectangle(row, (0, 0), (row_w - 1, cell_h - 1), (60, 60, 60), 1)
        cells.append(row)
    if not cells:
        cells.append(np.zeros((cell_h, row_w, 3), np.uint8))
    cols = []
    for i in range(0, len(cells), rows_per_col):
        chunk = cells[i : i + rows_per_col]
        chunk += [np.zeros_like(cells[0])] * (rows_per_col - len(chunk))
        cols.append(np.vstack(chunk))
    imwrite(path, np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 85])


_LINE = re.compile(r'^\s*t(\d+)\s*=\s*"([^"]*)"')


def apply_draft(draft: Path, logs_path: Path, out: Path, friends: list[str]) -> int:
    """读改好的草稿，按 logs.json 展开成 truth.jsonl；返回写了几行。取值不合法抛 ValueError（带行号）。"""
    logs = load_logs(logs_path)
    ok = set(friends) | set(RESERVED)
    rows: list[str] = []
    for no, line in enumerate(draft.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"草稿第 {no} 行看不懂：{line.strip()}")
        tid, who = int(m.group(1)), m.group(2)
        if who not in ok:
            raise ValueError(f"草稿第 {no} 行：「{who}」不是好友名，也不是 团子 / 陌生人 / 不要 / ?")
        if tid not in logs:
            raise ValueError(f"草稿第 {no} 行：logs.json 里没有轨迹 t{tid}")
        if who in ("不要", "?"):
            continue
        for frame, box in logs[tid].frames:
            rows.append(json.dumps({"frame": frame, "box": [box.x, box.y, box.w, box.h], "who": who}, ensure_ascii=False))
    out.write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    return len(rows)
