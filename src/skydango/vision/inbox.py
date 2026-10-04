"""难例收件箱（spec 2026-10-04-hardcase-inbox-design）：把各次运行存下的难例收进 datasets/inbox。

    inbox/<运行>/raw/*.jpg    运行目录 hard/ 里的难例原图
    inbox/<运行>/hard.jsonl   运行目录 hard.jsonl 的副本（原因、检测框）
    inbox/_index.jsonl        每行一条：收了哪个运行（run / collected_at / frames），之后的步骤往同一运行追加字段（processed_at 等）
"""

from __future__ import annotations

import json
import shutil
import time
import zlib
from pathlib import Path

INDEX = "_index.jsonl"


def split_of(run: str, every: int) -> str:
    """整次运行落在同一边：crc32(运行目录名) % every == 0 进验证集。"""
    return "val" if zlib.crc32(run.encode()) % every == 0 else "train"


def frame_name(run: str, file: str) -> str:
    """收件箱帧名 = <运行目录名>_<原文件名去后缀>（同 weaklabel.hard_images）。"""
    return f"{run}_{Path(file).stem}"


def _index_rows(inbox: Path) -> list[dict]:
    path = inbox / INDEX
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("run"):
            rows.append(row)
    return rows


def collect(run_dir: Path, inbox: Path) -> int:
    """把 run_dir/hard/*.jpg 和 hard.jsonl 复制进 inbox/<运行>/；已有的跳过。返回新复制的图片张数。"""
    run_dir, inbox = Path(run_dir), Path(inbox)
    hard = run_dir / "hard"
    images = sorted(hard.glob("*.jpg")) if hard.is_dir() else []
    if not images:
        return 0
    run = run_dir.name
    raw = inbox / run / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in images:
        dst = raw / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    meta = run_dir / "hard.jsonl"
    if meta.is_file() and not (inbox / run / "hard.jsonl").exists():
        shutil.copy2(meta, inbox / run / "hard.jsonl")
    if not any(r["run"] == run for r in _index_rows(inbox)):
        row = {"run": run, "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "frames": len(images)}
        with (inbox / INDEX).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return copied


def collect_all(runs: Path, inbox: Path) -> list[str]:
    """对 runs/*/ 逐个收，返回有新复制的运行名。"""
    runs = Path(runs)
    out = []
    if not runs.is_dir():
        return out
    for d in sorted(runs.iterdir()):
        if d.is_dir() and collect(d, inbox):
            out.append(d.name)
    return out


def pending_runs(inbox: Path) -> list[str]:
    """收了、还没 processed_at 的运行（同一运行多行，后面的行覆盖前面的字段）。"""
    merged: dict[str, dict] = {}
    for row in _index_rows(Path(inbox)):
        merged.setdefault(row["run"], {}).update(row)
    return [run for run, row in merged.items() if not row.get("processed_at")]
