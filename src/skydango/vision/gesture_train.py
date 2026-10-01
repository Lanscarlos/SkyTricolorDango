"""动作识别的训练数据：读确认过的片段、按录像切训练 / 验证集。

这一部分不依赖 torch（训练在后面加，torch 延迟导入）。
片段放在 `<数据集>/<标签>/<片段名>/00.jpg…`，片段名见 `gesture.clip_name`。
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import GestureConfig
from .gesture import SUFFIXES, recording_of

MIN_PER_CLASS = 20
SPLIT_FILE = "_split.json"
_START = re.compile(r"_t(\d+(?:\.\d+)?)s$")


@dataclass(frozen=True)
class Sample:
    clip: str
    label: str
    recording: str
    start: float
    path: Path


def _start_of(clip: str) -> float:
    m = _START.search(clip)
    return float(m.group(1)) if m else 0.0


def list_samples(root: Path, labels: list[str], frames: int = 16) -> tuple[list[Sample], list[str]]:
    """只看 labels 里的类别目录。帧数不等于 frames 的片段不算样本，名字放进第二个返回值。"""
    samples: list[Sample] = []
    skipped: list[str] = []
    for label in labels:
        folder = Path(root) / label
        if not folder.is_dir():
            continue
        for clip in sorted(p for p in folder.iterdir() if p.is_dir()):
            n = sum(1 for f in clip.iterdir() if f.suffix.lower() in SUFFIXES)
            if n != frames:
                skipped.append(clip.name)
                continue
            samples.append(Sample(clip.name, label, recording_of(clip.name), _start_of(clip.name), clip))
    return samples, skipped


def check_counts(samples: list[Sample], labels: list[str], names: dict[str, str] | None = None) -> list[str]:
    """每类至少 MIN_PER_CLASS 段；不够的给提示。"""
    names = {**GestureConfig().names, "none": "都不是", **(names or {})}
    count = {label: 0 for label in labels}
    for s in samples:
        if s.label in count:
            count[s.label] += 1
    return [
        f"{names.get(label, label)}只有 {n} 段，还差 {MIN_PER_CLASS - n} 段"
        for label, n in count.items() if n < MIN_PER_CLASS
    ]


def split(
    samples: list[Sample], labels: list[str], val_ratio: float = 0.2, gap: float = 2.0, seed: int = 0,
) -> dict[str, list]:
    """切训练 / 验证集：同一段录像整个进一边（防止相邻帧泄漏），同样的 seed 结果一样。

    - 每个有样本的类别：有 ≥ 2 段录像就至少有一段录像进验证集（且训练集里也留着一段）；
      其余录像随机补到验证集样本数约占 val_ratio。
    - 类别只有一段录像：该录像里这个类别的片段按 start 切——设 n 段按 start 排序，
      c = 第 ceil((1 − val_ratio)·n) 段（至多最后一段）的 start；train = start < c − gap，val = start ≥ c，
      中间 gap 秒内的不用。这段录像里别的类别的片段照常进训练集（它不进整录像验证候选）。
    - 进不了验证（或训练）集的类别写进 warnings。
    返回 {"train": [片段名], "val": [片段名], "warnings": [...], "seed": seed}。
    """
    rng = random.Random(seed)
    samples = [s for s in samples if s.label in labels]
    key = lambda s: s.recording or s.clip  # 老格式没有录像名：每段自成一组
    by_rec: dict[str, list[Sample]] = {}
    for s in samples:
        by_rec.setdefault(key(s), []).append(s)
    recs_of: dict[str, set[str]] = {}
    for s in samples:
        recs_of.setdefault(s.label, set()).add(key(s))
    pinned = {next(iter(r)) for r in recs_of.values() if len(r) == 1}  # 某类唯一的录像：只按时间切
    order = sorted(by_rec)
    rng.shuffle(order)
    val_recs: set[str] = set()

    def train_ok(label: str, extra: str) -> bool:
        return bool(recs_of[label] - val_recs - {extra})

    # 覆盖：录像少的类别先挑
    for label in sorted(recs_of, key=lambda lb: (len(recs_of[lb]), lb)):
        if len(recs_of[label]) < 2 or recs_of[label] & val_recs:
            continue
        for r in order:
            if r in recs_of[label] and r not in pinned and all(
                    train_ok(lb, r) for lb in {s.label for s in by_rec[r]} if len(recs_of[lb]) >= 2):
                val_recs.add(r)
                break
    # 补到比例
    target = val_ratio * len(samples)
    got = sum(len(by_rec[r]) for r in val_recs)
    for r in order:
        if got >= target:
            break
        if r in val_recs or r in pinned:
            continue
        if all(train_ok(lb, r) for lb in {s.label for s in by_rec[r]} if len(recs_of[lb]) >= 2):
            val_recs.add(r)
            got += len(by_rec[r])

    train: list[str] = []
    val: list[str] = []
    cut: dict[str, float] = {}  # 只有一段录像的类别 → 切点 c
    for label, recs in recs_of.items():
        if len(recs) == 1:
            starts = sorted(s.start for s in samples if s.label == label)
            if len(starts) >= 2:
                cut[label] = starts[min(math.ceil((1 - val_ratio) * len(starts)), len(starts) - 1)]
    for s in samples:
        if key(s) in val_recs:
            val.append(s.clip)
        elif s.label in cut:
            if s.start < cut[s.label] - gap:
                train.append(s.clip)
            elif s.start >= cut[s.label]:
                val.append(s.clip)
        else:
            train.append(s.clip)
    in_val = set(val)
    in_train = set(train)
    warnings = []
    for label in labels:
        mine = [s.clip for s in samples if s.label == label]
        if not mine:
            continue
        if not in_val.intersection(mine):
            warnings.append(f"{label} 进不了验证集（录像或片段太少），评估时看不到它")
        if not in_train.intersection(mine):
            warnings.append(f"{label} 进不了训练集")
    return {"train": train, "val": val, "warnings": warnings, "seed": seed}


def save_split(root: Path, result: dict, seed: int | None = None) -> None:
    data = dict(result)
    if seed is not None:
        data["seed"] = seed
    (Path(root) / SPLIT_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_split(root: Path) -> dict | None:
    path = Path(root) / SPLIT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
