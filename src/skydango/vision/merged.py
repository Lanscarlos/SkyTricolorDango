"""训练用的合并视图（spec 2026-11-04-yolo-boxes-only-design §2）：把 datasets/sky 里三类人合成一类。

datasets/sky 的类别：0 player、4 player_unlit、9 spirit 都是"人"。单类模型只学"这是个人"，
点没点火 / 先祖交给外形头。这里不动 datasets/sky：另建一个目录，图片硬链接（失败就复制）、
标签逐行把第一列按 `MERGE_IDS` 改写，写一份新的 data.yaml（类别号不变：0 person、4 / 9 占位）。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

MERGE_IDS = {4: 0, 9: 0}  # player_unlit、spirit → person
MERGED_NAMES = ["person", "name_tag", "social_ring", "self", "unused_4", "typing", "bench", "bonfire", "instrument", "unused_9"]
SPLITS = ("train", "val")


def _merge_line(line: str) -> str:
    """一行标签 "<类别> cx cy w h"：类别在 MERGE_IDS 里就改；空行 / 非数字开头的行原样。"""
    parts = line.split(None, 1)
    if not parts or not parts[0].lstrip("-").isdigit():
        return line
    new = MERGE_IDS.get(int(parts[0]))
    if new is None:
        return line
    return str(new) + line[len(parts[0]):]


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:  # 跨盘、文件系统不支持硬链接
        shutil.copy2(src, dst)


def build(dataset: Path, out: Path) -> Path:
    """在 out 建合并视图，返回 out / "data.yaml"。out 已存在先整个删掉（重训重跑不能留上次的标签）。"""
    dataset, out = Path(dataset), Path(out)
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        for img in sorted((dataset / "images" / split).glob("*")) if (dataset / "images" / split).is_dir() else []:
            if img.is_file():
                _link(img, out / "images" / split / img.name)
        labels = dataset / "labels" / split
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for txt in sorted(labels.glob("*.txt")) if labels.is_dir() else []:
            text = txt.read_text(encoding="utf-8")
            lines = [_merge_line(ln) for ln in text.split("\n")]
            (out / "labels" / split / txt.name).write_text("\n".join(lines), encoding="utf-8", newline="")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(MERGED_NAMES))
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return yaml
