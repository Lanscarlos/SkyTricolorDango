"""难例收件箱（spec 2026-10-04-hardcase-inbox-design）：把各次运行存下的难例收进 datasets/inbox。

    inbox/<运行>/raw/*.jpg    运行目录 hard/ 里的难例原图
    inbox/<运行>/hard.jsonl   运行目录 hard.jsonl 的副本（原因、检测框）
    inbox/_index.jsonl        每行一条：收了哪个运行（run / collected_at / frames），之后的步骤往同一运行追加字段（processed_at 等）
"""

from __future__ import annotations

import json
import os
import shutil
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import attrs
from .bubbles import Rect

INDEX = "_index.jsonl"

PAIR = {"player": "lit", "player_unlit": "unlit", "spirit": "spirit"}  # YOLO 人物类别 ↔ 外形头的类
MERGE_FORM = {"shared": "lit", "morph": "lit"}  # 并进 lit 再比
# 人在外形页判的类 → 最终标注的类别名（not_person 不在里面 = 去掉）
FORM_CLS = {"lit": "player", "morph": "player", "shared": "player", "unlit": "player_unlit", "spirit": "spirit"}


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


@dataclass
class Route:
    auto: str | None  # "agree" / "drop_low" / None（给人判）
    form: str  # 外形头的类（已并过 shared / morph）
    p: float  # 这个类的把握


def route(yolo_cls: str, score: float, probs: dict[str, float], conf: float, agree: float) -> Route:
    """人物框分流（spec §4 的表）：高分框只会自动确认一致的（判"不是人"也给人看，篝火不能自动删）；
    低分框一致自动补成正式框、判"不是人"且有把握自动丢，其余给人判。"""
    merged: dict[str, float] = {}
    for form, v in probs.items():
        k = MERGE_FORM.get(form, form)
        merged[k] = merged.get(k, 0.0) + float(v)
    form = max(merged, key=merged.get) if merged else "not_person"
    p = merged.get(form, 0.0)
    auto = None
    if p >= agree:
        if form == PAIR.get(yolo_cls):
            auto = "agree"
        elif form == "not_person" and score < conf:
            auto = "drop_low"
    return Route(auto, form, p)


def crop_place(attrs_root: Path, crop: str) -> str | None:
    """裁图现在在 datasets/attrs 的哪里：_unlabeled / _discard / form/<类>；都不在返回 None。"""
    root = Path(attrs_root)
    if (root / "_unlabeled" / crop).is_file():
        return "_unlabeled"
    if (root / "_discard" / crop).is_file():
        return "_discard"
    for form in attrs.FORMS:
        if (root / "form" / form / crop).is_file():
            return form
    return None


def frame_state(entry: dict, place: Callable[[str], str | None]) -> str:
    """帧状态（现算，不存盘，spec §2.2）：done / discarded / crops / edit / glance，另有 dup / error。"""
    if entry.get("dup_of"):
        return "dup"
    if entry.get("error"):
        return "error"
    what = (entry.get("decision") or {}).get("what")
    if what == "pass":
        return "done"
    if what == "discard":
        return "discarded"
    places = [place(b["crop"]) for b in entry.get("boxes", []) if b.get("crop")]
    if "_unlabeled" in places:
        return "crops"
    if "_discard" in places or entry.get("editing"):
        return "edit"
    return "glance"


def final_boxes(entry: dict, place: Callable[[str], str | None], classes: list[str]) -> list[tuple[int, Rect]]:
    """这一帧通过时写进标注的框（类别编号, 框），顺序同 boxes：自动一致的用 YOLO 的框和类别；
    drop_low 去掉；有裁图的人物框按人判的类（not_person / 不要 = 去掉）；还没人判的保持 YOLO 类别；非人物框原样。"""
    out: list[tuple[int, Rect]] = []
    for b in entry.get("boxes", []):
        cls = b["cls"]
        if cls in PAIR and b.get("auto") != "agree":
            if b.get("auto") == "drop_low":
                continue
            where = place(b["crop"]) if b.get("crop") else None
            if where == "_discard":
                continue
            if where in attrs.FORMS:
                cls = FORM_CLS.get(where)
                if cls is None:
                    continue
        x, y, w, h = (int(v) for v in b["box"])
        out.append((classes.index(cls), Rect(x, y, w, h)))
    return out


def load_frames(inbox: Path, run: str) -> dict:
    """inbox/<运行>/frames.json：帧名 → 条目；没有就是空。"""
    path = Path(inbox) / run / "frames.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_frames(inbox: Path, run: str, frames: dict) -> None:
    path = Path(inbox) / run / "frames.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(frames, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def pass_frame(
    inbox: Path,
    run: str,
    frame: str,
    attrs_root: Path,
    dataset: Path,
    classes: list[str],
    boxes: list[tuple[int, Rect]] | None = None,
) -> str:
    """整帧通过：写最终标注，复制原图和标注进 dataset；返回 "<split>/<frame>"。
    目标已存在抛 FileExistsError（什么都不写）；复制到一半失败删掉已复制的再抛。"""
    from ..imageio import imread
    from .weaklabel import yolo_line

    inbox, dataset = Path(inbox), Path(dataset)
    frames = load_frames(inbox, run)
    entry = frames[frame]
    split = entry["split"]
    dst_img = dataset / "images" / split / f"{frame}.jpg"
    dst_lbl = dataset / "labels" / split / f"{frame}.txt"
    if dst_img.exists() or dst_lbl.exists():
        raise FileExistsError(f"{split}/{frame} 已在数据集里")
    edited = boxes is not None
    if boxes is None:
        boxes = final_boxes(entry, lambda c: crop_place(attrs_root, c), classes)
    src_img = inbox / run / entry["file"]
    height, width = imread(src_img).shape[:2]
    text = "".join(yolo_line(c, b, width, height) + "\n" for c, b in boxes)
    label = inbox / run / "labels" / f"{frame}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(text, encoding="utf-8")
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(src_img, dst_img)
        shutil.copyfile(label, dst_lbl)
    except BaseException:
        dst_img.unlink(missing_ok=True)
        dst_lbl.unlink(missing_ok=True)
        raise
    entry["decision"] = {"what": "pass", "edited": edited, "t": time.time(), "dataset": f"{split}/{frame}"}
    entry["editing"] = False
    save_frames(inbox, run, frames)
    return f"{split}/{frame}"


def discard_frame(inbox: Path, run: str, frame: str) -> None:
    frames = load_frames(inbox, run)
    frames[frame]["decision"] = {"what": "discard", "t": time.time()}
    save_frames(inbox, run, frames)


def undo_frame(inbox: Path, run: str, frame: str, dataset: Path) -> None:
    """撤销决定；通过的要把 datasets/sky 里那两个文件删掉。"""
    frames = load_frames(inbox, run)
    entry = frames[frame]
    decision = entry.get("decision") or {}
    if decision.get("what") == "pass":
        split = decision["dataset"].split("/")[0]
        (Path(dataset) / "images" / split / f"{frame}.jpg").unlink(missing_ok=True)
        (Path(dataset) / "labels" / split / f"{frame}.txt").unlink(missing_ok=True)
    entry["decision"] = None
    save_frames(inbox, run, frames)


def set_editing(inbox: Path, run: str, frame: str, on: bool) -> None:
    frames = load_frames(inbox, run)
    frames[frame]["editing"] = bool(on)
    save_frames(inbox, run, frames)
