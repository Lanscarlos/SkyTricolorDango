"""第二层（外形头）的数据工具（`perception crops`）：从 YOLO 数据集和录像 / 难例目录裁人物图，确认后写回数据集。

输出目录（默认 datasets/attrs）：
  form/<外形>/       已知类别的框直接放这里（来自有标注的数据集框）
  _unlabeled/        检测器认出、但数据集里没标的人物框（等 Claude 初分 + 标注页确认）
  _crops.jsonl       每张裁图一行：crop / image / box / score / yolo_cls / source / split / group / known / size
重跑幂等：_crops.jsonl 里已有的裁图名不再生成（裁图被挪走了也不会又造一份）。
写回（`writeback`）：来自数据集、本来没标的框，被标注页挪进 form/<人形类>/ 就算确认，补进 labels/。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from ..imageio import imread, imwrite
from . import attrs
from .augment import IMAGE_EXTS, SUFFIXES
from .bubbles import Rect
from .track import iou

KNOWN = {0: "lit", 4: "unlit", 9: "spirit"}  # datasets/sky 类别编号 → 外形
PERSON_IDS = (0, 3, 4, 9)  # 数据集里算"人物"的类别：player / self / player_unlit / spirit
ID_NAMES = {0: "player", 3: "self", 4: "player_unlit", 9: "spirit"}
WRITEBACK_ID = {"lit": 0, "morph": 0, "shared": 0, "unlit": 4, "spirit": 9}
DETECT_CLASSES = ("player", "player_unlit")
UNLABELED_IOU = 0.4  # 检测框和任何人物标注的 IoU 低于这个才算"没标过"
DEDUP_IOU = 0.5  # 同一张图里重叠这么多的只留分高的；写回时同帧已有人物框重叠这么多就不写

log = logging.getLogger(__name__)
_FRAME = re.compile(r"^(.*)_\d+_[\d.]+s$")


def crop_name(frame_stem: str, box: Rect) -> str:
    return f"{frame_stem}__{box.x}_{box.y}_{box.w}_{box.h}.jpg"


def frame_group(stem: str) -> str:
    """数据集帧名 <录像>_<NNNN>_<T.TT>s 的录像前缀；旧格式 <NNNN>_<T.TT>s 没有前缀 → ""。"""
    m = _FRAME.match(stem)
    return m.group(1) if m else ""


def _folder_group(folder: Path) -> str:
    folder = Path(folder)
    if folder.name == "hard":  # runs/<运行>/hard
        return folder.resolve().parent.name
    return folder.resolve().name


def _images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []


def _read_rows(out: Path) -> list[dict]:
    path = out / "_crops.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _Writer:
    """裁图 + 追加 _crops.jsonl；已有的裁图名跳过。"""

    def __init__(self, out: Path, size: int, pad: float, keys: tuple[str, ...]):
        self.out, self.size, self.pad = Path(out), size, pad
        self.out.mkdir(parents=True, exist_ok=True)
        self.done = {r["crop"] for r in _read_rows(self.out)}
        self.counts: dict[str, int] = dict.fromkeys(keys, 0)
        self._fh = open(self.out / "_crops.jsonl", "a", encoding="utf-8")

    def add(self, img, stem: str, box: Rect, folder: str, row: dict) -> None:
        name = crop_name(stem, box)
        if name in self.done:
            return
        self.done.add(name)
        imwrite(self.out / folder / name, attrs.crop(img, box, self.pad, self.size))
        h, w = img.shape[:2]
        row = {"crop": name, **row, "box": [box.x, box.y, box.w, box.h], "size": [w, h]}
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        key = folder.split("/")[-1]
        self.counts[key] = self.counts.get(key, 0) + 1

    def close(self) -> dict:
        self._fh.close()
        return self.counts


def _label_boxes(path: Path, w: int, h: int) -> list[tuple[int, Rect]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1, y1 = max(0, round((cx - bw / 2) * w)), max(0, round((cy - bh / 2) * h))
        x2, y2 = min(w, round((cx + bw / 2) * w)), min(h, round((cy + bh / 2) * h))
        if x2 > x1 and y2 > y1:
            boxes.append((cls, Rect(x1, y1, x2 - x1, y2 - y1)))
    return boxes


def _people(detector, img, conf: float):
    return [d for d in detector.detect(img) if d.cls in DETECT_CLASSES and d.score >= conf]


def crops_from_dataset(root: Path, detector, out: Path, conf: float, size: int = 224, pad: float = attrs.CROP_PAD) -> dict:
    """已知类别的标注框 → out/form/<外形>/；检测器认出但标注里没有的人物 → out/_unlabeled/。返回各目录新增张数。"""
    root = Path(root)
    w = _Writer(out, size, pad, (*KNOWN.values(), "_unlabeled"))
    try:
        for split in ("train", "val"):
            for path in _images_in(root / "images" / split):
                if path.stem.endswith(SUFFIXES):  # 增强图
                    continue
                img = imread(path)
                h, wd = img.shape[:2]
                labeled = _label_boxes(root / "labels" / split / f"{path.stem}.txt", wd, h)
                base = {"image": path.resolve().as_posix(), "dataset": root.resolve().as_posix(), "split": split, "group": frame_group(path.stem), "source": "dataset"}
                for cls, box in labeled:
                    if cls in KNOWN:
                        w.add(img, path.stem, box, f"form/{KNOWN[cls]}",
                              {**base, "score": 1.0, "yolo_cls": ID_NAMES[cls], "known": True})
                people = [b for c, b in labeled if c in PERSON_IDS]
                for d in _people(detector, img, conf):
                    if all(iou(d.box, b) < UNLABELED_IOU for b in people):
                        w.add(img, path.stem, d.box, "_unlabeled",
                              {**base, "score": round(float(d.score), 4), "yolo_cls": d.cls, "known": False})
    finally:
        counts = w.close()
    return counts


def crops_from_images(folder: Path, detector, out: Path, conf: float, size: int = 224, pad: float = attrs.CROP_PAD) -> dict:
    """录像 / 难例目录：人物框全进 out/_unlabeled/；同一张图里重叠 ≥ 0.5 的只留分高的。"""
    folder = Path(folder)
    group = _folder_group(folder)
    w = _Writer(out, size, pad, ("_unlabeled",))
    try:
        for path in _images_in(folder):
            img = imread(path)
            kept = []
            for d in sorted(_people(detector, img, conf), key=lambda d: -d.score):
                if all(iou(d.box, k.box) < DEDUP_IOU for k in kept):
                    kept.append(d)
            for d in kept:
                w.add(img, f"{group}_{path.stem}", d.box, "_unlabeled",  # 不同目录可能有同名帧，裁图名带上目录名
                      {"image": path.resolve().as_posix(), "score": round(float(d.score), 4), "yolo_cls": d.cls,
                       "source": "images", "split": None, "group": group, "known": False})
    finally:
        counts = w.close()
    return counts


def writeback(dataset: Path, out: Path, now: datetime) -> dict:
    """把标注页确认过的人物写回 labels/：来自数据集（source == dataset）、本来没标（known == false）、
    现在躺在 out/form/<人形类>/ 里的裁图。写之前整个复制 labels/ 到 _backup/labels-<时间>。"""
    dataset, out = Path(dataset), Path(out)
    todo: dict[tuple[str, str], list[tuple[int, Rect, list[int] | None]]] = {}
    here = dataset.resolve().as_posix()
    old_rows = 0
    for r in _read_rows(out):
        if r.get("source") != "dataset" or r.get("known"):
            continue
        if "dataset" not in r:
            old_rows += 1
            continue
        if Path(r["dataset"]).resolve().as_posix() != here:  # 别的数据集的裁图，共用同一个 out
            continue
        form = next((f for f in attrs.PERSON_FORMS if (out / "form" / f / r["crop"]).is_file()), None)
        if form is None:
            continue
        x, y, bw, bh = r["box"]
        todo.setdefault((r["split"], Path(r["image"]).stem), []).append((WRITEBACK_ID[form], Rect(x, y, bw, bh), r.get("size")))
    if old_rows:
        log.warning("%d 条旧的 _crops.jsonl 记录没有 dataset 字段，写回时跳过", old_rows)
    if not todo:
        return {"frames": 0, "boxes": 0}
    labels = dataset / "labels"
    backup = dataset / "_backup" / f"labels-{now.strftime('%Y%m%d-%H%M%S')}"
    shutil.copytree(labels, backup)
    frames = boxes = 0
    for (split, stem), items in sorted(todo.items()):
        path = labels / split / f"{stem}.txt"
        size = items[0][2]
        if size is None:
            h, w = imread(dataset / "images" / split / f"{stem}.jpg").shape[:2]
        else:
            w, h = size
        existing = [b for c, b in _label_boxes(path, w, h) if c in PERSON_IDS]
        lines = []
        for cls, box, _ in items:
            if any(iou(box, b) >= DEDUP_IOU for b in existing):
                continue
            existing.append(box)
            lines.append(f"{cls} {(box.x + box.w / 2) / w:.6f} {(box.y + box.h / 2) / h:.6f} {box.w / w:.6f} {box.h / h:.6f}")
        if not lines:
            continue
        old = path.read_text(encoding="utf-8") if path.is_file() else ""
        if old and not old.endswith("\n"):
            old += "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(old + "\n".join(lines) + "\n", encoding="utf-8")
        frames += 1
        boxes += len(lines)
    return {"frames": frames, "boxes": boxes}
