"""训练集核对名单（`perception crops-audit`）：几个 YOLO 模型在训练帧上和标注对一遍，列出该人工核对的人物框，
写进 `datasets/attrs/_focus.json`，外形页的「重点核对」筛选读它。

四类（同一张裁图可能属于几类，名单去重）：
- missed：有检测器框到、标注里没有的人（裁进 `_unlabeled/`，记 `found_by`）；
- suspect：标注了人，但所有检测器都没框到（可能标错了位置 / 不是人）；
- partial：标了人，只有一部分检测器框到；
- head：所有检测器都框到了，但外形头说不是人 / 人形概率不够。
只处理 `images/train`、跳过增强图（`_blur` / `_dark`）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import attrs
from .attrs_data import (DEDUP_IOU, DETECT_CLASSES, ID_NAMES, KNOWN, PERSON_IDS, UNLABELED_IOU, _images_in, _label_boxes,
                         _read_rows, _Writer, crop_name, frame_group, imread)
from .augment import SUFFIXES
from .bubbles import Rect
from .track import iou

HIT_IOU = 0.5  # 标注框和检测框重叠这么多才算"框到了"
FOCUS_FILE = "_focus.json"


def _people_at(detections, thresh: float):
    """分 ≥ thresh 的人物框；同一个人的 player / player_unlit 两框按运行时的 merge_people 合成一个。"""
    from .perception import merge_people

    return merge_people([d for d in detections if d.cls in DETECT_CLASSES and d.score >= thresh])[0]


def _head_flag(head, img, cls: int, box: Rect, accept: float) -> bool:
    """外形头说它不是人（最高类不在 PERSON_FORMS）或人形概率之和 < accept。"""
    yolo_cls = "player_unlit" if cls == 4 else "player"
    res = head.predict([(yolo_cls, attrs.crop(img, box, head.pad("form"), head.size, head.keep("form")))])[0]
    p = res.get("form")
    if p is None:
        return False
    labels = head.labels("form")
    top = labels[int(max(range(len(labels)), key=lambda i: p[i]))]
    return top not in attrs.PERSON_FORMS or sum(float(p[i]) for i, lb in enumerate(labels) if lb in attrs.PERSON_FORMS) < accept


def audit(dataset: Path, attrs_root: Path, detectors: dict, head=None, conf: float = 0.25, low: float = 0.15,
          accept: float = 0.6) -> dict:
    """返回 {"missed", "suspect", "partial", "head", "focus"}（都是裁图名）。detectors = {名字: 检测器}，
    检测器出框门槛要不高于 low；缺的裁图会写进 attrs_root（missed → _unlabeled/，其余补裁进 form/<外形>/）。"""
    dataset, attrs_root = Path(dataset), Path(attrs_root)
    res: dict[str, list[str]] = {"missed": [], "suspect": [], "partial": [], "head": []}
    existing: dict[str, list[tuple[Rect, str]]] = {}  # 原图 → 已经裁过的 (框, 裁图名)，换模型重跑时认得出已有的
    for r in _read_rows(attrs_root):
        if isinstance(r.get("image"), str) and isinstance(r.get("box"), list) and len(r["box"]) == 4:
            existing.setdefault(r["image"], []).append((Rect(*r["box"]), r["crop"]))
    w = _Writer(attrs_root, 224, attrs.CROP_PAD, (*KNOWN.values(), "_unlabeled"))
    try:
        for path in _images_in(dataset / "images" / "train"):
            if path.stem.endswith(SUFFIXES):  # 增强图
                continue
            img = imread(path)
            h, wd = img.shape[:2]
            labeled = _label_boxes(dataset / "labels" / "train" / f"{path.stem}.txt", wd, h)
            people = [b for c, b in labeled if c in PERSON_IDS]
            image = path.resolve().as_posix()
            base = {"image": image, "dataset": dataset.resolve().as_posix(), "split": "train", "group": frame_group(path.stem), "source": "dataset"}
            raw = {name: det.detect(img) for name, det in detectors.items()}

            for name, dets in raw.items():  # 漏标：检测器框到、标注里没有
                for d in _people_at(dets, conf):
                    if any(iou(d.box, b) >= UNLABELED_IOU for b in people):
                        continue
                    hit = next((n for b, n in existing.get(image, []) if iou(d.box, b) >= DEDUP_IOU), None)
                    if hit is None:
                        hit = crop_name(path.stem, d.box)
                        w.add(img, path.stem, d.box, "_unlabeled",
                              {**base, "score": round(float(d.score), 4), "yolo_cls": d.cls, "known": False, "found_by": name})
                        existing.setdefault(image, []).append((d.box, hit))
                    res["missed"].append(hit)

            lows = {name: _people_at(dets, low) for name, dets in raw.items()}
            for cls, box in labeled:  # 标了的人：有几个检测器框到
                if cls not in PERSON_IDS or cls == 3 or cls not in KNOWN:
                    continue
                hits = sum(any(iou(d.box, box) >= HIT_IOU for d in ds) for ds in lows.values())
                kind = None
                if lows and hits == 0:
                    kind = "suspect"
                elif lows and hits < len(lows):
                    kind = "partial"
                elif head is not None and _head_flag(head, img, cls, box, accept):
                    kind = "head"
                if kind is None:
                    continue
                name = crop_name(path.stem, box)
                if name not in w.done:  # form/ 和 _unlabeled/ 里都没有：补裁
                    w.add(img, path.stem, box, f"form/{KNOWN[cls]}",
                          {**base, "score": 1.0, "yolo_cls": ID_NAMES[cls], "known": True})
                res[kind].append(name)
    finally:
        w.close()
    out = {k: list(dict.fromkeys(v)) for k, v in res.items()}
    out["focus"] = list(dict.fromkeys(n for k in ("missed", "suspect", "partial", "head") for n in out[k]))
    return out


def write_focus(attrs_root: Path, names: list[str]) -> Path | None:
    """写 `_focus.json`；已有的先改名成 `_focus-<时间>.json`。返回备份路径（没有旧文件 = None）。"""
    attrs_root = Path(attrs_root)
    attrs_root.mkdir(parents=True, exist_ok=True)
    path = attrs_root / FOCUS_FILE
    backup = None
    if path.exists():
        backup = attrs_root / f"_focus-{datetime.now():%Y%m%d-%H%M%S}.json"
        n = 1
        while backup.exists():  # 同一秒写两次
            backup = attrs_root / f"_focus-{datetime.now():%Y%m%d-%H%M%S}-{n}.json"
            n += 1
        path.rename(backup)
    path.write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    return backup
