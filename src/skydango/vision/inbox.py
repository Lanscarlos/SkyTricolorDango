"""难例收件箱（spec 2026-10-04-hardcase-inbox-design）：把各次运行存下的难例收进 datasets/inbox。

    inbox/<运行>/raw/*.jpg    运行目录 hard/ 里的难例原图
    inbox/<运行>/hard.jsonl   运行目录 hard.jsonl 的副本（原因、检测框）
    inbox/_index.jsonl        每行一条：收了哪个运行（run / collected_at / frames），之后的步骤往同一运行追加字段（processed_at 等）
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

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


def pending_frames(inbox: Path) -> int:
    """没整理的运行在 index.jsonl 里记的帧数之和。"""
    pending = set(pending_runs(inbox))
    last: dict[str, int] = {}
    for row in _index_rows(Path(inbox)):
        if row["run"] in pending and row.get("frames") is not None:
            last[row["run"]] = int(row["frames"])
    return sum(last.values())


def read_stats(inbox: Path) -> dict:
    """`_stats.json`（没有或坏了是空字典）。"""
    return _json_dict(Path(inbox) / STATS)


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
    label_existed = label.exists()
    label.write_text(text, encoding="utf-8")
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(src_img, dst_img)
        shutil.copyfile(label, dst_lbl)
        entry["decision"] = {"what": "pass", "edited": edited, "t": time.time(), "dataset": f"{split}/{frame}"}
        entry["editing"] = False
        save_frames(inbox, run, frames)
    except BaseException:
        dst_img.unlink(missing_ok=True)
        dst_lbl.unlink(missing_ok=True)
        if not label_existed:
            label.unlink(missing_ok=True)
        raise
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


# ---- 整理流水线（spec §4）----

STATS = "_stats.json"
GUESS_MODEL = "attrs-screen"
_YOLO_CN = {"player": "点亮的人", "player_unlit": "黑影", "spirit": "先祖"}
_FORM_CN = {"not_person": "不是人", "lit": "点亮的人", "unlit": "黑影", "spirit": "先祖", "shared": "共享空间", "morph": "变身"}
_CROP_SIZE = 224
_WRITER_KEYS = ("_unlabeled", "lit", "unlit", "spirit")
_TIME = re.compile(r"^(\d{2})(\d{2})(\d{2})")


def similar(a: np.ndarray, b: np.ndarray, diff: float) -> bool:
    """两张图各缩成 1/8 灰度，平均绝对差 < diff 算差不多。"""

    def small(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.resize(gray, (max(1, gray.shape[1] // 8), max(1, gray.shape[0] // 8)), interpolation=cv2.INTER_AREA)

    x, y = small(a), small(b)
    if x.shape != y.shape:
        return False
    return float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean()) < diff


def frame_time(file: str) -> float | None:
    """文件名开头的 HHMMSS → 当天第几秒；没有就是 None。"""
    m = _TIME.match(Path(file).name)
    if not m:
        return None
    h, mi, s = (int(v) for v in m.groups())
    return float(h * 3600 + mi * 60 + s)


def _hard_reasons(inbox: Path, run: str) -> dict[str, str | None]:
    path = Path(inbox) / run / "hard.jsonl"
    out: dict[str, str | None] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("file"):
            out[str(row["file"])] = row.get("reason")
    return out


def _json_dict(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _other_boxes(dets: list, pre: list[tuple[str, Rect]]) -> list[dict]:
    """非人物框：预标注（弱标注 + 检测 + 圆圈）里的名字标签、圆圈等；分数取同类 IoU > 0.5 的检测框的，没有就 1.0（来自 OCR）。"""
    from .track import iou

    out: list[dict] = []
    for cls, box in pre:
        if cls in PAIR:
            continue
        match = max((d.score for d in dets if d.cls == cls and iou(box, d.box) > 0.5), default=None)
        out.append({"cls": cls, "box": [box.x, box.y, box.w, box.h], "score": float(match if match is not None else 1.0),
                    "src": "yolo" if match is not None else "ocr", "crop": None, "auto": None})
    return out


def _process_frame(img, image: str, name: str, run: str, split: str, ctx: dict) -> tuple[list[dict], int, int]:
    """一帧：检测 + 弱标注 → 人物框裁图、外形头判、分流。返回 (boxes, 自动一致几个, 给人判几个)。"""
    from .attrs_data import FORM_PROMPT_VERSION, crop_name
    from .perception import merge_people
    from .track import iou

    cfg, writer = ctx["cfg"], ctx["writer"]
    dets = [d for d in ctx["detect"](img) if d.score >= cfg.perception.low_conf]
    sure = [d for d in dets if d.score >= cfg.perception.conf]
    boxes = _other_boxes(sure, ctx["weak"](img, sure))
    selfs = [d for d in sure if d.cls == "self"]
    # 团子身上常同时出 self 和 player 两个框：和 self 框重合的人物框是团子的重复框，不裁、不判、不写
    people = [d for d in merge_people([d for d in dets if d.cls in PAIR])[0]
              if not any(iou(d.box, s.box) >= 0.5 for s in selfs)]
    probs = ctx["judge"](img, [d.box for d in people]) if people else []
    auto = human = 0
    for d, p in zip(people, probs):
        r = route(d.cls, d.score, p, cfg.perception.conf, cfg.inbox.agree)
        entry = {"cls": d.cls, "box": [d.box.x, d.box.y, d.box.w, d.box.h], "score": float(d.score), "src": "yolo",
                 "crop": None, "auto": r.auto}
        boxes.append(entry)
        if r.auto == "drop_low":
            continue
        entry["crop"] = crop = crop_name(name, d.box)
        fresh = crop not in writer.done
        row = {"image": image, "score": float(d.score), "yolo_cls": d.cls, "source": "inbox", "split": split, "group": run, "known": False}
        writer.add(img, name, d.box, f"form/{r.form}" if r.auto == "agree" else "_unlabeled", row)
        if r.auto == "agree":
            auto += 1
            if fresh:
                ctx["labels_fh"].write(json.dumps({"t": time.time(), "crop": crop, "from": "_unlabeled", "to": r.form,
                                                   "by": "auto-agree", "p": round(r.p, 3)}, ensure_ascii=False) + "\n")
        else:
            human += 1
            ctx["guesses"][crop] = {
                "label": r.form, "confidence": round(r.p, 3), "model": GUESS_MODEL, "version": FORM_PROMPT_VERSION,
                "reason": f"YOLO 判{_YOLO_CN.get(d.cls, d.cls)} {d.score:.2f}，外形头判{_FORM_CN.get(r.form, r.form)} {r.p:.2f}"}
    return boxes, auto, human


def _state_counts(frames: dict, attrs_root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in frames.values():
        s = frame_state(entry, lambda c: crop_place(attrs_root, c))
        out[s] = out.get(s, 0) + 1
    return out


def process(
    inbox: Path,
    attrs_root: Path,
    runs: Path,
    cfg,
    detect: Callable,
    weak: Callable,
    judge: Callable,
    progress: Callable[[str], None] = lambda line: print(line, flush=True),
) -> dict:
    """整理收件箱（spec §4）：先收（强杀后补收），再对没整理完的运行逐帧去重、检测、外形头分流，每帧落一次盘（可续跑）。
    _stats.json 的 sec_per_frame = 最近一次整理里每张非重复帧的平均耗时（秒）。"""
    from ..imageio import imread
    from .attrs_data import _Writer

    inbox, attrs_root = Path(inbox), Path(attrs_root)
    collect_all(runs, inbox)
    total = {"runs": 0, "frames": 0, "dups": 0, "auto": 0, "to_judge": 0, "glance": 0}
    guess_path = attrs_root / "_unlabeled" / "claude.json"
    writer = _Writer(attrs_root, _CROP_SIZE, attrs.CROP_PAD, _WRITER_KEYS)
    labels_fh = (attrs_root / "_labels.jsonl").open("a", encoding="utf-8")
    try:
        for run in sorted(pending_runs(inbox)):
            frames = load_frames(inbox, run)
            reasons = _hard_reasons(inbox, run)
            files = sorted(p.name for p in (inbox / run / "raw").glob("*.jpg"))
            prev: tuple[str, np.ndarray, float | None] | None = None
            for name, e in frames.items():  # 续跑：上一张保留帧
                if not e.get("dup_of") and not e.get("error") and (inbox / run / e["file"]).is_file():
                    prev = (name, None, frame_time(e["file"]))
            seconds, done = 0.0, 0
            for i, file in enumerate(files, 1):
                name = frame_name(run, file)
                if name in frames:
                    continue
                started = time.time()
                split = split_of(run, cfg.inbox.val_every)
                base = {"file": f"raw/{file}", "reason": reasons.get(file), "split": split, "dup_of": None, "boxes": [],
                        "editing": False, "error": None, "decision": None}
                try:
                    img = imread(inbox / run / "raw" / file)
                    t = frame_time(file)
                    if prev is not None and prev[1] is None:  # 续跑后第一次要比较：现读上一张
                        prev = (prev[0], imread(inbox / run / frames[prev[0]]["file"]), prev[2])
                    if (prev is not None and similar(prev[1], img, cfg.inbox.dup_diff)
                            and (t is None or prev[2] is None or abs(t - prev[2]) <= cfg.inbox.dup_gap)):
                        frames[name] = {**base, "dup_of": prev[0]}
                        total["dups"] += 1
                    else:
                        ctx = {"cfg": cfg, "writer": writer, "detect": detect, "weak": weak, "judge": judge,
                               "labels_fh": labels_fh, "guesses": _json_dict(guess_path)}
                        image = (inbox / run / "raw" / file).resolve().as_posix()
                        boxes, auto, human = _process_frame(img, image, name, run, split, ctx)
                        frames[name] = {**base, "boxes": boxes}
                        labels_fh.flush()
                        writer.flush()
                        if human:
                            _write_json(guess_path, ctx["guesses"])
                        prev = (name, img, t)
                        total["frames"] += 1
                        total["auto"] += auto
                        total["to_judge"] += human
                        seconds += time.time() - started
                        done += 1
                except Exception as exc:  # 单帧出错记下来跳过，其余照常
                    frames[name] = {**base, "error": f"{type(exc).__name__}: {exc}"}
                save_frames(inbox, run, frames)
                progress(f"PROGRESS {i}/{len(files)} {run}")
            counts = _state_counts(frames, attrs_root)
            total["runs"] += 1
            total["glance"] += counts.get("glance", 0)
            with (inbox / INDEX).open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run": run, "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "counts": counts},
                                   ensure_ascii=False) + "\n")
            if done:
                _write_json(inbox / STATS, {**_json_dict(inbox / STATS), "sec_per_frame": round(seconds / done, 3)})
    finally:
        labels_fh.close()
        writer.close()
    return total


def status(inbox: Path, attrs_root: Path) -> dict:
    """各次运行各状态的帧数、来自整理但还在 _unlabeled 的裁图数、上次训练以来通过的帧数。"""
    from .attrs_data import _read_rows

    inbox, attrs_root = Path(inbox), Path(attrs_root)
    trained = float(_json_dict(inbox / STATS).get("trained_at") or 0)
    runs: dict[str, dict[str, int]] = {}
    passed = 0
    for d in sorted(p for p in inbox.iterdir() if p.is_dir()) if inbox.is_dir() else []:
        frames = load_frames(inbox, d.name)
        if not frames:
            continue
        runs[d.name] = _state_counts(frames, attrs_root)
        passed += sum(1 for e in frames.values()
                      if (e.get("decision") or {}).get("what") == "pass" and float(e["decision"].get("t") or 0) > trained)
    left = sum(1 for r in _read_rows(attrs_root) if r.get("source") == "inbox" and crop_place(attrs_root, r["crop"]) == "_unlabeled")
    return {"runs": runs, "judge_left": left, "passed_since_train": passed}
