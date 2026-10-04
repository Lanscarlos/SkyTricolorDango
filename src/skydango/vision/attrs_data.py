"""第二层（外形头）的数据工具（`perception crops`）：从 YOLO 数据集和录像 / 难例目录裁人物图，确认后写回数据集。

输出目录（默认 datasets/attrs）：
  form/<外形>/       已知类别的框直接放这里（来自有标注的数据集框）
  _unlabeled/        检测器认出、但数据集里没标的人物框（等 Claude 初分 + 标注页确认）
  _crops.jsonl       每张裁图一行：crop / image / box / score / yolo_cls / source / split / group / known / size
重跑幂等：_crops.jsonl 里已有的裁图名不再生成（裁图被挪走了也不会又造一份）。
写回（`writeback`）：来自数据集、本来没标的框，被标注页挪进 form/<人形类>/ 就算确认，补进 labels/。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..brain.images import image_block
from ..config import AssistConfig
from ..imageio import imread, imwrite
from . import attrs
from .assist import FrameInput, Protocol, extract_json
from .augment import IMAGE_EXTS, SUFFIXES
from .bubbles import Rect
from .gesture_label import Guess, _conf
from .track import iou

KNOWN = {0: "lit", 4: "unlit", 9: "spirit"}  # datasets/sky 类别编号 → 外形
PERSON_IDS = (0, 3, 4, 9)  # 数据集里算"人物"的类别：player / self / player_unlit / spirit
ID_NAMES = {0: "player", 3: "self", 4: "player_unlit", 9: "spirit"}
WRITEBACK_ID = {"lit": 0, "morph": 0, "shared": 0, "unlit": 4, "spirit": 9}
DETECT_CLASSES = ("player", "player_unlit")
UNLABELED_IOU = 0.4  # 检测框和任何人物标注的 IoU 低于这个才算"没标过"
DEDUP_IOU = 0.5  # 同一张图里重叠这么多的只留分高的；写回时同帧已有人物框重叠这么多就不写

LABEL_LOG = "_labels.jsonl"  # 标注页的记账（console/labeling.py 写）

log = logging.getLogger(__name__)
_FRAME = re.compile(r"^(.*)_\d+_[\d.]+s$")


def hand_labels(root: Path) -> dict[str, str]:
    """标注页记账里每张裁图最后一次有效操作的去处（撤销抵消它前面最近一条没被抵消的操作，同 labeling 的撤销）。
    裁图现在就在这个位置 = 人确认过（含原地确认 from == to）；datasets/sky 导进 form/ 的裁图没有记录 = 没人看过。"""
    try:
        lines = (Path(root) / LABEL_LOG).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    stack: list[dict] = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        if e.get("undo"):
            if stack:
                stack.pop()
        elif isinstance(e.get("crop"), str) and isinstance(e.get("to"), str):
            stack.append(e)
    return {e["crop"]: e["to"] for e in stack}


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
        rows = _read_rows(self.out)
        self.done = {r["crop"] for r in rows}
        self.boxes: dict[str, list[Rect]] = {}  # 原图 → 已经裁过的框（换模型重裁时，人判过的框位置差一点也不再裁）
        for r in rows:
            if isinstance(r.get("image"), str) and isinstance(r.get("box"), list) and len(r["box"]) == 4:
                self.boxes.setdefault(r["image"], []).append(Rect(*r["box"]))
        self.counts: dict[str, int] = dict.fromkeys(keys, 0)
        self._fh = open(self.out / "_crops.jsonl", "a", encoding="utf-8")

    def add(self, img, stem: str, box: Rect, folder: str, row: dict) -> None:
        name = crop_name(stem, box)
        if name in self.done:
            return
        self.done.add(name)
        if isinstance(row.get("image"), str):
            self.boxes.setdefault(row["image"], []).append(box)
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
    """人物框；同一个人的 player / player_unlit 两框按运行时的 merge_people 合成一个（留分高的）。"""
    from .perception import merge_people

    return merge_people([d for d in detector.detect(img) if d.cls in DETECT_CLASSES and d.score >= conf])[0]


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
                    cropped = w.boxes.get(base["image"], [])
                    if all(iou(d.box, b) < UNLABELED_IOU for b in people) and all(iou(d.box, b) < DEDUP_IOU for b in cropped):
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


def _line(cls: int, box: Rect, w: int, h: int) -> str:
    return f"{cls} {(box.x + box.w / 2) / w:.6f} {(box.y + box.h / 2) / h:.6f} {box.w / w:.6f} {box.h / h:.6f}"


def _edit_label(path: Path, w: int, h: int, fixes: list[tuple[Rect, int | None]], adds: list[tuple[int, Rect]],
                write: bool = True) -> tuple[int, int, int]:
    """改一个标注文件：fixes = 已有人物框（0 / 4 / 9）里和它 IoU ≥ DEDUP_IOU 最高的那行改类别（None 删掉）；
    adds = 和已有人物框都不重叠（< DEDUP_IOU）才追加。别的行原样不动。返回 (改类别几行, 删了几行, 加了几行)；write = False 只算不写。"""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    parsed = []  # [(原行, 类别, 框 | None)]
    for ln in lines:
        parts = ln.split()
        box = None
        cls = -1
        if len(parts) >= 5:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            box = Rect(round((cx - bw / 2) * w), round((cy - bh / 2) * h), round(bw * w), round(bh * h))
        parsed.append([ln, cls, box])
    relabeled = removed = added = 0
    for fbox, cid in fixes:
        cands = [i for i, (_, c, b) in enumerate(parsed) if b is not None and c in KNOWN]
        hit = max(cands, key=lambda i: iou(fbox, parsed[i][2]), default=None)
        if hit is None or iou(fbox, parsed[hit][2]) < DEDUP_IOU:
            continue
        if cid is None:
            parsed.pop(hit)
            removed += 1
        elif parsed[hit][1] != cid:
            parts = parsed[hit][0].split()
            parsed[hit] = [" ".join([str(cid), *parts[1:]]), cid, parsed[hit][2]]
            relabeled += 1
    existing = [b for _, c, b in parsed if b is not None and c in PERSON_IDS]
    for cls, box in adds:
        if any(iou(box, b) >= DEDUP_IOU for b in existing):
            continue
        existing.append(box)
        parsed.append([_line(cls, box, w, h), cls, box])
        added += 1
    if write and (relabeled or removed or added):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(ln + "\n" for ln, _, _ in parsed), encoding="utf-8")
    return relabeled, removed, added


def writeback(dataset: Path, out: Path, now: datetime) -> dict:
    """把标注页确认过的结果写回 labels/（来自这个数据集的裁图）：
    - 本来没标（known == false）、现在躺在 out/form/<人形类>/ 里 → 补一个框；
    - 本来标了（known == true）、标注页确认过（`hand_labels`，现在还在那）且类别变了 → 改那一行的类别，确认是不是人 → 删掉那一行；
      没人确认的（导入的）、「不要」（看不清，不等于不是人）、撤销掉的都不动。
    增强图（<帧>_blur / _dark，标注是原图的副本）一起改。真有要改的才先整个复制 labels/ 到 _backup/labels-<时间>（重跑不多备份）。
    返回 {frames 改了几帧（只数原图）, boxes 加了几个框, relabeled 改了几个类别, removed 删了几个框}。"""
    dataset, out = Path(dataset), Path(out)
    hand = hand_labels(out)
    todo: dict[tuple[str, str], dict] = {}
    here = dataset.resolve().as_posix()
    old_rows = 0
    for r in _read_rows(out):
        if r.get("source") != "dataset":
            continue
        if "dataset" not in r:
            old_rows += 1
            continue
        if Path(r["dataset"]).resolve().as_posix() != here:  # 别的数据集的裁图，共用同一个 out
            continue
        x, y, bw, bh = r["box"]
        key = (r["split"], Path(r["image"]).stem)
        if not r.get("known"):
            form = next((f for f in attrs.PERSON_FORMS if (out / "form" / f / r["crop"]).is_file()), None)
            if form is not None:
                todo.setdefault(key, {"fixes": [], "adds": [], "size": r.get("size")})["adds"].append((WRITEBACK_ID[form], Rect(x, y, bw, bh)))
            continue
        to = hand.get(r["crop"])
        if to not in attrs.FORMS or not (out / "form" / to / r["crop"]).is_file():
            continue
        cid = WRITEBACK_ID.get(to)  # not_person → None：删掉
        if cid is not None and ID_NAMES.get(cid) == r.get("yolo_cls"):
            continue
        todo.setdefault(key, {"fixes": [], "adds": [], "size": r.get("size")})["fixes"].append((Rect(x, y, bw, bh), cid))
    if old_rows:
        log.warning("%d 条旧的 _crops.jsonl 记录没有 dataset 字段，写回时跳过", old_rows)
    res = {"frames": 0, "boxes": 0, "relabeled": 0, "removed": 0}
    if not todo:
        return res
    labels = dataset / "labels"
    jobs = []  # [(要改的文件, 宽, 高, 任务, 是不是原图)]
    for (split, stem), job in sorted(todo.items()):
        if job["size"] is None:
            h, w = imread(dataset / "images" / split / f"{stem}.jpg").shape[:2]
        else:
            w, h = job["size"]
        jobs.append((labels / split / f"{stem}.txt", w, h, job, True))
        for suffix in SUFFIXES:  # 增强图的标注副本
            aug = labels / split / f"{stem}{suffix}.txt"
            if aug.is_file():
                jobs.append((aug, w, h, job, False))
    if not any(any(_edit_label(f, w, h, j["fixes"], j["adds"], write=False)) for f, w, h, j, _ in jobs):
        return res
    shutil.copytree(labels, dataset / "_backup" / f"labels-{now.strftime('%Y%m%d-%H%M%S')}")
    for path, w, h, job, original in jobs:
        rl, rm, ad_ = _edit_label(path, w, h, job["fixes"], job["adds"])
        if original and (rl or rm or ad_):
            res["frames"] += 1
            res["boxes"] += ad_
            res["relabeled"] += rl
            res["removed"] += rm
    return res


# ---- Claude 初分（`perception attrs-label`）：≤16 张裁图拼 4×4，Claude 按裁图名逐张判外形 ----

FORM_PROMPT_VERSION = 1
FORM_LABELS = (*attrs.FORMS, "unsure")
FORM_CELL = 160  # 拼图每格边长；4×4 → 640×640
FORM_PER_SHEET = 16
FORM_GUESS_FILE = "claude.json"  # 在 _unlabeled/ 里：{裁图名: {label, confidence, reason, model, version}}

FORM_SYSTEM = """你是游戏《光·遇》(Sky) 画面里"人物框"的标注员。每张图是一张 4×4 拼图：16 个检测器框出来的裁图，
顺序是从左到右、从上到下，每格左上角的数字是格号 0~15（消息里会给出格号对应的裁图名）。判断每一格里的东西属于哪一类：
- not_person：根本不是玩家 —— 树、椅子、UI 图标、篝火、茶壶、雕像、壁画、地面、光效等。
- lit：点过火的玩家：有头发、衣服、斗篷，样子和外观实心清楚（颜色鲜明、不透明）。
- unlit：没点火的玩家（黑影）：整个人是深色 / 灰黑的剪影，没有衣服细节。
- spirit：先祖：身形修长、偏透明、发光。
- shared：共享空间里的玩家：矮小、蓝色半透明。
- morph：变身的玩家：雪人、白鹿等变身造型，也是玩家。
- unsure：太小、被挡住、模糊，没法判断。
只输出一个 JSON 对象，每格一项，键是裁图名：
{"<裁图名>": {"label": "lit", "confidence": 0.8, "reason": "一句话说明看到了什么"}}
label 只能是 not_person / lit / unlit / spirit / shared / morph / unsure；confidence 是 0~1。不要输出别的文字。"""


def form_sheet(crops: list[np.ndarray]) -> np.ndarray:
    """≤16 张裁图 → 4×4 拼图（每格 160×160，不是这个尺寸先缩放），每格左上角白底黑字写格号 0~15；不足 16 张的格子留黑。"""
    if not 1 <= len(crops) <= FORM_PER_SHEET:
        raise ValueError(f"拼图要 1~{FORM_PER_SHEET} 张裁图，给了 {len(crops)}")
    c = FORM_CELL
    sheet = np.zeros((c * 4, c * 4, 3), np.uint8)
    for i, f in enumerate(crops):
        if f.shape[:2] != (c, c):
            f = cv2.resize(f, (c, c), interpolation=cv2.INTER_AREA)
        y, x = (i // 4) * c, (i % 4) * c
        sheet[y : y + c, x : x + c] = f
        label = str(i)
        cv2.rectangle(sheet, (x, y), (x + 10 * len(label) + 4, y + 16), (255, 255, 255), -1)
        cv2.putText(sheet, label, (x + 2, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return sheet


def sheet_stem(names: list[str]) -> str:
    """拼图的名字：由它包含的裁图名算出，同一组裁图每次都一样。"""
    return "sheet-" + hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:12]


def make_sheet_input(names: list[str], crops: list[np.ndarray]) -> FrameInput:
    """一张拼图 = Reviewer 的一帧；hints 放格号对应的裁图名。"""
    return FrameInput(sheet_stem(names), form_sheet(crops), [], hints=list(names))


def build_form_message(batch: list[FrameInput], cfg: AssistConfig) -> list[dict]:
    """一批拼图的内容块：每张一段文字（格号 → 裁图名）加图片。"""
    content: list[dict] = []
    for f in batch:
        cells = "；".join(f"{i}={n}" for i, n in enumerate(f.hints))
        content.append({"type": "text", "text": f"拼图 {f.stem}，格号对应的裁图：{cells}"})
        content.append(image_block(f.image, 85))
    return content


def parse_form_review(text: str, batch: list[FrameInput]) -> dict[str, Guess]:
    """Claude 的回答 → 裁图名 → Guess。只收 FORM_LABELS 里的 label；裁图名对不上、label 不认识的不在结果里（= 没核对）。"""
    names = {n for f in batch for n in f.hints}
    data = extract_json(text, names)
    if data is None:
        return {}
    out: dict[str, Guess] = {}
    for name, item in data.items():
        if not isinstance(item, dict) or item.get("label") not in FORM_LABELS:
            continue
        out[name] = Guess(item["label"], _conf(item.get("confidence")), str(item.get("reason") or "")[:60])
    return out


def _parse_by_sheet(text: str, batch: list[FrameInput]) -> dict[str, dict[str, Guess]]:
    """Reviewer 要的形状：拼图名 → {裁图名: Guess}（这张拼图一个都没认出来就不在结果里）。"""
    flat = parse_form_review(text, batch)
    out = {}
    for f in batch:
        got = {n: flat[n] for n in f.hints if n in flat}
        if got:
            out[f.stem] = got
    return out


FORM_PROTOCOL = Protocol(FORM_PROMPT_VERSION, FORM_SYSTEM, build_form_message, _parse_by_sheet)


def load_form_guesses(unlabeled: Path) -> dict[str, dict]:
    """读 _unlabeled/claude.json；没有 / 坏了返回空。"""
    try:
        d = json.loads((Path(unlabeled) / FORM_GUESS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def pending_crops(unlabeled: Path, existing: dict[str, dict], recheck: bool = False) -> list[str]:
    """_unlabeled/ 里还要初分的裁图名：没写过、或者写的是旧版提示词（--recheck 全部重做）。"""
    unlabeled = Path(unlabeled)
    names = sorted(p.name for p in unlabeled.iterdir() if p.suffix.lower() in IMAGE_EXTS) if unlabeled.is_dir() else []
    if recheck:
        return names
    return [n for n in names if (existing.get(n) or {}).get("version") != FORM_PROMPT_VERSION]


def write_form_guesses(unlabeled: Path, new: dict[str, Guess], model: str) -> None:
    """把这一批的结果并进 claude.json（读旧的再写，Claude 初分期间标注页挪走的裁图也不影响）。"""
    unlabeled = Path(unlabeled)
    merged = load_form_guesses(unlabeled)
    for name, g in new.items():
        merged[name] = {"label": g.label, "confidence": g.confidence, "reason": g.reason,
                        "model": model, "version": FORM_PROMPT_VERSION}
    tmp = unlabeled / (FORM_GUESS_FILE + ".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(unlabeled / FORM_GUESS_FILE)
