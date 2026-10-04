"""「整帧」页的后端（spec 2026-10-04-hardcase-inbox-design）：收件箱里每一帧的状态、原图、过目 / 编辑保存 / 不要 / 撤销。

帧状态见 `vision.inbox.frame_state`；这里只把它们摆给页面，并把页面的操作转成 `vision.inbox` 的函数。
接口对前端用类别**编号**（`[perception] classes` 的下标）。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ..imageio import imread
from ..vision import attrs
from ..vision.attrs_data import _Writer, crop_name
from ..vision.bubbles import Rect
from ..vision.inbox import (
    FORM_CLS,
    PAIR,
    crop_place,
    discard_frame,
    frame_state,
    load_frames,
    pass_frame,
    set_editing,
    undo_frame,
)
from ..vision.track import iou

MIN_SIDE = 4  # 编辑保存时宽或高小于它的框丢掉
SAME_BOX = 0.5  # 和原人物框 IoU 不到它（或类别变了）算新框 / 改过的框，要裁图给外形页
CROP_SIZE = 224
_WRITER_KEYS = ("_unlabeled", "lit", "unlit", "spirit")


def normalize_box(box, w: int, h: int) -> Rect | None:
    """负宽高翻正、裁进 [0, w) × [0, h)；宽或高 < MIN_SIDE 返回 None。"""
    x, y, bw, bh = (int(round(float(v))) for v in box)
    if bw < 0:
        x, bw = x + bw, -bw
    if bh < 0:
        y, bh = y + bh, -bh
    x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return Rect(x1, y1, x2 - x1, y2 - y1)


class FramesApi:
    def __init__(self, inbox: Path, attrs_root: Path, dataset: Path, classes: list[str]) -> None:
        self.inbox, self.attrs_root, self.dataset, self.classes = Path(inbox), Path(attrs_root), Path(dataset), list(classes)
        self._lock = threading.Lock()
        self._sizes: dict[Path, tuple[int, int]] = {}

    # ---- 读 ----

    def _runs(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not self.inbox.is_dir():
            return out
        for d in sorted(p for p in self.inbox.iterdir() if p.is_dir()):
            try:
                frames = load_frames(self.inbox, d.name)
            except (OSError, ValueError):
                continue
            if frames:
                out[d.name] = frames
        return out

    def _find(self, frame: str) -> tuple[str, dict] | None:
        for run, frames in self._runs().items():
            if frame in frames:
                return run, frames
        return None

    def _place(self, crop: str) -> str | None:
        return crop_place(self.attrs_root, crop)

    def _size(self, path: Path) -> tuple[int, int]:
        if path not in self._sizes:
            h, w = imread(path).shape[:2]
            self._sizes[path] = (w, h)
        return self._sizes[path]

    def _boxes(self, entry: dict) -> list[dict]:
        """按当前判断合成的框（`final_boxes` 之前的样子）：drop_low 标 dropped，人判成"不是人"/"不要"的不列；
        已通过的帧也是这个合成（不去读写进数据集的标注）。src：auto = 外形头自动确认 / 自动丢，judged = 人在外形页判过，yolo = 其余。"""
        out = []
        for b in entry.get("boxes", []):
            cls, src, dropped = b["cls"], "yolo", False
            if cls in PAIR and b.get("auto") != "agree":
                if b.get("auto") == "drop_low":
                    src, dropped = "auto", True
                else:
                    where = self._place(b["crop"]) if b.get("crop") else None
                    if where == "_discard":
                        continue
                    if where in attrs.FORMS:
                        cls, src = FORM_CLS.get(where), "judged"
                        if cls is None:
                            continue
            elif b.get("auto") == "agree":
                src = "auto"
            out.append({"cls": self.classes.index(cls), "box": [int(v) for v in b["box"]], "src": src, "dropped": dropped})
        return out

    def state(self) -> dict:
        frames, counts = [], {}
        for run, entries in self._runs().items():
            for name, e in entries.items():
                st = frame_state(e, self._place)
                if st == "dup":
                    continue
                counts[st] = counts.get(st, 0) + 1
                try:
                    w, h = self._size(self.inbox / run / e["file"])
                except (OSError, RuntimeError):
                    w = h = 0
                frames.append({"frame": name, "run": run, "state": st, "reason": e.get("reason"), "split": e.get("split"),
                               "boxes": self._boxes(e), "w": w, "h": h})
        return {"ok": True, "frames": frames, "counts": counts}

    def image(self, frame: str) -> Path | None:
        """只认 frames.json 里登记过的帧，且文件必须在收件箱目录里。"""
        hit = self._find(frame) if isinstance(frame, str) and frame else None
        if hit is None:
            return None
        run, frames = hit
        path = (self.inbox / run / frames[frame]["file"]).resolve()
        try:
            path.relative_to(self.inbox.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None

    # ---- 写 ----

    def act(self, body: dict) -> tuple[int, dict]:
        frame, do = body.get("frame"), body.get("do")
        with self._lock:
            hit = self._find(frame) if isinstance(frame, str) else None
            if hit is None:
                return 404, {"ok": False, "text": "没有这一帧"}
            run, frames = hit
            try:
                if do == "pass":
                    return self._pass(run, frame, frames[frame], body.get("boxes"))
                if do == "discard":
                    discard_frame(self.inbox, run, frame)
                elif do == "undo":
                    undo_frame(self.inbox, run, frame, self.dataset)
                elif do in ("edit", "cancel_edit"):
                    set_editing(self.inbox, run, frame, do == "edit")
                else:
                    return 400, {"ok": False, "text": "不认识这个操作"}
            except OSError as exc:
                return 500, {"ok": False, "text": f"写盘失败：{exc}"}
            return 200, {"ok": True}

    def _pass(self, run: str, frame: str, entry: dict, boxes) -> tuple[int, dict]:
        final = None
        if boxes is not None:
            parsed = self._parse(run, entry, boxes)
            if isinstance(parsed, str):
                return 400, {"ok": False, "text": parsed}
            final = parsed
        try:
            where = pass_frame(self.inbox, run, frame, self.attrs_root, self.dataset, self.classes, final)
        except FileExistsError:
            return 409, {"ok": False, "text": "datasets/sky 里已经有同名的帧"}
        except OSError as exc:
            return 500, {"ok": False, "text": f"写盘失败：{exc}"}
        if final is not None:
            try:
                self._crop_edited(run, frame, entry, final)
            except OSError as exc:
                return 500, {"ok": False, "text": f"帧已通过，但裁图写盘失败：{exc}"}
        return 200, {"ok": True, "dataset": where}

    def _parse(self, run: str, entry: dict, boxes) -> list[tuple[int, Rect]] | str:
        if not isinstance(boxes, list):
            return "boxes 要是列表"
        try:
            w, h = self._size(self.inbox / run / entry["file"])
        except (OSError, RuntimeError):
            return "读不了这一帧的图"
        out: list[tuple[int, Rect]] = []
        for b in boxes:
            try:
                cls, box = b["cls"], b["box"]
                if isinstance(cls, bool) or not isinstance(cls, int) or cls not in range(len(self.classes)) or len(box) != 4:
                    return "类别或框不对"
                rect = normalize_box(box, w, h)
            except (KeyError, TypeError, ValueError):
                return "类别或框不对"
            if rect is not None:
                out.append((cls, rect))
        return out

    def _crop_edited(self, run: str, frame: str, entry: dict, final: list[tuple[int, Rect]]) -> None:
        """人物类里和原框 IoU < 0.5 或类别变了的框裁一张图进外形页（form/<类>/，记 frame-edit）。"""
        olds = [(b["cls"], Rect(*(int(v) for v in b["box"]))) for b in entry.get("boxes", []) if b["cls"] in PAIR]
        fresh = [(c, r) for c, r in final if self.classes[c] in PAIR
                 and not any(oc == self.classes[c] and iou(orect, r) >= SAME_BOX for oc, orect in olds)]
        if not fresh:
            return
        img = imread(self.inbox / run / entry["file"])
        image = (self.inbox / run / entry["file"]).resolve().as_posix()
        writer = _Writer(self.attrs_root, CROP_SIZE, attrs.CROP_PAD, _WRITER_KEYS)
        labels = (self.attrs_root / "_labels.jsonl").open("a", encoding="utf-8")
        try:
            for c, r in fresh:
                name, form = self.classes[c], PAIR[self.classes[c]]
                crop = crop_name(frame, r)
                is_new = crop not in writer.done
                writer.add(img, frame, r, f"form/{form}", {"image": image, "score": 1.0, "yolo_cls": name, "source": "inbox",
                                                           "split": entry.get("split"), "group": run, "known": False})
                if is_new:
                    labels.write(json.dumps({"t": time.time(), "crop": crop, "from": "_unlabeled", "to": form, "by": "frame-edit"},
                                            ensure_ascii=False) + "\n")
        finally:
            labels.close()
            writer.close()
