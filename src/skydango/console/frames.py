"""「整帧」页的后端（spec 2026-10-04-hardcase-inbox-design）：收件箱里每一帧的状态、原图、过目 / 编辑保存 / 不要 / 撤销。

帧状态见 `vision.inbox.frame_state`；这里只把它们摆给页面，并把页面的操作转成 `vision.inbox` 的函数。
接口对前端用类别**编号**（`[perception] classes` 的下标）。
"""

from __future__ import annotations

import json
import logging
import shutil
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
from .labeling import _LOCK  # 和外形页同一把锁：frames.json、_labels.jsonl 的 读-改-写 整段串行（每个请求现建实例，锁必须在模块级）

log = logging.getLogger(__name__)
MIN_SIDE = 4  # 编辑保存时宽或高小于它的框丢掉
SAME_BOX = 0.5  # 和原人物框 IoU 不到它算新框
CROP_SIZE = 224
_WRITER_KEYS = ("_unlabeled", "lit", "unlit", "spirit")
_SIZES: dict[Path, tuple[int, int]] = {}  # 图片宽高缓存（模块级：实例每个请求现建）


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
        if path not in _SIZES:
            h, w = imread(path).shape[:2]
            _SIZES[path] = (w, h)
        return _SIZES[path]

    def _resolve(self, b: dict) -> tuple[str | None, str, bool]:
        """一个框现在算什么：(类别名 / None = 不要, src, dropped)。drop_low 保留类别、标 dropped；
        有裁图的人物框按裁图现在在哪（not_person / 不要 = None）：自动确认、还在原类的算 auto，人改判过的算 judged。"""
        cls = b["cls"]
        if cls in PAIR:
            if b.get("auto") == "drop_low":
                return cls, "auto", True
            where = self._place(b["crop"]) if b.get("crop") else None
            if where == "_discard":
                return None, "judged", False
            if where in attrs.FORMS:
                now = FORM_CLS.get(where)
                return now, "auto" if b.get("auto") == "agree" and now == cls else "judged", False
        return cls, "auto" if b.get("auto") == "agree" else "yolo", False

    def _fixed(self, boxes) -> list[dict]:
        """存好的最终框（decision["boxes"] / last_boxes，编号 + 框）摆给页面；编号不对的跳过。"""
        out = []
        for b in boxes:
            try:
                cls, box = b["cls"], [int(v) for v in b["box"]]
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(cls, int) and not isinstance(cls, bool) and 0 <= cls < len(self.classes) and len(box) == 4:
                out.append({"cls": cls, "box": box, "src": "edited", "dropped": False})
        return out

    def _boxes(self, entry: dict) -> list[dict]:
        """这一帧要显示的框。已通过的 = 通过时写进去的最终框（decision["boxes"]，src：编辑过 edited / 没编辑 final）；
        撤销过编辑过的通过 = last_boxes（src edited，再编辑从它开始）；其余 = 按当前判断合成（`final_boxes` 之前的样子），
        不列人判成"不是人"/"不要"的，src：auto = 外形头自动确认 / 自动丢，judged = 人在外形页判过，yolo = 其余。"""
        decision = entry.get("decision") or {}
        if decision.get("what") == "pass" and isinstance(decision.get("boxes"), list):
            out = self._fixed(decision["boxes"])
            if not decision.get("edited"):
                for b in out:
                    b["src"] = "final"
            return out
        if isinstance(entry.get("last_boxes"), list):
            return self._fixed(entry["last_boxes"])
        out = []
        for b in entry.get("boxes", []):
            cls, src, dropped = self._resolve(b)
            if cls is None:
                continue
            if cls not in self.classes:
                log.warning("整帧页：类别 %s 不在 [perception] classes 里，跳过这个框", cls)
                continue
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
        with _LOCK:
            hit = self._find(frame) if isinstance(frame, str) else None
            if hit is None:
                return 404, {"ok": False, "text": "没有这一帧"}
            run, frames = hit
            entry = frames[frame]
            passed = (entry.get("decision") or {}).get("what") == "pass"
            try:
                if do == "pass":
                    return self._pass(run, frame, entry, body.get("boxes"))
                if do in ("discard", "edit") and passed:
                    return 409, {"ok": False, "text": "这一帧已经通过了，先撤销通过"}
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
        if boxes is None:  # 不编辑直接通过：还有裁图没判 / 整理出错的不行
            state = frame_state(entry, self._place)
            if state == "crops":
                return 409, {"ok": False, "text": "还有裁图没在外形页判"}
            if state == "error":
                return 409, {"ok": False, "text": "这一帧整理时出错"}
            if state == "dup":
                return 409, {"ok": False, "text": "这是重复帧"}
            if state == "edit":  # 有裁图被判了不要（或别处正在编辑）：直接通过会把那个人悄悄去掉
                return 409, {"ok": False, "text": "这一帧要先编辑：有裁图被判了不要（或正在编辑），按 E 改好框再保存"}
        else:
            parsed = self._parse(run, entry, boxes)
            if isinstance(parsed, str):
                return 400, {"ok": False, "text": parsed}
            final = parsed
        try:
            where = pass_frame(self.inbox, run, frame, self.attrs_root, self.dataset, self.classes, final)
        except FileExistsError:
            return 409, {"ok": False, "text": "datasets/sky 里已经有同名的帧"}
        except ValueError as exc:  # final_boxes：框的类别不在 [perception] classes 里（写任何东西之前）
            return 409, {"ok": False, "text": f"通过不了：{exc}"}
        except RuntimeError as exc:  # 原图读不了（写任何东西之前）
            return 500, {"ok": False, "text": f"读不了这一帧的原图：{exc}"}
        except OSError as exc:
            return 500, {"ok": False, "text": f"写盘失败：{exc}"}
        if final is not None:
            try:
                self._crop_edited(run, frame, entry, final)
            except (OSError, RuntimeError) as exc:
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
            except (KeyError, TypeError, ValueError, OverflowError):
                return "类别或框不对"
            if rect is not None:
                out.append((cls, rect))
        return out

    def _crop_edited(self, run: str, frame: str, entry: dict, final: list[tuple[int, Rect]]) -> None:
        """编辑保存后让外形页的数据跟上最终标注：
        - 原人物框已有裁图（和最终框 IoU >= 0.5）：类别和现在展示的一样就不动；变了就把裁图挪到新类的 form/ 目录
          （变成非人物类 -> form/not_person），`_labels.jsonl` 记 frame-edit；
        - 其余人物类的框（新画的）裁一张进 form/<类>/，`_crops.jsonl` 记 source = inbox。"""
        names = [(self.classes[c], r) for c, r in final]
        claimed: set[int] = set()
        moves: list[tuple[str, str, str]] = []  # (裁图, 现在在哪, 挪到哪个外形类)
        for b in entry.get("boxes", []):
            if b["cls"] not in PAIR or not b.get("crop"):
                continue
            here = self._place(b["crop"])
            if here is None:
                continue
            orig = Rect(*(int(v) for v in b["box"]))
            near = [i for i, (_, r) in enumerate(names) if iou(orig, r) >= SAME_BOX]
            if not near:
                continue
            now = self._resolve(b)[0]
            same = [i for i in near if names[i][0] == now]
            if same:
                claimed.update(same)
                continue
            people = [i for i in near if names[i][0] in PAIR]
            pick = max(people, key=lambda i: iou(orig, names[i][1])) if people else None
            claimed.update(near if pick is None else [pick])
            to = PAIR[names[pick][0]] if pick is not None else "not_person"
            if here != to:
                moves.append((b["crop"], here, to))
        fresh = [(n, r) for i, (n, r) in enumerate(names) if i not in claimed and n in PAIR]
        if not moves and not fresh:
            return
        self.attrs_root.mkdir(parents=True, exist_ok=True)
        labels = (self.attrs_root / "_labels.jsonl").open("a", encoding="utf-8")
        try:
            for crop, here, to in moves:
                src = self.attrs_root / (here if here in ("_unlabeled", "_discard") else f"form/{here}") / crop
                dst = self.attrs_root / "form" / to / crop
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                labels.write(json.dumps({"t": time.time(), "crop": crop, "from": here, "to": to, "by": "frame-edit"}, ensure_ascii=False) + "\n")
            if fresh:
                img = imread(self.inbox / run / entry["file"])
                image = (self.inbox / run / entry["file"]).resolve().as_posix()
                writer = _Writer(self.attrs_root, CROP_SIZE, attrs.CROP_PAD, _WRITER_KEYS)
                try:
                    for name, r in fresh:
                        form, crop = PAIR[name], crop_name(frame, r)
                        is_new = crop not in writer.done
                        writer.add(img, frame, r, f"form/{form}", {"image": image, "score": 1.0, "yolo_cls": name, "source": "inbox",
                                                                   "split": entry.get("split"), "group": run, "known": False})
                        if is_new:
                            labels.write(json.dumps({"t": time.time(), "crop": crop, "from": "_unlabeled", "to": form, "by": "frame-edit"},
                                                    ensure_ascii=False) + "\n")
                finally:
                    writer.close()
        finally:
            labels.close()
