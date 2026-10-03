"""标注页的后端：动作片段（spec 2026-10-01-gesture-labeling-training §3）和人物外形裁图（`FormLabels`，感知层第二层）。

动作页：数据目录 root 下 `_unlabeled/` 刚切的、每个类别一个目录、`_discard/` 丢弃的；每个片段是一个文件夹（16 张 jpg + 可选 claude.json）。
外形页：`datasets/attrs/` 下 `_unlabeled/`、`form/<类别>/`、`_discard/` 里的 jpg 文件。
确认 / 改类别 / 丢弃 = 把条目挪到目标目录（`_Moves`，两页共用），操作记在 `_labels.jsonl`（撤销也记一条）。
条目名永远不拼进路径：先遍历合法目录比对名字，找得到才用。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..vision.attrs import FORMS
from ..vision.gesture import recording_of
from ..vision.gesture_label import BLIND_FILE, load_guess

UNLABELED, DISCARD, LOG = "_unlabeled", "_discard", "_labels.jsonl"
FRAMES = 16
FORM_DIR = "form"
# 外形页出按钮的类别（10-04：认人阶段只分这三类；先祖 / 共享空间 / 变身训练时不够数本来就并掉，筛选里照旧列出）
FORM_BUTTONS = ("not_person", "lit", "unlit")
CONTEXT_W = 480  # 外形页的原图缩略宽度
_LOCK = threading.Lock()  # 每个请求现建一个 GestureLabels / FormLabels，所以锁放模块级：label / undo 的 找→挪→记账 整段串行


class _Moves:
    """挪条目 + `_labels.jsonl` 记账 + 撤销（动作页的条目是文件夹、外形页的是 jpg 文件）。

    子类给：root、`_places()`（所有位置名）、`_dir(where)`（位置对应的目录）、`_is_entry(path)`、`_key`（记账里条目名的字段）。
    持锁是调用方（子类的 label / undo）的事。
    """

    _key = "clip"

    def _places(self) -> list[str]:
        raise NotImplementedError

    def _dir(self, where: str) -> Path:
        return self.root / where

    def _is_entry(self, p: Path) -> bool:
        return p.is_dir()

    def _find(self, name: str) -> str | None:
        """条目当前所在的位置（遍历比对，不拼请求里的路径）。"""
        if not isinstance(name, str):
            return None
        for d in self._places():
            base = self._dir(d)
            if not base.is_dir():
                continue
            for p in base.iterdir():
                if p.name == name and self._is_entry(p):
                    return d
        return None

    def _names(self, where: str) -> list[str]:
        base = self._dir(where)
        return sorted(p.name for p in base.iterdir() if self._is_entry(p)) if base.is_dir() else []

    def _log(self, entry: dict) -> None:
        with (self.root / LOG).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _logged(self, entry: dict, name: str, src: str, dst: str) -> bool:
        """写记录；写不进去就把刚挪的条目挪回 src（记录和目录不能对不上）。"""
        try:
            self._log(entry)
            return True
        except OSError:
            self._move(name, dst, src)
            return False

    def _move(self, name: str, src: str, dst: str) -> tuple[int, str]:
        if src == dst:
            return 409, "已经在这个类别里了"
        target = self._dir(dst)
        if (target / name).exists():
            return 409, "目标目录里已经有同名条目"
        try:
            target.mkdir(parents=True, exist_ok=True)
            # 同一个数据目录（同一个盘）里直接改名：要么挪过去、要么原地不动。shutil.move 改名失败会退回 复制 + 删除，
            # Windows 上文件被占用时可能删一半，条目两边都有
            (self._dir(src) / name).rename(target / name)
        except OSError as exc:
            return 409, f"挪不动：{exc}"
        return 200, ""

    def _relabel(self, name: str, dst: str) -> tuple[int, dict]:
        """把条目挪到 dst 并记账。成功 (200, {})，失败 (状态码, {"ok": False, "text": …})。"""
        src = self._find(name)
        if src is None:
            return 409, {"ok": False, "text": "找不到这一条（可能被手动挪走了），刷新一下"}
        code, text = self._move(name, src, dst)
        if code != 200:
            return code, {"ok": False, "text": text}
        if not self._logged({"t": time.time(), self._key: name, "from": src, "to": dst}, name, src, dst):
            return 500, {"ok": False, "text": "记录写不进去，已挪回原处"}
        return 200, {}

    def _entries(self) -> list[dict]:
        try:
            lines = (self.root / LOG).read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict):
                out.append(e)
        return out

    def _revert(self) -> tuple[int, dict, str]:
        """撤销最近一条没被抵消的操作。返回 (状态码, body, 条目名)：成功时 body = {"where": 回到的位置}，失败时是错误 body。"""
        stack: list[dict] = []  # 撤销记录抵消它前面最近一条没被抵消的操作
        for e in self._entries():
            if e.get("undo"):
                if stack:
                    stack.pop()
            else:
                stack.append(e)
        if not stack:
            return 409, {"ok": False, "text": "没有可撤销的操作"}, ""
        e = stack[-1]
        name, src, dst = e.get(self._key), e.get("to"), e.get("from")
        places = self._places()
        if src not in places or dst not in places or self._find(name) != src:
            return 409, {"ok": False, "text": "已经不在原来的位置（可能被手动挪过），撤销不了"}, ""
        code, text = self._move(name, src, dst)
        if code != 200:
            return code, {"ok": False, "text": text}, ""
        if not self._logged({"t": time.time(), self._key: name, "from": src, "to": dst, "undo": True}, name, src, dst):
            return 500, {"ok": False, "text": "记录写不进去，已挪回原处"}, ""
        return 200, {"where": dst}, name


class GestureLabels(_Moves):
    def __init__(self, root: Path, labels: list[str], names: dict[str, str] | None = None) -> None:
        self.root, self.labels, self.names = Path(root), list(labels), dict(names or {})

    def _places(self) -> list[str]:
        return [UNLABELED, *self.labels, DISCARD]

    def _item(self, clip: str, where: str) -> dict:
        d = self.root / where / clip
        g = load_guess(d, BLIND_FILE) or load_guess(d)  # 不看录像名的那份更可信（看名字时 Claude 常照着名字判）
        guess = {"label": g.label, "confidence": g.confidence, "reason": g.reason} if g else None
        return {"clip": clip, "recording": recording_of(clip), "where": where, "guess": guess}

    def state(self) -> dict:
        if not self.root.is_dir():
            return {"ok": False, "text": "还没有片段：先跑 perception clips"}
        counts: dict[str, int] = {}
        clips: list[dict] = []
        for d in self._places():
            names = self._names(d)
            counts[d] = len(names)
            clips.extend(self._item(n, d) for n in names)
        return {"ok": True, "labels": self.labels, "names": self.names, "counts": counts, "clips": clips}

    def frame(self, clip: str, i: int) -> bytes | None:
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < FRAMES:
            return None
        where = self._find(clip)
        if where is None:
            return None
        files = sorted((self.root / where / clip).glob("*.jpg"))
        if i >= len(files):
            return None
        try:
            return files[i].read_bytes()
        except OSError:
            return None

    def label(self, clip: str, to: str) -> tuple[int, dict]:
        with _LOCK:
            return self._label(clip, to)

    def _label(self, clip: str, to: str) -> tuple[int, dict]:
        dst = DISCARD if to == "discard" else to if to in self.labels else None
        if dst is None:
            return 400, {"ok": False, "text": f"不认识的类别：{to}"}
        code, body = self._relabel(clip, dst)
        return (200, {"ok": True, **self._item(clip, dst)}) if code == 200 else (code, body)

    def undo(self) -> tuple[int, dict]:
        with _LOCK:
            code, body, clip = self._revert()
            return (200, {"ok": True, **self._item(clip, body["where"])}) if code == 200 else (code, body)


class FormLabels(_Moves):
    """外形页：`root/_unlabeled/*.jpg` 待确认、`root/form/<类别>/*.jpg` 已确认、`root/_discard/*.jpg` 丢弃；
    `_crops.jsonl` 每行记着裁图的来源（原图路径、框），`_unlabeled/claude.json` 是 Claude 的初分。"""

    _key = "crop"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _places(self) -> list[str]:
        return [UNLABELED, *FORMS, DISCARD]

    def _dir(self, where: str) -> Path:
        return self.root / (where if where in (UNLABELED, DISCARD) else f"{FORM_DIR}/{where}")

    def _is_entry(self, p: Path) -> bool:
        return p.suffix.lower() == ".jpg" and p.is_file()  # _unlabeled/claude.json 之类不算裁图

    def _rows(self) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        try:
            lines = (self.root / "_crops.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError:
            return rows
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and isinstance(r.get("crop"), str):
                rows[r["crop"]] = r
        return rows

    def _guesses(self) -> dict:
        try:
            d = json.loads((self.root / UNLABELED / "claude.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return d if isinstance(d, dict) else {}

    @staticmethod
    def _guess(g) -> dict | None:
        if not isinstance(g, dict) or not isinstance(g.get("label"), str):
            return None
        try:
            conf = float(g.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        return {"label": g["label"], "confidence": conf, "reason": str(g.get("reason") or "")}

    def _item(self, crop: str, where: str, rows: dict, guesses: dict) -> dict:
        r = rows.get(crop) or {}
        return {"crop": crop, "where": where, "guess": self._guess(guesses.get(crop)), "image": r.get("image"), "box": r.get("box")}

    def state(self) -> dict:
        if not self.root.is_dir():
            return {"ok": False, "text": "还没有裁图：先跑 perception crops"}
        rows, guesses = self._rows(), self._guesses()
        counts: dict[str, int] = {}
        items: list[dict] = []
        for d in self._places():
            names = self._names(d)
            counts[d] = len(names)
            items.extend(self._item(n, d, rows, guesses) for n in names)
        return {"ok": True, "forms": list(FORMS), "buttons": list(FORM_BUTTONS), "counts": counts, "items": items}

    def crop(self, name: str) -> bytes | None:
        where = self._find(name)
        if where is None:
            return None
        try:
            return (self._dir(where) / name).read_bytes()
        except OSError:
            return None

    def context(self, name: str) -> bytes | None:
        """裁图的来源原图：缩到宽 480、画出框（路径取自 `_crops.jsonl`，不取自请求）。"""
        if self._find(name) is None:
            return None
        r = self._rows().get(name)
        if not r or not isinstance(r.get("image"), str):
            return None
        try:
            img = cv2.imdecode(np.fromfile(r["image"], np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            return None
        if img is None:
            return None
        s = CONTEXT_W / img.shape[1]
        img = cv2.resize(img, (CONTEXT_W, max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        box = r.get("box")
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
            x, y, bw, bh = (v * s for v in box)
            cv2.rectangle(img, (round(x), round(y)), (round(x + bw), round(y + bh)), (80, 80, 255), 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def label(self, name: str, to: str) -> tuple[int, dict]:
        dst = DISCARD if to == "discard" else to if to in FORMS else None
        if dst is None:
            return 400, {"ok": False, "text": f"不认识的类别：{to}"}
        with _LOCK:
            code, body = self._relabel(name, dst)
        return (200, {"ok": True, **self._item(name, dst, self._rows(), self._guesses())}) if code == 200 else (code, body)

    def undo(self) -> tuple[int, dict]:
        with _LOCK:
            code, body, name = self._revert()
        return (200, {"ok": True, **self._item(name, body["where"], self._rows(), self._guesses())}) if code == 200 else (code, body)
