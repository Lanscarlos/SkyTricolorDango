"""动作片段标注的后端（spec 2026-10-01-gesture-labeling-training §3）。

数据目录 root 下：`_unlabeled/` 刚切的、每个类别一个目录、`_discard/` 丢弃的；每个片段是一个文件夹（16 张 jpg + 可选 claude.json）。
确认 / 改类别 / 丢弃 = 把片段文件夹挪到目标目录，操作记在 `_labels.jsonl`（撤销也记一条）。
片段名永远不拼进路径：先遍历合法目录比对名字，找得到才用。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ..vision.gesture import recording_of
from ..vision.gesture_label import BLIND_FILE, load_guess

UNLABELED, DISCARD, LOG = "_unlabeled", "_discard", "_labels.jsonl"
FRAMES = 16
_LOCK = threading.Lock()  # 每个请求现建一个 GestureLabels，所以锁放模块级：label / undo 的 找→挪→记账 整段串行


class GestureLabels:
    def __init__(self, root: Path, labels: list[str], names: dict[str, str] | None = None) -> None:
        self.root, self.labels, self.names = Path(root), list(labels), dict(names or {})

    def _dirs(self) -> list[str]:
        return [UNLABELED, *self.labels, DISCARD]

    def _find(self, clip: str) -> str | None:
        """片段当前所在的目录名（遍历比对，不拼请求里的路径）。"""
        if not isinstance(clip, str):
            return None
        for d in self._dirs():
            base = self.root / d
            if not base.is_dir():
                continue
            for p in base.iterdir():
                if p.name == clip and p.is_dir():
                    return d
        return None

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
        for d in self._dirs():
            base = self.root / d
            names = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
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

    def _log(self, entry: dict) -> None:
        with (self.root / LOG).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _logged(self, entry: dict, clip: str, src: str, dst: str) -> bool:
        """写记录；写不进去就把刚挪的片段挪回 src（记录和目录不能对不上）。"""
        try:
            self._log(entry)
            return True
        except OSError:
            self._move(clip, dst, src)
            return False

    def _move(self, clip: str, src: str, dst: str) -> tuple[int, str]:
        if src == dst:
            return 409, "已经在这个类别里了"
        target = self.root / dst
        if (target / clip).exists():
            return 409, "目标目录里已经有同名片段"
        try:
            target.mkdir(parents=True, exist_ok=True)
            # 同一个数据目录（同一个盘）里直接改名：要么挪过去、要么原地不动。shutil.move 改名失败会退回 复制 + 删除，
            # Windows 上文件被占用时可能删一半，片段两边都有
            (self.root / src / clip).rename(target / clip)
        except OSError as exc:
            return 409, f"挪不动：{exc}"
        return 200, ""

    def label(self, clip: str, to: str) -> tuple[int, dict]:
        with _LOCK:
            return self._label(clip, to)

    def _label(self, clip: str, to: str) -> tuple[int, dict]:
        dst = DISCARD if to == "discard" else to if to in self.labels else None
        if dst is None:
            return 400, {"ok": False, "text": f"不认识的类别：{to}"}
        src = self._find(clip)
        if src is None:
            return 409, {"ok": False, "text": "找不到这个片段（可能被手动挪走了），刷新一下"}
        code, text = self._move(clip, src, dst)
        if code != 200:
            return code, {"ok": False, "text": text}
        if not self._logged({"t": time.time(), "clip": clip, "from": src, "to": dst}, clip, src, dst):
            return 500, {"ok": False, "text": "记录写不进去，已把片段挪回原处"}
        return 200, {"ok": True, **self._item(clip, dst)}

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

    def undo(self) -> tuple[int, dict]:
        with _LOCK:
            return self._undo()

    def _undo(self) -> tuple[int, dict]:
        stack: list[dict] = []  # 撤销记录抵消它前面最近一条没被抵消的操作
        for e in self._entries():
            if e.get("undo"):
                if stack:
                    stack.pop()
            else:
                stack.append(e)
        if not stack:
            return 409, {"ok": False, "text": "没有可撤销的操作"}
        e = stack[-1]
        clip, src, dst = e.get("clip"), e.get("to"), e.get("from")
        if src not in self._dirs() or dst not in self._dirs() or self._find(clip) != src:
            return 409, {"ok": False, "text": "片段已经不在原来的位置（可能被手动挪过），撤销不了"}
        code, text = self._move(clip, src, dst)
        if code != 200:
            return code, {"ok": False, "text": text}
        if not self._logged({"t": time.time(), "clip": clip, "from": src, "to": dst, "undo": True}, clip, src, dst):
            return 500, {"ok": False, "text": "记录写不进去，已把片段挪回原处"}
        return 200, {"ok": True, **self._item(clip, dst)}
