"""没认出的名字（三期 §5）：名字标签读得很清楚、但 friends.md 里没有的（新加的好友、改了昵称），记下来给用户看。

    runs/<这次>/unknown_names/names.jsonl   每行一个名字：出现次数、第一次 / 最后一次时间、裁剪图文件名
    runs/<这次>/unknown_names/001.jpg       每个名字一张名字标签的裁剪图

`perception unknown-names` 汇总最近几次运行。**只列出，不自动写 friends.md**：用户看了决定要不要加。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..chat.tracker import normalize, similar
from ..imageio import imwrite

log = logging.getLogger(__name__)

SAME = 0.75  # 和已记下的名字这么像就算同一个（OCR 偶尔多读 / 少读一个字）
_RUN_NAME_PREFIX = 8  # 运行目录名以 YYYYmmdd 开头


def _stamp(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


class UnknownNames:
    def __init__(self, folder: Path, names: Callable[[], list[str]], wall: Callable[[], float] = time.time) -> None:
        self.folder = folder
        self.names = names
        self.wall = wall
        self.entries: dict[str, dict] = {}
        self._lock = threading.Lock()

    def add(self, text: str, crop: np.ndarray) -> None:
        text = text.strip()
        if len(normalize(text)) < 2 or any(similar(text, n, SAME) for n in self.names()):
            return
        now = _stamp(self.wall())
        with self._lock:
            key = next((k for k in self.entries if similar(text, k, SAME)), None)
            if key is None:
                image = f"{len(self.entries) + 1:03d}.jpg"
                self.folder.mkdir(parents=True, exist_ok=True)
                imwrite(self.folder / image, crop)
                self.entries[text] = {"name": text, "count": 1, "first": now, "last": now, "image": image}
                log.info("读到一个不在好友名单里的名字：%s", text)
            else:
                entry = self.entries[key]
                entry["count"] += 1
                entry["last"] = now
            self._write()

    def _write(self) -> None:
        lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.entries.values())
        (self.folder / "names.jsonl").write_text(lines, encoding="utf-8")


def collect(runs_root: Path, last: int, friends: list[str]) -> list[dict]:
    """最近 last 次运行里记下的名字：同名合并（次数相加），去掉现在已经对得上好友的，按次数从多到少。"""
    if not runs_root.is_dir():
        return []
    runs = sorted(p for p in runs_root.iterdir() if p.is_dir() and p.name[:_RUN_NAME_PREFIX].isdigit())
    merged: dict[str, dict] = {}
    for run in runs[-max(1, last):]:
        path = run / "unknown_names" / "names.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = normalize(row.get("name", ""))
            if not key:
                continue
            image = str(run / "unknown_names" / row.get("image", ""))
            entry = merged.get(key)
            if entry is None:
                merged[key] = {**row, "image": image}
                continue
            entry["count"] += row.get("count", 0)
            if row.get("last", "") >= entry.get("last", ""):
                entry["last"], entry["image"] = row.get("last", ""), image
    out = [e for e in merged.values() if not any(similar(e["name"], f, SAME) for f in friends)]
    return sorted(out, key=lambda e: (-e["count"], e["name"]))
