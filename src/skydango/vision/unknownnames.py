"""没认出的名字（三期 §5）：名字标签读得很清楚、但 friends.md 里没有的（新加的好友、改了昵称），记下来给用户看。

    runs/<这次>/unknown_names/names.jsonl   每行一个名字：出现次数、第一次 / 最后一次时间、裁剪图文件名
    runs/<这次>/unknown_names/001.jpg       每个名字一张名字标签的裁剪图

`perception unknown-names` 汇总最近几次运行。**只列出，不自动写 friends.md**：用户看了决定要不要加。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..chat.tracker import normalize, similar
from ..imageio import imwrite

log = logging.getLogger(__name__)

SAME = 0.75  # 和已记下的名字这么像就算同一个（OCR 偶尔多读 / 少读一个字）
NAME_MAX = 12  # 名字最多这么多字（去掉空白和符号后）；头顶聊天气泡常比这长
PUNCT = set("，。！？!?、：:;；…~～\"“”‘’'『』「」()（）《》,.-－—")  # 名字里基本不会有；气泡、面板的行（"话 - 名字"）、系统提示常有
SYSTEM_WORDS = ("点亮了", "陌生人", "遇境", "心火", "留影", "延迟", "屏蔽", "解锁",  # 系统提示、面板、界面按钮里的字
                "关注", "客服", "退出", "错误", "回忆", "击掌", "鞠躬")
UI_EXACT = {"举报", "商店", "活动", "邮箱", "好友码", "好友"}  # 整个就是这几个字的
CHAT_CHARS = set("你我")  # 名字里很少有，头顶聊天气泡里到处都是
PARTICLES = set("啊呀吧呢吗啦嘛哦哈呐的了")  # 气泡常以语气词结尾
FRAGMENT = 3  # 和好友名有这么长的一段一样（又不是他）：气泡和名字标签叠在一起读出来的
_TIME = re.compile(r"^(上午|下午|凌晨|早上|中午|晚上)?\s*\d{1,2}\s*[:：]\s*\d{2}$")
_RUN_NAME_PREFIX = 8  # 运行目录名以 YYYYmmdd 开头


def _stamp(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def looks_like_name(text: str, friends: list[str]) -> bool:
    """像不像一个名字标签（10-02 攒下的"没认出的名字"几乎全是头顶聊天气泡、系统提示、聊天面板的行）：
    太短 / 太长、带标点或"-"、有系统 / 界面用语、是时间、带"你 / 我"或语气词结尾（气泡）、
    是好友名的一截或者和好友名有一大段一样（气泡和名字标签读成了一行）的都不算。宁可漏记：这只是给用户看的清单。"""
    text = text.strip()
    norm = normalize(text)
    if not 2 <= len(norm) <= NAME_MAX:
        return False
    if any(c in PUNCT for c in text) or _TIME.match(text) or any(w in text for w in SYSTEM_WORDS) or norm in UI_EXACT:
        return False
    if any(c in CHAT_CHARS for c in norm) or norm[-1] in PARTICLES:
        return False
    for f in friends:
        f = normalize(f)
        if len(f) < 2 or norm == f:
            continue
        if norm in f or _common(norm, f) >= min(FRAGMENT, len(f)):  # 好友名的一截 / 和好友名有一大段一样
            return False
    return True


def _common(a: str, b: str) -> int:
    """a、b 最长的公共连续片段有几个字。"""
    best = 0
    for i in range(len(a)):
        for j in range(i + best + 1, len(a) + 1):
            if a[i:j] in b:
                best = j - i
            else:
                break
    return best


class UnknownNames:
    def __init__(self, folder: Path, names: Callable[[], list[str]], wall: Callable[[], float] = time.time) -> None:
        self.folder = folder
        self.names = names
        self.wall = wall
        self.entries: dict[str, dict] = {}
        self._lock = threading.Lock()

    def add(self, text: str, crop: np.ndarray) -> None:
        text = text.strip()
        friends = self.names()
        if not looks_like_name(text, friends) or any(similar(text, n, SAME) for n in friends):
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
