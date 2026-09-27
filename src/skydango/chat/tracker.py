"""消息去重：同一个气泡会在屏幕上停留好几秒、被截到很多次，OCR 结果还会有细微抖动。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """只用于比较：全半角统一、去空白和标点、转小写。"""
    return _STRIP.sub("", unicodedata.normalize("NFKC", text)).lower()


def similar(a: str, b: str, threshold: float) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    short, long_ = sorted((na, nb), key=len)
    # 气泡淡出 / 被遮挡时 OCR 只读到一部分
    if short in long_ and len(short) >= 0.6 * len(long_):
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


@dataclass
class _Entry:
    text: str
    last_seen: float


class SeenTracker:
    """判断一句话是不是“新”的。只要它还在屏幕上被持续看到，就一直算旧的。"""

    def __init__(self, ttl: float, threshold: float) -> None:
        self.ttl = ttl
        self.threshold = threshold
        self._entries: list[_Entry] = []

    def observe(self, text: str, now: float) -> bool:
        self._entries = [e for e in self._entries if now - e.last_seen <= self.ttl]
        for entry in self._entries:
            if similar(entry.text, text, self.threshold):
                entry.last_seen = now
                if len(text) > len(entry.text):
                    entry.text = text
                return False
        self._entries.append(_Entry(text, now))
        return True


class SelfFilter:
    """记住自己刚发出去的话，避免读到自己头顶的气泡再回复自己。"""

    def __init__(self, window: float, threshold: float, prefix: str = "") -> None:
        self.window = window
        self.threshold = threshold
        self.prefix = prefix
        self._sent: list[tuple[str, float]] = []

    def remember(self, text: str, now: float) -> None:
        self._sent.append((text, now))

    def is_self(self, text: str, now: float) -> bool:
        self._sent = [(t, ts) for t, ts in self._sent if now - ts <= self.window]
        candidates = [text]
        if self.prefix and normalize(self.prefix) and normalize(text).startswith(normalize(self.prefix)):
            candidates.append(text.replace(self.prefix, "", 1))
        return any(similar(sent, c, self.threshold) for sent, _ in self._sent for c in candidates)
