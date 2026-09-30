"""沙盒的聊天记录（brain-sandbox spec §3）：内存里最近 limit 行，每行 {seq, t, kind, who, text}。

kind：heard（冒充的发言）/ said（团子说的）/ act（动作、走路、气泡）/ event（来了走了、快进、反思……）/ blocked（被护栏拦下的话）。
`cond` 在 add 时 notify_all，给 /state 长轮询用。
"""

from __future__ import annotations

import threading
from collections import deque

KINDS = ("heard", "said", "act", "event", "blocked")


class Transcript:
    def __init__(self, clock, limit: int = 500) -> None:
        self.clock = clock  # 有 wall() 的东西（SimClock）：行上记沙盒墙上时间
        self.cond = threading.Condition()
        self.version = 0
        self._rows: deque[dict] = deque(maxlen=limit)

    def add(self, kind: str, text: str, who: str = "") -> dict:
        if kind not in KINDS:
            raise ValueError(f"不认识的聊天记录类型：{kind}")
        with self.cond:
            self.version += 1
            row = {"seq": self.version, "t": self.clock.wall(), "kind": kind, "who": who, "text": text}
            self._rows.append(row)
            self.cond.notify_all()
        return row

    def since(self, seq: int) -> list[dict]:
        """seq 之后的行（最多 limit 行，更早的已经丢了）。"""
        with self.cond:
            return [dict(r) for r in self._rows if r["seq"] > seq]
