"""聊天记录（沙盒 brain-sandbox spec §3；真机 console-live-page spec §3.2）：内存里最近 limit 行，每行 {seq, t, kind, who, text}。

kind：heard（听到的 / 冒充的发言）/ said（团子说的）/ act（动作、走路、气泡）/ event（来了走了、快进、反思、心里……）/ blocked（被护栏拦下的话）。
`cond` 在 add 时 notify_all，给沙盒 /state 和 viewer /chat 长轮询用。
"""

from __future__ import annotations

import threading
from collections import deque

KINDS = ("heard", "said", "act", "event", "blocked")


class Transcript:
    def __init__(self, clock, limit: int = 500) -> None:
        self.clock = clock  # 有 wall() 的东西：沙盒是 SimClock（沙盒墙上时间），真机是包着身体墙钟的小对象
        self.cond = threading.Condition()
        self.version = 0
        self._rows: deque[dict] = deque(maxlen=limit)

    def add(self, kind: str, text: str, who: str = "", why: str = "") -> dict:
        """why：blocked 行被拦下的原因（有才带这个键）。"""
        if kind not in KINDS:
            raise ValueError(f"不认识的聊天记录类型：{kind}")
        with self.cond:
            self.version += 1
            row = {"seq": self.version, "t": self.clock.wall(), "kind": kind, "who": who, "text": text}
            if why:
                row["why"] = why
            self._rows.append(row)
            self.cond.notify_all()
        return row

    def since(self, seq: int) -> list[dict]:
        """seq 之后的行（最多 limit 行，更早的已经丢了）。"""
        with self.cond:
            return [dict(r) for r in self._rows if r["seq"] > seq]

    def wait_since(self, seq: int, timeout: float) -> tuple[int, list[dict]]:
        """长轮询：seq 之后有新行马上返回，没有就最多等 timeout 秒。seq 比最新的还大（这边重启过）当 0。返回 (最新 seq, 新行)。"""
        with self.cond:
            if seq > self.version:
                seq = 0
            self.cond.wait_for(lambda: self.version > seq, max(0.0, timeout))
            return self.version, [dict(r) for r in self._rows if r["seq"] > seq]


def event_line(kind: str, text: str, who: str = "") -> str | None:
    """真机聊天记录里的分隔线：身体放给大脑的事件里，值得在聊天旁边看到的几种（来去、牵手、黑屏）；别的返回 None。"""
    if kind == "arrive":
        return f"── {who or text} 来到身边 ──"
    if kind == "leave":
        return f"── {who} 走开了 ──" if who else f"── {text} ──"
    if kind == "return" or (kind == "lull" and who):  # lull 带 who = 冷场后他回来了
        return f"── {who} 回来了 ──" if who else f"── {text} ──"
    if kind in ("holding", "released", "stranger_back"):
        return f"── {text} ──"
    if kind == "scene_change":
        if text.startswith("画面整屏黑了"):
            return "── 画面黑了（可能在切场景） ──"
        if text.startswith("画面恢复了"):
            return "── 画面恢复了 ──"
    return None
