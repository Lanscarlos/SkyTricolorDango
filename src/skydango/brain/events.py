"""身体 → 大脑的事件队列：身体线程往里放，大脑线程醒来一次取走。"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace

log = logging.getLogger(__name__)

# 背景事件：身体已经处理了 / 只是周围在变，自己不叫醒大脑，攒着等下一次醒来一起给（最多等 brain.background_wait 秒）。
# 2026-09-29 真机 dry-run 5 分钟醒了 42 次，41 次是这些（陌生人数量来回跳、好友走出画面又回来、没接的互动请求反复报）
BACKGROUND = frozenset({"stranger", "leave", "return", "request", "accepted", "holding", "released", "scene_change", "dropped", "reflex",
                        "stranger_back", "outfit"})
CANCELS = {"leave": "return", "return": "leave"}  # 同一个人攒着的"走开"和"回来"互相抵消：大脑不用知道他离开过


@dataclass(frozen=True)
class Event:
    kind: str  # chat / owner_command / arrive / return / leave / stranger / approach / gesture / request / accepted / holding / released / scene_change / panel / error / fallback / dropped / task_done / task_failed / notice / stranger_back / outfit
    text: str  # 给大脑看的一行
    t: float
    count: int = 1  # 重复了几次（同一种错误连续出现只占一行；背景事件不挨着也合并）
    who: str = ""  # arrive / return / leave 的好友名字（抵消用）；outfit 也带

    def line(self) -> str:
        return self.text + (f"（×{self.count}）" if self.count > 1 else "")


class EventQueue:
    def __init__(self, limit: int = 200, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.clock = clock
        self.last_put = float("-inf")  # 大脑按它攒一小会儿再醒（连发的几句一起看）
        self._items: list[Event] = []
        self._dropped = 0
        self._cond = threading.Condition()
        self._listeners: list[Callable[[str], None]] = []
        self.history: deque[Event] = deque(maxlen=20)  # 最近的事件（可视化页面显示用，drain 不清）

    def subscribe(self, fn: Callable[[str], None]) -> None:
        """每放一个事件就调 fn(kind)（在放事件的线程里调）；眼睛用它知道有人来了、画面变了。"""
        self._listeners.append(fn)

    def put(self, kind: str, text: str, who: str = "") -> None:
        with self._cond:
            now = self.clock()
            last = self._items[-1] if self._items else None
            same = self._same_background(kind, text)
            opposite = self._find(lambda e: who and e.who == who and e.kind == CANCELS.get(kind))
            if opposite is not None:
                self._items.pop(opposite)
            elif kind == "stranger":  # 陌生人数量来回跳：只留最新的一条，时间留最早的（兜底叫醒不往后推）
                first = min((e.t for e in self._items if e.kind == "stranger"), default=now)
                self._items = [e for e in self._items if e.kind != "stranger"] + [Event(kind, text, first)]
                self.history.append(Event(kind, text, now))
            elif same is not None:  # 背景事件留第一次的时间：一直重复也不会把兜底叫醒往后推
                self._items[same] = replace(self._items[same], count=self._items[same].count + 1)
                if not (self.history and self.history[-1].kind == kind and self.history[-1].text == text):
                    self.history.append(Event(kind, text, now, who=who))
            elif last is not None and last.kind == kind and last.text == text:
                self._items[-1] = replace(last, count=last.count + 1, t=now)
                if self.history and self.history[-1] is last:
                    self.history[-1] = self._items[-1]
            else:
                self._items.append(Event(kind, text, now, who=who))
                self.history.append(self._items[-1])
                if len(self._items) > self.limit:  # 大脑很久没醒（离线 / 退避）：丢最旧的
                    self._items.pop(0)
                    self._dropped += 1
            self.last_put = now
            self._cond.notify_all()
        for fn in list(self._listeners):
            try:
                fn(kind)
            except Exception:
                log.exception("事件订阅者出错")

    def _find(self, pred) -> int | None:
        return next((i for i, e in enumerate(self._items) if pred(e)), None)

    def _same_background(self, kind: str, text: str) -> int | None:
        if kind not in BACKGROUND:
            return None
        return self._find(lambda e: e.kind == kind and e.text == text)

    def urgent(self) -> bool:
        """攒着的事件里有没有要马上叫醒大脑的。"""
        with self._cond:
            return any(e.kind not in BACKGROUND for e in self._items)

    def oldest_background(self) -> float | None:
        """最早那条还攒着的背景事件是什么时候放进来的（兜底叫醒用）。"""
        with self._cond:
            return min((e.t for e in self._items if e.kind in BACKGROUND), default=None)

    def recent(self, n: int) -> list[Event]:
        """最近 n 个事件（从旧到新；大脑取走的也在）。"""
        with self._cond:
            return list(self.history)[-n:]

    def drain(self) -> list[Event]:
        with self._cond:
            items, self._items = self._items, []
            if self._dropped:
                t = items[0].t if items else self.clock()
                items.insert(0, Event("dropped", f"事件太多，丢了 {self._dropped} 个较早的", t))
                self._dropped = 0
            return items

    def wait(self, timeout: float) -> bool:
        """等到有事件或超时；返回有没有事件。"""
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            return bool(self._items)

    def has(self, kind: str) -> bool:
        with self._cond:
            return any(e.kind == kind for e in self._items)

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
