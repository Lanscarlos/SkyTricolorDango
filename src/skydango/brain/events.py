"""身体 → 大脑的事件队列：身体线程往里放，大脑线程醒来一次取走。"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    kind: str  # chat / owner_command / arrive / leave / request / accepted / holding / released / scene_change / panel / error / fallback / dropped
    text: str  # 给大脑看的一行
    t: float
    count: int = 1  # 连着重复了几次（同一种错误连续出现只占一行）

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

    def subscribe(self, fn: Callable[[str], None]) -> None:
        """每放一个事件就调 fn(kind)（在放事件的线程里调）；眼睛用它知道有人来了、画面变了。"""
        self._listeners.append(fn)

    def put(self, kind: str, text: str) -> None:
        with self._cond:
            now = self.clock()
            last = self._items[-1] if self._items else None
            if last is not None and last.kind == kind and last.text == text:
                self._items[-1] = replace(last, count=last.count + 1, t=now)
            else:
                self._items.append(Event(kind, text, now))
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
