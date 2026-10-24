"""团子跑道（管理面板「真机团子」页中栏）：身体和大脑每走一步记一条结构化事件，给 viewer /flow 长轮询。

一条事件：{id, t, node, label, outcome, speaker?, value?, model?, cost?}，没有的字段不写。
- node：跑道上的十个节点（NODES），outcome：ok / skip / unsure / ask / sent
- 大脑一轮（begin_turn ~ end_turn）里身体发的「选一选 / 能不能 / 发出去」（decision=True）先攒着，
  这一轮结束时排在「想一想（想完了）」「想不想」后面一起放出来：跑道上的顺序是想完了才决定说不说
- 用量（UsageMeter 的监听）：眼睛的一笔 = 一次「看一眼」；大脑 / 压缩的钱记到这一轮「想完了」上；别的用处不上跑道
- 没有判定数字的地方（跟谁说是规则、能不能发是护栏）只给结果，不编 value
只在内存里、最近 limit 条；所有入口由调用方兜住异常，出错不影响团子本身。
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from collections.abc import Callable

NODES = ("look", "has_new", "read", "to_me", "batch", "reply_or_not", "think", "say_or_act", "can_send", "send")
OUTCOMES = ("ok", "skip", "unsure", "ask", "sent")
LIMIT = 300
TURN_COST = ("brain", "recap")  # 记到这一轮「想完了」上的用处
DONT = "不说："


def _quiet_reason(text: str) -> str:
    """大脑这一轮最后的文字里「不说：原因」那一行；没有就是「这轮没开口」。"""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith(DONT):
            return line[:60]
    return "这轮没开口"


class FlowLog:
    def __init__(self, wall: Callable[[], float], limit: int = LIMIT) -> None:
        self.wall = wall
        self.boot = uuid.uuid4().hex[:8]  # 每个进程不同：网页发现变了就清空重来
        self._cond = threading.Condition()
        self._events: deque[dict] = deque(maxlen=limit)
        self._version = 0
        self._turn: list[dict] | None = None  # 大脑这一轮里攒着的决定；None = 不在一轮里
        self._turn_cost: float | None = None

    def emit(self, node: str, label: str, outcome: str, *, speaker: str | None = None, value=None,
             model: str | None = None, cost: float | None = None, decision: bool = False) -> dict:
        if node not in NODES:
            raise ValueError(f"跑道上没有这个节点：{node}")
        if outcome not in OUTCOMES:
            raise ValueError(f"不认识的结果：{outcome}")
        e: dict = {"node": node, "label": str(label), "outcome": outcome}
        for k, v in (("speaker", speaker), ("value", value), ("model", model), ("cost", cost)):
            if v is not None and v != "":
                e[k] = v
        with self._cond:
            if decision and self._turn is not None:
                self._turn.append(e)
                return e
            return self._add(e)

    # ---- 大脑一轮 ----
    def begin_turn(self) -> None:
        with self._cond:
            self._flush()  # 上一轮没收尾（被打断）：先放出去
            self._add({"node": "think", "label": "在想…", "outcome": "ok", "model": "brain"})
            self._turn, self._turn_cost = [], None

    def end_turn(self, seconds: float, text: str, error: str = "") -> None:
        with self._cond:
            held = self._turn or []
            done: dict = {"node": "think", "outcome": "skip" if error else "ok", "model": "brain",
                          "label": f"这一轮失败：{error}"[:60] if error else f"想了 {seconds:.1f} 秒"}
            if self._turn_cost is not None:
                done["cost"] = round(self._turn_cost, 6)
            self._turn, self._turn_cost = None, None
            self._add(done)
            if not error:
                acted = next((e for e in held if e["node"] == "say_or_act"), None)
                if acted is not None:
                    self._add({"node": "reply_or_not", "label": "回", "outcome": "ok"})
                else:
                    self._add({"node": "reply_or_not", "label": _quiet_reason(text), "outcome": "skip"})
            for e in held:
                self._add(e)

    # ---- 用量（UsageMeter.listeners） ----
    def on_usage(self, use: str, provider: str, model: str, cost: float | None, ok: bool) -> None:
        if use == "eyes":
            self.emit("look", "看了一眼" if ok else "没看成", "ok" if ok else "skip", model="eye", cost=cost)
        elif use in TURN_COST and cost is not None:
            with self._cond:
                if self._turn is not None:
                    self._turn_cost = (self._turn_cost or 0.0) + cost

    # ---- 网页这边 ----
    def wait_since(self, seq: int, timeout: float) -> tuple[str, int, list[dict]]:
        """长轮询：seq 之后有新事件马上返回，没有就最多等 timeout 秒。seq 比最新的还大（这边重启过）当 0。"""
        with self._cond:
            if seq > self._version:
                seq = 0
            self._cond.wait_for(lambda: self._version > seq, max(0.0, timeout))
            return self.boot, self._version, [dict(e) for e in self._events if e["id"] > seq]

    # ---- 内部（锁里） ----
    def _add(self, e: dict) -> dict:
        self._version += 1
        row = {"id": self._version, "t": self.wall(), **e}
        self._events.append(row)
        self._cond.notify_all()
        return row

    def _flush(self) -> None:
        for e in self._turn or []:
            self._add(e)
        self._turn, self._turn_cost = None, None
