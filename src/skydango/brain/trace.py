"""大脑活动记录：在内存里记最近几十轮大脑（常驻 Claude Code）的来龙去脉，给可视化网页的大脑时间线用（/brain）。

- 数据来源：Brain.wake / farewell 开一轮（叫醒原因 + 发出去的消息）、收尾（result 或错误）；
  BrainSession.on_message 的每条 stream-json 输出拆成步骤（思考 / 说的话 / 调工具 / 工具返回）
- 只在 `run --brain --view` 时创建，不写盘；设计见 docs/superpowers/specs/2026-09-28-viewer-brain-trace-design.md
- 每个入口都兜住异常：监控出问题不能影响大脑本身
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable

log = logging.getLogger(__name__)

MAX_TURNS = 50  # 最多留几轮
MAX_CHARS = 20000  # 单条文本最多留几个字
TOOL_PREFIX = "mcp__sky__"
INTERRUPTED = "被打断（下一轮开始了）"


def _clip(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    return f"{text[:MAX_CHARS]}（截断，原长 {len(text)} 字）"


def _result_text(content) -> str:
    """tool_result 的 content：字符串，或文字 / 图片块的列表（图片只记个占位）。"""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image":
            parts.append("[图片]")
        elif block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(parts)


def _steps(m: dict) -> list[dict]:
    """一条 stream-json 消息 → 步骤列表；不关心的类型返回空。"""
    kind = m.get("type")
    if kind not in ("assistant", "user"):
        return []
    content = (m.get("message") or {}).get("content")
    if not isinstance(content, list):
        return []
    steps: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if kind == "assistant" and t == "thinking" and (block.get("thinking") or "").strip():
            steps.append({"kind": "thinking", "text": _clip(block["thinking"].strip())})
        elif kind == "assistant" and t == "text" and (block.get("text") or "").strip():
            steps.append({"kind": "text", "text": _clip(block["text"].strip())})
        elif kind == "assistant" and t == "tool_use":
            name = str(block.get("name") or "")
            steps.append({"kind": "tool", "id": block.get("id"), "name": name.removeprefix(TOOL_PREFIX), "input": block.get("input")})
        elif kind == "user" and t == "tool_result":
            steps.append({
                "kind": "result",
                "tool_use_id": block.get("tool_use_id"),
                "text": _clip(_result_text(block.get("content"))),
                "error": bool(block.get("is_error")),
            })
    return steps


def _result(result: dict) -> dict:
    usage = result.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    return {
        "subtype": result.get("subtype"),
        "num_turns": result.get("num_turns"),
        "cost": result.get("total_cost_usd"),  # 订阅不按这个收费，只做参考
        "tokens": {
            "input": usage.get("input_tokens"),
            "output": usage.get("output_tokens"),
            "cache_read": usage.get("cache_read_input_tokens"),
            "cache_write": usage.get("cache_creation_input_tokens"),
            "thinking": details.get("thinking_tokens"),
        },
    }


class BrainTrace:
    def __init__(self, wall: Callable[[], float] = time.time) -> None:
        self.wall = wall
        self.state: Callable[[], dict] | None = None  # 栏头的总体状态（Brain.trace_state）
        self._cond = threading.Condition()
        self._turns: deque[dict] = deque()
        self._outside: dict | None = None  # 两轮之间收到的消息（id = 0）
        self._current: dict | None = None
        self._next_id = 1
        self._version = 0
        self._oldest: int | None = None
        self.boot = uuid.uuid4().hex[:8]  # 每个进程不同：网页发现变了就清空重来（新进程的版本号可能已经超过网页记的）

    # ---- 大脑这边 ----
    def begin(self, reason: str, prompt: str) -> None:
        try:
            with self._cond:
                if self._current is not None:
                    self._close(self._current, error=INTERRUPTED)
                turn = self._new(self._next_id, reason, prompt)
                self._next_id += 1
                self._turns.append(turn)
                while len(self._turns) > MAX_TURNS:
                    self._turns.popleft()
                self._oldest = self._turns[0]["id"]
                self._current = turn
                self._touch(turn)
        except Exception:
            log.debug("大脑记录出错（begin）", exc_info=True)

    def feed(self, message: dict) -> None:
        try:
            steps = _steps(message) if isinstance(message, dict) else []
            if not steps:
                return
            with self._cond:
                turn = self._current
                if turn is None:
                    if self._outside is None:
                        self._outside = self._new(0, "outside", "")
                    turn = self._outside
                turn["steps"] += steps
                turn["tools"] += [s["name"] for s in steps if s["kind"] == "tool"]
                self._touch(turn)
        except Exception:
            log.debug("大脑记录出错（feed）", exc_info=True)

    def finish(self, result: dict, seconds: float) -> None:
        try:
            with self._cond:
                if self._current is not None:
                    self._current["result"] = _result(result or {})
                    self._close(self._current, seconds=seconds)
        except Exception:
            log.debug("大脑记录出错（finish）", exc_info=True)

    def fail(self, error: str, seconds: float) -> None:
        try:
            with self._cond:
                if self._current is not None:
                    self._close(self._current, seconds=seconds, error=error)
        except Exception:
            log.debug("大脑记录出错（fail）", exc_info=True)

    def note(self, text: str) -> None:
        """大脑会话自己的一件事（OpenAI 兼容大脑的历史压缩换上 / 失败）：当前轮里多一步，没有当前轮记进「轮外」。"""
        try:
            with self._cond:
                turn = self._current
                if turn is None:
                    if self._outside is None:
                        self._outside = self._new(0, "outside", "")
                    turn = self._outside
                turn["steps"].append({"kind": "compact", "text": _clip(str(text))})
                self._touch(turn)
        except Exception:
            log.debug("大脑记录出错（note）", exc_info=True)

    def chain(self, fn: Callable[[dict], None]) -> Callable[[dict], None]:
        """BrainSession.on_message 用：先交给 fn（打日志），再记下来；fn 出错也照样记，且不往上抛（别打断大脑这一轮）。"""

        def handler(m: dict) -> None:
            try:
                fn(m)
            except Exception:
                log.debug("on_message 出错", exc_info=True)
            self.feed(m)

        return handler

    # ---- 网页这边 ----
    def since(self, after: int, timeout: float) -> dict:
        """等到有比 after 新的变化（最多 timeout 秒），返回这之后变过的轮次 + 最新的总体状态；没变化时 turns 为空。"""
        with self._cond:
            if after > self._version:  # 网页记的版本号比这边大：程序重启过，从头来
                after = 0
            self._cond.wait_for(lambda: self._version > after, timeout)
            pool = ([self._outside] if self._outside is not None else []) + list(self._turns)
            turns = copy.deepcopy([t for t in pool if t["updated"] > after])
            version, oldest, count = self._version, self._oldest, self._next_id - 1
        state: dict = {}
        if self.state is not None:
            try:
                state = dict(self.state())
            except Exception:
                log.debug("大脑状态读不到", exc_info=True)
        state["turns"] = count
        return {"boot": self.boot, "version": version, "oldest": oldest, "state": state, "turns": turns}

    # ---- 内部（都在锁里调） ----
    def _new(self, turn_id: int, reason: str, prompt: str) -> dict:
        return {
            "id": turn_id, "reason": reason, "start": self.wall(), "end": None, "seconds": None,
            "prompt": _clip(prompt), "steps": [], "tools": [], "result": None, "error": None, "updated": 0,
        }

    def _close(self, turn: dict, seconds: float | None = None, error: str | None = None) -> None:
        turn["end"] = self.wall()
        turn["seconds"] = seconds
        turn["error"] = error
        self._current = None
        self._touch(turn)

    def _touch(self, turn: dict) -> None:
        self._version += 1
        turn["updated"] = self._version
        self._cond.notify_all()
