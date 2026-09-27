"""大脑：常驻线程。有事件（攒一小会儿）或到了心跳就醒来，把事件和状态交给 Claude，执行它调用的工具，
直到它不再调工具或到了上限，然后接着睡。API 一直失败或花费超了，聊天交给身体的备用回复。"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

from ..chat.memory import format_date
from ..config import BrainConfig, ChatConfig
from .budget import Budget
from .context import Context, wake_message
from .events import Event, EventQueue
from .prompt import COMPACT_REQUEST
from .tools import ACTIONS, TOOLS

log = logging.getLogger(__name__)

BACKOFF = (10.0, 30.0, 60.0)  # 连续失败后隔多久再试
AUTO_LOOK_KINDS = {"arrive", "leave", "scene_change"}  # 这些事件发生时自动附一张截图


class Brain:
    def __init__(
        self,
        cfg: BrainConfig,
        chat: ChatConfig,
        client,
        context: Context,
        toolbox,
        events: EventQueue,
        budget: Budget,
        nearby: Callable[[float], list[str]],
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        run=None,  # runlog.RunDir
        store=None,  # chat.memory.MemoryStore：live 时把摘要记进 inbox.md
    ) -> None:
        self.cfg = cfg
        self.chat = chat
        self.client = client
        self.context = context
        self.toolbox = toolbox
        self.events = events
        self.budget = budget
        self.nearby = nearby
        self.clock = clock
        self.wall = wall
        self.run_dir = run
        self.store = store
        self.last_wake = float("-inf")  # 刚上线马上醒一次，看看周围
        self._idle = 0  # 连着几次醒来什么都没做（心跳逐档退后）
        self.failures = 0
        self.failing_since: float | None = None
        self.backoff_until = float("-inf")
        self.last_prompt_tokens = 0

    # ---- 什么时候醒 ----
    def offline(self, now: float) -> bool:
        """聊天要不要交给备用回复：花费到了暂停线，或 API 连续失败太久。"""
        if self.budget.level(now) == "paused":
            return True
        return self.failing_since is not None and now - self.failing_since >= self.cfg.offline_fallback

    def heartbeat(self, now: float) -> float:
        beats = self.cfg.heartbeat
        start = 0 if self.nearby(now) else 1  # 身边有好友时醒得勤一点
        return beats[min(start + self._idle, len(beats) - 1)]

    def due(self, now: float) -> str | None:
        if now < self.backoff_until:
            return None
        level = self.budget.level(now)
        if level == "paused":
            return None
        if len(self.events) and now - self.events.last_put >= self.chat.debounce:
            if level == "ok" or self.events.has("chat"):
                return "events"
        if level == "ok" and now - self.last_wake >= self.heartbeat(now):
            return "heartbeat"
        return None

    def run(self, stop: threading.Event) -> None:
        log.info("大脑上线（%s，effort=%s）", self.cfg.model, self.cfg.effort)
        while not stop.is_set():
            now = self.clock()
            reason = self.due(now)
            if reason is None:
                if len(self.events):  # 在攒消息或在退避：稍等再看
                    stop.wait(0.2)
                else:
                    self.events.wait(0.5)
                continue
            try:
                self.wake(now, reason)
            except Exception as exc:
                log.exception("大脑这一轮出错")
                self._failed(now, exc)
        log.info("大脑下线")

    # ---- 醒来一次 ----
    def wake(self, now: float, reason: str) -> None:
        self.last_wake = now
        level = self.budget.level(now)
        events = self.events.drain()
        images = self._auto_look(events, now) if level == "ok" else []
        status, _ = self.toolbox.run("status", {})
        w = self.wall()
        stamp = f"{format_date(w)} {time.strftime('%H:%M:%S', time.localtime(w))}"
        self.context.add_user(wake_message(stamp, events, str(status), images))
        acted = self._turn(now)
        self._idle = 0 if (reason == "events" or acted) else self._idle + 1
        self.context.prune()
        if self.last_prompt_tokens > self.cfg.compact_tokens:
            self.compact()

    def _auto_look(self, events: list[Event], now: float) -> list[dict]:
        body = self.toolbox.body
        if body.blackout:
            return []
        since = now - body.last_look
        wanted = since >= self.cfg.auto_look_max or (
            since >= self.cfg.auto_look_min and any(e.kind in AUTO_LOOK_KINDS for e in events)
        )
        if not wanted:
            return []
        content, error = self.toolbox.run("look", {})
        return [] if error else list(content)

    def _turn(self, now: float) -> bool:
        """调模型 → 执行工具 → 把结果给它 → ……，直到它不再调工具或到了上限。返回有没有用手脚工具。"""
        acted, calls, says = False, 0, 0
        while True:
            try:
                resp = self.client.create(self.context.system(), self.context.messages, TOOLS)
            except Exception as exc:
                self._failed(now, exc)
                return acted
            self._ok()
            usage = resp.get("usage") or {}
            cost = self.budget.record(usage, now)
            self.last_prompt_tokens = sum(
                usage.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
            )
            content = resp.get("content") or []
            self.context.add_assistant(content)
            self._log(resp, cost)
            stop = resp.get("stop_reason")
            if stop in ("refusal", "max_tokens"):
                log.warning("大脑这一轮没说完（%s），什么都不做", stop)
                self.context.drop_last_assistant()
                return acted
            uses = [b for b in content if b.get("type") == "tool_use"]
            if not uses:
                return acted
            results = []
            for use in uses:
                name, args = use.get("name"), use.get("input") or {}
                if calls >= self.cfg.max_steps:
                    out, err = "这一轮做的事够多了，先停下，下次醒来再说", True
                elif name == "say" and says >= self.cfg.max_says:
                    out, err = "这一轮已经说得够多了，别刷屏", True
                else:
                    calls += 1
                    out, err = self.toolbox.run(name, args)
                    if not err and name in ACTIONS:
                        acted = True
                    if not err and name == "say":
                        says += 1
                shown = out if isinstance(out, str) else "[图片]"
                log.info("大脑 → %s %s → %s", name, json.dumps(args, ensure_ascii=False), shown)
                result = {"type": "tool_result", "tool_use_id": use["id"], "content": out}
                if err:
                    result["is_error"] = True
                results.append(result)
            self.context.add_tool_results(results)
            if calls >= self.cfg.max_steps:
                return acted

    def compact(self) -> bool:
        """对话记录太长：让它写摘要，从摘要开一段新记录；live 时摘要也记进 inbox.md。退出时也调一次。"""
        if len(self.context.messages) < 2:
            return False
        self.context.add_user([{"type": "text", "text": COMPACT_REQUEST}])
        try:
            resp = self.client.create(self.context.system(), self.context.messages, TOOLS)
        except Exception as exc:
            log.warning("写摘要失败，下次再试：%s", exc)
            self.context.messages.pop()
            return False
        self.budget.record(resp.get("usage") or {}, self.clock())
        summary = "".join(b.get("text", "") for b in resp.get("content") or [] if b.get("type") == "text").strip()
        if not summary:
            self.context.messages.pop()
            return False
        self.context.reset(summary)
        self.last_prompt_tokens = 0
        if self.store is not None:
            self.store.add_memos([f"{format_date(self.wall())} 的经过：" + " ".join(summary.split())])
        log.info("对话记录压缩成摘要（%d 字）", len(summary))
        return True

    # ---- 失败 / 日志 ----
    def _failed(self, now: float, exc: Exception | None = None) -> None:
        self.failures += 1
        if self.failing_since is None:
            self.failing_since = now
        delay = BACKOFF[min(self.failures, len(BACKOFF)) - 1]
        self.backoff_until = now + delay
        log.warning("调用 Claude 失败（第 %d 次），%.0f 秒后再试：%s", self.failures, delay, exc)

    def _ok(self) -> None:
        self.failures = 0
        self.failing_since = None
        self.backoff_until = float("-inf")

    def _log(self, resp: dict, cost: float) -> None:
        content = resp.get("content") or []
        thoughts = " ".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
        if thoughts:
            log.info("大脑想：%s", thoughts)
        if self.run_dir is not None:
            self.run_dir.record_brain({
                "stop_reason": resp.get("stop_reason"),
                "usage": resp.get("usage"),
                "usd": round(cost, 5),
                "tools": [b.get("name") for b in content if b.get("type") == "tool_use"],
                "text": thoughts,
            })
