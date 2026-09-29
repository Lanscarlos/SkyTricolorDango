"""大脑：常驻线程。有事件（攒一小会儿）或到了心跳就醒来，把事件、身体状态、眼睛最近的描述拼成一条文字消息，
交给常驻的 Claude Code（session），它自己调 MCP 工具，直到这一轮 result。失败退避；连续失败太久算离线（聊天交给备用回复）。"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

from ..chat.memory import format_date
from ..config import BrainConfig, ChatConfig
from .claude import ClaudeError
from .events import Event, EventQueue
from .prompt import SUMMARY_REQUEST

log = logging.getLogger(__name__)

BACKOFF = (10.0, 30.0, 60.0)  # 连续失败后隔多久再试


def log_brain_message(m: dict) -> None:
    """把大脑一轮里想了什么、调了什么工具打进日志（agent.log）。"""
    if m.get("type") != "assistant":
        return
    for block in (m.get("message") or {}).get("content") or []:
        if block.get("type") == "text" and (block.get("text") or "").strip():
            log.info("大脑想：%s", block["text"].strip())
        elif block.get("type") == "tool_use":
            log.debug("大脑调用 %s %s", block.get("name"), json.dumps(block.get("input"), ensure_ascii=False))


class Brain:
    def __init__(
        self,
        cfg: BrainConfig,
        chat: ChatConfig,
        session,  # brain.session.BrainSession：send(text) -> result
        toolbox,  # brain.tools.ToolBox：begin_turn()、status()、acted、used
        events: EventQueue,
        nearby: Callable[[float], list[str]],
        eyes=None,  # brain.eyes.Eyes：summary(now)
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        run=None,  # runlog.RunDir
        store=None,  # chat.memory.MemoryStore：live 时退出前把经过记进 inbox.md
        trace=None,  # brain.trace.BrainTrace：可视化网页的大脑时间线（run --brain --view）
    ) -> None:
        self.cfg = cfg
        self.chat = chat
        self.session = session
        self.toolbox = toolbox
        self.events = events
        self.nearby = nearby
        self.eyes = eyes
        self.clock = clock
        self.wall = wall
        self.run_dir = run
        self.store = store
        self.trace = trace
        if trace is not None:
            trace.state = self.trace_state
        self.last_wake = float("-inf")  # 刚上线马上醒一次
        self._idle = 0  # 连着几次醒来什么都没做（心跳逐档退后）
        self.failures = 0
        self.failing_since: float | None = None
        self.backoff_until = float("-inf")
        self.chat_turn = False  # 正在回聊天（这一轮取走了聊天 / 主人命令，还没结束）：身体据此别把聊天面板当成安静关掉

    # ---- 什么时候醒 ----
    def offline(self, now: float) -> bool:
        """聊天要不要交给备用回复：连续失败太久。"""
        return self.failing_since is not None and now - self.failing_since >= self.cfg.offline_fallback

    def heartbeat(self, now: float) -> float:
        beats = self.cfg.heartbeat
        start = 0 if self.nearby(now) else 1  # 身边有好友时醒得勤一点
        return beats[min(start + self._idle, len(beats) - 1)]

    def due(self, now: float) -> str | None:
        if now < self.backoff_until:
            return None
        if len(self.events) and now - self.events.last_put >= self.chat.debounce:
            return "events"
        if now - self.last_wake >= self.heartbeat(now):
            return "heartbeat"
        return None

    def run(self, stop: threading.Event) -> None:
        log.info("大脑上线（Claude Code：%s，effort=%s）", self.cfg.model, self.cfg.effort)
        while not stop.is_set():
            try:
                now = self.clock()
                reason = self.due(now)
                if reason is None:
                    if len(self.events):  # 在攒消息或在退避：稍等再看
                        stop.wait(0.2)
                    else:
                        self.events.wait(0.5)
                    continue
                self.wake(now, reason)
            except Exception as exc:  # 大脑线程不能悄悄死掉：记下来，歇一秒接着跑
                log.exception("大脑这一轮出错")
                self._failed(self.clock(), exc)
                stop.wait(1.0)
        log.info("大脑下线")

    # ---- 醒来一次 ----
    def message(self, now: float, events: list[Event]) -> str:
        w = self.wall()
        stamp = f"{format_date(w)} {time.strftime('%H:%M:%S', time.localtime(w))}"
        lines = [f"[{stamp}] " + ("事件：" if events else "没有新事件（定时醒来）")]
        lines += [f"- {e.line()}" for e in events]
        lines.append("状态：" + self.toolbox.status())
        lines.append(self.eyes.summary(now) if self.eyes is not None else "场景：（没开眼睛）")
        return "\n".join(lines)

    def wake(self, now: float, reason: str) -> None:
        self.last_wake = now
        events = self.events.drain()
        text = self.message(now, events)
        self.toolbox.begin_turn()
        self._trace("begin", reason, text)
        start = self.clock()
        self.chat_turn = any(e.kind in ("chat", "owner_command") for e in events)
        try:
            result = self.session.send(text)
        except ClaudeError as exc:
            self._trace("fail", str(exc), self.clock() - start)
            self._failed(self.clock(), exc)
            return
        except Exception as exc:  # 进程起不来之类：run() 兜住退避；时间线上这一轮也得收尾
            self._trace("fail", f"{type(exc).__name__}: {exc}", self.clock() - start)
            raise
        finally:
            self.chat_turn = False
        self._trace("finish", result, self.clock() - start)
        self._ok()
        self._idle = 0 if (reason == "events" or self.toolbox.acted) else self._idle + 1
        self._log(result)

    def farewell(self) -> bool:
        """退出前让它写一份这次的经过，记进 inbox.md（只在 live、没在失败时）。"""
        if self.store is None or self.failing_since is not None:
            return False
        self.toolbox.begin_turn()
        self._trace("begin", "farewell", SUMMARY_REQUEST)
        start = self.clock()
        try:
            result = self.session.send(SUMMARY_REQUEST)
        except ClaudeError as exc:
            self._trace("fail", str(exc), self.clock() - start)
            log.warning("退出前写经过失败：%s", exc)
            return False
        except Exception as exc:
            self._trace("fail", f"{type(exc).__name__}: {exc}", self.clock() - start)
            raise
        self._trace("finish", result, self.clock() - start)
        text = " ".join((result.get("result") or "").split())
        if not text:
            return False
        self.store.add_memos([f"{format_date(self.wall())} 的经过：{text}"])
        log.info("这次的经过记进了 inbox.md（%d 字）", len(text))
        return True

    # ---- 可视化网页的大脑时间线 ----
    def trace_state(self) -> dict:
        """栏头的总体状态。"""
        now = self.clock()
        return {
            "model": self.cfg.model,
            "effort": self.cfg.effort,
            "failures": self.failures,
            "retry_in": self.backoff_until - now if self.backoff_until > now else None,
            "offline": self.offline(now),
        }

    def _trace(self, name: str, *args) -> None:
        if self.trace is None:
            return
        try:
            getattr(self.trace, name)(*args)
        except Exception:
            log.debug("大脑记录出错（%s）", name, exc_info=True)

    # ---- 失败 / 日志 ----
    def _failed(self, now: float, exc: Exception | None = None) -> None:
        self.failures += 1
        if self.failing_since is None:
            self.failing_since = now
        limit = bool(getattr(exc, "limit", False))
        delay = self.cfg.limit_retry if limit else BACKOFF[min(self.failures, len(BACKOFF)) - 1]
        self.backoff_until = now + delay
        what = "订阅额度用完了" if limit else "大脑这一轮失败"
        log.warning("%s（第 %d 次），%.0f 秒后再试：%s", what, self.failures, delay, exc)

    def _ok(self) -> None:
        self.failures = 0
        self.failing_since = None
        self.backoff_until = float("-inf")

    def _log(self, result: dict) -> None:
        text = (result.get("result") or "").strip()
        if text:
            log.info("大脑这一轮最后说：%s", text)
        if self.run_dir is None:
            return
        try:
            self.run_dir.record_brain({
                "subtype": result.get("subtype"),
                "num_turns": result.get("num_turns"),
                "total_cost_usd": result.get("total_cost_usd"),  # 订阅不按这个收费，只做参考
                "usage": result.get("usage"),
                "tools": list(self.toolbox.used),
                "text": text,
            })
        except Exception:
            log.exception("记大脑日志出错")
