"""大脑的上下文：系统提示词（两段，分别缓存）、只往后追加的对话记录、删旧截图、从摘要重开。"""

from __future__ import annotations

from collections.abc import Callable

from ..config import BrainConfig
from .events import Event

REMOVED = {"type": "text", "text": "[截图已移除]"}


class Context:
    def __init__(self, cfg: BrainConfig, static: str, memory: Callable[[], str]) -> None:
        self.cfg = cfg
        self.static = static  # 规则：整个运行期不变
        self._memory_fn = memory
        self.memory = memory()  # 人设和记忆：只在开新记录时重读，免得每次都让后面的缓存失效
        self.messages: list[dict] = []

    def system(self) -> list[dict]:
        return [
            {"type": "text", "text": self.static, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": self.memory or "（没有人设和记忆）", "cache_control": {"type": "ephemeral"}},
        ]

    def add_user(self, content: list[dict] | str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: list[dict]) -> None:
        self.messages.append({"role": "assistant", "content": content})  # 原样放回（含 thinking 块）

    def add_tool_results(self, results: list[dict]) -> None:
        self.messages.append({"role": "user", "content": results})

    def drop_last_assistant(self) -> None:
        """这一轮被拒绝 / 被截断：扔掉它，免得留下没有结果的 tool_use（下次请求会 400）。"""
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()

    def images(self) -> list[tuple[list, int]]:
        """所有图片块的位置：(所在的 content 列表, 下标)，按出现顺序。"""
        found: list[tuple[list, int]] = []
        for msg in self.messages:
            _collect(msg["content"], found)
        return found

    def prune(self) -> int:
        found = self.images()
        if len(found) < self.cfg.prune_at:
            return 0
        old = found[: len(found) - self.cfg.keep_images]
        for items, i in old:
            items[i] = dict(REMOVED)
        return len(old)

    def reset(self, summary: str) -> None:
        self.memory = self._memory_fn()
        self.messages = [
            {"role": "user", "content": [{"type": "text", "text": "[之前的经过（你自己写的摘要）]\n" + summary.strip()}]}
        ]


def _collect(content, found: list) -> None:
    if not isinstance(content, list):
        return
    for i, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image":
            found.append((content, i))
        elif block.get("type") == "tool_result":
            _collect(block.get("content"), found)


def wake_message(stamp: str, events: list[Event], status: str, images: list[dict]) -> list[dict]:
    """每次醒来追加的那条 user 消息：事件、状态，可能附一张截图。"""
    lines = [f"[{stamp}] " + ("事件：" if events else "没有新事件（定时醒来）")]
    lines += [f"- {e.line()}" for e in events]
    lines.append("状态：" + status)
    return [{"type": "text", "text": "\n".join(lines)}, *images]
