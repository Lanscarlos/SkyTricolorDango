"""主人命令：`#` 开头的消息，在读取阶段拦截，走独立的确认回复，不进大模型的对话历史。

见 docs/superpowers/specs/2026-09-27-owner-commands-design.md。
"""

from __future__ import annotations

from collections.abc import Callable

from .memory import MemoryStore

_KNOWN = {"friend", "remember", "pause", "resume", "status"}


def is_command(text: str) -> bool:
    """speaker 已经确认是主人的前提下，判断这行文本是不是命令（`#`/`＃` 开头）。"""
    return text.strip()[:1] in ("#", "＃")


class CommandRouter:
    """解析并执行一条主人命令，返回要回复的确认/报错文案（永远回一句，不调模型）。"""

    def __init__(
        self,
        store: MemoryStore | None,
        on_pause: Callable[[bool], None],
        status: Callable[[], str],
    ) -> None:
        self.store = store
        self.on_pause = on_pause
        self.status = status

    def handle(self, text: str) -> str:
        body = text.strip().lstrip("#＃").strip()
        word, _, rest = body.partition(" ")
        rest = rest.strip()
        if word == "friend":
            return self._friend(rest)
        if word == "remember":
            return self._remember(rest)
        if word == "pause":
            self.on_pause(True)
            return "先歇会儿，不自动理别人了"
        if word == "resume":
            self.on_pause(False)
            return "好，继续陪聊"
        if word == "status":
            return self.status()
        return "没这个命令"

    def _friend(self, rest: str) -> str:
        name, _, note = rest.partition(" ")
        note = note.strip()
        if not name or not note:
            return "格式不对，是 #friend 昵称 备注内容"
        if self.store is None:
            return "没开记忆功能"
        self.store.add_friend_note(name, note)
        return "记好啦"

    def _remember(self, rest: str) -> str:
        if not rest:
            return "格式不对，是 #remember 内容"
        if self.store is None:
            return "没开记忆功能"
        self.store.add_memos([rest])
        return "记下了"
