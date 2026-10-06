"""OpenAI 兼容大脑的历史压缩（docs/superpowers/specs/2026-10-06-brain-compact-design.md）：压缩指令、前情提要的格式、
唤醒消息的时间戳、中途新记的 inbox。纯函数和小部件；状态机在 brain/toolloop.py 的 ToolLoopBrain 里。"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

log = logging.getLogger(__name__)

RECAP_ACK = "（知道了）"  # 前情提要后面的占位 assistant：别让两条 user 挨着（deepseek-reasoner 不收）

COMPACT_REQUEST = """（这不是事件，是整理：先别调工具、也别回应上面的事。）
把开头的「前情提要」（有的话）和{where}的那些轮，写成一份新的「这次上线到现在的前情提要」；那条消息和它之后的不用写，它们会原样留着。
写这些：这段时间谁来过、谁走了（大概几点）；跟谁聊了什么（具体的事，名字、数字、日子照原话写）；你答应过什么还没做；谁说过难过 / 不舒服、要对他收着点；聊出来的约定和老梗；你刚记下的事。
不写：心情和精力、状态里每次都有的东西、你调了什么工具。
用「你」称呼你自己，按时间顺序写，最多 {max} 字。只写前情提要本身，不要开场白。"""

_STAMP = re.compile(r"^\[([^\]]+)\]")
_HHMM = re.compile(r"(\d{1,2}:\d{2})(?::\d{2})?")


def wake_stamp(text: str) -> str:
    """唤醒消息开头 [...] 里的时间（Brain.message 的格式）；没有返回空串。"""
    m = _STAMP.match(text or "")
    return m.group(1) if m else ""


def until_of(stamp: str) -> str:
    """时间戳里最后一个 HH:MM；没有返回空串。"""
    found = _HHMM.findall(stamp or "")
    return found[-1] if found else ""


def compact_request(first_kept: str, keep: int, recap_max: int) -> str:
    """压缩指令：first_kept 是第一条留原话的唤醒消息。"""
    stamp = wake_stamp(first_kept)
    where = f"[{stamp}] 那条消息之前" if stamp else f"最后 {keep} 轮之前"
    return COMPACT_REQUEST.format(where=where, max=recap_max)


def recap_message(text: str, n: int, until: str, keep: int) -> str:
    """历史开头那条前情提要（user 消息）。"""
    upto = f"写到 {until} 为止" if until else f"写到最后 {keep} 轮之前为止"
    return f"（这次上线到现在的前情提要，第 {n} 次整理，{upto}；之后的原话在后面）\n{text}"


def clean_recap(text: str | None, recap_max: int) -> str:
    """去首尾空白；超了 1.5 倍才截到 recap_max 字（宁可截也不丢）。"""
    text = (text or "").strip()
    if len(text) > recap_max * 1.5:
        text = text[:recap_max] + "……"
    return text


def inbox_note(lines: list[str]) -> str:
    """接在唤醒消息末尾的「你刚记下：」；没有新行是空串。"""
    if not lines:
        return ""
    return "你刚记下：\n" + "\n".join(f"- {line.removeprefix('- ').strip()}" for line in lines)


class InboxWatch:
    """inbox.md 里这次会话还没见过的行：按行内容记（整理 notes.md 会把行挪走，不能按位置）。
    建的时候读到的算见过（系统提示词里已经有了）。读出错只记日志、当这一轮没有新行。"""

    def __init__(self, read: Callable[[], str]) -> None:
        self.read = read
        self.seen: set[str] = set()
        self.seen.update(self._lines())

    def _lines(self) -> list[str]:
        try:
            text = self.read() or ""
        except Exception:  # noqa: BLE001 读 inbox 出错不能打断大脑
            log.warning("读 inbox.md 出错，这一轮不带新记的", exc_info=True)
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    def fresh(self) -> list[str]:
        out: list[str] = []
        for line in self._lines():
            if line not in self.seen:
                self.seen.add(line)
                out.append(line)
        return out
