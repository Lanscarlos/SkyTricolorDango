"""从 history.jsonl 回填（spec §3）：people.json 还不存在时启动做一次。

history 只记团子开口的那几轮，每轮 user 是 responder.format_incoming 的格式（“名字：「内容」”一行一句）。
推得出：第一次 / 最后一次说话、说过几句、最后一句、哪几天、大致哪几次上线；推不出：待了多久、跟团子说的几句。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..chat.memory import SKIP, Turn
from .ledger import LINE_CHARS, Card, Session, day_of, match_friend

_LINE = re.compile(r"^(.+?)：「(.*)」$")
_HEADER = "新的聊天消息："


def _lines(turn: Turn) -> list[tuple[str, str]]:
    """一轮里听到的每句 (说话人, 内容)；拆不出说话人的说话人是空。"""
    out = []
    for line in turn.user.splitlines():
        line = line.strip()
        if not line or line == _HEADER:
            continue
        m = _LINE.match(line)
        out.append((m.group(1), m.group(2)) if m else ("", line))
    return out


def backfill(turns: list[Turn], friends: Sequence[str], session_gap: float) -> tuple[dict[str, Card], list[Session]]:
    cards: dict[str, Card] = {}
    sessions: list[Session] = []
    for turn in sorted(turns, key=lambda t: t.t):
        if not sessions or turn.t - (sessions[-1].end or sessions[-1].start) > session_gap:
            sessions.append(Session(start=turn.t, end=turn.t, live=True, ended="backfill"))
        s = sessions[-1]
        s.end = turn.t
        if turn.reply and turn.reply != SKIP:
            s.said += 1
        for speaker, text in _lines(turn):
            s.heard += 1
            name = match_friend(speaker, friends)
            if name is None:
                continue
            card = cards.setdefault(name, Card(first_met=turn.t))
            if name not in s.friends:
                s.friends.append(name)
                card.visits += 1  # 近似“见过几次”：出现过的回填上线段数
            card.lines += 1
            card.last_seen = turn.t
            card.last_line = {"t": turn.t, "text": text[:LINE_CHARS]}
            if day_of(turn.t) not in card.days:
                card.days.append(day_of(turn.t))
    return cards, sessions
