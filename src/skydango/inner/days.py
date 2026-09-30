"""日子（spec §4 ①、⑤）：系统提示词里的「日子」一节，memory show 的“最近几次上线”。纯文字，不碰文件。"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ..chat.memory import format_date
from .ledger import Card, Session, ago, day_of

WEEK = 7 * 86400
MAX_LONG = 3  # “很久没见的好友”最多列几个
SUMMARY_CHARS = 40
ENDED = {"normal": "正常下线", "crash": "意外断了", "backfill": "回填"}


def days_prompt(history: list[Session], cards: dict[str, Card], friends: Sequence[str], now: float, long_gap_days: float) -> str:
    today = day_of(now)
    lines = ["## 日子"]
    head = f"今天 {format_date(now)}。"
    if not history:
        lines.append(head + "这是你第一次上线。")
    else:
        last = history[-1]
        nth_today = sum(1 for s in history if day_of(s.start) == today) + 1
        end = last.end if last.end is not None else (last.saved or last.start)
        crash = "（意外断了）" if last.ended == "crash" else ""
        week = {day_of(s.start) for s in history if s.start >= now - WEEK} | {today}
        which = "今天第一次" if nth_today == 1 else f"今天第 {nth_today} 次"
        lines.append(
            f"{head}这是你第 {len(history) + 1} 次上线，{which}；上次下线是 {ago(now - end)}前{crash}。最近 7 天上线了 {len(week)} 天。"
        )
        last_bits = []
        if last.friends:
            last_bits.append(f"上次见到了：{'、'.join(last.friends)}。")
        if last.summary:
            last_bits.append(f"上次的经过：{last.summary}")
        if last_bits:
            lines.append("".join(last_bits))
    far = sorted(
        (now - cards[n].last_seen, n)
        for n in dict.fromkeys(friends)
        if n in cards and cards[n].last_seen is not None and now - cards[n].last_seen > long_gap_days * 86400
    )[:MAX_LONG]
    if far:
        lines.append("很久没见的好友：" + "、".join(f"{n}（{ago(gap)}前）" for gap, n in far))
    lines.append("这些是给你心里有数的，别跟人报数字。")
    return "\n".join(lines)


def recent_days(history: list[Session], n: int = 10) -> list[str]:
    """最近 n 次上线，新的在前（memory show 用）。"""
    out = []
    for s in reversed(history[-n:] if n > 0 else []):
        d = time.localtime(s.start)
        span = "没有结束时间" if s.end is None else ago(s.end - s.start)
        who = "、".join(s.friends) or "没见到好友"
        line = f"{d.tm_mon}月{d.tm_mday}日 {d.tm_hour:02d}:{d.tm_min:02d}  {span}  {who}  {ENDED.get(s.ended, '没结束')}"
        if s.summary:
            summary = s.summary if len(s.summary) <= SUMMARY_CHARS else s.summary[:SUMMARY_CHARS] + "…"
            line += f"  {summary}"
        out.append(line)
    return out
