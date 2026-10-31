"""记忆条目：notes.md / inbox.md 的每一行带日期、长期标记和可选的键（纯函数，无 I/O）。

行格式：`- 正文 〔YYYY-MM-DD｜长期｜键〕`（全角方括号和分隔线，字段顺序固定，长期和键可省）。
没有方括号的行（手写的、旧笔记）= 长期、日期不详；方括号写坏的行整行当正文，同样按长期、日期不详。
设计见 docs/superpowers/specs/2026-10-31-memory-entries-design.md。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from datetime import date

log = logging.getLogger(__name__)

SELF = "自己"
PINNED_WORD = "长期"  # 保留字，不能当键
_BRACKET = re.compile(r"\s*〔([^〔〕]*)〕\s*$")
_SPEAKER_MAX = 20  # inbox 里「谁：」最长多少字，再长就当正文里的冒号


@dataclass(frozen=True)
class Entry:
    section: str  # 所在节（人名；`自己`；节外的是 ""）
    text: str  # 正文（不含方括号）
    date: date | None  # 最后一次被提起 / 确认的日期
    pinned: bool  # True = 长期，False = 近况
    key: str  # 事实键（同一节内相同的视为同一件事），可空
    raw: str  # 原始行去掉 "- " 和首尾空白


def _parse_body(section: str, body: str) -> Entry:
    """`body` 是 "- " 之后的内容（inbox 里含 "谁："）。方括号不合格时整行当正文。"""
    body = body.strip()
    m = _BRACKET.search(body)
    if m:
        fields = [f.strip() for f in m.group(1).split("｜")]
        try:
            d = date.fromisoformat(fields[0])
        except ValueError:
            d = None
        if d is not None:
            rest = [f for f in fields[1:] if f]
            pinned = PINNED_WORD in rest
            key = next((f for f in rest if f != PINNED_WORD), "")
            return Entry(section, body[: m.start()].strip(), d, pinned, key, body)
        log.debug("记忆条目的方括号看不懂，整行当正文: %s", body[:60])
    return Entry(section, body, None, True, "", body)


def parse_notes(text: str) -> list[Entry]:
    out: list[Entry] = []
    section = ""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- ") and line[2:].strip():
            out.append(_parse_body(section, line[2:]))
    return out


def _split_speaker(body: str) -> tuple[str, str]:
    m = _BRACKET.search(body)
    head = body[: m.start()] if m else body
    for i, ch in enumerate(head):
        if ch in "：:":
            who = head[:i].strip()
            if who and i <= _SPEAKER_MAX and "〔" not in who:
                return who, body[i + 1 :].strip()
            break
    return "", body


def parse_inbox(text: str) -> list[Entry]:
    out: list[Entry] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not (line.startswith("- ") and line[2:].strip()):
            continue
        body = line[2:].strip()
        who, rest = _split_speaker(body)
        e = _parse_body(SELF if who == "我" else who, rest)
        out.append(replace(e, raw=body))
    return out


def entry_line(e: Entry) -> str:
    fields = []
    if e.date is not None:
        fields = [e.date.isoformat()]
        if e.pinned:
            fields.append(PINNED_WORD)
        if e.key:
            fields.append(e.key)
    tail = f" 〔{'｜'.join(fields)}〕" if fields else ""
    return f"- {e.text}{tail}"


def render_notes(entries: list[Entry]) -> str:
    order: list[str] = []
    groups: dict[str, list[Entry]] = {}
    for e in entries:
        if e.section not in groups:
            groups[e.section] = []
            order.append(e.section)
        groups[e.section].append(e)
    if "" in groups:  # 节外的放最前、无标题
        order.remove("")
        order.insert(0, "")
    blocks = []
    for sec in order:
        lines = ([f"## {sec}"] if sec else []) + [entry_line(e) for e in groups[sec]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _when(d: date, today: date) -> str:
    n = (today - d).days
    if n == 0:
        return "今天"
    if n == 1:
        return "昨天"
    if 2 <= n <= 13:
        return f"{n} 天前"
    same_month = (d.year, d.month) == (today.year, today.month)
    if n < 0 or same_month:  # 未来的日期（笔误）和本月早些时候：写绝对日期
        return f"{d.month} 月 {d.day} 日" if d.year == today.year else f"{d.year} 年 {d.month} 月 {d.day} 日"
    prev = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    if (d.year, d.month) == prev:
        return f"上个月 {d.month:02d}-{d.day:02d}"
    if d.year == today.year:
        return f"{d.month} 月 {d.day} 日"
    if d.year == today.year - 1:
        return f"去年 {d.month:02d}-{d.day:02d}"
    return f"{d.year} 年 {d.month} 月 {d.day} 日"


def show(e: Entry, today: date) -> str:
    """给大脑看的样子：去掉方括号，日期写成相对时间；日期不详只给正文。"""
    return e.text if e.date is None else f"{e.text}（{_when(e.date, today)}）"
