"""记忆条目：notes.md / inbox.md 的每一行带日期、长期标记和可选的键（纯函数，无 I/O）。

行格式：`- 正文 〔YYYY-MM-DD｜长期｜键〕`（全角方括号和分隔线，字段顺序固定，长期和键可省）。
没有方括号的行（手写的、旧笔记）= 长期、日期不详；方括号写坏的行整行当正文，同样按长期、日期不详。
设计见 docs/superpowers/specs/2026-10-31-memory-entries-design.md。
"""

from __future__ import annotations

import difflib
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


# ---- 写盘前的兜底：取代、补日期、近况淡出、整理把关、常驻切分 ----
SIMILAR = 0.8  # 补日期时，新旧两条文字相似度的门槛
_MIN_OLD_FOR_SHRINK = 6  # 旧笔记不到这么多条时不管条目骤减


def _redate(e: Entry, d: date) -> Entry:
    e2 = replace(e, date=d)
    return replace(e2, raw=entry_line(e2)[2:])


def settle(old: list[Entry], new: list[Entry], today: date, recent_days: int) -> tuple[list[Entry], list[tuple[Entry, str]]]:
    """整理拿到模型输出后、写盘前：补日期 → 同键取代 → 近况淡出。返回（留下的，[(挪走的, 原因)]）。"""
    gone: list[tuple[Entry, str]] = []
    # 1. 补日期：模型漏写日期时，借旧笔记同一节里最像的那条的
    filled: list[Entry] = []
    for e in new:
        if e.date is None:
            best, best_r = None, SIMILAR
            for o in old:
                if o.section != e.section or o.date is None:
                    continue
                r = difflib.SequenceMatcher(None, o.text, e.text).ratio()
                if r >= best_r:
                    best, best_r = o, r
            if best is not None:
                e = _redate(e, best.date)
        filled.append(e)
    # 2. 同一节、同一个键：留日期最新的（一样新留写得靠后的）
    winner: dict[tuple[str, str], int] = {}
    for i, e in enumerate(filled):
        if not e.key:
            continue
        k = (e.section, e.key)
        j = winner.get(k)
        if j is None or (e.date or date.min) >= (filled[j].date or date.min):
            winner[k] = i
    keep: list[Entry] = []
    for i, e in enumerate(filled):
        if e.key and winner[(e.section, e.key)] != i:
            gone.append((e, f"被取代：{filled[winner[(e.section, e.key)]].text}"))
        else:
            keep.append(e)
    # 3. 近况淡出：长期的不按时间淡出；未来的日期（笔误）不当过期
    out: list[Entry] = []
    for e in keep:
        if not e.pinned and e.date is not None and (today - e.date).days > recent_days:
            gone.append((e, "近况过期"))
        else:
            out.append(e)
    return out, gone


def check_rewrite(old_text: str, new_text: str) -> str | None:
    """整理结果像不像样；该拒绝就返回原因。"""
    lines = [ln.strip() for ln in (new_text or "").splitlines() if ln.strip()]
    body = [ln for ln in lines if not ln.startswith("## ")]
    junk = [ln for ln in body if not ln.startswith("- ")]
    if body and len(junk) * 2 > len(body):
        return f"{len(body)} 行里有 {len(junk)} 行解析不了"
    n_old, n_new = len(parse_notes(old_text)), len(parse_notes(new_text))
    if n_old >= _MIN_OLD_FOR_SHRINK and n_new * 2 < n_old:
        return f"条目从 {n_old} 条骤减到 {n_new} 条"
    return None


def split_pinned(entries: list[Entry], today: date, pinned_max: int) -> tuple[list[Entry], list[Entry]]:
    """长期条目渲染后超过 pinned_max 字时不删，只把超出的挪出常驻：日期最旧的先挪（日期不详最旧），`自己` 最后。
    返回（常驻的，挪出常驻的）。"""
    pinned = [e for e in entries if e.pinned]
    total = sum(len(entry_line(e)) for e in pinned)
    moved: list[Entry] = []
    while pinned and total > pinned_max:
        pool = [e for e in pinned if e.section != SELF] or pinned
        victim = min(pool, key=lambda e: e.date or date.min)  # min 取第一个：一样旧的先挪文件里靠前的
        pinned.remove(victim)
        moved.append(victim)
        total -= len(entry_line(victim))
    return pinned, moved


def archive_line(e: Entry, today: date, reason: str) -> str:
    return f"- 〔挪走 {today.isoformat()}：{reason}〕{e.raw}"


# ---- 给大脑看：常驻的长期条目 / 全部条目 ----
def _render_shown(entries: list[Entry], today: date) -> str:
    order: list[str] = []
    groups: dict[str, list[Entry]] = {}
    for e in entries:
        if e.section not in groups:
            groups[e.section] = []
            order.append(e.section)
        groups[e.section].append(e)
    if "" in groups:
        order.remove("")
        order.insert(0, "")
    blocks = []
    for sec in order:
        lines = ([f"## {sec}"] if sec else []) + [f"- {show(e, today)}" for e in groups[sec]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def pinned_view(notes: str, inbox: str, today: date, pinned_max: int) -> tuple[str, list[Entry]]:
    """系统提示词里的「记得的事」：只放长期条目（notes 和 inbox 里标了长期的，按节、相对日期），超过 pinned_max 字的挪出去。
    返回（正文，候选池）；候选池 = 近况（notes 和 inbox 里没标长期的）+ 挪出常驻的长期条目，给 Recollector 按这一轮的话挑。"""
    everything = parse_notes(notes) + parse_inbox(inbox)
    pinned, moved = split_pinned(everything, today, pinned_max)
    if moved:
        log.info("长期记忆超过 %d 字，%d 条改成按需想起来（不删）：%s", pinned_max, len(moved), "；".join(e.text[:12] for e in moved[:5]))
    pool = [e for e in everything if not e.pinned] + moved
    return _render_shown(pinned, today), pool


def render_shown(notes: str, inbox: str, today: date) -> str:
    """全部条目（笔记 + 随手记）都用相对日期渲染：给 --no-brain 的回复和反思用。"""
    return _render_shown(parse_notes(notes) + parse_inbox(inbox), today)


def section_lines(notes: str, inbox: str, names: list[str], today: date, limit: int = 10) -> list[str]:
    """这些人那几节的条目（笔记 + 随手记），`节：正文（相对日期）`，每人最多 limit 条：反思材料用。
    条目写在 `## 名字` 下面、行里不带名字，所以要按节取，不能按“行里有没有名字”筛。"""
    everything = parse_notes(notes) + parse_inbox(inbox)
    out: list[str] = []
    for n in names:
        out += [f"{e.section}：{show(e, today)}" for e in everything if e.section == n][:limit]
    return out


_ARCHIVED = re.compile(r"^-\s*〔挪走\s*([^：〕]*)：([^〕]*)〕(.*)$")


def parse_archive(text: str) -> list[tuple[Entry, str]]:
    """notes_archive.md 的行 → (条目, 挪走的原因)。archive 行不记所在节，section 是 ""。"""
    out: list[tuple[Entry, str]] = []
    for line in (text or "").splitlines():
        m = _ARCHIVED.match(line.strip())
        if m and m.group(3).strip():
            out.append((_parse_body("", m.group(3)), m.group(2).strip()))
    return out


def memory_stats(notes: str, inbox: str, today: date, pinned_max: int) -> tuple[int, int, int]:
    """（常驻条数，常驻字数，候选池条数）：`memory show` 用。"""
    text, pool = pinned_view(notes, inbox, today, pinned_max)
    n = sum(1 for ln in text.splitlines() if ln.startswith("- "))
    return n, len(text), len(pool)
