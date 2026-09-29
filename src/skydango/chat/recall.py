"""翻以前的聊天（大脑的 recall 工具）：在 history.jsonl 里按关键词 / 人名 / 天数找原话，再附上笔记里相关的行。

只读，不调模型：好友说“昨天你不是说要……”时，大脑先查原话再答，查不到就说记不清，别顺着编。
"""

from __future__ import annotations

import time

from .memory import SKIP, MemoryStore, Turn

MAX_DAYS = 90
MAX_CHARS = 200  # 每轮最多显示这么多字
MAX_NOTE_LINES = 10
_HEADER = "新的聊天消息：\n"


def _heard(turn: Turn) -> str:
    text = turn.user[len(_HEADER) :] if turn.user.startswith(_HEADER) else turn.user
    return " / ".join(line.strip() for line in text.splitlines() if line.strip())


def _render(turn: Turn) -> str:
    d = time.localtime(turn.t)
    text = _heard(turn)
    if turn.reply and turn.reply != SKIP:
        text += f" → 我：{turn.reply}"
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS - 1] + "…"
    return f"{d.tm_mon}月{d.tm_mday}日 {d.tm_hour:02d}:{d.tm_min:02d}  {text}"


def recall(
    store: MemoryStore, query: str = "", who: str = "", days: int = 14, now: float | None = None, limit: int = 8
) -> str:
    """query：空格分开的关键词，中任意一个就算；who：只看这个人说的 / 对他说的（子串）；days：往前查几天（1~90）。"""
    words = [w.lower() for w in query.split() if w]
    who = who.strip().lower()
    if not words and not who:
        raise ValueError("给几个关键词（query）或者一个名字（who）")
    days = max(1, min(int(days), MAX_DAYS))
    now = time.time() if now is None else now
    turns = [t for t in store.history.all() if now - t.t <= days * 86400]

    scored = []
    for t in turns:
        text = f"{t.user}\n{t.reply}".lower()
        if who and who not in text:
            continue
        hits = sum(w in text for w in words)
        if words and not hits:
            continue
        scored.append((hits, t.t, t))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    found = sorted((s[2] for s in scored[:limit]), key=lambda t: t.t)

    keys = words or [who]
    notes = [
        line.strip() for line in f"{store.notes()}\n{store.inbox()}".splitlines()
        if line.strip() and not line.startswith("#") and any(k in line.lower() for k in keys)
    ][:MAX_NOTE_LINES]

    what = "、".join(filter(None, [" ".join(words) and f"“{' '.join(words)}”", who and f"和“{who}”有关的"]))
    parts = []
    if found:
        parts.append(f"最近 {days} 天（共 {len(turns)} 轮）里找到 {len(found)} 轮（{what}），“我”是你自己说的：")
        parts += [_render(t) for t in found]
    else:
        parts.append(f"最近 {days} 天（共 {len(turns)} 轮）里没找到{what}的聊天。")
    if notes:
        parts.append("笔记里：")
        parts += notes
    if not found and not notes:
        parts.append("别顺着对方编：可以换个说法再查一次，查不到就说记不清了。")
    return "\n".join(parts)
