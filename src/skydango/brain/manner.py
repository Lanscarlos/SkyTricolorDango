"""回法（spec 2026-10-07-chat-pacing §2）：把大脑最近的习惯摆到它眼前，拼成唤醒消息里「回法：…」那一行。纯计算。

说什么、贱不贱还是大脑看上下文定；这里只给建议：贱的分寸（三选一）、这批可以只回个动作、攒了好几句挑一句回。"""

from __future__ import annotations

import random

from ..config import PacingConfig
from ..inner.mind import sounds_upset
from .pacing import is_question

EMOTE_CAP = 0.6  # 困了提示动作的概率乘 1.5，最多到这里
LOW_MOODS = ("低落", "烦")


def manner(batch: list[tuple[str, str]], recent: list[bool], mood: str | None, sleepy: bool,
           rng: random.Random, cfg: PacingConfig) -> str | None:
    """batch：这批聊天 (说话人, 原文)；recent：最近说的几句贱不贱（从旧到新）；mood：心情档；sleepy：精力档是困。
    没什么好提醒的返回 None。"""
    parts: list[str] = []
    jabs = sum(recent)
    if jabs >= cfg.jab_limit or (recent and recent[-1]):
        parts.append(f"最近 {len(recent)} 句里贱了 {jabs} 句，这轮好好说")
    elif mood in LOW_MOODS or sleepy:
        parts.append("没力气贫，懒一点回")
    elif mood == "开心" and jabs <= 1:
        parts.append("想贫可以贫一句")
    texts = [text for _, text in batch]
    if texts and not any(is_question(t) or sounds_upset(t) for t in texts) \
            and all(len("".join(t.split())) <= cfg.short_len for t in texts):
        chance = min(cfg.emote_chance * 1.5, EMOTE_CAP) if sleepy else cfg.emote_chance
        if rng.random() < chance:
            parts.append("这几句可以只回个动作，不用说话")
    if len(batch) >= 2:
        who = "他们" if len({w for w, _ in batch}) > 1 else "他"
        parts.append(f"{who}连着说了 {len(batch)} 句，挑最想接的回一句就行，不用每句都回")
    return "回法：" + "；".join(parts) if parts else None


def jab_note(recent: list[bool]) -> str:
    """status 那一项：「最近说的：6 句里贱了 2 句」；还没说过是空的。"""
    return f"最近说的：{len(recent)} 句里贱了 {sum(recent)} 句" if recent else ""
