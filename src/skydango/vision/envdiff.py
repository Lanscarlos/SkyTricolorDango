"""两轮聊天之间环境变了什么（谁来了、谁走了、陌生人多了少了、到了哪）：写进那一轮的用户消息，跟着聊天记录走。

完整的"现在的环境"照旧放在系统提示词里（每次现取、不进历史）；这里只记变化，模型才知道前后发生了什么。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSnapshot:
    friends: frozenset[str]
    strangers: int
    place: str | None


def snapshot(env, now: float) -> EnvSnapshot:
    """EnvWatcher 认不出陌生人（没有 strangers()），算 0。"""
    strangers = env.strangers(now) if hasattr(env, "strangers") else 0
    return EnvSnapshot(frozenset(env.nearby(now)), strangers, getattr(env, "place", None) or None)


def diff_line(before: EnvSnapshot | None, after: EnvSnapshot) -> str:
    """没有"之前"（刚启动）或没变化时返回空串。地名提示过期不算离开：只报到了新地方。"""
    if before is None:
        return ""
    parts = [f"{n}来了" for n in sorted(after.friends - before.friends)]
    parts += [f"{n}走了" for n in sorted(before.friends - after.friends)]
    if after.strangers != before.strangers:
        parts.append(f"陌生人 {before.strangers}→{after.strangers} 个")
    if after.place and after.place != before.place:
        parts.append(f"到了{after.place}")
    return f"（这之间：{'；'.join(parts)}）" if parts else ""
