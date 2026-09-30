"""后果（spec 2026-09-30-inner-phase2 §5）：心情和精力 → 倍数。只管“说多少、做多少”（主动开口额度、反射、心跳），
“怎么说”交给提示词；接话永远照常。数字都是估的，真机跑过再决定开不开放成配置。"""

from __future__ import annotations

from dataclasses import dataclass

QUOTA_FLOOR = 0.25


@dataclass(frozen=True)
class Effects:
    quota: float = 1.0  # 主动开口额度
    addressed: float = 1.0  # 被叫到时做小动作的概率
    idle: float = 1.0  # 闲着的小动作间隔（< 1 更勤）
    slow: bool = False  # 没事时醒得慢一档（困）


NEUTRAL = Effects()

# 档位 → (额度, 被叫到的小动作, 闲着间隔, 心跳慢一档)
_MOOD = {"开心": (1.5, 1.5, 1.0, False), "平常": (1.0, 1.0, 1.0, False), "低落": (0.5, 1.0, 1.0, False), "烦": (0.5, 0.0, 1.0, False)}
_ENERGY = {"精神": (1.0, 1.0, 1.0, False), "还行": (1.0, 1.0, 1.0, False), "有点累": (0.75, 1.0, 1.0, False), "困": (0.5, 1.0, 0.6, True)}
_PLAIN = (1.0, 1.0, 1.0, False)


def effects(mood_level: str, energy_level: str) -> Effects:
    """心情和精力的倍数相乘；额度下限 QUOTA_FLOOR。认不出的档位当平常 / 精神。"""
    m = _MOOD.get(mood_level, _PLAIN)
    e = _ENERGY.get(energy_level, _PLAIN)
    return Effects(max(QUOTA_FLOOR, m[0] * e[0]), m[1] * e[1], m[2] * e[2], m[3] or e[3])
