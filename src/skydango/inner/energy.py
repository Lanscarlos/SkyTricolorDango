"""精力（spec 2026-09-30-inner-phase2 §3）：现实时间 + 连续挂了多久 + 热闹程度，身体每圈现算，不调模型、不存盘。数字都是估的。"""

from __future__ import annotations

from dataclasses import dataclass

from .ledger import Session

# (开始小时, 结束小时, 分数, 原因)：一天里的底子
_BANDS = ((0, 2, 60, "半夜了"), (2, 7, 40, "凌晨了"), (7, 9, 80, "刚起"), (9, 22, 100, ""), (22, 24, 80, "挺晚了"))
AWAKE_STEP = 10  # 每连着挂 60 分钟扣多少
AWAKE_MAX = 40
CHEER = 10  # 最近 10 分钟有好友跟团子说话
BUSY_FREE = 30  # 最近 60 分钟热闹超过这么多分钟才开始累
BUSY_STEP = 5  # 之后每 10 分钟扣多少
BUSY_MAX = 15


@dataclass(frozen=True)
class Energy:
    level: str  # 精神 / 还行 / 有点累 / 困
    score: int  # 0~100
    note: str  # 给大脑看的：“有点累（半夜了，连着挂了 2 个多小时）”


def energy_parts(hour: float, awake_min: float, cheered: bool, busy_min: float) -> list[tuple[int, str]]:
    """逐项：第一项 (基础分, 原因)，之后 (加减分, 原因)，只列非 0 项（introspect 给团子看为什么累）。"""
    hour %= 24
    score, reason = next((s, r) for lo, hi, s, r in _BANDS if lo <= hour < hi)
    parts = [(score, reason or "白天")]
    hours = int(max(0.0, awake_min) // 60)
    if hours >= 1 and min(AWAKE_MAX, hours * AWAKE_STEP):
        parts.append((-min(AWAKE_MAX, hours * AWAKE_STEP), f"连着挂了 {hours} 个多小时"))
    if cheered:
        parts.append((CHEER, "有人陪着聊"))
    tired = min(BUSY_MAX, int(max(0.0, busy_min - BUSY_FREE) // 10) * BUSY_STEP)
    if tired > 0:
        parts.append((-tired, "闹了好一阵"))
    return parts


def format_parts(parts: list[tuple[int, str]], e: Energy) -> str:
    """“基础 100（白天）− 20（连着挂了 2 个多小时）+ 10（有人陪着聊）= 90，精神”；夹过 0~100 的照 e.score 写。"""
    (base, why), rest = parts[0], parts[1:]
    text = f"基础 {base}（{why}）" + "".join(f"{'+' if d > 0 else '−'} {abs(d)}（{r}）" for d, r in rest)
    return f"{text}= {e.score}，{e.level}"


def energy(hour: float, awake_min: float, cheered: bool, busy_min: float) -> Energy:
    """hour：本地时间（带小数）；awake_min：连着挂了多久；cheered：最近有好友跟团子说话；busy_min：最近 60 分钟热闹了几分钟。"""
    parts = energy_parts(hour, awake_min, cheered, busy_min)
    score = max(0, min(100, sum(d for d, _ in parts)))
    reasons = [r for i, (_, r) in enumerate(parts) if not (i == 0 and r == "白天")]
    level = "精神" if score >= 70 else "还行" if score >= 50 else "有点累" if score >= 30 else "困"
    return Energy(level, score, level + (f"（{'，'.join(reasons)}）" if reasons else ""))


def awake_minutes(now: float, start: float, history: list[Session], rest_gap: float) -> float:
    """连着挂了多久（分钟）：这一次的时长；上一次下线到这次上线不到 rest_gap（没睡），再加上一次的时长。"""
    minutes = max(0.0, now - start) / 60
    if history:
        last = history[-1]
        end = last.end if last.end is not None else (last.saved or last.start)
        if 0 <= start - end < rest_gap:
            minutes += max(0.0, end - last.start) / 60
    return minutes
