"""画面里的人和物品：感知层给身体 / 技能 / 大脑用的统一说法（谁 / 什么、在画面哪边、多远）。

只依赖 Rect：身体导入它不会把 YOLO 拉进来。
"""

from __future__ import annotations

from dataclasses import dataclass

from .bubbles import Rect

WHO = {"stranger": "陌生人", "unlit": "没点火的陌生人"}
OBJECT_NAMES = {"bench": "座位", "bonfire": "篝火", "instrument": "乐器", "spirit": "先祖"}  # 物品类别 → 中文（顺序即类别顺序）


@dataclass(frozen=True)
class Person:
    track_id: int
    kind: str  # "friend" / "stranger" / "unlit"
    name: str | None  # 好友才有
    box: Rect  # 整张截图坐标
    side: str  # "左边" / "前面" / "右边"（框中心在画面三等分的哪一份）
    distance: str  # "近" / "中" / "远"


def side_of(cx: float, width: int) -> str:
    return "左边" if cx < width / 3 else ("右边" if cx > width * 2 / 3 else "前面")


def describe_people(people: list[Person]) -> str:
    """"小明（左边·近）、陌生人（右边·远）"；没人返回空串。"""
    return "、".join(f"{p.name or WHO.get(p.kind, '陌生人')}（{p.side}·{p.distance}）" for p in people)


@dataclass(frozen=True)
class Thing:
    """画面里的物品（座位 / 篝火 / 乐器 / 先祖）。"""

    track_id: int
    kind: str  # OBJECT_NAMES 的键
    box: Rect  # 整张截图坐标
    side: str  # 同 Person
    distance: str  # "近" / "中" / "远"（按框底边，见 object_distance）


def object_distance(bottom: float, height: int, near: float, far: float) -> str:
    """物品大小差得多，不按框高分远近：框底边越靠画面下方越近。"""
    return "近" if bottom >= height * near else ("中" if bottom >= height * far else "远")


def describe_things(things: list[Thing]) -> str:
    """"座位（左边·近）、先祖（前面·远）"；没有返回空串。"""
    return "、".join(f"{OBJECT_NAMES.get(t.kind, t.kind)}（{t.side}·{t.distance}）" for t in things)
