"""画面里的人和物品：感知层给身体 / 技能 / 大脑用的统一说法（谁 / 什么、在画面哪边、多远）。

只依赖 Rect：身体导入它不会把 YOLO 拉进来。
"""

from __future__ import annotations

from dataclasses import dataclass

from .bubbles import Rect

WHO = {"stranger": "陌生人", "unlit": "没点火的陌生人"}
MOVING = ("走近", "走远", "往左走", "往右走")  # 写进 status 的运动方向（站着不写）
OBJECT_NAMES = {"bench": "座位", "bonfire": "篝火", "instrument": "乐器", "spirit": "先祖"}  # 物品类别 → 中文（顺序即类别顺序）


@dataclass(frozen=True)
class Person:
    track_id: int
    kind: str  # "friend" / "stranger" / "unlit"
    name: str | None  # 好友才有
    box: Rect  # 整张截图坐标
    side: str  # "左边" / "前面" / "右边"（框中心在画面三等分的哪一份）
    distance: str  # "近" / "中" / "远"
    sure: bool = True  # False = 没看到名字标签、按外观认的好友（"像小明"）
    sid: str | None = None  # 点过火的陌生人按外观给的编号（"陌生人A"）
    look: str = ""  # 陌生人的装扮描述（有才有）
    motion: str | None = None  # 走近 / 走远 / 往左走 / 往右走 / 站着（团子画面里的方向）；None = 拿不准


def side_of(cx: float, width: int) -> str:
    return "左边" if cx < width / 3 else ("右边" if cx > width * 2 / 3 else "前面")


def _describe_person(p: Person) -> str:
    where = f"{p.side}·{p.distance}" + (f"，正在{p.motion}" if p.motion in MOVING else "")
    if p.kind == "friend" and not p.sure:
        return f"像{p.name}（没看到名字，{where}）"
    if p.sid:
        return f"{p.sid}（{p.look}，{where}）" if p.look else f"{p.sid}（{where}）"
    return f"{p.name or WHO.get(p.kind, '陌生人')}（{where}）"


def describe_people(people: list[Person]) -> str:
    """"小明（左边·近）、像小红（没看到名字，右边·远）、陌生人A（白斗篷，前面·中）"；没人返回空串。"""
    return "、".join(_describe_person(p) for p in people)


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


@dataclass(frozen=True)
class Seen:
    """按 Q 喊一声时亮出名字的好友在哪（spec 2026-10-01-q-call §1.2）。"""

    side: str  # "左边" / "前面" / "右边"
    distance: str | None  # "近" / "中" / "远"；只看到名字标签、没对上人的是 None
    on_screen: bool = True  # False = 名字贴在屏幕边上：人在画面外


@dataclass
class CallSeen:
    """一次呼喊窗口的结果：窗口里亮出名字的好友、结束时还剩几个没挂名字的人。"""

    at: float  # 按键时刻（身体拿它取结果）
    friends: dict[str, Seen]
    unnamed: int = 0
    ended: bool = False
