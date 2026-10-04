"""把一份无障碍快照（`device/a11y.py`）分成聊天面板的行、头顶名字标签、头顶气泡、输入栏。

纯计算、没有状态：同一份快照永远分出同一个结果。「这一行是不是新消息」「气泡说完没有」这类
要看前后快照的事由读聊天的那一层管。规则和坐标见 spec 2026-10-04-a11y-chat-reader-design.md §2.1
（坐标按 1920×1080 标定，其它分辨率按比例缩放）。
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from skydango.device.a11y import SEP, Node, Snapshot, split_speaker

GAME_PACKAGE = "com.netease.sky.vivo"

_DOTS = set(".。…")
_PLACEHOLDER = "聊天……"


@dataclass(frozen=True)
class PanelRow:
    speaker: str  # 自己的行是 ""
    text: str  # 被屏蔽（内容全是点点）时是 ""
    is_self: bool
    masked: bool
    box: tuple[int, int, int, int]
    visible: bool  # False = 滚出面板可视范围的历史行

    def key(self) -> tuple[bool, str, str]:
        return (self.is_self, self.speaker, self.text)


@dataclass(frozen=True)
class Tag:
    name: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Bubble:
    text: str
    box: tuple[int, int, int, int]
    speaker: str | None  # None = 挂不上名字标签
    typing_only: bool  # 全是点点和空格（正在输入）


@dataclass(frozen=True)
class UiView:
    in_game: bool
    panel_open: bool
    rows: tuple[PanelRow, ...]
    tags: tuple[Tag, ...]
    bubbles: tuple[Bubble, ...]
    input_text: str | None  # None = 输入栏没开


_EMPTY = UiView(False, False, (), (), (), None)


def is_dots(text: str) -> bool:
    """去掉空白后非空且全是 . 。 …（别人正在输入 / 被屏蔽的内容）。"""
    t = "".join(text.split())
    return bool(t) and all(c in _DOTS for c in t)


def bubble_key(text: str) -> str:
    """气泡文字去掉末尾的点和空格：别人还在输入时气泡会带着点点，说完变成原文。"""
    return re.sub(r"[ .]*$", "", text).strip()


def classify(snap: Snapshot | None, width: int, height: int, friends: Collection[str]) -> UiView:
    if snap is None or snap.package != GAME_PACKAGE:
        return _EMPTY
    s, v = width / 1920, height / 1080
    side = 0.335 * width  # 聊天面板右边界

    input_text: str | None = None
    placeholder_seen = False
    others: list[tuple[Node, tuple[str, str]]] = []
    rest: list[Node] = []  # 没被认成输入栏 / 占位 / 别人的行的节点（顺序保持）

    for n in snap.nodes:
        l, t, r, b = n.box
        if n.visible and (r - l) > 0.9 * width and b > 0.85 * height:
            input_text = n.text
            continue
        if n.text == _PLACEHOLDER and l < 0.1 * width and b > 0.9 * height and (r - l) < 0.15 * width:
            placeholder_seen = placeholder_seen or n.visible
            continue
        if SEP in n.text and l <= 30 * s and r <= side:
            sp = split_speaker(n.text)
            if sp is not None:
                others.append((n, sp))
                continue
        rest.append(n)

    panel_open = placeholder_seen or any(n.visible for n, _ in others)

    # 面板行：保持快照顺序（别人的和自己的混在一起，按 y 排在快照里本来就是从上到下）
    row_nodes: dict[int, PanelRow] = {}
    for n, (content, speaker) in others:
        masked = is_dots(content)
        row_nodes[id(n)] = PanelRow(speaker, "" if masked else content, False, masked, n.box, n.visible)
    candidates: list[Node] = []
    self_rows: dict[int, PanelRow] = {}
    for n in rest:
        l, t, r, b = n.box
        if panel_open and l > 30 * s and side - 60 * s <= r <= side and SEP not in n.text:
            self_rows[id(n)] = PanelRow("", n.text, True, False, n.box, n.visible)
        elif n.visible:
            candidates.append(n)
    rows = tuple(
        row_nodes.get(id(n)) or self_rows[id(n)] for n in snap.nodes if id(n) in row_nodes or id(n) in self_rows
    )

    # 3D 画面里的候选：名字标签 / 头顶气泡
    def tag_like(n: Node) -> bool:
        h = n.box[3] - n.box[1]
        return 40 * v <= h <= 46 * v and not is_dots(n.text) and n.text in friends

    def cx(n: Node) -> float:
        return (n.box[0] + n.box[2]) / 2

    tag_like_nodes = [n for n in candidates if tag_like(n)]
    tag_nodes = [
        n
        for n in tag_like_nodes
        if not any(o is not n and abs(cx(o) - cx(n)) <= 25 * s and o.box[1] < n.box[1] for o in tag_like_nodes)
    ]
    tags = tuple(Tag(n.text, n.box) for n in tag_nodes)

    bubbles: list[Bubble] = []
    for n in candidates:
        if any(n is t for t in tag_nodes):
            continue
        above = [t for t in tag_nodes if abs(cx(t) - cx(n)) <= 25 * s and t.box[3] < n.box[3]]
        owner = min(above, key=lambda t: n.box[3] - t.box[3], default=None)
        bubbles.append(Bubble(n.text, n.box, owner.text if owner else None, is_dots(n.text)))

    return UiView(True, panel_open, rows, tags, tuple(bubbles), input_text)
