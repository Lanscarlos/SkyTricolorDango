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
EDGE_BAND = 0.06  # 名字标签中心在最左 / 最右这么宽（占屏宽）里 = 贴边（同感知层 [perception] edge_band）
COLUMN_X = 25  # 同一列：x 中心相差多少 px 以内（1920 宽）
COLUMN_GAP = 60  # 同一列：竖着挨着，下一个节点框顶到上面最近一个框底不超过多少 px（1080 高）。
# 实测（10-04 录像）标签底到第一句、一摞气泡之间都是 20~24 px，动画时还会叠在一起；60 留了一倍多余量，
# 又远小于两个人上下站开时的间距（同一条竖线上远处另一个人的标签不会认领近处这人的气泡）


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
    known: bool = True  # False = 不在好友名单里（游戏里的好友、friends.md 没写）
    edge: bool = False  # True = 贴在屏幕最左 / 最右（好友在画面外，光遇把标签贴边）：不算在画面里


@dataclass(frozen=True)
class Bubble:
    text: str
    box: tuple[int, int, int, int]
    speaker: str | None  # None = 挂不上好友名单里、不贴边的名字标签
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


def strip_typing(text: str) -> str:
    """去掉首尾空白和末尾的打字动画（一串空格和半角点）：报出去的原文用它。"""
    return re.sub(r"[ .]*$", "", text.strip()).strip()


def bubble_key(text: str) -> str:
    """比较键：去掉所有空白（折行、多空格）和末尾的点。别人还在输入时气泡会带着点点，说完变成原文；
    折行的气泡和面板行文字可能差一个换行。去重的比较一律用它（spec §3.2：只差末尾点算同一句）。"""
    return re.sub(r"\.*$", "", "".join(text.split()))


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

    # 输入栏开着时不拿左下角的占位判面板（面板关着按 Enter 开输入框时它可能也在），只认看得见的别人的行
    if input_text is not None:
        placeholder_seen = False
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
        is_self_row = (
            panel_open
            and l > 30 * s
            and side - 60 * s <= r <= side
            and SEP not in n.text
            and n.text not in friends  # 面板后面好友的名字标签刚好落在这一带
        )
        if is_self_row:
            self_rows[id(n)] = PanelRow("", n.text, True, False, n.box, n.visible)
        elif n.visible:
            candidates.append(n)
    # 按框的底边排，不信树里的顺序：面板刚打开时游戏在复用 TextView，顺序暂时乱（10-04 真机，乱序快照当了基准、旧话又报一遍）。
    # 滚出去的行框被裁成顶 0、底是负数，越往上越负；sorted 是稳定的，底边一样时保持树里的顺序
    rows = tuple(
        sorted(
            (row_nodes.get(id(n)) or self_rows[id(n)] for n in snap.nodes if id(n) in row_nodes or id(n) in self_rows),
            key=lambda r: r.box[3],
        )
    )

    # 3D 画面里的候选：按列分组（x 中心挨着、竖着一个挨一个），一列里最上面的标签样子的节点是名字标签
    def cx(n: Node) -> float:
        return (n.box[0] + n.box[2]) / 2

    def tag_like(n: Node) -> bool:
        h = n.box[3] - n.box[1]
        if n.box[1] <= 1 and h <= 46 * v and n.text in friends:
            return True  # 被屏幕上沿切掉一截（框顶 0，实测高 38 / 28）：只认好友名单里的名字，免得被切的气泡冒充
        return 40 * v <= h <= 46 * v and not is_dots(n.text)

    columns: list[list[Node]] = []
    bottoms: list[int] = []
    for n in sorted(candidates, key=lambda n: n.box[1]):
        near = [
            i
            for i, col in enumerate(columns)
            if abs(cx(col[0]) - cx(n)) <= COLUMN_X * s and n.box[1] - bottoms[i] <= COLUMN_GAP * v
        ]
        if near:
            i = max(near, key=lambda i: bottoms[i])  # 挨得最近的那一列
            columns[i].append(n)
            bottoms[i] = max(bottoms[i], n.box[3])
        else:
            columns.append([n])
            bottoms.append(n.box[3])

    band = EDGE_BAND * width
    order = {id(n): i for i, n in enumerate(candidates)}
    tags: list[tuple[int, Tag]] = []  # (快照里的次序, 标签)：按快照顺序给出
    owner_of: dict[int, str | None] = {}
    for col in columns:
        top = next((n for n in col if tag_like(n)), None)
        speaker = None
        if top is not None:
            t = Tag(top.text, top.box, known=top.text in friends, edge=not band <= cx(top) <= width - band)
            tags.append((order[id(top)], t))
            if t.known and not t.edge:
                speaker = t.name  # 名单外 / 贴边的标签：他列里的气泡挂不上已知好友
        below = False
        for n in col:
            if n is top:
                below = True
                continue
            owner_of[id(n)] = speaker if below else None  # 标签上面叠着的（动画里）挂不上

    bubbles = tuple(
        Bubble(n.text, n.box, owner_of[id(n)], is_dots(n.text)) for n in candidates if id(n) in owner_of
    )
    ordered = tuple(t for _, t in sorted(tags, key=lambda x: x[0]))
    return UiView(True, panel_open, rows, ordered, bubbles, input_text)
