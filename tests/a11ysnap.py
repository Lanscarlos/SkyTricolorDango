"""合成无障碍快照的小工具（坐标照 spec §1，1920×1080）：给 a11yui 和读聊天的测试共用。"""

from __future__ import annotations

from skydango.device.a11y import Node, Snapshot

GAME_PACKAGE = "com.netease.sky.vivo"


def node(text: str, box: tuple[int, int, int, int], visible: bool = True) -> Node:
    return Node(text, None, "android.widget.TextView", None, box, visible)


def snap(*nodes: Node, pkg: str | None = GAME_PACKAGE, at: float = 0.0) -> Snapshot:
    return Snapshot(at, 0, pkg, tuple(nodes))


def row(text: str, speaker: str, y: int, visible: bool = True) -> Node:
    """别人的面板行：「内容 - 说话人」。"""
    full = f"{text} - {speaker}"
    return node(full, (21, y, 21 + 20 * len(full), y + 38), visible)


def self_row(text: str, y: int) -> Node:
    """自己的面板行：只有内容、靠右。"""
    return node(text, (560, y, 618, y + 40))


def placeholder(visible: bool = True) -> Node:
    """面板打开时左下角的「聊天……」。"""
    return node("聊天……", (77, 1021, 154, 1051), visible)


def input_bar(text: str = "聊天……") -> Node:
    return node(text, (14, 962, 1906, 1032))


def tag(name: str, cx: int, y: int) -> Node:
    return node(name, (cx - 78, y, cx + 78, y + 42))


def bubble(text: str, cx: int, y: int, lines: int = 1) -> Node:
    h = 38 if lines == 1 else 34 * lines + 4
    half = max(20 * len(text), 40) // 2
    return node(text, (cx - half, y, cx + half, y + h))
