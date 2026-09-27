"""聊天记录面板（光遇里按 C 打开）的解析。

面板在屏幕左侧，一行一条，新消息加在底部、旧的往上滚：
- 别人的消息：深色半透明底 + 白字，格式“内容 - 说话人”，靠左
- 自己的消息：浅色气泡 + 深色字，靠右，没有说话人
- 没解锁聊天的陌生人，内容被游戏替换成省略号（“........ - 陌生人”）

比起在 3D 画面里找头顶气泡，面板背景固定、还带说话人，识别稳定得多。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .bubbles import Rect
from .ocr import OcrLine, join_lines

_ONLY_DOTS = re.compile(r"^[\s.。·…,，_~\-—]*$")


@dataclass(frozen=True)
class LogRow:
    speaker: str  # 自己的消息为 ""
    text: str  # 被屏蔽（省略号）时为 ""
    is_self: bool
    box: Rect

    @property
    def masked(self) -> bool:
        return not self.is_self and not self.text

    def key(self) -> str:
        """用于前后帧对齐的比较键。"""
        return ("[我]" if self.is_self else self.speaker + "：") + self.text

    def display(self) -> str:
        if self.is_self:
            return f"[我] {self.text}"
        return f"{self.speaker or '?'}：{self.text or '（被屏蔽）'}"


def _union(boxes: list[Rect]) -> Rect:
    x1, y1 = min(b.x for b in boxes), min(b.y for b in boxes)
    x2, y2 = max(b.x2 for b in boxes), max(b.y2 for b in boxes)
    return Rect(x1, y1, x2 - x1, y2 - y1)


def _group_rows(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """按纵向位置把 OCR 片段分成视觉行（同一行可能被拆成好几段）。"""
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda l: l.box.y + l.box.h / 2):
        center = line.box.y + line.box.h / 2
        if rows:
            ref = rows[-1][0].box
            if abs(center - (ref.y + ref.h / 2)) <= max(ref.h, line.box.h) * 0.5:
                rows[-1].append(line)
                continue
        rows.append([line])
    return rows


def _is_light(img: np.ndarray, box: Rect, min_value: int) -> bool:
    """自己的消息是浅色气泡：文字框里占多数的背景像素是亮的。"""
    patch = box.crop(img)
    if patch.size == 0:
        return False
    value = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(np.median(value)) >= min_value


def _split_speaker(text: str) -> tuple[str, str]:
    """“内容 - 说话人” → (说话人, 内容)。

    被屏蔽的行“........ - 陌生人”里，省略号和短横线经常一起被 OCR 丢掉，只剩“陌生人”。
    所以没有分隔符时当成只读到了说话人、内容为空：宁可漏回一句，也不把名字当成消息。
    """
    idx = text.rfind("-")
    if idx < 0:
        return text.strip(), ""
    return text[idx + 1 :].strip(), text[:idx].strip()


def parse_rows(img: np.ndarray, lines: list[OcrLine], self_min_value: int) -> list[LogRow]:
    """img 是面板区域的截图，lines 是对它做 OCR 的结果（坐标相对 img）。按从上到下返回。"""
    width = img.shape[1]
    rows: list[LogRow] = []
    for group in _group_rows(lines):
        text = join_lines(group)
        if not text:
            continue
        box = _union([l.box for l in group])
        # 按面板排版过滤：别人的消息贴左边，自己的消息贴右边。
        # 飘在 3D 场景里的名字标签、头顶气泡两头都不贴，面板关着时也就读不出“行”
        if _is_light(img, box, self_min_value):
            if box.x2 >= width * 0.85:
                rows.append(LogRow("", text, True, box))
            continue
        if box.x > width * 0.3:  # 被屏蔽的行省略号读不出来，文字框从“陌生人”开始，会往右偏一点
            continue
        speaker, content = _split_speaker(text)
        if _ONLY_DOTS.match(content):
            content = ""
        rows.append(LogRow(speaker, content, False, box))
    return rows


def new_rows(prev: list[str], cur: list[str], same: Callable[[str, str], bool]) -> list[int]:
    """对齐前后两帧的行，返回 cur 里新增行的下标。

    新消息只会加在底部，所以在 cur 里找上一帧最后一行的位置（要求往上连续匹配得越多越好），
    它下面的就是新的。prev 为空（刚启动 / 面板刚打开）或完全对不上时，一律不算新消息：
    宁可漏一句，也不要把整屏历史当成新消息去回复。
    """
    if not prev:
        return []
    best_j, best_run = -1, 0
    for j in range(len(cur)):
        run = 0
        while run <= j and run < len(prev) and same(cur[j - run], prev[-1 - run]):
            run += 1
        if run > 0 and run >= best_run:
            best_j, best_run = j, run
    if best_run == 0:
        return []
    return list(range(best_j + 1, len(cur)))
