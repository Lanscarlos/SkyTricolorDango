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

_DASHES = "-－—–―"
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


def _merge_wrapped(
    groups: list[list[OcrLine]], img: np.ndarray, width: int, self_min_value: int
) -> list[list[OcrLine]]:
    """把一条长消息折出来的几行拼回去。

    实测同一个框里的折行几乎贴着（上一行下沿到下一行上沿 0~2 px），而且上一行一定写满了；
    不同消息之间隔着 25~30 px。只看间距不够：两条短消息也可能挨得近，所以还要求上一行够宽、底色深浅一致。
    """
    merged: list[list[OcrLine]] = []
    for group in groups:
        if merged:
            prev, cur = _union([l.box for l in merged[-1]]), _union([l.box for l in group])
            # 上一个 group 可能已经合并过好几行，拿它最后一行的宽度判断
            last = _union([l.box for l in _group_rows(merged[-1])[-1]])
            gap = cur.y - prev.y2
            if (
                last.w >= width * 0.75
                and gap < 0.4 * min(last.h, cur.h)
                and _is_light(img, last, self_min_value) == _is_light(img, cur, self_min_value)
            ):
                merged[-1] = merged[-1] + group
                continue
        merged.append(group)
    return merged


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
    # OCR 有时把“ - ”读成全角“－”或长破折号
    idx = max(text.rfind(d) for d in _DASHES)
    if idx < 0:  # 偶尔还会读成间隔号；内容里也可能有“·”，所以只在找不到横线时才用
        idx = text.rfind("·")
    if idx < 0:
        return text.strip(), ""
    return text[idx + 1 :].strip(), text[:idx].strip()


def parse_rows(img: np.ndarray, lines: list[OcrLine], self_min_value: int) -> list[LogRow]:
    """img 是面板区域的截图，lines 是对它做 OCR 的结果（坐标相对 img）。按从上到下返回。"""
    width = img.shape[1]
    rows: list[LogRow] = []
    for group in _merge_wrapped(_group_rows(lines), img, width, self_min_value):
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


def find_input_top(panel: np.ndarray) -> int | None:
    """找面板底部“聊天……”胶囊输入框的上沿（y，相对 panel），找不到返回 None。

    面板的高度会变：输入框关着时胶囊上沿在 y≈1006，按 Enter 打开输入框后整个面板上移到 y≈932。
    固定的读取区域总有一种状态会切掉最新一条，所以每帧找胶囊。它是个空心框（关着时白色描边、开着时蓝色）：
    - 上沿是一条横贯面板的细亮线，线上、线下都是暗的
    - 往下 57~60 px 还有一条同样亮的下沿
    只看“亮线 + 下面变暗”不够：自己发的长回复是几乎横贯面板的浅色气泡，它的下沿也长这样（实测因此漏读）；
    气泡下沿的上面是亮的气泡本体，也没有第二条线，用这两点排除。
    """
    height, width = panel.shape[:2]
    value = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)[:, int(width * 0.125) : int(width * 0.87), 2]
    bright = (value > 170).mean(axis=1)
    for y in range(int(height * 0.6), height - 50):
        if bright[y] <= 0.8 or bright[y + 8] >= 0.3 or bright[y - 8] >= 0.5:
            continue
        if bright[y + 50 : y + 67].max() > 0.8:  # 胶囊的下沿
            return y
    # 面板开着一会儿后胶囊会变暗（2026-09-28 实测）：灰色细线（V≈84）外面贴着一圈深色边、里面深色（V≈26），
    # 往下约 60 px 还有一条；变暗的过程中线是 V≈159、里面 V≈88。亮度不固定，按相对亮度找：
    # 两条都是比上下紧挨着的行亮 30 以上的细线，中间整段都比线暗 30 以上
    med = np.median(value, axis=1)
    thin = lambda y: med[y] - max(med[y - 2], med[y + 2]) >= 30  # noqa: E731
    for y in range(int(height * 0.6), height - 67):
        if not thin(y) or med[y + 3 : y + 50].max() > med[y] - 30:
            continue
        if any(thin(b) for b in range(y + 55, y + 65)):
            return y
    return None


def text_signature(panel: np.ndarray) -> np.ndarray:
    """面板文字的“指纹”：亮像素（白字、自己的浅色气泡）缩到一半大小。用来判断要不要重新 OCR。

    面板是半透明的，背后的 3D 场景一直在动，所以不直接比整块画面，只比亮到像文字的像素。
    """
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    small = cv2.resize(gray, (max(1, gray.shape[1] // 2), max(1, gray.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    return small > 180


def changed_pixels(a: np.ndarray, b: np.ndarray) -> int:
    """两个指纹之间变了多少像素；尺寸不同（面板高度变了）算全变。"""
    if a.shape != b.shape:
        return a.size + b.size
    return int(np.count_nonzero(a != b))


def _run_back(prev: list[str], cur: list[str], i: int, j: int, same: Callable[[str, str], bool]) -> int:
    """cur[j] 对上 prev[i]，往上连续对上几行。"""
    run = 0
    while run <= j and run <= i and same(cur[j - run], prev[i - run]):
        run += 1
    return run


def bottom_clipped(prev: list[str], cur: list[str], same: Callable[[str, str], bool]) -> bool:
    """这一帧底部被裁掉了一截：它最底下几行正好是上一帧倒数第二行（或更上面）往上的几行，
    而且比"上一帧最后一行在这一帧里、下面是新消息"的对法对上得更多。

    新消息只会加在底部、旧消息只会从顶上滚出去，上一帧最底下的行不会自己消失 ——
    消失了就是面板在重绘 / 输入框打开挤掉了底部（2026-09-30 实测），这一帧不能当基准。
    """
    if not prev or not cur:
        return False
    normal = max((_run_back(prev, cur, len(prev) - 1, j, same) for j in range(len(cur))), default=0)
    clipped = max((_run_back(prev, cur, i, len(cur) - 1, same) for i in range(len(prev) - 1)), default=0)
    return clipped >= 2 and clipped > normal


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
