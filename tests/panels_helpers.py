"""面板识别测试共用的小工具：写特征卡、画十字模板、按行给结果的假 OCR。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from skydango.vision.bubbles import Rect
from skydango.vision.ocr import OcrLine

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "assets" / "panels"


def cross(size: int = 40) -> np.ndarray:
    """黑底上的白色十字（剪影就是十字本身，边长 size）。"""
    img = np.zeros((size, size, 3), np.uint8)
    t = max(2, size // 5)
    c = size // 2
    img[:, c - t // 2 : c - t // 2 + t] = 255
    img[c - t // 2 : c - t // 2 + t, :] = 255
    return img


def write_card(root: Path, name: str, text: str, images: tuple[str, ...] = ()) -> Path:
    """在 root/name/ 写一张卡（card.toml 内容 = text），images 里的文件名各画一个 40×40 十字（外面留 10 px 黑边）。"""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "card.toml").write_text(text, encoding="utf-8")
    for image in images:
        canvas = np.zeros((60, 60, 3), np.uint8)
        canvas[10:50, 10:50] = cross(40)
        cv2.imwrite(str(folder / image), canvas)
    return folder


def draw_cross(frame: np.ndarray, at: tuple[float, float], size: float) -> np.ndarray:
    """在 frame 的归一化位置 at（中心）画一个边长 size 的十字（周围垫一块黑底）。"""
    h, w = frame.shape[:2]
    s = max(4, round(size))
    cx, cy = round(at[0] * w), round(at[1] * h)
    x, y = cx - s // 2, cy - s // 2
    frame = frame.copy()
    pad = s // 3
    frame[y - pad : y + s + pad, x - pad : x + s + pad] = 0
    frame[y : y + s, x : x + s] = cv2.resize(cross(40), (s, s), interpolation=cv2.INTER_NEAREST)
    return frame


def line(text: str, x: int, y: int, w: int | None = None, h: int = 30, score: float = 0.95) -> OcrLine:
    return OcrLine(text, score, Rect(x, y, w if w is not None else 30 * len(text), h))


class ListOcr:
    """recognize 按调用顺序返回预设的行列表（用完之后一直返回最后一个）；元素是异常时抛出它。"""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls = 0

    def recognize(self, img):
        self.calls += 1
        if not self.results:
            return []
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return list(result)
