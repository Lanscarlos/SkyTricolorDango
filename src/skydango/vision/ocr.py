"""OCR 适配层。默认用 RapidOCR（PaddleOCR 模型的 ONNX 版，CPU 上一个气泡几十毫秒）。"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .bubbles import Rect


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float
    box: Rect


class OcrEngine(Protocol):
    def recognize(self, img: np.ndarray) -> list[OcrLine]: ...


def _box_from_points(points: Any) -> Rect:
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return Rect(int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)))


class RapidOcrEngine:
    """兼容 rapidocr_onnxruntime 1.x 和 rapidocr 2.x/3.x 两套 API。"""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # 1.x：模型随包附带，可离线

            self._engine = RapidOCR()
            self._legacy = True
        except ImportError:
            try:
                from rapidocr import RapidOCR  # 2.x/3.x：首次运行会下载模型
            except ImportError as exc:
                raise ImportError("没有安装 OCR：pip install \"skydango[ocr]\"") from exc
            self._engine = RapidOCR()
            self._legacy = False

    def recognize(self, img: np.ndarray) -> list[OcrLine]:
        if img.size == 0:
            return []
        if self._legacy:
            result, _ = self._engine(img)
            return [OcrLine(str(text), float(score), _box_from_points(box)) for box, text, score in result or []]
        out = self._engine(img)
        boxes, txts, scores = getattr(out, "boxes", None), getattr(out, "txts", None), getattr(out, "scores", None)
        if boxes is None or txts is None:
            return []
        return [OcrLine(str(t), float(s), _box_from_points(b)) for b, t, s in zip(boxes, txts, scores)]


def _is_wide(ch: str) -> bool:
    """中日韩文字和全角标点：拼接时两边不需要空格。"""
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _concat(parts: list[str]) -> str:
    out = ""
    for part in parts:
        if out and not (_is_wide(out[-1]) or _is_wide(part[0])):
            out += " "
        out += part
    return out


def join_lines(lines: list[OcrLine]) -> str:
    """把一个气泡里的多个识别片段按阅读顺序拼成一句话。"""
    if not lines:
        return ""
    ordered = sorted(lines, key=lambda l: (l.box.y + l.box.h / 2, l.box.x))
    rows: list[list[OcrLine]] = []
    for line in ordered:
        center = line.box.y + line.box.h / 2
        if rows:
            ref = rows[-1][0]
            if abs(center - (ref.box.y + ref.box.h / 2)) <= ref.box.h * 0.5:
                rows[-1].append(line)
                continue
        rows.append([line])
    parts: list[str] = []
    for row in rows:
        kept: list[OcrLine] = []
        for line in sorted(row, key=lambda l: (l.box.x, -l.box.w)):
            text = line.text.strip()
            if not text:
                continue
            # OCR 有时会把标点单独再识别一次，框落在前一个片段里面，丢掉
            if kept:
                prev = kept[-1].box
                overlap = min(prev.x2, line.box.x2) - max(prev.x, line.box.x)
                if overlap >= 0.6 * line.box.w:
                    continue
            kept.append(line)
            parts.append(text)
    return _concat(parts)


def make_ocr(engine: str) -> OcrEngine:
    if engine == "rapidocr":
        return RapidOcrEngine()
    raise ValueError(f"不支持的 OCR 引擎: {engine}")
