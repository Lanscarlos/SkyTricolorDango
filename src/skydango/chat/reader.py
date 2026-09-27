"""从一帧画面里读出“新的”聊天消息。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from ..config import ChatConfig, OcrConfig, VisionConfig
from ..vision.bubbles import Rect, find_bubbles, roi_rect
from ..vision.ocr import OcrEngine, OcrLine, join_lines
from .tracker import SeenTracker, SelfFilter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    text: str
    box: Rect
    seen_at: float


@dataclass(frozen=True)
class Detection:
    """一帧里识别到的一个文本块（不论新旧），调试和标注用。"""

    text: str
    box: Rect


class ChatReader:
    def __init__(
        self,
        ocr: OcrEngine,
        vision: VisionConfig,
        ocr_cfg: OcrConfig,
        chat: ChatConfig,
        self_filter: SelfFilter,
    ) -> None:
        self.ocr = ocr
        self.vision = vision
        self.ocr_cfg = ocr_cfg
        self.tracker = SeenTracker(chat.dedupe_ttl, chat.similarity)
        self.self_filter = self_filter
        self.ignore = [re.compile(p) for p in chat.ignore_patterns]

    def _good(self, lines: list[OcrLine]) -> list[OcrLine]:
        return [l for l in lines if l.score >= self.ocr_cfg.min_score and l.text.strip()]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """识别画面里所有文本块（不做新旧判断）。"""
        height, width = frame.shape[:2]
        detections: list[Detection] = []
        if self.vision.mode == "bubble":
            for rect in find_bubbles(frame, self.vision.bubble, self.vision.roi):
                crop_rect = rect.pad(self.vision.bubble.padding, width, height)
                text = join_lines(self._good(self.ocr.recognize(crop_rect.crop(frame))))
                if text:
                    detections.append(Detection(text, rect))
        elif self.vision.mode == "roi":
            area = roi_rect(self.vision.roi, width, height)
            for line in self._good(self.ocr.recognize(area.crop(frame))):
                box = Rect(line.box.x + area.x, line.box.y + area.y, line.box.w, line.box.h)
                detections.append(Detection(line.text.strip(), box))
        else:
            raise ValueError(f"未知的 vision.mode: {self.vision.mode}")
        return detections

    def read(self, frame: np.ndarray, now: float) -> list[Message]:
        """返回这一帧里新出现的、不是自己说的消息。"""
        fresh: list[Message] = []
        for det in self.detect(frame):
            text = det.text
            if len(text.replace(" ", "")) < self.ocr_cfg.min_chars:
                continue
            if any(p.search(text) for p in self.ignore):
                continue
            # 先更新去重状态，再判断是不是自己：自己的气泡也要标记为“已见”
            if not self.tracker.observe(text, now):
                continue
            if self.self_filter.is_self(text, now):
                log.debug("忽略自己发的消息: %s", text)
                continue
            fresh.append(Message(text, det.box, now))
        return fresh
