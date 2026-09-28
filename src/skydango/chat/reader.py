"""从一帧画面里读出“新的”聊天消息。"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ChatConfig, OcrConfig, VisionConfig
from ..vision.bubbles import Rect, find_bubbles, roi_rect
from ..vision.chatlog import LogRow, changed_pixels, find_input_top, new_rows, parse_rows, text_signature
from ..vision.ocr import OcrEngine, OcrLine, join_lines
from .tracker import SeenTracker, SelfFilter, similar

log = logging.getLogger(__name__)

UNKNOWN_SPEAKERS = ("", "陌生人")  # 聊天记录面板里读不出说话人 / 陌生人的消息只显示"陌生人"


@dataclass(frozen=True)
class Message:
    text: str
    box: Rect
    seen_at: float
    speaker: str = ""  # 只有聊天记录面板能读到说话人


@dataclass(frozen=True)
class Detection:
    """一帧里识别到的一个文本块（不论新旧），调试和标注用。"""

    text: str
    box: Rect



def with_speaker_hint(env, messages: list[Message], now: float) -> list[Message]:
    """陌生人（或看不出是谁）的消息后面加注"说话的可能是画面上哪个人"（YOLO 感知层的 typing 气泡，二期 §3）。"""
    hint_of = getattr(env, "speaker_hint", None)
    if hint_of is None or not any(m.speaker in UNKNOWN_SPEAKERS for m in messages):
        return messages
    hint = hint_of(now)
    if not hint:
        return messages
    return [dataclasses.replace(m, text=m.text + hint) if m.speaker in UNKNOWN_SPEAKERS else m for m in messages]


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
        self.similarity = chat.similarity
        self._prev_keys: list[str] = []  # log 模式：上一帧面板里的行
        self._sig: np.ndarray | None = None  # 上次 OCR 时面板文字的“指纹”
        self._ocr_at = float("-inf")
        self._shrunk = 0  # 连续几帧行数骤减
        self.panel_closed_since: float | None = None  # 从什么时候开始看不到面板
        self._confirming = False  # 上一帧冒出了新行，等这一帧确认
        self.self_filter = self_filter
        self.ignore = [re.compile(p) for p in chat.ignore_patterns]
        self.trace_path: Path | None = None  # 设了就把面板每次变化记下来（运行目录里的 rows.log）

    def _good(self, lines: list[OcrLine]) -> list[OcrLine]:
        return [l for l in lines if l.score >= self.ocr_cfg.min_score and l.text.strip()]

    def _log_area(self, frame: np.ndarray) -> tuple[Rect, bool]:
        """(读取区域, 看没看到面板底部的输入框)。看到了就读到它上沿为止（面板高度随输入框开关变化）。"""
        height, width = frame.shape[:2]
        area = roi_rect(self.vision.log_roi, width, height)
        top = find_input_top(frame[:, area.x : area.x2])
        if top is None or top <= area.y:
            return area, False
        return Rect(area.x, area.y, area.w, top - 4 - area.y), True

    def panel_visible(self, frame: np.ndarray) -> bool:
        """聊天记录面板开着没：看面板底部的“聊天……”输入框（它是面板的一部分，面板关了就没有）。

        不能看“读不读得出字”：面板关着时 3D 场景里的名字标签也会被读成一行（实测因此没按 C，整轮读不到消息）。
        """
        return self._log_area(frame)[1]

    def log_rows(self, frame: np.ndarray, area: Rect | None = None) -> list[LogRow]:
        """log 模式：读出聊天记录面板里的所有行（坐标换算回整张图）。"""
        area = area or self._log_area(frame)[0]
        crop = area.crop(frame)
        rows = parse_rows(crop, self._good(self.ocr.recognize(crop)), self.vision.log_self_min_value)
        return [
            LogRow(r.speaker, r.text, r.is_self, Rect(r.box.x + area.x, r.box.y + area.y, r.box.w, r.box.h))
            for r in rows
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """识别画面里所有文本块（不做新旧判断）。"""
        height, width = frame.shape[:2]
        detections: list[Detection] = []
        if self.vision.mode == "log":
            detections = [Detection(r.display(), r.box) for r in self.log_rows(frame)]
        elif self.vision.mode == "bubble":
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

    def _trace(self, now: float, rows: list[LogRow], added: list[int]) -> None:
        """调试：面板内容每次变化都记一笔（每行的识别结果、哪几行被判成新的），排查漏读 / 晚读。"""
        lines = [f"== {time.strftime('%H:%M:%S')} t={now:.1f} 新增行={added}"]
        lines += [f"  {'*' if i in added else ' '} y={r.box.y:4d} {r.display()}" for i, r in enumerate(rows)]
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _read_log(self, frame: np.ndarray, now: float) -> list[Message]:
        area, visible = self._log_area(frame)
        if self.vision.log_require_panel:
            # 面板关着就不跑 OCR：不会误读场景里的字，也不会因为 3D 画面一直在动而每帧都 OCR（实测 CPU 拉满）
            if not visible:
                if self.panel_closed_since is None:
                    self.panel_closed_since = now
                    log.warning("没看到聊天记录面板（底部的“聊天……”输入框），暂停读取")
                return []
            if self.panel_closed_since is not None:
                log.info("聊天记录面板又出现了，继续读取")
                self.panel_closed_since = None
        sig = text_signature(area.crop(frame))
        # 面板里的文字没变就不跑 OCR（截图约 9 ms，OCR 约 0.4 s）；隔一阵还是强制识别一次，以防万一
        if (
            not self._confirming
            and self._sig is not None
            and changed_pixels(self._sig, sig) < self.vision.log_change_pixels
            and now - self._ocr_at < self.vision.log_max_skip
        ):
            return []
        self._sig, self._ocr_at = sig, now
        rows = self.log_rows(frame, area)
        if not rows:  # 面板被挡住 / 关掉了：保留上一帧的状态
            return []
        keys = [r.key() for r in rows]
        # 实测界面滚动 / 重绘的瞬间会截到只读出一两行的坏帧；拿它当基准，下一帧会把读过的行又当成新的。
        # 行数骤减先忽略；连续几帧都这样才认为面板真的变短了，接受为新基准（不回复里面的行）
        if len(self._prev_keys) >= 4 and len(rows) < 0.5 * len(self._prev_keys):
            self._shrunk += 1
            log.debug("面板行数从 %d 骤减到 %d（第 %d 帧）", len(self._prev_keys), len(rows), self._shrunk)
            if self._shrunk >= 3:
                self._prev_keys, self._shrunk = keys, 0
            return []
        self._shrunk = 0
        added = new_rows(self._prev_keys, keys, lambda a, b: similar(a, b, self.similarity))
        if self.trace_path and keys != self._prev_keys:
            self._trace(now, rows, added)
        # 新消息淡入时名字比内容晚出现（实测第一帧读成“?：嗯应该是正太”）：新行第一次出现先不报、
        # 也不更新基准，下一帧强制再识别一次，用完整的版本
        if added and not self._confirming:
            self._confirming = True
            return []
        self._confirming = False
        self._prev_keys = keys
        fresh: list[Message] = []
        for row in (rows[i] for i in added):
            if row.is_self or row.masked:
                continue
            if len(row.text.replace(" ", "")) < self.ocr_cfg.min_chars:
                continue
            if any(p.search(row.text) for p in self.ignore):
                continue
            if self.self_filter.is_self(row.text, now):
                continue
            fresh.append(Message(row.text, row.box, now, row.speaker))
        return fresh

    def read(self, frame: np.ndarray, now: float) -> list[Message]:
        """返回这一帧里新出现的、不是自己说的消息。"""
        if self.vision.mode == "log":
            return self._read_log(frame, now)
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
