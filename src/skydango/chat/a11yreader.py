"""用无障碍节点读聊天（代替 OCR）：面板行精确对齐，不会读错字、不用等确认。

接口照 `ChatReader`（`read` / `panel_visible` / `detect` / `log_rows` / `panel_closed_since` / `settling` / `trace_path`）。
设计见 spec 2026-10-04-a11y-chat-reader-design.md §3.1。头顶气泡的部分在后面的任务里加。
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

from ..config import ChatConfig, OcrConfig
from ..device.a11y import Snapshot
from ..vision.a11yui import PanelRow, UiView, classify
from ..vision.bubbles import Rect
from ..vision.chatlog import LogRow
from .reader import Detection, Message
from .tracker import SelfFilter

log = logging.getLogger(__name__)

_EMPTY = UiView(False, False, (), (), (), None)


def _rect(box: tuple[int, int, int, int]) -> Rect:
    l, t, r, b = box
    return Rect(l, t, r - l, b - t)


def align(prev: list, cur: list) -> int | None:
    """新一列里“上一列最后几行”的结尾下标 j（j 之后是新行）；对不上返回 None。

    找使 cur[j-k] == prev[-1-k] 连续成立的 k 最多的 j，并列取最大的 j（同一句连发两次时不会漏）。
    """
    best_k, best_j = 0, None
    for j in range(len(cur)):
        k = 0
        while k < len(prev) and j - k >= 0 and cur[j - k] == prev[-1 - k]:
            k += 1
        if k > 0 and k >= best_k:
            best_k, best_j = k, j
    return best_j


class A11yChatReader:
    settling = False  # 没有淡入动画要等
    reads_bubbles = True

    def __init__(
        self,
        source: Callable[[], Snapshot | None],
        friends: Callable[[], list[str]],
        chat: ChatConfig,
        ocr_cfg: OcrConfig,
        self_filter: SelfFilter,
    ) -> None:
        self.source = source
        self.friends = friends
        self.ocr_cfg = ocr_cfg
        self.self_filter = self_filter
        self.ignore = [re.compile(p) for p in chat.ignore_patterns]
        self.panel_closed_since: float | None = None
        self.trace_path: Path | None = None
        self._prev: list[tuple] = []  # 上一次看到面板时的全部行（含滚出可视范围的历史）
        self._view: UiView = _EMPTY
        self._traced: list[tuple] | None = None

    # ---- 取快照 / 分类 ----
    def _classify(self, frame) -> UiView:
        height, width = frame.shape[:2] if frame is not None else (1080, 1920)
        return classify(self.source(), width, height, self.friends())

    def view(self, frame=None) -> UiView:
        """最近一次 `read()` 分好类的结果。"""
        return self._view

    def panel_visible(self, frame=None) -> bool:
        return self._classify(frame).panel_open

    def log_rows(self, frame=None) -> list[LogRow]:
        return [
            LogRow(r.speaker, r.text, r.is_self, _rect(r.box)) for r in self._classify(frame).rows if r.visible
        ]

    def detect(self, frame=None) -> list[Detection]:
        return [Detection(r.display(), r.box) for r in self.log_rows(frame)]

    # ---- 调试记录 ----
    def _trace(self, now: float, rows: list[PanelRow], added: set[int]) -> None:
        lines = [f"== {time.strftime('%H:%M:%S')} t={now:.1f} 新增行={sorted(added)}"]
        for i, r in enumerate(rows):
            lr = LogRow(r.speaker, r.text, r.is_self, _rect(r.box))
            lines.append(f"  {'*' if i in added else ' '} y={r.box[1]:4d} {lr.display()}")
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    # ---- 面板行 ----
    def _panel_messages(self, view: UiView, now: float) -> list[Message]:
        """面板行和上一次对齐，把新行变成消息（Task 4 在这里接“头顶气泡已经报过”的去重）。"""
        cur = [r.key() for r in view.rows]
        added_idx: list[int] = []
        j = align(self._prev, cur) if self._prev else None
        if j is None:
            if self._prev and cur:
                log.info("面板历史对不上，当新基准")
        else:
            added_idx = list(range(j + 1, len(cur)))
        self._prev = cur
        if self.trace_path:
            visible = [r for r in view.rows if r.visible]
            vkeys = [r.key() for r in visible]
            if vkeys != self._traced:
                self._traced = vkeys
                added = {i for i, r in enumerate(visible) if any(r is view.rows[a] for a in added_idx)}
                self._trace(now, visible, added)
        fresh: list[Message] = []
        for i in added_idx:
            row = view.rows[i]
            if row.is_self or row.masked:
                continue
            if len(row.text.replace(" ", "")) < self.ocr_cfg.min_chars:
                continue
            if any(p.search(row.text) for p in self.ignore):
                continue
            if self.self_filter.is_self(row.text, now):
                continue
            fresh.append(Message(row.text, _rect(row.box), now, row.speaker, source="panel"))
        return fresh

    def read(self, frame, now: float) -> list[Message]:
        """返回这一次新出现的、不是自己说的消息。"""
        view = self._classify(frame)
        self._view = view
        if not view.panel_open:
            if self.panel_closed_since is None:
                self.panel_closed_since = now
                log.warning("没看到聊天记录面板，暂停读取")
            return []
        if self.panel_closed_since is not None:
            log.info("聊天记录面板又出现了，继续读取")
            self.panel_closed_since = None
        return self._panel_messages(view, now)
