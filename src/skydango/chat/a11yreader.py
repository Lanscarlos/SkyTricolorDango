"""用无障碍节点读聊天（代替 OCR）：面板行精确对齐，不会读错字、不用等确认。

接口照 `ChatReader`（`read` / `panel_visible` / `detect` / `log_rows` / `panel_closed_since` / `settling` / `trace_path`），
另外有 `tags_in_view` / `typing` / `want_peek` 给面板管理器用。
设计见 spec 2026-10-04-a11y-chat-reader-design.md §3：面板开着读面板行（§3.1），关着读好友头顶的气泡（§3.2），
两边读到同一句只报一次（§3.3）。
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, deque
from collections.abc import Callable
from pathlib import Path

from ..config import ChatConfig, OcrConfig
from ..device.a11y import Snapshot
from ..vision.a11yui import Bubble, PanelRow, UiView, bubble_key, classify
from ..vision.bubbles import Rect
from ..vision.chatlog import LogRow
from .reader import Detection, Message
from .tracker import SelfFilter

log = logging.getLogger(__name__)

_EMPTY = UiView(False, False, (), (), (), None)

REPORTED_KEEP = 180.0  # 气泡报过的话留多久，等面板行来抵（§3.3）
BUBBLE_KEEP = 30.0  # 名字标签看不见（闪一下 / 走出画面）时，他头上的气泡计数还记多久（气泡 20 多秒淡掉）


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
        # 头顶气泡（§3.2）
        self._bubbles_ready = False  # 看过第一份游戏画面了：那时已有的气泡都算见过
        self._counts: dict[str, Counter[str]] = {}  # 说话人 → 他头上每句（比较键）现在几个
        self._tag_seen: dict[str, float] = {}  # 说话人 → 最后一次看到他的名字标签
        self._orig: dict[tuple[str, str], str] = {}  # (说话人, 比较键) → 这句第一次出现时的原文
        self._loose: set[str] = set()  # 挂不上名字的原文气泡（比较键）：同一句只记一次“想看一眼”
        self._peek: str | None = None
        self._typing: list[str] = []
        self._reported: deque[tuple[str, str, float]] = deque()  # 气泡报过的 (说话人, 原文, 时间)（§3.3）

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

    # ---- 给面板管理器 ----
    def tags_in_view(self) -> list[str]:
        """最近一次 `read()` 看得见的好友名字标签。"""
        return list(dict.fromkeys(t.name for t in self._view.tags))

    def typing(self) -> list[str]:
        """最近一次 `read()` 里正在打字的好友：头上有只有点点的气泡，或者某句后面挂上了点点。"""
        return list(self._typing)

    def want_peek(self) -> str | None:
        """有挂不上名字的原文气泡时返回 "bubble_text"（想看一眼面板认说话人）；取一次就清掉。"""
        peek, self._peek = self._peek, None
        return peek

    # ---- 调试记录 ----
    def _trace(self, now: float, rows: list[PanelRow], added: set[int]) -> None:
        lines = [f"== {time.strftime('%H:%M:%S')} t={now:.1f} 新增行={sorted(added)}"]
        for i, r in enumerate(rows):
            lr = LogRow(r.speaker, r.text, r.is_self, _rect(r.box))
            lines.append(f"  {'*' if i in added else ' '} y={r.box[1]:4d} {lr.display()}")
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _trace_bubble(self, now: float, speaker: str, text: str) -> None:
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(f"== {time.strftime('%H:%M:%S')} t={now:.1f} 气泡 {speaker}：{text}\n")

    # ---- 过滤 / 去重 ----
    def _passes(self, text: str, now: float) -> bool:
        """`min_chars`、`ignore_patterns`、`SelfFilter`（面板行和气泡共用）。"""
        if len(text.replace(" ", "")) < self.ocr_cfg.min_chars:
            return False
        if any(p.search(text) for p in self.ignore):
            return False
        return not self.self_filter.is_self(text, now)

    def _expire_reported(self, now: float) -> None:
        while self._reported and now - self._reported[0][2] > REPORTED_KEEP:
            self._reported.popleft()

    def _take_reported(self, speaker: str, text: str, now: float) -> bool:
        """气泡报过同一句（说话人、原文相同）就划掉一条、返回 True：一条抵一条（§3.3）。"""
        self._expire_reported(now)
        text = text.strip()
        for item in self._reported:
            if item[0] == speaker and item[1] == text:
                self._reported.remove(item)
                return True
        return False

    # ---- 头顶气泡 ----
    def _bubble_messages(self, view: UiView, now: float, report: bool) -> list[Message]:
        """更新每个好友头上的气泡计数；`report` 为假（面板开着）或第一份快照时只记成见过、不报。"""
        report = report and self._bubbles_ready
        self._bubbles_ready = True

        groups: dict[str, list[Bubble]] = {t.name: [] for t in view.tags}
        loose: set[str] = set()
        new_loose = False
        for b in view.bubbles:
            if b.speaker is not None:
                groups.setdefault(b.speaker, []).append(b)
                continue
            if b.typing_only:
                continue  # 陌生人 / 挂不上名字的点点：不管
            key = bubble_key(b.text)
            if not key:
                continue
            loose.add(key)
            if key not in self._loose and not self.self_filter.is_self(b.text.strip(), now):
                new_loose = True  # 团子自己刚说的丢掉；别的（好友离远了、标签淡了）等面板行带着说话人报
        self._loose = loose
        if new_loose and report:
            self._peek = "bubble_text"
            log.debug("有挂不上名字的气泡，想看一眼面板")

        # 名字标签一时看不见的：计数先留着（标签闪一下不重报），太久没见才忘
        for name in list(self._counts):
            if name not in groups and now - self._tag_seen.get(name, now) > BUBBLE_KEEP:
                del self._counts[name]
                for k in [k for k in self._orig if k[0] == name]:
                    del self._orig[k]

        fresh: list[Message] = []
        typing: list[str] = []
        for name, bubbles in groups.items():
            self._tag_seen[name] = now
            prev = self._counts.get(name, Counter())
            cur: Counter[str] = Counter()
            boxes: dict[str, tuple[int, int, int, int]] = {}
            is_typing = False
            for b in bubbles:
                if b.typing_only:
                    is_typing = True
                    continue
                text = b.text.strip()
                key = bubble_key(text)
                if not key:
                    continue
                cur[key] += 1
                boxes[key] = b.box
                orig = self._orig.setdefault((name, key), text)
                if text == key and orig != key:
                    self._orig[(name, key)] = orig = key  # 第一次看到时正挂着打字的点：点是动画
                if text != orig:
                    is_typing = True  # 这句后面挂上了点点：他在打下一句
            for k in [k for k in self._orig if k[0] == name and k[1] not in cur]:
                del self._orig[k]
            self._counts[name] = cur
            if is_typing:
                typing.append(name)
            if not report:
                continue
            for key, n in cur.items():
                for _ in range(n - prev[key]):
                    text = self._orig[(name, key)]
                    if not self._passes(text, now):
                        continue
                    fresh.append(Message(text, _rect(boxes[key]), now, name, source="bubble"))
                    self._expire_reported(now)
                    self._reported.append((name, text, now))
                    if self.trace_path:
                        self._trace_bubble(now, name, text)
        self._typing = typing
        return fresh

    # ---- 面板行 ----
    def _panel_messages(self, view: UiView, now: float) -> list[Message]:
        """面板行和上一次对齐，把新行变成消息；头顶气泡已经报过的同一句不再报。"""
        cur = [r.key() for r in view.rows]
        if not cur:
            return []  # 开着但一行都没有（刚登录、历史清空）：基准留着，等第一句进来对得上
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
            if row.is_self or row.masked or not self._passes(row.text, now):
                continue
            if self._take_reported(row.speaker, row.text, now):
                continue
            fresh.append(Message(row.text, _rect(row.box), now, row.speaker, source="panel"))
        return fresh

    def read(self, frame, now: float) -> list[Message]:
        """返回这一次新出现的、不是自己说的消息：面板开着读面板行，关着读好友头顶的气泡。"""
        view = self._classify(frame)
        self._view = view
        if not view.in_game:
            self._typing = []  # 气泡计数留着：回到游戏后还在的气泡不重报
            if self.panel_closed_since is None:
                self.panel_closed_since = now
                log.warning("没看到游戏画面，暂停读取")
            return []
        if not view.panel_open:
            if self.panel_closed_since is None:
                self.panel_closed_since = now
                log.info("聊天记录面板关着，改读好友头顶的气泡")
            return self._bubble_messages(view, now, report=True)
        if self.panel_closed_since is not None:
            log.info("聊天记录面板又出现了，读面板行")
            self.panel_closed_since = None
        self._bubble_messages(view, now, report=False)
        return self._panel_messages(view, now)
