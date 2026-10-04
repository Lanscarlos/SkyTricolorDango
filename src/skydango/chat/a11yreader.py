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
PANEL_KEEP = 30.0  # 面板行报过的话留多久，等他头上同一句气泡来抵（约一条气泡的寿命）


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
        self._panel_reported: deque[tuple[str, str, float]] = deque()  # 面板行报过的，等气泡来抵（§3.3）

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

    def _take_panel_reported(self, speaker: str, key: str, now: float) -> bool:
        """面板行报过他这句（比较键相同）就划掉一条、返回 True：一条抵一条（§3.3，反过来那一半）。"""
        while self._panel_reported and now - self._panel_reported[0][2] > PANEL_KEEP:
            self._panel_reported.popleft()
        for item in self._panel_reported:
            if item[0] == speaker and bubble_key(item[1]) == key:
                self._panel_reported.remove(item)
                return True
        return False

    # ---- 头顶气泡 ----
    def _counted(self, key: str, now: float) -> bool:
        """这句现在还记在某个好友名下（他的标签 `BUBBLE_KEEP` 秒内看到过）。"""
        return any(
            key in counts and now - self._tag_seen.get(name, now) <= BUBBLE_KEEP
            for name, counts in self._counts.items()
        )

    def _note_loose(self, bubbles: list[Bubble], now: float, report: bool) -> None:
        """挂不上名字的原文气泡：新冒出来的、不是团子自己的、也没认出是谁的 → 想看一眼面板。"""
        loose: set[str] = set()
        new = False
        for b in bubbles:
            key = bubble_key(b.text)
            if not key:
                continue
            loose.add(key)
            if key in self._loose or self._counted(key, now):
                continue  # 同一句只记一次；标签闪一下 / 人走远了，这句已经知道是谁的
            if not self.self_filter.is_self(b.text.strip(), now):
                new = True  # 好友离远了、标签淡了：等面板行带着说话人报
        self._loose = loose
        if new and report:
            self._peek = "bubble_text"
            log.debug("有挂不上名字的气泡，想看一眼面板")

    def _forget_stale(self, in_view: set[str], now: float) -> None:
        """名字标签一时看不见的：计数先留着（标签闪一下不重报），`BUBBLE_KEEP` 秒没见才忘。"""
        for name in list(self._counts):
            if name not in in_view and now - self._tag_seen.get(name, now) > BUBBLE_KEEP:
                del self._counts[name]
                for k in [k for k in self._orig if k[0] == name]:
                    del self._orig[k]

    def _count(self, name: str, bubbles: list[Bubble]) -> tuple[Counter[str], dict[str, tuple], bool]:
        """他头上每句（比较键）几个、每句的框、是不是在打字；顺带记 / 清每句第一次出现时的原文。"""
        cur: Counter[str] = Counter()
        boxes: dict[str, tuple] = {}
        typing = False
        for b in bubbles:
            if b.typing_only:
                typing = True
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
                typing = True  # 这句后面挂上了点点：他在打下一句
        for k in [k for k in self._orig if k[0] == name and k[1] not in cur]:
            del self._orig[k]
        return cur, boxes, typing

    def _bubble_messages(self, view: UiView, now: float, report: bool) -> list[Message]:
        """更新每个好友头上的气泡计数；`report` 为假（面板开着）或第一份快照时只记成见过、不报。"""
        report = report and self._bubbles_ready
        self._bubbles_ready = True

        groups: dict[str, list[Bubble]] = {t.name: [] for t in view.tags}
        for b in view.bubbles:
            if b.speaker is not None:
                groups.setdefault(b.speaker, []).append(b)
        # 陌生人 / 挂不上名字的点点不管
        self._note_loose([b for b in view.bubbles if b.speaker is None and not b.typing_only], now, report)
        self._forget_stale(set(groups), now)

        fresh: list[Message] = []
        self._typing = []
        for name, bubbles in groups.items():
            self._tag_seen[name] = now
            prev = self._counts.get(name, Counter())
            cur, boxes, typing = self._count(name, bubbles)
            self._counts[name] = cur
            if typing:
                self._typing.append(name)
            for key, n in cur.items():
                for _ in range(n - prev[key]):
                    if not report:
                        self._take_panel_reported(name, key, now)  # 面板开着时看到了：面板那条抵掉
                        continue
                    text = self._orig[(name, key)]
                    if not self._passes(text, now) or self._take_panel_reported(name, key, now):
                        continue
                    fresh.append(Message(text, _rect(boxes[key]), now, name, source="bubble"))
                    self._expire_reported(now)
                    self._reported.append((name, text, now))
                    if self.trace_path:
                        self._trace_bubble(now, name, text)
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
            self._panel_reported.append((row.speaker, row.text.strip(), now))
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
        fresh = self._panel_messages(view, now)
        self._bubble_messages(view, now, report=False)  # 在面板行之后：同一份里的气泡能抵掉刚报的面板行
        return fresh


# ---- 读不到自动退回 OCR（spec §5）----
RESTART_DELAYS = (2.0, 5.0, 10.0)  # 客户端死了以后第几次重启前等多久；用完了又死 → 这次 run 剩下的时间都用 OCR
STALE = 30.0  # 客户端活着、这么久没收到快照（心跳每秒一份）→ 退回 OCR
MAX_AGE = 2.0  # 比这旧的快照不算（客户端死了以后最后一份会一直留着）


class FallbackReader:
    """先读无障碍节点（`A11yChatReader`），读不到自动退回 OCR（`ChatReader`），接口同两者。

    - 健康（`client.latest(max_age=MAX_AGE)` 拿得到）：照常交给 `a11y`
    - 客户端死了：按 `RESTART_DELAYS` 重启，重启后收到快照次数清零；用完了又死 → 切到 OCR
    - 活着但 `STALE` 秒没快照 → 切到 OCR
    - 不健康期间 `read()` 返回空，面板开没开借 OCR reader 看截图（便宜，不跑 OCR），免得面板管理器乱按键
    - 切到 OCR 后这次 run 剩下的时间都交给 `ocr`（它从空基准开始，不会把历史当新消息；切换期间漏的几句不补）
    """

    def __init__(self, client, a11y: A11yChatReader, ocr) -> None:
        self.client = client  # device.a11y.A11yReader
        self.a11y = a11y
        self.ocr = ocr  # chat.reader.ChatReader
        self.using_ocr = False
        self.reason = ""  # 退回 OCR 的原因
        self._ok = False  # 最近一次 read() 时无障碍是健康的
        self._good_at: float | None = None  # 最后一次拿到新鲜快照
        self._dead_at: float | None = None  # 发现客户端死了的时间（等重启）
        self._restarts = 0
        self._start_error = ""
        self._stopped = False
        self._closed_since: float | None = None  # 无障碍模式下自己维护的“面板从什么时候关着”
        self._trace_path: Path | None = None

    # ---- 起停 ----
    def start(self) -> None:
        try:
            self.client.start()
        except Exception as exc:
            self._to_ocr(str(exc) or type(exc).__name__)

    def stop(self) -> None:
        self._stopped = True  # 收尾时身体还可能借 reader 看面板：别再重启客户端
        try:
            self.client.stop()
        except Exception:
            log.warning("停无障碍客户端出错", exc_info=True)

    def describe(self) -> str:
        return f"OCR（无障碍读不到：{self.reason}）" if self.using_ocr else "无障碍"

    def _to_ocr(self, reason: str) -> None:
        self.using_ocr, self.reason, self._ok = True, reason, False
        log.warning("读聊天退回 OCR：%s", reason)
        self.ocr.panel_closed_since = self._closed_since  # 面板关着的时间接着算，不跳
        self.stop()

    # ---- 健康检查 ----
    def _restart(self, now: float) -> None:
        self._restarts += 1
        self._dead_at = None
        self._good_at = now  # 重启后给它 STALE 秒出第一份快照
        log.warning("无障碍客户端退出了，第 %d 次重启", self._restarts)
        try:
            self.client.start()
        except Exception as exc:  # 起不来就当又死了一次：下一圈看到 alive 为假接着等
            self._start_error = str(exc) or type(exc).__name__
            log.warning("重启无障碍客户端失败：%s", self._start_error)

    def _check(self, now: float) -> bool:
        """这一圈无障碍能不能用；顺带重启 / 切到 OCR。"""
        client = self.client
        if not client.alive:
            if self._stopped:
                return False
            if self._dead_at is None:
                if self._restarts >= len(RESTART_DELAYS):
                    self._to_ocr(client.error or self._start_error or "客户端反复退出")
                    return False
                self._dead_at = now
            if now >= self._dead_at + RESTART_DELAYS[self._restarts]:
                self._restart(now)
            return False
        if client.latest(max_age=MAX_AGE) is not None:
            if self._restarts:
                log.info("无障碍客户端恢复了")
            self._good_at, self._restarts, self._start_error = now, 0, ""
            return True
        if self._good_at is None:
            self._good_at = now
        if now - self._good_at > STALE:
            self._to_ocr(f"{STALE:.0f} 秒没收到快照")
        return False

    def _live(self) -> bool:
        return not self.using_ocr and self.client.alive and self.client.latest(max_age=MAX_AGE) is not None

    # ---- 读 ----
    def read(self, frame, now: float) -> list[Message]:
        if not self.using_ocr:
            self._ok = self._check(now)
        if self.using_ocr:
            return self.ocr.read(frame, now)
        if self._ok:
            fresh = self.a11y.read(frame, now)
            self._closed_since = self.a11y.panel_closed_since
            return fresh
        if frame is not None:  # 不健康：借 OCR reader 看截图里的输入框
            if self.ocr.panel_visible(frame):
                self._closed_since = None
            elif self._closed_since is None:
                self._closed_since = now
        return []

    def panel_visible(self, frame=None) -> bool:
        if self._live():
            return self.a11y.panel_visible(frame)
        if frame is None:
            return self.panel_closed_since is None
        return self.ocr.panel_visible(frame)

    @property
    def panel_closed_since(self) -> float | None:
        return self.ocr.panel_closed_since if self.using_ocr else self._closed_since

    @property
    def settling(self) -> bool:
        return self.ocr.settling if self.using_ocr else self.a11y.settling

    @property
    def reads_bubbles(self) -> bool:
        return not self.using_ocr

    @property
    def trace_path(self) -> Path | None:
        return self._trace_path

    @trace_path.setter
    def trace_path(self, path: Path | None) -> None:
        self._trace_path = self.a11y.trace_path = self.ocr.trace_path = path

    def _current(self):
        return self.ocr if self.using_ocr else self.a11y

    def log_rows(self, frame=None) -> list[LogRow]:
        return self._current().log_rows(frame)

    def detect(self, frame=None) -> list[Detection]:
        return self._current().detect(frame)

    # ---- 给面板管理器（OCR 模式 / 不健康时没有）----
    def tags_in_view(self) -> list[str]:
        return self.a11y.tags_in_view() if self._ok else []

    def typing(self) -> list[str]:
        return self.a11y.typing() if self._ok else []

    def want_peek(self) -> str | None:
        return self.a11y.want_peek() if self._ok else None
