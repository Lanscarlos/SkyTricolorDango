"""YOLO 感知层：持续检测人物 / 名字标签 / 互动圆圈，汇总成"身边有谁、谁在发起互动、有没有陌生人"。

对外接口和 vision.env.EnvWatcher 一样（observe / nearby / describe / requests / labels / circles / last_seen），
身体、社交、眼睛不用改；`[perception] enabled = false` 就退回 EnvWatcher 的定时整图 OCR。

每一帧：
1. YOLO 出框（player / name_tag / social_ring / self），去掉面板挡住的、底部输入栏的、团子自己
2. 追踪：框接成轨迹
3. 名字标签轨迹：新出现就裁小图只跑 OCR 识别（不跑检测），多读几次投票，对上 friends.md 就是那个好友
   —— 身份跟着名字走，不跟着轨迹 id 走：转视角轨迹断了，新轨迹一读出名字就接回同一个人
4. 圆圈：挂到正上方的名字标签上，用现有的剪影模板认图标（牵手 / 拥抱 / ……）；上面没有标签的圆圈算陌生人的
5. 人物：一直没有名字标签、离得不太远 → 陌生人

以上规则和阈值都还**没在真机验证**（模型还没训练），见 docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable

import numpy as np

from ..chat.tracker import normalize, similar
from ..config import EnvConfig, PerceptionConfig
from ..game.social import IDLE, Request
from .bubbles import Rect, roi_rect
from .detect import Detection, Detector
from .ocr import OcrEngine, join_lines
from .track import Track, Tracker, iou

log = logging.getLogger(__name__)

STRANGER = "陌生人"  # 陌生人头顶没有名字：发起的请求用这个名字（社交规则里按 stranger 处理）
ICON_OFFSET = 2.23  # 圆圈中心在名字标签上沿往下这么多倍标签高度（game-ops §6 实测）


def _center(r: Rect) -> tuple[float, float]:
    return r.x + r.w / 2, r.y + r.h / 2


def _inside(point: tuple[float, float], rect: Rect) -> bool:
    return rect.x <= point[0] < rect.x2 and rect.y <= point[1] < rect.y2


class PerceptionWatcher:
    def __init__(
        self,
        detector: Detector,
        ocr: OcrEngine,
        cfg: PerceptionConfig,
        env_cfg: EnvConfig,
        names: Callable[[], list[str]],
        log_roi: list[float],
        icons=None,  # game.social.IconClassifier
        background: bool = True,
        capture: Callable[[], np.ndarray] | None = None,  # capture = "own" 时感知线程自己截图用
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.names = names
        self.log_roi = log_roi
        self.icons = icons
        self.background = background
        self.capture = capture
        self.clock = clock
        self.keep = cfg.keep  # 身体说"走开了"时用
        self.tracker = Tracker(cfg.track_buffer, cfg.track_iou)
        self.requests: dict = {}  # 名字 → game.social.Request
        self.labels: dict[str, tuple[int, int, int, int, float]] = {}
        self.circles: dict[str, tuple[str | None, float]] = {}
        self.last_seen: dict[str, float] = {}
        self.place = ""  # YOLO 不认地名；留着和 EnvWatcher 接口一致
        self.place_at = float("-inf")
        self.last_dets: list[Detection] = []  # 最近一帧的检测（调试画框用）
        self.last_tracks: list[Track] = []
        self.timings: deque[tuple[float, float]] = deque(maxlen=300)  # (检测 ms, 整帧 ms)
        self._strangers: deque[tuple[float, int]] = deque()  # (时间, 这一帧有几个陌生人)
        self._panel_visible = False
        self._last_run = float("-inf")
        self._busy = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- 帧从哪来 ----
    def observe(self, frame: np.ndarray, now: float, panel_visible: bool) -> None:
        """身体主循环每圈调一次。capture = "own" 时只记下面板开没开，帧由感知线程自己截。"""
        self._panel_visible = panel_visible
        if self.cfg.capture == "own" and self.capture is not None:
            self._ensure_thread()
            return
        if now - self._last_run < 1.0 / max(self.cfg.fps, 0.1):
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self._last_run = now
        if self.background:
            threading.Thread(target=self._guarded, args=(frame, now, panel_visible), name="perception", daemon=True).start()
        else:
            self._guarded(frame, now, panel_visible)

    def _guarded(self, frame: np.ndarray, now: float, panel_visible: bool) -> None:
        try:
            self.process(frame, now, panel_visible)
        except Exception:
            log.exception("感知出错")
        finally:
            with self._lock:
                self._busy = False

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="perception", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / max(self.cfg.fps, 0.1)
        while not self._stop.is_set():
            started = self.clock()
            try:
                self.process(self.capture(), started, self._panel_visible)
            except Exception:
                log.exception("感知出错")
                self._stop.wait(1.0)
            self._stop.wait(max(0.0, period - (self.clock() - started)))

    def stop(self) -> None:
        self._stop.set()

    # ---- 一帧 ----
    def process(self, frame: np.ndarray, now: float, panel_visible: bool) -> None:
        started = time.perf_counter()
        height, width = frame.shape[:2]
        dets = self._filter(self.detector.detect(frame), width, height, panel_visible)
        detected = time.perf_counter()
        tracks = self.tracker.update(dets, now)
        selfs = [t for t in tracks if t.cls == "self"]
        players = [t for t in tracks if t.cls == "player" and not self._is_self(t, selfs)]
        tags = [t for t in tracks if t.cls == "name_tag"]
        rings = [t for t in tracks if t.cls == "social_ring"]

        friends = self.names()
        for tag in tags:
            self._read_name(frame, tag, now, friends)

        owner: dict[int, Track] = {}  # 名字标签轨迹 id → 它下面的圆圈
        orphans: list[Track] = []  # 上面没有名字标签的圆圈
        for ring in rings:
            tag = self._tag_above(ring, tags)
            if tag is not None and tag.id not in owner:
                owner[tag.id] = ring
            else:
                orphans.append(ring)

        seen = []
        for tag in tags:
            name = tag.data.get("name")
            if not name:
                continue
            b = tag.box
            self.last_seen[name] = now
            self.labels[name] = (b.x, b.y, b.w, b.h, now)
            seen.append(name)
            ring = owner.get(tag.id)
            kind = self._classify(frame, ring) if ring is not None else None  # 牵着手时圆圈会消失 → None
            self.circles[name] = (kind, now)
            if kind and kind != IDLE:
                if name not in self.requests or self.requests[name].kind != kind:
                    log.info("%s 发起了互动：%s", name, kind)
                self.requests[name] = Request(name, kind, self._ring_center(ring), now)
            else:
                self.requests.pop(name, None)

        stranger_req = None
        for ring in orphans:
            kind = self._classify(frame, ring)
            if kind and kind != IDLE:
                stranger_req = Request(STRANGER, kind, self._ring_center(ring), now)
                break
        if stranger_req is not None:
            if STRANGER not in self.requests or self.requests[STRANGER].kind != stranger_req.kind:
                log.info("陌生人发起了互动：%s", stranger_req.kind)
            self.requests[STRANGER] = stranger_req
        else:
            self.requests.pop(STRANGER, None)

        strangers = 0
        for player in players:
            if self._tag_over(player, tags) is not None:
                player.data["tagged"] = True  # 这条轨迹上出现过名字标签：不是陌生人（标签被挡一下不改判）
            is_stranger = (
                not player.data.get("tagged")
                and now - player.first >= self.cfg.stranger_after
                and player.box.h >= self.cfg.stranger_min_height * height
            )
            player.data["stranger"] = is_stranger
            strangers += is_stranger
        with self._lock:  # 身体线程会同时读（strangers()）
            self._strangers.append((now, strangers))
            while self._strangers and now - self._strangers[0][0] > self.cfg.keep:
                self._strangers.popleft()

        self.last_dets, self.last_tracks = dets, tracks
        self.timings.append(((detected - started) * 1000, (time.perf_counter() - started) * 1000))
        if seen:
            log.debug("旁边看到: %s", "、".join(seen))

    def _filter(self, dets: list[Detection], width: int, height: int, panel_visible: bool) -> list[Detection]:
        area = roi_rect(self.env_cfg.roi, width, height)  # 底部输入栏不要
        panel = roi_rect(self.log_roi, width, height) if panel_visible else None  # 面板里的"- 名字"不是名字标签
        mine = roi_rect(self.cfg.self_roi, width, height) if self.cfg.self_roi else None
        out = []
        for det in dets:
            c = _center(det.box)
            if not _inside(c, area) or (panel is not None and _inside(c, panel)):
                continue
            if mine is not None and det.cls == "player" and _inside(c, mine):
                continue
            out.append(det)
        return out

    @staticmethod
    def _is_self(player: Track, selfs: list[Track]) -> bool:
        return any(iou(player.box, s.box) >= 0.5 for s in selfs)

    def _read_name(self, frame: np.ndarray, tag: Track, now: float, friends: list[str]) -> None:
        """名字标签轨迹：刚出现就读一次，之后隔 ocr_retry 秒再读，读够 ocr_votes 次（且认出是谁）就不再读。"""
        votes: Counter = tag.data.setdefault("votes", Counter())
        tries = tag.data.get("tries", 0)
        if tries and (now - tag.data.get("ocr_at", float("-inf")) < self.cfg.ocr_retry):
            return
        if tries >= self.cfg.ocr_votes and votes:
            return
        tag.data["tries"], tag.data["ocr_at"] = tries + 1, now
        text = self._ocr(tag.box.pad(4, frame.shape[1], frame.shape[0]).crop(frame))
        if len(normalize(text)) < 2:
            return
        tag.data["text"] = text
        name = next((n for n in friends if similar(text, n, 0.75)), None)
        if name:
            votes[name] += 1
            tag.data["name"] = votes.most_common(1)[0][0]

    def _ocr(self, crop: np.ndarray) -> str:
        read_line = getattr(self.ocr, "read_line", None)
        if read_line is not None:
            line = read_line(crop)
            return line.text if line is not None else ""
        return join_lines(self.ocr.recognize(crop))

    @staticmethod
    def _tag_above(ring: Track, tags: list[Track]) -> Track | None:
        """圆圈正上方的名字标签：圆圈中心在标签上沿往下约 2.23 倍标签高度、水平居中。"""
        rx, ry = _center(ring.box)
        best, best_err = None, float("inf")
        for tag in tags:
            tx = tag.box.x + tag.box.w / 2
            dy = (ry - tag.box.y) / tag.box.h
            if not 1.2 <= dy <= 3.5 or abs(rx - tx) > max(tag.box.w / 2, ring.box.w):
                continue
            err = abs(dy - ICON_OFFSET) + abs(rx - tx) / tag.box.h
            if err < best_err:
                best, best_err = tag, err
        return best

    @staticmethod
    def _tag_over(player: Track, tags: list[Track]) -> Track | None:
        """人物头顶的名字标签：水平上差不多对齐，标签在人物上沿附近（上方一个半身高到身体上半截之间）。"""
        p = player.box
        px = p.x + p.w / 2
        best, best_err = None, float("inf")
        for tag in tags:
            tx = tag.box.x + tag.box.w / 2
            if not p.x - p.w * 0.25 <= tx <= p.x2 + p.w * 0.25:
                continue
            if not p.y - 1.5 * p.h <= tag.box.y2 <= p.y + 0.5 * p.h:
                continue
            err = abs(tx - px) + abs(tag.box.y2 - p.y)
            if err < best_err:
                best, best_err = tag, err
        return best

    @staticmethod
    def _ring_center(ring: Track) -> tuple[int, int]:
        cx, cy = _center(ring.box)
        return int(cx), int(cy)

    def _classify(self, frame: np.ndarray, ring: Track) -> str | None:
        if self.icons is None:
            return None
        cx, cy = self._ring_center(ring)
        r = round(56 * frame.shape[0] / 1080)
        region = frame[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
        return self.icons.classify(region)[0]

    # ---- 给身体 / 提示词用的（和 EnvWatcher 一样） ----
    def nearby(self, now: float) -> list[str]:
        return [n for n in self.names() if now - self.last_seen.get(n, float("-inf")) <= self.cfg.keep]

    def strangers(self, now: float) -> int:
        """最近 keep 秒里同一帧最多看到几个陌生人（轨迹会断，所以不数轨迹条数）。"""
        with self._lock:
            recent = list(self._strangers)
        return max((n for t, n in recent if now - t <= self.cfg.keep), default=0)

    def describe(self, now: float) -> str:
        parts = []
        people = self.nearby(now)
        if people:
            parts.append(f"- 你身边现在有：{'、'.join(people)}（画面上能看到他们头顶的名字）")
        n = self.strangers(now)
        if n:
            parts.append(f"- 身边还有 {n} 个陌生人（头顶没有名字）")
        if people or n:
            parts.append("- 别的人看不到不代表不在（可能被挡住或离得远）")
        return "\n".join(parts)
