"""YOLO 感知层：持续检测人物 / 名字标签 / 互动圆圈，汇总成"身边有谁、谁在发起互动、有没有陌生人"。

对外接口和 vision.env.EnvWatcher 一样（observe / nearby / describe / requests / labels / circles / last_seen），
身体、社交、眼睛不用改；`[perception] enabled = false` 就退回 EnvWatcher 的定时整图 OCR。

每一帧：
1. YOLO 出框（player / player_unlit / name_tag / social_ring / self），去掉面板挡住的、底部输入栏的、团子自己
2. 追踪：框接成轨迹
3. 名字标签轨迹：新出现就裁小图只跑 OCR 识别（不跑检测），多读几次投票，对上 friends.md 就是那个好友
   —— 身份跟着名字走，不跟着轨迹 id 走：转视角轨迹断了，新轨迹一读出名字就接回同一个人
4. 圆圈：挂到正上方的名字标签上，用现有的剪影模板认图标（牵手 / 拥抱 / ……）；上面没有标签的圆圈算陌生人的
5. 人物：未点火的黑影（player_unlit）一定是陌生人；点过火的陌生人外观和好友一样，
   只能靠头顶有没有名字标签分 —— 一直没有名字标签、离得不太远 → 陌生人
   （还有个办法是点一下人物，右边会打开好友树面板，看得出是不是好友；要点屏幕，没做）

以上规则和阈值都还**没在真机验证**（模型还没训练），见 docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np

from ..brain.images import difference, thumb
from ..chat.tracker import normalize, similar
from ..config import EnvConfig, PerceptionConfig
from ..game.social import IDLE, KIND_NAMES, Request
from .bubbles import Rect, roi_rect
from .detect import Detection, Detector
from .ocr import OcrEngine, join_lines
from .track import Track, Tracker, iou

log = logging.getLogger(__name__)

UNLIT = "player_unlit"  # 没点火的陌生人：黑色剪影
STRANGER = "陌生人"  # 陌生人头顶没有名字：发起的请求用这个名字（社交规则里按 stranger 处理）
ICON_OFFSET = 2.23  # 圆圈中心在名字标签上沿往下这么多倍标签高度（game-ops §6 实测）
OCCLUSION = "occlusion"  # 多人同时消失 + 画面大变：玩家自己开了全屏界面（地图、商店……），暂停计时但检测照跑


def detector_conf(cfg: PerceptionConfig) -> float:
    """检测器的出框阈值：收集难例时要看到 low_conf ~ conf 之间的框，判定仍按 conf。"""
    return min(cfg.low_conf, cfg.conf) if cfg.hardcases else cfg.conf


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
        scene_change: float = 0.25,  # 集体消失时，画面差异超过这个才算开了全屏界面（同 brain.scene_change）
        hardcases=None,  # vision.hardcases.HardCaseCollector：可能认错的画面存下来
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
        self.scene_change = scene_change
        self.hardcases = hardcases
        self.keep = cfg.keep  # 身体说"走开了"时用
        self.tracker = Tracker(cfg.track_buffer, cfg.track_iou, cross=frozenset({"player", UNLIT}))  # 同一个人可能两类来回变
        self.requests: dict = {}  # 名字 → game.social.Request
        self.labels: dict[str, tuple[int, int, int, int, float]] = {}
        self.circles: dict[str, tuple[str | None, float]] = {}
        self.last_seen: dict[str, float] = {}
        self.place = ""  # YOLO 不认地名；留着和 EnvWatcher 接口一致
        self.place_at = float("-inf")
        self.last_dets: list[Detection] = []  # 最近一帧的检测（调试画框用）
        self.last_tracks: list[Track] = []
        self.last_low: list[Detection] = []  # 最近一帧置信度在 conf 以下的框（不进追踪，给难例收集看）
        self.timings: deque[tuple[float, float]] = deque(maxlen=300)  # (检测 ms, 整帧 ms)
        self._strangers: deque[tuple[float, int]] = deque()  # (时间, 这一帧有几个陌生人)
        self._panel_visible = False
        self._holds: Counter = Counter()  # 暂停的原因 → 次数（可以嵌套）
        self._held_since: float | None = None
        self._occluded_at = 0.0
        self._prev_count = 0  # 上一帧看到几个人物 + 名字标签
        self._prev_thumb: np.ndarray | None = None
        self._last_run = float("-inf")
        self._busy = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- 画面被挡时暂停计时（一期设计 §4） ----
    @property
    def paused(self) -> bool:
        return self._held_since is not None

    def hold(self, reason: str) -> None:
        """开始暂停：不跑检测、不累计"没看到"的时间。原因可以叠加，全部 release 才恢复。"""
        with self._lock:
            if self._held_since is None:
                self._held_since = self.clock()
                log.debug("感知暂停：%s", reason)
            self._holds[reason] += 1

    def release(self, reason: str) -> None:
        with self._lock:
            if self._holds[reason] <= 0:
                self._holds.pop(reason, None)
                return
            self._holds[reason] -= 1
            if self._holds[reason] <= 0:
                del self._holds[reason]
            if self._holds or self._held_since is None:
                return
            since, self._held_since = self._held_since, None
        self._resume(since, reason)

    @contextmanager
    def held(self, reason: str) -> Iterator[None]:
        self.hold(reason)
        try:
            yield
        finally:
            self.release(reason)

    def _release_all(self, why: str) -> None:
        with self._lock:
            if self._held_since is None:
                return
            since, self._held_since = self._held_since, None
            self._holds.clear()
        self._resume(since, why)

    def _resume(self, since: float, why: str) -> None:
        """暂停了 d 秒：把所有"最后看到"的时间往后挪 d，恢复后要再过 keep 秒没看到才算走开。"""
        now = self.clock()
        d = now - since
        log.debug("感知恢复（%s），暂停了 %.1f 秒", why, d)
        self._prev_count, self._prev_thumb = 0, None  # 暂停前那一帧不能拿来判"集体消失"（镜头可能已经转走了）
        if d <= 0:
            return
        for name, t in list(self.last_seen.items()):
            self.last_seen[name] = min(t + d, now)
        for name, (x, y, w, h, t) in list(self.labels.items()):
            self.labels[name] = (x, y, w, h, min(t + d, now))
        for name, (kind, t) in list(self.circles.items()):
            self.circles[name] = (kind, min(t + d, now))
        with self._lock:
            self._strangers = deque((min(t + d, now), n, u) for t, n, u in self._strangers)
        self.tracker.shift(d)

    def _check_hold_max(self) -> None:
        since = self._held_since
        if since is not None and self.clock() - since > self.cfg.hold_max:
            log.warning("感知暂停超过 %.0f 秒（%s），自动恢复", self.cfg.hold_max, "、".join(self._holds) or "?")
            self._release_all("hold_max")

    def _blocked(self) -> bool:
        """暂停且要跳过检测（只因为集体消失而暂停时检测照跑，等人重新出现）。"""
        return self.paused and set(self._holds) != {OCCLUSION}

    def _occlusion(self, frame: np.ndarray, dets: list[Detection]) -> bool:
        """集体消失规则。返回 True = 这一帧什么都看不到、正在暂停，不更新状态。"""
        count = sum(d.cls in ("player", UNLIT, "name_tag") for d in dets)
        small = thumb(frame)
        prev_count, prev_thumb = self._prev_count, self._prev_thumb
        self._prev_count, self._prev_thumb = count, small
        if self._holds.get(OCCLUSION):
            if count:
                self.release(OCCLUSION)
                return False
            if self.clock() - self._occluded_at > self.cfg.occlusion_hold:
                log.info("人还是都看不到（%.0f 秒），不再等了，照常计时", self.cfg.occlusion_hold)
                self.release(OCCLUSION)
            return True
        if prev_count >= 2 and count == 0 and prev_thumb is not None and difference(prev_thumb, small) > self.scene_change:
            log.info("%d 个人同时不见了、画面大变（可能开了全屏界面），暂停计时", prev_count)
            self._occluded_at = self.clock()
            self.hold(OCCLUSION)
            return True
        return False

    def _frozen(self, now: float) -> float:
        """暂停期间按暂停开始那一刻算：给身体的结果停在暂停前。"""
        since = self._held_since
        return min(now, since) if since is not None else now

    # ---- 帧从哪来 ----
    def observe(self, frame: np.ndarray, now: float, panel_visible: bool) -> None:
        """身体主循环每圈调一次。capture = "own" 时只记下面板开没开，帧由感知线程自己截。"""
        self._panel_visible = panel_visible
        self._check_hold_max()
        if self._blocked():
            return
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
            self._check_hold_max()
            if self._blocked():
                self._stop.wait(period)
                continue
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
        low = [d for d in dets if d.score < self.cfg.conf]
        dets = [d for d in dets if d.score >= self.cfg.conf]
        if self._occlusion(frame, dets):
            return
        tracks = self.tracker.update(dets, now)
        selfs = [t for t in tracks if t.cls == "self"]
        players = [t for t in tracks if t.cls in ("player", UNLIT) and not self._is_self(t, selfs)]
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
            if ring is not None:
                ring.data["kind"] = kind
            self.circles[name] = (kind, now)
            if kind and kind != IDLE:
                if name not in self.requests or self.requests[name].kind != kind:
                    log.info("%s 发起了互动：%s", name, kind)
                self.requests[name] = Request(name, kind, self._ring_center(ring), now)
            else:
                self.requests.pop(name, None)

        stranger_req = None
        for ring in orphans:
            kind = ring.data["kind"] = self._classify(frame, ring)
            if kind and kind != IDLE:
                stranger_req = Request(STRANGER, kind, self._ring_center(ring), now)
                break
        if stranger_req is not None:
            if STRANGER not in self.requests or self.requests[STRANGER].kind != stranger_req.kind:
                log.info("陌生人发起了互动：%s", stranger_req.kind)
            self.requests[STRANGER] = stranger_req
        else:
            self.requests.pop(STRANGER, None)

        strangers = unlit = 0
        for player in players:
            if player.cls == UNLIT:  # 没点火的黑影：一定是陌生人，远近都算
                player.data["stranger"] = True
                strangers += 1
                unlit += 1
                continue
            tag = self._tag_over(player, tags)
            if tag is not None:
                if tag.data.get("name"):
                    player.data["name"] = tag.data["name"]
                player.data["tagged"] = True  # 这条轨迹上出现过名字标签：不是陌生人（标签被挡一下不改判）
            is_stranger = (
                not player.data.get("tagged")
                and now - player.first >= self.cfg.stranger_after
                and player.box.h >= self.cfg.stranger_min_height * height
            )
            player.data["stranger"] = is_stranger
            strangers += is_stranger
        with self._lock:  # 身体线程会同时读（strangers()）
            self._strangers.append((now, strangers, unlit))
            while self._strangers and now - self._strangers[0][0] > self.cfg.keep:
                self._strangers.popleft()

        self.last_dets, self.last_tracks, self.last_low = dets, tracks, low
        if self.hardcases is not None and not self.paused:
            try:
                self.hardcases.check(frame, now, tracks, low, set(seen), panel_visible)
            except Exception:
                log.exception("收集难例出错")
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
            if mine is not None and det.cls in ("player", UNLIT) and _inside(c, mine):
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

    # ---- 可视化（vision/viewer.py） ----
    def overlay(self, now: float) -> list[dict]:
        """最近一帧认出了什么：每个框一条 {x, y, w, h, kind, label}（整张截图坐标）。"""
        out = []
        for t in list(self.last_tracks):
            d = t.data
            if t.cls == "player":
                kind = "stranger" if d.get("stranger") else ("friend" if d.get("name") else "player")
                label = d.get("name") or ("陌生人" if d.get("stranger") else "")
            elif t.cls == UNLIT:
                kind, label = "unlit", "陌生人（没点火）"
            elif t.cls == "self":
                kind, label = "self", "团子"
            elif t.cls == "name_tag":
                kind = "name" if d.get("name") else "tag"
                label = d.get("name") or (f"?{d['text']}" if d.get("text") else "?")
            elif t.cls == "social_ring":
                ring = d.get("kind")
                kind = "request" if ring and ring != IDLE else "ring"
                label = KIND_NAMES.get(ring, "✦" if ring == IDLE else "?")
            else:
                kind, label = t.cls, t.cls
            b = t.box
            out.append({"x": b.x, "y": b.y, "w": b.w, "h": b.h, "kind": kind, "label": label, "score": round(t.score, 2)})
        return out

    # ---- 给身体 / 提示词用的（和 EnvWatcher 一样） ----
    def nearby(self, now: float) -> list[str]:
        now = self._frozen(now)
        return [n for n in self.names() if now - self.last_seen.get(n, float("-inf")) <= self.cfg.keep]

    def strangers(self, now: float) -> int:
        """最近 keep 秒里同一帧最多看到几个陌生人（轨迹会断，所以不数轨迹条数）。"""
        now = self._frozen(now)
        with self._lock:
            recent = list(self._strangers)
        return max((n for t, n, _ in recent if now - t <= self.cfg.keep), default=0)

    def unlit(self, now: float) -> int:
        """其中还没点火（黑影）的有几个。"""
        now = self._frozen(now)
        with self._lock:
            recent = list(self._strangers)
        return max((u for t, _, u in recent if now - t <= self.cfg.keep), default=0)

    def describe(self, now: float) -> str:
        parts = []
        people = self.nearby(now)
        if people:
            parts.append(f"- 你身边现在有：{'、'.join(people)}（画面上能看到他们头顶的名字）")
        n = self.strangers(now)
        if n:
            dark = self.unlit(now)
            note = f"，其中 {dark} 个还没点火（黑影，看不到外观）" if dark else ""
            parts.append(f"- 身边还有 {n} 个陌生人（头顶没有名字{note}）")
        if people or n:
            parts.append("- 别的人看不到不代表不在（可能被挡住或离得远）")
        return "\n".join(parts)
