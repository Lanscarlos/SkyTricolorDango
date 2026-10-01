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

二期（docs/superpowers/specs/2026-09-28-perception-phase2-design.md）：
- `sweep()`：转一圈截下的帧单独汇总成"哪个方向有谁"（vision/sweep.py 算方位、合并），顺带认出团子（self_box，之后代替 self_roi）
- 远近：人物框高 ÷ 团子框高；框持续变大、往画面中间走 → 朝团子走过来（`pop_approaches`，身体发 approach 事件）
- `typing`（头顶"正在输入"气泡）：挂到正下方的人身上；陌生人的消息靠它猜是画面上哪个人说的（`speaker_hint`）

认装扮（docs/superpowers/specs/2026-10-01-appearance-design.md，挂了 AppearanceBook 才有）：
- 人物框裁图算外观特征，挂着名字标签时学进记忆簿；名字标签看不到的人按外观认成好友（`maybe`，"像小明"），不算陌生人
- 点过火的陌生人按外观编号（"陌生人A"），走开又回来时认得出（`pop_stranger_backs`）

以上规则和阈值都还**没在真机验证**（模型还没训练），见 docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np

from ..brain.images import difference, thumb
from ..chat.tracker import normalize, similar
from ..config import AppearanceConfig, EnvConfig, GestureConfig, PerceptionConfig, SpinConfig
from ..game.social import IDLE, KIND_NAMES, Request, is_request
from .appearance import describe_crop, good_crop
from .bubbles import Rect, roi_rect
from .detect import Detection, Detector
from .embed import unit
from .gesture import ClipBuffer, eligible, person_crop
from .ocr import OcrEngine, join_lines
from .people import OBJECT_NAMES, Person, Thing, object_distance, side_of
from .sweep import STRANGER_WHO, UNKNOWN_WHO, UNLIT_WHO, Sighting, SweepResult, bearing, distance, find_self, merge
from .track import Track, Tracker, iou

log = logging.getLogger(__name__)

UNLIT = "player_unlit"  # 没点火的陌生人：黑色剪影
STRANGER = "陌生人"  # 陌生人头顶没有名字：发起的请求用这个名字（社交规则里按 stranger 处理）
ICON_OFFSET = 2.23  # 圆圈中心在名字标签上沿往下这么多倍标签高度（game-ops §6 实测）
PLACE_GAP = 3.0  # 画面大变后认地图：离上次至少隔这么久（转镜头时画面一直在变）
FAR_RETRY = 1.0  # 远处二次检测：同一条轨迹最多隔这么久裁一次（还要比 track_buffer 短）；挂上名字标签后这么久内也不裁
FAR_MISSES = 3  # 远处二次检测连着这么多次没找到名字标签……
FAR_BACKOFF = 5.0  # ……之后这么久才再裁一次
FAR_DUP_IOU = 0.3  # 二次检测找到的框和原图里已有的同类框重叠这么多就算同一个
UNKNOWN_MIN_SCORE = 0.9  # 没认出的名字：OCR 至少这么有把握才记下（三期 §5）
REQUEST_HOLD = 1.5  # 陌生人的圆圈还在原地、只是这一帧没认出图标（火焰会晃）：请求再留这么久，免得一闪就没、身体来不及点
PEOPLE_STALE = 1.0  # people()：最近一帧比这更旧（被挡住、没跑检测）就不再报里面的人
OCCLUSION = "occlusion"  # 多人同时消失 + 画面大变：玩家自己开了全屏界面（地图、商店……），暂停计时但检测照跑


def detector_conf(cfg: PerceptionConfig) -> float:
    """检测器的出框阈值：收集难例时要看到 low_conf ~ conf 之间的框，判定仍按 conf。"""
    return min(cfg.low_conf, cfg.conf) if cfg.hardcases else cfg.conf


def approaching(hist: list[tuple[float, float, float]], width: int, grow: float) -> bool:
    """(时间, 框高, 框中心 x) 的序列：前 1/3 和后 1/3 比，框高变大 ≥ grow，且往画面中间走（或一直在中间）。"""
    if len(hist) < 3:
        return False
    k = max(1, len(hist) // 3)
    h0 = sum(h for _, h, _ in hist[:k]) / k
    h1 = sum(h for _, h, _ in hist[-k:]) / k
    if h0 <= 0 or h1 < h0 * (1 + grow):
        return False
    c0 = sum(abs(x - width / 2) for _, _, x in hist[:k]) / k
    c1 = sum(abs(x - width / 2) for _, _, x in hist[-k:]) / k
    return c1 <= c0 or c1 < 0.15 * width


def far_region(p: Rect, width: int, height: int) -> Rect | None:
    """远处小人头顶要再检测一次的区域：宽 3 倍框宽、高 2.5 倍框高（上方 2 个身高到身体上半截），水平居中，夹到画面内。"""
    x1, y1 = round(p.x + p.w / 2 - 1.5 * p.w), round(p.y - 2 * p.h)
    x2, y2 = x1 + 3 * p.w, y1 + round(2.5 * p.h)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return Rect(x1, y1, x2 - x1, y2 - y1)


def _near(a: tuple[int, int], b: tuple[int, int], dist: int = 80) -> bool:
    return abs(a[0] - b[0]) <= dist and abs(a[1] - b[1]) <= dist


def one_self(dets: list[Detection]) -> list[Detection]:
    """一帧只有一个团子：留分数最高的 self，其余改成 player（实测模型会把躺在地上的别人也认成团子）。"""
    selfs = sorted((d for d in dets if d.cls == "self"), key=lambda d: d.score, reverse=True)
    extra = {id(d) for d in selfs[1:]}
    return [Detection("player", d.box, d.score) if id(d) in extra else d for d in dets]


def promote_weak_self(dets: list[Detection], low: list[Detection]) -> list[Detection]:
    """没有像样的 self 时，和低分 self 框几乎重合（IoU ≥ 0.5）的那个 player 就是团子，改成 self。

    实测（2026-09-30，好友站在团子正后方、镜头贴得近）：团子被认成 player 0.74、self 只有 0.29，
    好友的名字标签挂到了团子身上，look_person 裁了团子的背影，也找不到团子框、不换角度。只有团子会拿到 self 分数。"""
    if any(d.cls == "self" for d in dets):
        return dets
    pairs = [(iou(d.box, w.box), i) for i, d in enumerate(dets) if d.cls == "player" for w in low if w.cls == "self"]
    overlap, i = max(pairs, default=(0.0, -1))
    if overlap < 0.5:
        return dets
    out = list(dets)
    out[i] = Detection("self", dets[i].box, dets[i].score)
    return out


def people_boxes(dets: list[Detection]) -> list[Detection]:
    """一帧里的人（player / player_unlit / self）。模型可能在团子身上同时出 self 和 player 框：player 那个不算另一个人。"""
    selfs = [d for d in dets if d.cls == "self"]
    return selfs + [d for d in dets if d.cls in ("player", UNLIT) and not any(iou(d.box, s.box) >= 0.5 for s in selfs)]


def _center(r: Rect) -> tuple[float, float]:
    return r.x + r.w / 2, r.y + r.h / 2


def _inside(point: tuple[float, float], rect: Rect) -> bool:
    return rect.x <= point[0] < rect.x2 and rect.y <= point[1] < rect.y2


@dataclass(frozen=True)
class Talker:
    """头顶正冒着气泡的人（空闲注意力用）。"""

    track_id: int
    name: str | None  # 好友名；陌生人 None
    friend: bool
    x: float  # 人物框中心 x（整图像素）
    last: float  # 最近一次看到气泡
    start: float  # 这次气泡开始（气泡断开超过 bubble_gap 再出现算新的一句）


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
        unknown=None,  # vision.unknownnames.UnknownNames：读得清楚但不在好友名单里的名字（三期 §5）
        places=None,  # vision.places.PlaceRecognizer：认地图（三期 §2）
        place_interval: float = 30.0,  # 每隔这么久认一次地图（画面大变后也认一次）
        gestures=None,  # vision.gesture.GestureClassifier：别人对团子做的动作（三期 §3，研究性质）
        gesture_cfg: GestureConfig | None = None,
        appearance=None,  # vision.appearance.AppearanceBook：认装扮（None = 不认，行为照旧）
        embedder=None,  # 外观特征模型（appearance.make_embedder）
        appearance_cfg: AppearanceConfig | None = None,
        saver=None,  # 攒训练数据（Task 5 接上）
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
        self.unknown = unknown
        self.places = places
        self.place_interval = place_interval
        self._place_run = float("-inf")
        self._place_thumb: np.ndarray | None = None
        self._place_misses = 0  # 连着几次认不出
        self.gestures = gestures
        self.gesture_cfg = gesture_cfg or GestureConfig()
        self._gestures: deque[tuple[str, str]] = deque(maxlen=50)  # (好友名, 动作)，身体取走；没人取（普通 Agent）时只留最近的
        self._gesture_at: dict[tuple[str, str], float] = {}
        self.appearance = appearance
        self.embedder = embedder
        self.appearance_cfg = appearance_cfg or AppearanceConfig()
        self.saver = saver
        self._stranger_backs: list[str] = []  # 走开又回来的陌生人编号，身体取走
        self.keep = cfg.keep  # 身体说"走开了"时用
        self.tracker = Tracker(cfg.track_buffer, cfg.track_iou, cross=frozenset({"player", UNLIT}))  # 同一个人可能两类来回变
        self.requests: dict = {}  # 名字 → game.social.Request
        self.labels: dict[str, tuple[int, int, int, int, float]] = {}
        self.circles: dict[str, tuple[str | None, float]] = {}
        self.last_seen: dict[str, float] = {}
        self.place = ""  # 认地图认出的地方（三期 §2，没挂 places 时一直空）；和 EnvWatcher 同名
        self.place_at = float("-inf")
        self.last_dets: list[Detection] = []  # 最近一帧的检测（调试画框用）
        self.last_tracks: list[Track] = []
        self.last_low: list[Detection] = []  # 最近一帧置信度在 conf 以下的框（不进追踪，给难例收集看）
        self.timings: deque[tuple[float, float]] = deque(maxlen=300)  # (检测 ms, 整帧 ms)
        self.far_runs = 0  # 远处二次检测跑了几次（测速 / compare 用）
        self.self_box: Rect | None = None  # 转圈认出的团子（sweep）；有它就不用 self_roi，下次转圈前一直用
        self._frame_h = 1080  # 最近一帧的高度（算远近用）
        self._frame_w = 1920  # 最近一帧的宽度（算在画面哪边用）
        self._approaches: list[str] = []  # 朝团子走过来的人（好友名 / STRANGER），身体取走
        self._approach_at: dict[str, float] = {}
        self._approach_log: deque[tuple[str, int, float]] = deque()  # (谁, 轨迹 id, 时间)：空闲注意力用，不取走
        self._typing: deque[tuple] = deque()  # (时间, 轨迹 id, 是好友, 没点火, 框中心 x, 框高, 画面宽, 画面高)
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
        self._infer = threading.Lock()  # 检测器 / OCR：sweep（身体线程）和后台感知线程可能同时用
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
        for track in list(self.tracker.tracks.values()):  # 暂停前后的框高不能连起来判"走过来"（镜头可能动过）
            track.data.pop("hist", None)
            track.data.pop("clip", None)  # 动作片段也不能跨暂停拼起来
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
        self._frame_h, self._frame_w = height, width
        dets = one_self(self._filter(self._detect(frame), width, height, panel_visible))
        detected = time.perf_counter()
        low = [d for d in dets if d.score < self.cfg.conf]
        dets = promote_weak_self([d for d in dets if d.score >= self.cfg.conf], low)
        if self._occlusion(frame, dets):
            return
        tracks = self.tracker.update(dets, now)
        selfs = [t for t in tracks if t.cls == "self"]
        players = [t for t in tracks if t.cls in ("player", UNLIT) and not self._is_self(t, selfs)]
        tags = [t for t in tracks if t.cls == "name_tag"]
        rings = [t for t in tracks if t.cls == "social_ring"]
        bubbles = [t for t in tracks if t.cls == "typing"]
        extra = self._far_tags(frame, players, tags, now, width, height, panel_visible)
        if extra:
            tags += [t for t in extra if t.cls == "name_tag"]
            rings += [t for t in extra if t.cls == "social_ring"]
            tracks = tracks + extra

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
            if is_request(kind):
                if name not in self.requests or self.requests[name].kind != kind:
                    log.info("%s 发起了互动：%s", name, kind)
                self.requests[name] = Request(name, kind, self._ring_center(ring), now)
            else:
                self.requests.pop(name, None)

        stranger_req = None
        for ring in orphans:
            kind = ring.data["kind"] = self._classify(frame, ring)
            if is_request(kind):
                stranger_req = Request(STRANGER, kind, self._ring_center(ring), now)
                break
        old = self.requests.get(STRANGER)
        if stranger_req is not None:
            if old is None or old.kind != stranger_req.kind:
                log.info("陌生人发起了互动：%s", stranger_req.kind)
            self.requests[STRANGER] = stranger_req
        elif old is not None and now - old.seen_at <= REQUEST_HOLD and any(
            ring.data.get("kind") is None and _near(self._ring_center(ring), old.pos) for ring in orphans
        ):
            pass  # 圆圈还在原地、这一帧没认出图标：先留着（seen_at 不更新，最多留 REQUEST_HOLD 秒）
        else:
            self.requests.pop(STRANGER, None)

        strangers = unlit = 0
        tagged = self._assign_tags([p for p in players if p.cls != UNLIT], tags)
        shown = {t.data["name"] for t in tags if t.data.get("name")}
        for player in players:
            name = player.data.get("name")
            if name in shown and player.id not in tagged:
                # 这个名字的标签此刻清清楚楚在别处：之前是挂错了（好友从他身后走过），摘掉。
                # 标签只是被挡住（这一帧没看到）时不摘
                log.debug("轨迹 %d 不是 %s（标签在别处），摘掉名字", player.id, name)
                for key in ("name", "tagged", "tag_at"):
                    player.data.pop(key, None)
            tag = tagged.get(player.id)
            if player.cls != UNLIT and tag is not None:
                player.data["tag_at"] = now
                if tag.data.get("name"):
                    player.data["name"] = tag.data["name"]
                player.data["tagged"] = True  # 这条轨迹上出现过名字标签：不是陌生人（标签被挡一下不改判）
        fresh = {}
        if self.appearance is not None:
            fresh = self._appearance_features(frame, players, selfs, now, width, height, panel_visible)
            self._appearance_identify(frame, players, selfs, tagged, shown, fresh, now, tracks)
        for player in players:
            if player.cls == UNLIT:  # 没点火的黑影：一定是陌生人，远近都算
                player.data["stranger"] = True
                strangers += 1
                unlit += 1
                continue
            maybe = player.data.get("maybe")
            is_stranger = (
                not player.data.get("tagged")
                and not maybe
                and now - player.first >= self.cfg.stranger_after
                and player.box.h >= self.cfg.stranger_min_height * height
            )
            player.data["stranger"] = is_stranger
            strangers += is_stranger
            if maybe and now - self.last_seen.get(maybe, float("-inf")) <= self.cfg.keep:
                self.last_seen[maybe] = now  # 好友还在身边、只是名字标签被挡住：别冒出"走开了"（已经走开的不靠外观接回来）
            if is_stranger and self.appearance is not None:
                self._appearance_stranger(frame, player, fresh.get(player.id), now)
        if self.appearance is not None:
            self.appearance.forget(now)
        self._watch_typing(bubbles, players, selfs, now, width, height)
        if not self.paused:
            self._watch_approach(players, now, width)
            if self.gestures is not None:
                self._watch_gestures(frame, players, now, width, height)
        with self._lock:  # 身体线程会同时读（strangers()）
            self._strangers.append((now, strangers, unlit))
            while self._strangers and now - self._strangers[0][0] > self.cfg.keep:
                self._strangers.popleft()

        self.last_dets, self.last_tracks, self.last_low = dets, tracks, low
        if self.places is not None and not self.paused:
            self._watch_place(frame, dets, now)
        if self.hardcases is not None and not self.paused:
            try:
                self.hardcases.check(frame, now, tracks, low, set(seen), panel_visible)
            except Exception:
                log.exception("收集难例出错")
        self.timings.append(((detected - started) * 1000, (time.perf_counter() - started) * 1000))
        if seen:
            log.debug("旁边看到: %s", "、".join(seen))

    def _far_tags(self, frame: np.ndarray, players: list[Track], tags: list[Track], now: float,
                  width: int, height: int, panel_visible: bool) -> list[Track]:
        """三期 §1：远处的小人没挂上名字标签 → 在它头顶裁一块再检测一次（只要 name_tag / social_ring），返回新接上的轨迹。

        裁剪直接交给检测器，它自己 letterbox 放大到 imgsz，等于"放大再检测"。"""
        if self.cfg.far_crops <= 0:
            return []
        limit = self.cfg.far_height * height
        # 远处的标签只在裁剪那一帧出现：下次裁剪要赶在标签轨迹被追踪器删掉（track_buffer）之前，
        # 不然每次都是新轨迹，名字投票、"同一条标签轨迹只记一次"都不起作用
        retry = min(FAR_RETRY, 0.8 * self.cfg.track_buffer)

        def due(p: Track) -> bool:
            # 连着几次都没找到标签（多半是点过火的陌生人）：放慢到 FAR_BACKOFF 秒一次，别一直占推理
            wait = FAR_BACKOFF if p.data.get("far_miss", 0) >= FAR_MISSES else retry
            return now - p.data.get("tag_at", float("-inf")) >= retry and now - p.data.get("far_at", float("-inf")) >= wait

        def tag_inside(area: Rect | None) -> bool:  # 原图里已经框到了标签（只是位置没挂上人）：裁了也只会找到它
            return area is not None and any(_inside(_center(t.box), area) for t in tags)

        todo = [
            p for p in players
            if p.cls == "player" and p.box.h < limit and self._tag_over(p, tags) is None and due(p)
            and not tag_inside(far_region(p.box, width, height))
        ]
        todo.sort(key=lambda p: p.data.get("far_at", float("-inf")))
        found: list[Detection] = []
        for p in todo[: self.cfg.far_crops]:
            p.data["far_at"] = now
            area = far_region(p.box, width, height)
            if area is None:
                continue
            self.far_runs += 1
            hits = 0
            for d in self._detect(area.crop(frame)):
                if d.cls not in ("name_tag", "social_ring") or d.score < self.cfg.conf:
                    continue
                box = Rect(d.box.x + area.x, d.box.y + area.y, d.box.w, d.box.h)
                found.append(Detection(d.cls, box, d.score))
                hits += d.cls == "name_tag"
            p.data["far_miss"] = 0 if hits else p.data.get("far_miss", 0) + 1
        found = self._filter(found, width, height, panel_visible, drop_self=False)
        seen = {  # 这一帧原图里已经有的框：二次检测又找到一遍的不要（不能多出一条轨迹、多读一次 OCR）
            cls: [t.box for t in self.tracker.tracks.values() if t.cls == cls and t.last == now]
            for cls in ("name_tag", "social_ring")
        }
        fresh: list[Detection] = []
        for d in found:
            same = seen[d.cls]
            if any(iou(d.box, b) >= FAR_DUP_IOU for b in same) or any(
                f.cls == d.cls and iou(d.box, f.box) >= FAR_DUP_IOU for f in fresh
            ):
                continue
            fresh.append(d)
        return self.tracker.update(fresh, now) if fresh else []

    # ---- 认装扮（设计 §3.2、§4） ----
    def _appearance_features(self, frame: np.ndarray, players: list[Track], selfs: list[Track], now: float,
                             width: int, height: int, panel_visible: bool) -> dict[int, tuple[np.ndarray, Rect]]:
        """点过火的人和团子：每 every 帧裁一次好样本、算特征，平滑进 data["feat"]、好样本数进 data["samples"]。
        一帧最多算 max_per_frame 个（最久没算的先算）。返回这一帧新算的 {轨迹 id: (这次的特征, 框)}，学进记忆簿时用。"""
        acfg = self.appearance_cfg
        every = max(1, acfg.every)
        due = [t for t in [p for p in players if p.cls == "player"] + selfs if t.hits % every == 0]
        due.sort(key=lambda t: t.data.get("feat_at", float("-inf")))
        blocked = [roi_rect(self.log_roi, width, height)] if panel_visible else []
        out: dict[int, tuple[np.ndarray, Rect]] = {}
        for t in due:
            if len(out) >= acfg.max_per_frame:
                break
            t.data["feat_at"] = now
            others = [o.box for o in players + selfs if o.id != t.id]
            crop = good_crop(frame, t.box, others, blocked, acfg.min_height, acfg.max_overlap)
            if crop is None:
                continue
            try:
                feat = unit(self.embedder.embed(crop))
            except Exception:
                log.exception("算外观特征出错")
                continue
            old = t.data.get("feat")
            t.data["feat"] = feat if old is None else unit((1 - acfg.ema) * old + acfg.ema * feat)
            t.data["samples"] = t.data.get("samples", 0) + 1
            out[t.id] = (feat, t.box)
        return out

    def _learn(self, kind: str, who: str, sample: tuple[np.ndarray, Rect], frame: np.ndarray, now: float) -> None:
        feat, box = sample
        self.appearance.learn(kind, who, feat, now, crop=(box.h, describe_crop(frame, box).copy()))

    def _appearance_identify(self, frame: np.ndarray, players: list[Track], selfs: list[Track],
                             tagged: dict[int, Track], shown: set[str], fresh: dict, now: float,
                             tracks: list[Track]) -> None:
        """学（挂着名字标签的好友、团子）→ 名字标签说了算（摘掉 maybe / 编号）→ 没标签的按外观认好友（maybe）。"""
        book, acfg = self.appearance, self.appearance_cfg
        for me in selfs:
            if me.id in fresh:
                self._learn("me", "", fresh[me.id], frame, now)
        cands: dict[int, np.ndarray] = {}
        for p in players:
            d = p.data
            if p.cls != "player":  # 变成黑影了：没有外观可比
                d.pop("maybe", None)
                d.pop("miss", None)
                continue
            tag = tagged.get(p.id)
            if tag is not None:  # 名字标签永远说了算
                name = tag.data.get("name")
                maybe = d.pop("maybe", None)
                d.pop("miss", None)
                d.pop("sid", None)
                if maybe and name and maybe != name:
                    log.info("轨迹 %d 按外观认成 %s，名字标签是 %s", p.id, maybe, name)
                    self._report(frame, now, f"按外观认成 {maybe}，名字标签是 {name}", tracks)
                if name and p.id in fresh:
                    self._learn("friend", name, fresh[p.id], frame, now)
                continue
            if d.get("tagged") or d.get("name"):
                continue  # 之前挂过标签、这一帧被挡住：身份照旧
            maybe = d.get("maybe")
            if maybe:
                if maybe in shown:  # 他的名字标签此刻清清楚楚挂在别处：不是他
                    log.debug("轨迹 %d 不像 %s 了（标签在别处）", p.id, maybe)
                    d.pop("maybe", None)
                    d.pop("miss", None)
                elif p.id in fresh:
                    d["miss"] = 0 if book.still_like(d["feat"], maybe) else d.get("miss", 0) + 1
                    if d["miss"] >= acfg.recheck:
                        log.debug("轨迹 %d 连着 %d 次不像 %s，摘掉", p.id, d["miss"], maybe)
                        d.pop("maybe", None)
                        d.pop("miss", None)
                continue
            if d.get("samples", 0) >= acfg.min_samples:
                cands[p.id] = d["feat"]
        if not cands:
            return
        taken = shown | {v for p in players for k in ("name", "maybe") if (v := p.data.get(k))}
        byid = {p.id: p for p in players}
        for tid, name in book.assign_friends(cands, taken).items():
            d = byid[tid].data
            d["maybe"], d["miss"] = name, 0
            d.pop("sid", None)
            log.info("轨迹 %d 没看到名字，按外观像 %s", tid, name)

    def _appearance_stranger(self, frame: np.ndarray, player: Track, sample, now: float) -> None:
        """判成陌生人的轨迹：好样本够了就编号（认回以前的、或新编号），之后接着学他的外观。"""
        d = player.data
        sid = d.get("sid")
        if sid is None:
            if d.get("samples", 0) < self.appearance_cfg.min_samples:
                return
            sid, back = self.appearance.stranger_id(d["feat"], now)
            d["sid"] = sid
            log.debug("轨迹 %d 是 %s", player.id, sid)
            if back:
                log.info("%s 又回来了", sid)
                with self._lock:
                    self._stranger_backs.append(sid)
        elif sample is not None:
            self._learn("stranger", sid, sample, frame, now)

    def _report(self, frame: np.ndarray, now: float, detail: str, tracks: list[Track]) -> None:
        if self.hardcases is None or self.paused:
            return
        try:
            self.hardcases.report(frame, now, "appearance", detail, tracks)
        except Exception:
            log.exception("收集难例出错")

    def _watch_place(self, frame: np.ndarray, dets: list[Detection], now: float) -> None:
        """三期 §2：每 place_interval 秒、或画面比上次认地图时大变（隔 ≥ PLACE_GAP 秒）认一次。
        偶尔一次认不出先留着原来的；画面大变后认不出、或连着两次认不出就清掉。"""
        small = thumb(frame)
        since = now - self._place_run
        changed = self._place_thumb is not None and difference(self._place_thumb, small) > self.scene_change
        if since < self.place_interval and not (changed and since >= PLACE_GAP):
            return
        self._place_run, self._place_thumb = now, small
        try:
            # 只遮人物 / 名字标签 / 圆圈 / 气泡；长椅、钢琴这些物品是地标，不遮（v4 建的图库也没遮它们）
            match = self.places.recognize(frame, [d.box for d in dets if d.cls not in OBJECT_NAMES])
        except Exception:
            log.exception("认地图出错")
            return
        if match.name is None:
            log.debug("认不出在哪（最像 %s %.2f，第二像 %.2f）", match.best, match.score, match.second)
            self._place_misses += 1
            # 宁可不说也不说错：画面大变后认不出（可能到了图库里没有的地方）、或连着两次认不出，就不再说原来的地名
            if self.place and (changed or self._place_misses >= 2):
                log.info("认不出在哪了，不再说在%s", self.place)
                self.place, self.place_at = "", float("-inf")
            return
        self._place_misses = 0
        if match.name != self.place:
            log.info("看起来到了：%s（相似度 %.2f）", match.name, match.score)
        self.place, self.place_at = match.name, now

    def _detect(self, frame: np.ndarray) -> list[Detection]:
        with self._infer:
            return self.detector.detect(frame)

    def _filter(
        self, dets: list[Detection], width: int, height: int, panel_visible: bool, drop_self: bool = True
    ) -> list[Detection]:
        area = roi_rect(self.env_cfg.roi, width, height)  # 底部输入栏不要
        panel = roi_rect(self.log_roi, width, height) if panel_visible else None  # 面板里的"- 名字"不是名字标签
        mine = self._self_areas(width, height) if drop_self else []
        out = []
        for det in dets:
            c = _center(det.box)
            if not _inside(c, area) or (panel is not None and _inside(c, panel)):
                continue
            if det.cls in ("player", UNLIT) and any(_inside(c, r) for r in mine):
                continue
            out.append(det)
        return out

    def _self_areas(self, width: int, height: int) -> list[Rect]:
        """团子所在的区域：转圈认出的团子框（四周各放宽 25%，镜头跟随会有点晃）+ 配置的 self_roi。"""
        areas = []
        if self.self_box is not None:
            b = self.self_box
            areas.append(Rect(round(b.x - b.w / 4), round(b.y - b.h / 4), round(b.w * 1.5), round(b.h * 1.5)))
        if self.cfg.self_roi:
            areas.append(roi_rect(self.cfg.self_roi, width, height))
        return areas

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
        crop = tag.box.pad(4, frame.shape[1], frame.shape[0]).crop(frame)
        text, score = self._ocr(crop)
        if len(normalize(text)) < 2:
            return
        tag.data["text"] = text
        name = self._match_name(text, friends)
        if name:
            votes[name] += 1
            tag.data["name"] = votes.most_common(1)[0][0]
        elif self.unknown is not None and score >= UNKNOWN_MIN_SCORE and not tag.data.get("unknown_logged"):
            tag.data["unknown_logged"] = True  # 同一条标签轨迹只记一次
            try:
                self.unknown.add(text, crop)
            except Exception:
                log.exception("记没认出的名字出错")

    @staticmethod
    def _match_name(text: str, friends: list[str]) -> str | None:
        return next((n for n in friends if similar(text, n, 0.75)), None)

    def _ocr(self, crop: np.ndarray) -> tuple[str, float]:
        """名字标签的字和置信度（整块识别时取各段里最低的）。"""
        with self._infer:
            read_line = getattr(self.ocr, "read_line", None)
            if read_line is not None:
                line = read_line(crop)
                return (line.text, line.score) if line is not None else ("", 0.0)
            lines = self.ocr.recognize(crop)
            return join_lines(lines), min((l.score for l in lines), default=0.0)

    def _ref_height(self, frame_h: int) -> float:
        """算远近的参照：团子的框高（转圈认出的 → 最近一帧的 self 框 → 按 self_height 估）。"""
        if self.self_box is not None:
            return float(self.self_box.h)
        selfs = [t.box.h for t in list(self.last_tracks) if t.cls == "self"]
        if selfs:
            return float(max(selfs))
        return self.cfg.self_height * frame_h

    # ---- 二期：环绕扫描（设计 §1） ----
    def sweep(self, frames: list[tuple[float, np.ndarray]], spin: SpinConfig) -> SweepResult:
        """转一圈截下的帧 [(按住后第几秒, 图)] → 每个方向有谁。独立的一次观察：不碰追踪器、last_seen、请求。"""
        friends = self.names()
        people: list[list[Detection]] = []
        tags: list[list[Detection]] = []
        width = height = 0
        for _, img in frames:
            height, width = img.shape[:2]
            dets = [d for d in self._filter(self._detect(img), width, height, False, drop_self=False) if d.score >= self.cfg.conf]
            people.append(people_boxes(dets))
            tags.append([d for d in dets if d.cls == "name_tag"])
        found = find_self([[d.box for d in frame] for frame in people], width, spin.self_motion)
        mine = self._self_areas(width, height)  # 帧太少认不出团子时，靠之前认出的团子框 / self_roi 排除
        sightings: list[Sighting] = []
        for fi, (t, img) in enumerate(frames):
            where = lambda box: bearing(t, box.x + box.w / 2, width, spin.seconds_per_turn, spin.hfov)  # noqa: E731
            frame_people = [(bi, d) for bi, d in enumerate(people[fi]) if d.cls != "self"]
            under: dict[int, tuple[int, Detection]] = {}  # 标签序号 → 它下面的人
            for bi, d in frame_people:
                tag = self._tag_over(d, tags[fi])
                if tag is not None:
                    ti = tags[fi].index(tag)
                    if ti not in under or abs(tag.box.y2 - d.box.y) < abs(tag.box.y2 - under[ti][1].box.y):
                        under[ti] = (bi, d)
            for ti, tag in enumerate(tags[fi]):
                text, _ = self._ocr(tag.box.pad(4, width, height).crop(img))
                name = self._match_name(text, friends) if len(normalize(text)) >= 2 else None
                bi, body = under.get(ti, (None, None))
                sightings.append(Sighting(
                    where(tag.box), name or UNKNOWN_WHO, fi, float(body.box.h) if body is not None else None,
                    beside=bi is not None and (fi, bi) in found.static,
                ))
            tagged = {bi for bi, _ in under.values()}
            for bi, d in frame_people:
                if bi in tagged or (fi, bi) in found.static or any(_inside(_center(d.box), r) for r in mine):
                    continue
                if d.cls == UNLIT:
                    sightings.append(Sighting(where(d.box), UNLIT_WHO, fi, float(d.box.h)))
                elif d.box.h >= self.cfg.stranger_min_height * height:
                    sightings.append(Sighting(where(d.box), STRANGER_WHO, fi, float(d.box.h)))
        if found.box is not None:
            self.self_box = found.box
            log.info("转圈认出了团子：%s", found.box)
        ref = self._ref_height(height or self._frame_h)
        entries = merge(sightings, spin.merge_deg, ref, self.cfg.near, self.cfg.far)
        return SweepResult(entries, found.box, len(frames), frames[-1][0] if frames else 0.0)

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
    def _tag_err(player, tag) -> float | None:
        """名字标签在不在这个人头顶：水平上差不多对齐，标签在人物上沿附近（上方一个半身高到身体上半截之间）。
        在的话返回对不齐的程度（越小越正），不在返回 None。"""
        p = player.box
        tx = tag.box.x + tag.box.w / 2
        if not p.x - p.w * 0.25 <= tx <= p.x2 + p.w * 0.25:
            return None
        if not p.y - 1.5 * p.h <= tag.box.y2 <= p.y + 0.5 * p.h:
            return None
        return abs(tx - (p.x + p.w / 2)) + abs(tag.box.y2 - p.y)

    @classmethod
    def _tag_over(cls, player, tags: list) -> object | None:
        """这个人头顶对得最正的名字标签（不管有没有分给别人；远处重裁、转圈扫描用）。"""
        scored = [(err, i) for i, t in enumerate(tags) if (err := cls._tag_err(player, t)) is not None]
        return tags[min(scored)[1]] if scored else None

    def _assign_tags(self, players: list[Track], tags: list[Track]) -> dict[int, Track]:
        """人物轨迹 id → 头顶的名字标签。一个标签只给对得最正的那个人（实测团子和好友挤在一起时两个人都挂上了好友的名字）。"""
        pairs = sorted(
            (err, i, j) for i, p in enumerate(players) for j, t in enumerate(tags)
            if (err := self._tag_err(p, t)) is not None
        )
        out: dict[int, Track] = {}
        used: set[int] = set()
        for _, i, j in pairs:
            if players[i].id not in out and j not in used:
                out[players[i].id] = tags[j]
                used.add(j)
        return out

    @staticmethod
    def _person_below(bubble: Rect, people: list):
        """气泡正下方的人：气泡水平中心落在人物框（左右各放宽 25%）内，气泡下沿在人物上沿往上 2.5 个身高到身体上部之间。"""
        bx = bubble.x + bubble.w / 2
        best, best_gap = None, float("inf")
        for person in people:
            p = person.box
            if not p.x - p.w * 0.25 <= bx <= p.x2 + p.w * 0.25:
                continue
            if not p.y - 2.5 * p.h <= bubble.y2 <= p.y + 0.3 * p.h:
                continue
            gap = abs(p.y - bubble.y2)
            if gap < best_gap:
                best, best_gap = person, gap
        return best

    def _watch_typing(self, bubbles: list[Track], players: list[Track], selfs: list[Track], now: float,
                      width: int, height: int) -> None:
        """头顶的"正在输入"气泡挂到下面的人身上（团子自己的不算），记下来给 speaker_hint 用。"""
        me = [s.box for s in selfs] + ([self.self_box] if self.self_box is not None else [])
        for b in bubbles:
            if self._person_below(b.box, [Track(0, "self", r, 1.0, now, now) for r in me]) is not None:
                continue  # 团子自己在打字（输入框开着时头顶也有气泡）
            owner = self._person_below(b.box, players)
            if owner is None:
                continue
            if now - owner.data.get("typing_at", float("-inf")) > self.cfg.bubble_gap:
                owner.data["bubble_start"] = now  # 新的一句
            owner.data["typing_at"] = now
            friend = bool(owner.data.get("name") or owner.data.get("tagged") or owner.data.get("maybe"))
            self._typing.append((now, owner.id, friend, owner.cls == UNLIT, owner.box.x + owner.box.w / 2, owner.box.h,
                                 width, height))
        while self._typing and now - self._typing[0][0] > self.cfg.typing_window:
            self._typing.popleft()

    def _watch_approach(self, players: list[Track], now: float, width: int) -> None:
        for p in players:
            hist: deque = p.data.setdefault("hist", deque())
            hist.append((now, float(p.box.h), p.box.x + p.box.w / 2))
            while hist and now - hist[0][0] > self.cfg.approach_window:
                hist.popleft()
            maybe = p.data.get("maybe")
            who = (
                p.data.get("name")
                or (maybe if maybe and maybe in self.nearby(now) else None)  # 按外观认的：他还在身边才算（同 last_seen 的规矩）
                or (STRANGER if p.data.get("stranger") else None)
            )
            if who is None or (who == STRANGER and not self.cfg.approach_strangers):
                continue
            if now - self._approach_at.get(who, float("-inf")) < self.cfg.approach_cooldown:
                continue
            if approaching(list(hist), width, self.cfg.approach_grow):
                self._approach_at[who] = now
                log.info("%s 朝团子走过来了", who)
                with self._lock:
                    self._approaches.append(who)
                    self._approach_log.append((who, p.id, now))
                    while self._approach_log and now - self._approach_log[0][2] > 30.0:
                        self._approach_log.popleft()

    def _watch_gestures(self, frame: np.ndarray, players: list[Track], now: float, width: int, height: int) -> None:
        """三期 §3：认出名字、近 / 中、在画面中间的好友，攒够一段（frames 张）后每 interval 秒判一次动作。"""
        cfg = self.gesture_cfg
        ref = self._ref_height(height)
        for p in players:
            name = p.data.get("name")
            if p.cls != "player" or not eligible(name, p.box, width, ref, self.cfg.near, self.cfg.far):
                p.data.pop("clip", None)  # 走开 / 走远了：之前攒的不能和回来后的拼在一起
                continue
            buf: ClipBuffer = p.data.setdefault("clip", ClipBuffer(cfg.frames, cfg.fps))
            buf.push(now, person_crop(frame, p.box, cfg.size))
            if not buf.ready() or now - p.data.get("gesture_at", float("-inf")) < cfg.interval:
                continue
            p.data["gesture_at"] = now
            try:
                with self._infer:
                    label, prob = self.gestures.classify(buf.clip())
            except Exception:
                log.exception("认动作出错")
                continue
            if label == "none" or prob < cfg.min_prob:
                continue
            key = (name, label)
            if now - self._gesture_at.get(key, float("-inf")) < cfg.cooldown:
                continue
            self._gesture_at[key] = now
            log.info("%s 对团子做了动作：%s（%.2f）", name, label, prob)
            with self._lock:
                self._gestures.append(key)

    def talkers(self, now: float) -> list[Talker]:
        """1 秒内头顶冒着气泡的人（不含团子自己），带位置和这次气泡开始的时间。"""
        out = []
        for t in list(self.last_tracks):
            at = t.data.get("typing_at")
            if at is None or now - at > 1.0:
                continue
            name = t.data.get("name") or t.data.get("maybe")
            friend = bool(name or t.data.get("tagged"))
            out.append(Talker(t.id, name, friend, t.box.x + t.box.w / 2, at, t.data.get("bubble_start", at)))
        return out

    def recent_approaches(self, now: float, within: float = 5.0) -> list[tuple[str, float, float]]:
        """within 秒内朝团子走过来的人：(谁, 现在的框中心 x, 时间)；谁同 pop_approaches（好友名 / STRANGER）。不取走。
        x 按那条轨迹现在的位置取（人会接着走、镜头会转）；轨迹 1 秒没更新（走没了）就不报。"""
        tracks = {t.id: t for t in list(self.last_tracks) if now - t.last <= 1.0}
        with self._lock:
            log_ = [a for a in self._approach_log if now - a[2] <= within]
        return [(who, tracks[tid].box.x + tracks[tid].box.w / 2, t) for who, tid, t in log_ if tid in tracks]

    def pop_gestures(self) -> list[tuple[str, str]]:
        """取走"谁对团子做了什么动作"，身体变成 gesture 事件。"""
        with self._lock:
            out = list(self._gestures)
            self._gestures.clear()
        return out

    def pop_approaches(self) -> list[str]:
        """取走"朝团子走过来"的人（好友名 / 陌生人），身体变成 approach 事件。"""
        with self._lock:
            out, self._approaches = self._approaches, []
        return out

    def pop_stranger_backs(self) -> list[str]:
        """取走"走开又回来了"的陌生人编号（"陌生人A"），身体变成 stranger_back 事件。"""
        with self._lock:
            out, self._stranger_backs = self._stranger_backs, []
        return out

    def my_look(self) -> str:
        """团子自己现在的装扮描述（没有就空）。"""
        return self.appearance.look("me", "") if self.appearance is not None else ""

    def looks(self, names: list[str]) -> dict[str, str]:
        """好友现在的装扮描述：{名字: 描述}，没描述的不给。"""
        if self.appearance is None:
            return {}
        return {n: desc for n in names if (desc := self.appearance.look("friend", n))}

    def nearest(self, now: float) -> tuple[str, str] | None:
        """最近一帧里离团子最近（框最高）、认得出是谁的人：(好友名 / "陌生人", 近 / 中 / 远)。"""
        known = [
            (t.box.h, t.data.get("name") or t.data.get("maybe") or STRANGER)
            for t in list(self.last_tracks)
            if t.cls in ("player", UNLIT) and (t.data.get("name") or t.data.get("maybe") or t.data.get("stranger"))
        ]
        if not known:
            return None
        h, who = max(known)
        return who, distance(h, self._ref_height(self._frame_h), self.cfg.near, self.cfg.far)

    def people(self, now: float) -> list[Person]:
        """最近一帧里认得出是谁的人（好友 / 陌生人 / 没点火的黑影），左到右、同一边的近的在前。

        暂停中、或者这一帧已经过时（被挡住 / 没跑检测）返回空：别让技能盯着暂停前的人。
        """
        if self.paused:
            return []
        ref = self._ref_height(self._frame_h)
        out = []
        for t in list(self.last_tracks):
            if now - t.last > PEOPLE_STALE:
                continue
            d = t.data
            name, sure, sid, look = d.get("name"), True, None, ""
            if t.cls == UNLIT and d.get("stranger"):  # 和团子框重叠的黑影不算（process 里没给它记 stranger）
                kind = "unlit"
            elif t.cls == "player" and name:
                kind = "friend"
            elif t.cls == "player" and d.get("maybe"):  # 没看到名字、按外观认的好友
                kind, name, sure = "friend", d["maybe"], False
            elif t.cls == "player" and d.get("stranger"):
                kind, sid = "stranger", d.get("sid")
                if sid and self.appearance is not None:
                    look = self.appearance.look("stranger", sid)
            else:
                continue
            side = side_of(t.box.x + t.box.w / 2, self._frame_w)
            out.append(Person(t.id, kind, name if kind == "friend" else None, t.box, side,
                              distance(t.box.h, ref, self.cfg.near, self.cfg.far), sure=sure, sid=sid, look=look))
        order = {"左边": 0, "前面": 1, "右边": 2}
        return sorted(out, key=lambda p: (order[p.side], -p.box.h))

    def typing_seen(self, now: float, within: float = 1.0, strangers: bool = False) -> bool:
        """最近 within 秒里有好友（strangers 时陌生人也算）头顶冒着"正在输入"气泡（团子自己的不算）：聊天面板该打开看了。"""
        return any(now - r[0] <= within and (r[2] or strangers) for r in list(self._typing))

    def speaker_hint(self, now: float) -> str | None:
        """陌生人的消息是谁说的：最近 typing_window 秒内头顶冒过气泡、又不是好友的人恰好一个 → 说出他在画面哪儿。"""
        recent = [r for r in list(self._typing) if now - r[0] <= self.cfg.typing_window and not r[2]]
        if len({r[1] for r in recent}) != 1:
            return None
        _, _, _, dark, cx, h, width, height = recent[-1]
        side = side_of(cx, width)
        far = distance(h, self._ref_height(height), self.cfg.near, self.cfg.far) == "远"
        who = "没点火的陌生人" if dark else "陌生人"
        return f"（说话的可能是{side}{'远处' if far else '近处'}那个{who}）"

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
            if t.cls == "player" and d.get("maybe") and not d.get("name"):
                kind, label = "maybe", f"像{d['maybe']}?"
            elif t.cls == "player":
                kind = "stranger" if d.get("stranger") else ("friend" if d.get("name") else "player")
                label = d.get("name") or ((d.get("sid") or "陌生人") if d.get("stranger") else "")
            elif t.cls == UNLIT:
                kind, label = "unlit", "陌生人（没点火）"
            elif t.cls == "self":
                kind, label = "self", "团子"
            elif t.cls == "typing":
                kind, label = "typing", "正在输入"
            elif t.cls == "name_tag":
                kind = "name" if d.get("name") else "tag"
                label = d.get("name") or (f"?{d['text']}" if d.get("text") else "?")
            elif t.cls in OBJECT_NAMES:
                kind, label = t.cls, OBJECT_NAMES[t.cls]
            elif t.cls == "social_ring":
                ring = d.get("kind")
                kind = "request" if is_request(ring) else "ring"
                label = KIND_NAMES.get(ring, "✦" if ring == IDLE else "?")
            else:
                kind, label = t.cls, t.cls
            b = t.box
            entry = {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "kind": kind, "label": label, "score": round(t.score, 2)}
            if self.appearance is not None and (desc := self._desc(kind, d)):
                entry["desc"] = desc
            out.append(entry)
        return out

    def _desc(self, kind: str, d: dict) -> str:
        """框对应的人的装扮描述（没有就空）。"""
        if kind == "friend":
            return self.appearance.look("friend", d["name"])
        if kind == "maybe":
            return self.appearance.look("friend", d["maybe"])
        if kind == "stranger" and d.get("sid"):
            return self.appearance.look("stranger", d["sid"])
        if kind == "self":
            return self.appearance.look("me", "")
        return ""

    def objects(self, now: float) -> list[Thing]:
        """最近一帧里的物品（座位 / 篝火 / 乐器 / 先祖），左到右、同一边近的在前。

        连续看到 object_min_hits 帧才算；暂停中、或者这一帧已经过时返回空（同 people()）。
        """
        if self.paused:
            return []
        out = []
        for t in list(self.last_tracks):
            if t.cls not in OBJECT_NAMES or now - t.last > PEOPLE_STALE or t.hits < self.cfg.object_min_hits:
                continue
            side = side_of(t.box.x + t.box.w / 2, self._frame_w)
            out.append(Thing(t.id, t.cls, t.box, side,
                             object_distance(t.box.y2, self._frame_h, self.cfg.object_near, self.cfg.object_far)))
        order = {"左边": 0, "前面": 1, "右边": 2}
        return sorted(out, key=lambda o: (order[o.side], -o.box.y2))

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
        if self.place and self._frozen(now) - self.place_at <= self.env_cfg.place_keep:
            parts.append(f"- 看起来在：{self.place}（按画面认的，可能不准）")
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
