"""身份板：每条人物轨迹"这个框是谁"（spec 2026-10-22-identity-board-design.md）。

纯计算：不碰画面、不调模型、不看墙钟、没有随机数；同一串 FrameIn 重放结果一样。
perception 每帧把证据（标签候选、外观分、团子线索、喊一声的结果）拼成一个 FrameIn 交给 step()，
拿回带记性的标签分配、每条轨迹的 Verdict、这一帧哪些样本能学进底库。

1. 标签分配：按误差贪心，上一帧挂着的那一对误差乘 tag_stick；名字连续 tag_hold 秒挂在另一条轨迹上才换主人，
   原来的主人失去它 → 拿不准（不是陌生人）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import AppearanceConfig, IdentityConfig
from .bubbles import Rect

FRIEND_TAG = "friend_tag"  # 名字标签证实
FRIEND_LOOK = "friend_look"  # 看着像
DANGO = "dango"
UNSURE = "unsure"  # 拿不准：等喊一声
STRANGER = "stranger"
PENDING = "pending"  # 刚出现、证据还不够

EPS = 1e-9  # 时间比较的容差（0.6 - 0.1 之类）


@dataclass(frozen=True)
class TagCand:
    key: str  # 读出来的名字；读不出时 "#<标签轨迹 id>"
    name: str | None
    err: float


@dataclass(frozen=True)
class Scores:
    dino: dict[str, float]  # 每个好友底库 DINOv2 最像一张（没分的好友不列）
    color: dict[str, float]  # 同上，颜色
    me: float | None  # 团子底库 DINOv2 最像一张


@dataclass(frozen=True)
class TrackIn:
    id: int
    box: Rect
    first: float
    unlit: bool = False
    under_panel: bool = False
    tags: tuple[TagCand, ...] = ()
    scores: Scores | None = None  # 只在这一帧算出好样本时有
    self_score: float | None = None  # 和它同一个身体的 YOLO self 框分数
    with_bg: bool | None = None  # 镜头转了时跟不跟背景走；没转 = None
    relink: str | None = None  # 按位置接回的名字
    sid: str | None = None  # 陌生人编号
    called_at: float | None = None  # 这条轨迹等的那一声是什么时候喊的


@dataclass(frozen=True)
class CallEnd:
    at: float  # 喊的时刻
    shown: dict[int, str]  # 窗口里哪条轨迹亮出了哪个名字


@dataclass(frozen=True)
class FrameIn:
    now: float
    width: int
    height: int
    panel: bool
    sliding: bool  # 聊天面板开关的横移动画里
    moved: bool  # 走路 / 黑屏 / 转圈之后
    tracks: tuple[TrackIn, ...]
    call_end: CallEnd | None = None


@dataclass
class Verdict:
    kind: str
    name: str | None = None
    score: float = 0.0
    since: float = 0.0


@dataclass
class StepOut:
    tags: dict[int, str]  # 轨迹 id → 这一帧分到的标签 key
    verdicts: dict[int, Verdict]
    learn: dict[int, tuple[str, str]]  # 轨迹 id → (friend / me / stranger, 谁)


@dataclass
class _Track:
    """一条轨迹攒下的证据。"""

    first: float
    last: float  # 最后一次出现
    seg_key: str | None = None  # 正在挂的这一段标签
    seg_name: str | None = None
    seg_start: float = 0.0
    seg_last: float = 0.0
    held: str | None = None  # 稳定持有的 key
    held_name: str | None = None
    tagged: bool = False  # 稳定持有过名字
    lost: str | None = None  # 失去的名字（读得出的）
    kind: str = PENDING
    name: str | None = None
    since: float = 0.0
    extra: dict = field(default_factory=dict)


class IdentityBoard:
    def __init__(
        self,
        cfg: IdentityConfig,
        acfg: AppearanceConfig,
        *,
        stranger_after: float,
        grace: float,
        stranger_min_height: float,
        track_buffer: float,
        call_window: float,
    ):
        self.cfg, self.acfg = cfg, acfg
        self.stranger_after, self.grace = stranger_after, grace
        self.stranger_min_height = stranger_min_height
        self.track_buffer, self.call_window = track_buffer, call_window
        self._tracks: dict[int, _Track] = {}
        self._holder: dict[str, int] = {}  # key → 稳定持有它的轨迹
        self._prev: dict[str, int] = {}  # 上一帧的分配 key → 轨迹
        self.dango_id: int | None = None

    # ── 每帧 ──

    def step(self, frame: FrameIn) -> StepOut:
        now = frame.now
        seen = {t.id for t in frame.tracks}
        self._forget(now, seen)
        for t in frame.tracks:
            st = self._tracks.get(t.id)
            if st is None:
                st = self._tracks[t.id] = _Track(first=t.first, last=now, since=now)
            st.last = now
        tags = self._assign(frame.tracks, exclude={self.dango_id} if self.dango_id is not None else set())
        self._hold(tags, frame.tracks, now)
        verdicts = {t.id: self._verdict(t, now) for t in frame.tracks}
        return StepOut(tags=tags, verdicts=verdicts, learn={})

    # ── 清理 ──

    def _forget(self, now: float, seen: set[int]) -> None:
        for tid in [i for i, st in self._tracks.items() if i not in seen and now - st.last > self.track_buffer]:
            del self._tracks[tid]  # 时间往回拨时差是负的，不删
            for key in [k for k, v in self._holder.items() if v == tid]:
                del self._holder[key]
            if self.dango_id == tid:
                self.dango_id = None

    # ── 标签 ──

    def _assign(self, tracks: tuple[TrackIn, ...], exclude: set[int]) -> dict[int, str]:
        """这一帧所有（轨迹, 候选）按误差升序贪心；上一帧同一对的误差乘 tag_stick。"""
        pairs = []
        for t in tracks:
            if t.id in exclude:
                continue
            for c in t.tags:
                err = c.err * self.cfg.tag_stick if self._prev.get(c.key) == t.id else c.err
                pairs.append((err, t.id, c.key))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))
        out: dict[int, str] = {}
        used: set[str] = set()
        for _, tid, key in pairs:
            if tid in out or key in used:
                continue
            out[tid] = key
            used.add(key)
        self._prev = {key: tid for tid, key in out.items()}
        return out

    def _hold(self, tags: dict[int, str], tracks: tuple[TrackIn, ...], now: float) -> None:
        """记每条轨迹挂着的这一段；满 tag_hold 稳定持有，原来的主人失去它。"""
        names = {c.key: c.name for t in tracks for c in t.tags}
        for tid, key in tags.items():
            st = self._tracks[tid]
            if st.seg_key == key and now - st.seg_last <= self.cfg.tag_hold + EPS:
                st.seg_last = now
            else:
                st.seg_key, st.seg_start, st.seg_last = key, now, now
            st.seg_name = names.get(key)
            if self._holder.get(key) != tid and now - st.seg_start >= self.cfg.tag_hold - EPS:
                self._take(tid, key, st.seg_name, now)

    def _take(self, tid: int, key: str, name: str | None, now: float) -> None:
        old = self._holder.get(key)
        if old is not None and old != tid and old in self._tracks:
            o = self._tracks[old]
            o.held = o.held_name = None
            if name is not None:
                o.lost = name  # 读不出名字的 key 失去了就回 pending
        st = self._tracks[tid]
        if st.held is not None and st.held != key:
            self._holder.pop(st.held, None)
        self._holder[key] = tid
        st.held, st.held_name, st.tagged, st.lost = key, name, True, None

    # ── 判定 ──

    def _verdict(self, t: TrackIn, now: float) -> Verdict:
        st = self._tracks[t.id]
        if st.held is not None:
            kind, name = FRIEND_TAG, st.held_name
        elif st.lost is not None:
            kind, name = UNSURE, st.lost
        else:
            kind, name = PENDING, None
        return self._mark(st, kind, name, 0.0, now)

    def _mark(self, st: _Track, kind: str, name: str | None, score: float, now: float) -> Verdict:
        """since = 这个结论从什么时候开始；拿不准换了最像的人不重新计时。"""
        if kind != st.kind or (name != st.name and kind != UNSURE):
            st.since = now
        st.kind, st.name = kind, name
        return Verdict(kind, name, score, st.since)
