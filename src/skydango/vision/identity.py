"""身份板：每条人物轨迹"这个框是谁"（spec 2026-10-22-identity-board-design.md）。

纯计算：不碰画面、不调模型、不看墙钟、没有随机数；同一串 FrameIn 重放结果一样。
perception 每帧把证据（标签候选、外观分、团子线索、喊一声的结果）拼成一个 FrameIn 交给 step()，
拿回带记性的标签分配、每条轨迹的 Verdict、这一帧哪些样本能学进底库。

1. 标签分配：按误差贪心，上一帧挂着的那一对误差乘 tag_stick；名字连续 tag_hold 秒挂在另一条轨迹上才换主人，
   原来的主人失去它 → 拿不准（不是陌生人）
2. 外观：每条轨迹最近 window 个样本对每个好友的中位分；像（带滞回）/ 拿不准；一个名字同一时刻只给一条轨迹，
   稳定标签永远盖过外观；接回的名字外观不反对就算像
3. 陌生人：黑影等 stranger_after；明显不像任何好友和团子；一直没结论等满 stranger_after + grace。
   挂过名字的、头顶最近有别人的名字标签候选的不判，进拿不准等喊一声：亮了 = 他，没亮 / 一直没喊成 = 陌生人。
   判了陌生人只有名字（稳定标签 / 喊一声亮出来）能改回来
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from ..config import AppearanceConfig, IdentityConfig
from .bubbles import Rect

FRIEND_TAG = "friend_tag"  # 名字标签证实
FRIEND_LOOK = "friend_look"  # 看着像
DANGO = "dango"
UNSURE = "unsure"  # 拿不准：等喊一声
STRANGER = "stranger"
PENDING = "pending"  # 刚出现、证据还不够

EPS = 1e-9  # 时间比较的容差（0.6 - 0.1 之类）
CALL_SLACK = 2.0  # 喊过一声、窗口结束后最多再等这么久收到结果，再久就当没喊（按拿不准超时算）


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
    lost: str | None = None  # 拿不准的名字：失去的（读得出的），或者一直挂在旁边人身上的
    stranger: bool = False  # 判过陌生人：只有名字能改回来
    panel_at: float | None = None  # 最后一次在面板后面
    cand_at: float | None = None  # 最后一次头顶有读出名字的标签候选
    cand_name: str | None = None
    kind: str = PENDING
    name: str | None = None
    since: float = 0.0
    samples: deque = field(default_factory=deque)  # 最近 window 个样本的好友分
    mes: deque = field(default_factory=deque)  # 同上，团子底库分（None 不存）


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
            if t.under_panel:
                st.panel_at = now
            named = [c for c in t.tags if c.name is not None]
            if named:
                st.cand_at, st.cand_name = now, min(named, key=lambda c: (c.err, c.key)).name
        tags = self._assign(frame.tracks, exclude={self.dango_id} if self.dango_id is not None else set())
        self._hold(tags, frame.tracks, now)
        if frame.call_end is not None:
            for tid, name in sorted(frame.call_end.shown.items()):
                if tid in seen and tid != self.dango_id:
                    self._take(tid, name, name, now)  # 喊一声亮出名字：当场算稳定持有
                    self._prev[name] = tid
        for t in frame.tracks:
            self._sample(t)
        looks = self._looks(frame.tracks)
        verdicts = {t.id: self._verdict(t, looks.get(t.id), frame) for t in frame.tracks}
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
        st.held, st.held_name, st.tagged, st.lost, st.stranger = key, name, True, None, False

    # ── 外观 ──

    def _friends(self, s: Scores) -> dict[str, float]:
        """按 friend_feature 挑好友分：both 两份都有时取平均、只有一份取那份。"""
        feat = self.cfg.friend_feature
        if feat == "color":
            return dict(s.color)
        if feat == "both":
            out = {}
            for name in set(s.dino) | set(s.color):
                vals = [d[name] for d in (s.dino, s.color) if name in d]
                out[name] = sum(vals) / len(vals)
            return out
        return dict(s.dino)

    def _sample(self, t: TrackIn) -> None:
        if t.scores is None:
            return
        st = self._tracks[t.id]
        st.samples.append(self._friends(t.scores))
        if t.scores.me is not None:
            st.mes.append(t.scores.me)
        while len(st.samples) > self.cfg.window:
            st.samples.popleft()
        while len(st.mes) > self.cfg.window:
            st.mes.popleft()

    def _medians(self, st: _Track) -> dict[str, float] | None:
        """样本够 min_samples 时每个好友的中位分（只算有分的样本）；不够 = None。"""
        if len(st.samples) < max(1, self.cfg.min_samples):
            return None
        names = {n for d in st.samples for n in d}
        return {n: median(d[n] for d in st.samples if n in d) for n in names}

    def _looks(self, tracks: tuple[TrackIn, ...]) -> dict[int, tuple[str, str, float]]:
        """外观结论：轨迹 → (FRIEND_LOOK / UNSURE, 名字, 中位分)。稳定持有标签的、团子不算。"""
        a = self.acfg
        cands: dict[int, tuple[str, float]] = {}
        out: dict[int, tuple[str, str, float]] = {}
        for t in tracks:
            st = self._tracks[t.id]
            if st.held is not None or t.id == self.dango_id:
                continue
            meds = self._medians(st)
            if t.relink is not None:
                med = None if meds is None else meds.get(t.relink)
                if med is None or med >= a.unsure:  # 外观不反对（样本不够 / 他没底库也算不反对）
                    cands[t.id] = (t.relink, med or 0.0)
                    continue
            if not meds:
                continue
            ranked = sorted(meds.items(), key=lambda kv: (-kv[1], kv[0]))
            name, top = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            if st.kind == FRIEND_LOOK and st.name in meds and meds[st.name] >= a.match - self.cfg.drop:
                cands[t.id] = (st.name, meds[st.name])  # 已经像的，掉到 match - drop 以下才算不像
            elif top >= a.match and top - second >= a.margin:
                cands[t.id] = (name, top)
            elif top >= a.unsure:
                out[t.id] = (UNSURE, name, top)
        held = {st.held_name for st in self._tracks.values() if st.held is not None and st.held_name is not None}
        best: dict[str, tuple[float, int]] = {}
        for tid, (name, score) in cands.items():
            if name not in held and (name not in best or (score, -tid) > (best[name][0], -best[name][1])):
                best[name] = (score, tid)
        for tid, (name, score) in cands.items():
            won = best.get(name, (0.0, None))[1] == tid
            out[tid] = (FRIEND_LOOK if won else UNSURE, name, score)  # 名字被别人占着：拿不准
        return out

    # ── 判定 ──

    def _verdict(self, t: TrackIn, look: tuple[str, str, float] | None, frame: FrameIn) -> Verdict:
        now = frame.now
        st = self._tracks[t.id]
        meds = self._medians(st)
        top = max(meds.values()) if meds else 0.0
        if st.held is not None:
            name = st.held_name
            return self._mark(st, FRIEND_TAG, name, (meds or {}).get(name, 0.0) if name else 0.0, now)
        if st.stranger:
            return self._mark(st, STRANGER, None, top, now)
        if look is not None:
            kind, name, score = look
        elif st.lost is not None:
            kind, name, score = UNSURE, st.lost, (meds or {}).get(st.lost, 0.0)
        else:
            kind, name, score = PENDING, None, 0.0
        # 面板挡着标签、框太小（远处）的不判陌生人；黑影照旧（远近、面板都算）
        judge = not t.under_panel and t.box.h >= self.stranger_min_height * frame.height
        base = max(t.first, st.panel_at) if st.panel_at is not None else t.first
        after = self.stranger_after
        if kind == UNSURE:
            if judge and self._rejected(t, st, frame):
                return self._stranger(st, top, now)
            return self._mark(st, kind, name, score, now)
        if kind != PENDING or st.tagged:
            return self._mark(st, kind, name, score, now)
        if t.unlit:
            if now - t.first >= after - EPS:
                return self._stranger(st, top, now)
        elif judge and now - base >= after - EPS:
            if self._nobody(st, meds):
                return self._stranger(st, top, now)
            if st.cand_at is not None and now - st.cand_at <= after + EPS:
                st.lost = st.cand_name  # 名字标签一直被旁边的人占着：拿不准，等喊一声
                return self._mark(st, UNSURE, st.lost, (meds or {}).get(st.lost, 0.0), now)
            if now - base >= after + self.grace - EPS:
                return self._stranger(st, top, now)
        return self._mark(st, PENDING, None, 0.0, now)

    def _nobody(self, st: _Track, meds: dict[str, float] | None) -> bool:
        """明显不像：有好友底库、对每个好友和团子都低于 stranger_low。"""
        low = self.cfg.stranger_low if self.cfg.stranger_low is not None else self.acfg.unsure
        if not meds or any(v >= low for v in meds.values()):
            return False
        return not st.mes or median(st.mes) < low

    def _rejected(self, t: TrackIn, st: _Track, frame: FrameIn) -> bool:
        """拿不准的收尾：喊过没亮出他；或者一直没喊成、挂满 unsure_wait + call_window。"""
        now, end = frame.now, frame.call_end
        if t.called_at is not None:
            if end is not None and abs(end.at - t.called_at) <= EPS:
                return t.id not in end.shown
            if now - t.called_at <= self.call_window + CALL_SLACK:
                return False  # 他的那一声还在途中
        since = st.since if st.kind == UNSURE else now
        return now - since >= self.acfg.unsure_wait + self.call_window - EPS

    def _stranger(self, st: _Track, score: float, now: float) -> Verdict:
        st.stranger, st.lost = True, None
        return self._mark(st, STRANGER, None, score, now)

    def _mark(self, st: _Track, kind: str, name: str | None, score: float, now: float) -> Verdict:
        """since = 这个结论从什么时候开始；拿不准换了最像的人不重新计时。"""
        if kind != st.kind or (name != st.name and kind != UNSURE):
            st.since = now
        st.kind, st.name = kind, name
        return Verdict(kind, name, score, st.since)
