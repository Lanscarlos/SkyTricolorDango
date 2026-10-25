"""身份板：只吃证据，不碰画面、模型和墙钟，可按帧重放。"""

from dataclasses import dataclass, field, replace
from statistics import median

from skydango.config import AppearanceConfig, IdentityConfig
from .bubbles import Rect

FRIEND_TAG = "friend_tag"
FRIEND_LOOK = "friend_look"
DANGO = "dango"
UNSURE = "unsure"
STRANGER = "stranger"
PENDING = "pending"
LOOK_SPAN = 0.15
CALL_GRACE = 2.0  # 标了喊过、这么久还没收到结果 = 其实没喊成（dry-run、被拦）


@dataclass(frozen=True)
class TagCand:
    key: str
    name: str | None
    err: float


@dataclass(frozen=True)
class Scores:
    dino: dict[str, float]
    color: dict[str, float]
    me: float | None


@dataclass(frozen=True)
class TrackIn:
    id: int
    box: Rect
    first: float
    unlit: bool = False
    under_panel: bool = False
    tags: tuple[TagCand, ...] = ()
    scores: Scores | None = None
    self_score: float | None = None
    with_bg: bool | None = None
    relink: str | None = None
    sid: str | None = None
    called_at: float | None = None


@dataclass(frozen=True)
class CallEnd:
    at: float
    shown: dict[int, str]


@dataclass(frozen=True)
class FrameIn:
    now: float
    width: int
    height: int
    panel: bool
    sliding: bool
    moved: bool
    tracks: tuple[TrackIn, ...]
    call_end: CallEnd | None = None
    tag: str = ''  # 离线录像文件名；运行时留空


@dataclass
class Verdict:
    kind: str
    name: str | None = None
    score: float = 0.0
    since: float = 0.0


@dataclass
class StepOut:
    tags: dict[int, str]
    verdicts: dict[int, Verdict]
    learn: dict[int, tuple[str, str]]


@dataclass
class Evidence:
    last: float
    key: str | None = None
    tag_start: float = 0.0
    tag_last: float = float('-inf')
    stable: TagCand | None = None
    verdict: Verdict = field(default_factory=lambda: Verdict(PENDING))
    samples: list[dict[str, float]] = field(default_factory=list)
    ever_tagged: bool = False
    panel_at: float = float('-inf')
    nearby_name: tuple[float, str] | None = None
    me_samples: list[float | None] = field(default_factory=list)
    look: float | None = None
    smooth: float | None = None


class IdentityBoard:
    def __init__(self, cfg: IdentityConfig, acfg: AppearanceConfig, *,
                 stranger_after: float, grace: float, stranger_min_height: float,
                 track_buffer: float, call_window: float):
        self.cfg, self.acfg = cfg, acfg
        self.stranger_after, self.grace = stranger_after, grace
        self.stranger_min_height, self.track_buffer = stranger_min_height, track_buffer
        self.call_window = call_window
        self.dango_id: int | None = None
        self._evidence: dict[int, Evidence] = {}
        self._previous: dict[str, int] = {}
        self._spots: dict[bool, Rect] = {}
        self._challenger: tuple[int, float] | None = None
        self._weak_since: float | None = None

    @staticmethod
    def _set(e: Evidence, kind: str, name: str | None, now: float, score: float = 0.0):
        since = e.verdict.since if (e.verdict.kind, e.verdict.name) == (kind, name) else now
        e.verdict = Verdict(kind, name, score, since)

    def step(self, frame: FrameIn) -> StepOut:
        now = frame.now
        present = {t.id for t in frame.tracks}
        for id, e in list(self._evidence.items()):
            if id not in present and now - e.last > self.track_buffer:
                del self._evidence[id]
        for t in frame.tracks:
            self._evidence.setdefault(t.id, Evidence(now)).last = now

        pairs = sorted((c.err * (self.cfg.tag_stick if self._previous.get(c.key) == t.id else 1),
                        t.id, c.key, c) for t in frame.tracks if t.id != self.dango_id for c in t.tags)
        assigned: dict[int, TagCand] = {}
        used, before = set(), self._previous
        for _, id, key, cand in pairs:
            if id not in assigned and key not in used:
                assigned[id] = cand
                used.add(key)
        self._previous = {c.key: id for id, c in assigned.items()}
        for id, cand in assigned.items():
            e = self._evidence[id]
            # 上一帧这个标签在别人身上也算断了：来回换的标签谁都攒不满 tag_hold
            if (e.key != cand.key or now - e.tag_last > self.cfg.tag_hold + 1e-9
                    or before.get(cand.key, id) != id):
                e.key, e.tag_start = cand.key, now
            e.tag_last = now
            if now - e.tag_start + 1e-9 >= self.cfg.tag_hold:
                for other_id, other in self._evidence.items():
                    if other_id != id and other.stable and other.stable.key == cand.key:
                        name = other.stable.name
                        other.stable = None
                        self._set(other, UNSURE if name else PENDING, name, now)
                e.stable = cand
                e.ever_tagged = True
        for t in frame.tracks:
            e = self._evidence[t.id]
            if e.stable:
                self._set(e, FRIEND_TAG, e.stable.name, now)
        self._friends(frame)
        self._strangers(frame)
        self._dango(frame)
        assigned.pop(self.dango_id, None)
        learn = {}
        for t in frame.tracks:
            if t.scores is None:
                continue
            e = self._evidence[t.id]
            v = e.verdict
            confirmed = frame.call_end and frame.call_end.shown.get(t.id) == v.name
            if (v.kind == FRIEND_TAG and v.name and
                    (confirmed or (t.id in assigned and assigned[t.id].key == e.stable.key
                                   and now - e.tag_start + 1e-9 >= self.cfg.tag_hold))):
                learn[t.id] = ('friend', v.name)
            elif v.kind == DANGO and v.score >= self.cfg.dango_learn:
                learn[t.id] = ('me', '')
            elif v.kind == STRANGER and t.sid:
                learn[t.id] = ('stranger', t.sid)
        return StepOut({id: c.key for id, c in assigned.items()},
                       {t.id: self._evidence[t.id].verdict for t in frame.tracks}, learn)

    def _scores(self, scores: Scores) -> dict[str, float]:
        if self.cfg.friend_feature == 'both':
            return {name: (sum(v[name] for v in (scores.dino, scores.color) if name in v)
                           / sum(name in v for v in (scores.dino, scores.color)))
                    for name in scores.dino.keys() | scores.color.keys()}
        return dict(scores.color if self.cfg.friend_feature == 'color' else scores.dino)

    def _medians(self, e: Evidence) -> dict[str, float]:
        if len(e.samples) < self.cfg.min_samples:
            return {}
        names = set().union(*e.samples)
        return {name: median(s.get(name, 0.0) for s in e.samples) for name in names}

    def _friends(self, frame: FrameIn):
        claimed = {e.stable.name for e in self._evidence.values() if e.stable and e.stable.name}
        candidates = []
        for t in frame.tracks:
            e = self._evidence[t.id]
            if t.scores is not None:
                e.samples.append(self._scores(t.scores))
                e.samples = e.samples[-self.cfg.window:]
                e.me_samples.append(t.scores.me)
                e.me_samples = e.me_samples[-self.cfg.window:]
            if e.stable or e.verdict.kind == STRANGER or t.id == self.dango_id:
                continue
            scores = self._medians(e)
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            name, score = ranked[0] if ranked else (None, 0.0)
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            old = e.verdict
            if (old.kind == FRIEND_LOOK and old.name in scores
                    and scores[old.name] >= self.acfg.match - self.cfg.drop):
                candidates.append((scores[old.name], t.id, old.name))
            elif name and score >= self.acfg.match and score - second >= self.acfg.margin:
                candidates.append((score, t.id, name))
            elif t.relink and (len(e.samples) < self.cfg.min_samples
                               or scores.get(t.relink, 0.0) >= self.acfg.unsure):
                candidates.append((scores.get(t.relink, 0.0), t.id, t.relink))
            elif name and score >= self.acfg.unsure:
                self._set(e, UNSURE, name, frame.now, score)
            elif old.kind != UNSURE:
                self._set(e, PENDING, None, frame.now)
        for score, id, name in sorted(candidates, key=lambda item: (-item[0], item[1])):
            e = self._evidence[id]
            if name not in claimed:
                self._set(e, FRIEND_LOOK, name, frame.now, score)
                claimed.add(name)
            else:
                self._set(e, UNSURE, name, frame.now, score)

    def _strangers(self, frame: FrameIn):
        now = frame.now
        low = self.acfg.unsure if self.cfg.stranger_low is None else self.cfg.stranger_low
        for t in frame.tracks:
            e = self._evidence[t.id]
            if t.under_panel:
                e.panel_at = now
            names = [c for c in t.tags if c.name]
            if names:
                e.nearby_name = (now, min(names, key=lambda c: (c.err, c.key)).name)
            v = e.verdict
            call = frame.call_end
            if call and v.kind in (UNSURE, STRANGER):
                name = call.shown.get(t.id)
                if name:
                    for other in self._evidence.values():
                        if other is not e and other.stable and other.stable.key == name:
                            other.stable = None
                            self._set(other, UNSURE, name, now)
                    e.stable = TagCand(name, name, 0.0)
                    e.ever_tagged = True
                    e.key, e.tag_start, e.tag_last = name, now - self.cfg.tag_hold, now
                    self._set(e, FRIEND_TAG, name, now)
                    continue
                # 一声确认画面里所有拿不准的人：喊之前就拿不准、喊的时候就在的，没亮出名字也算
                if t.called_at == call.at or (v.kind == UNSURE and v.since <= call.at and t.first <= call.at):
                    self._set(e, STRANGER, None, now)
                    continue
            if v.kind in (FRIEND_TAG, FRIEND_LOOK, STRANGER):
                continue
            # 遮挡和远处不凭“没有标签”下结论。
            if not t.unlit and (t.under_panel or t.box.h < self.stranger_min_height * frame.height):  # 黑影照旧远近、面板后面都判
                if v.kind != UNSURE:
                    self._set(e, PENDING, None, now)
                continue
            if v.kind == UNSURE:
                calling = t.called_at is not None and now - t.called_at <= self.call_window + CALL_GRACE
                if not calling and now - v.since >= self.acfg.unsure_wait + self.call_window:
                    self._set(e, STRANGER, None, now)
                continue
            age = now - (t.first if t.unlit else max(t.first, e.panel_at))
            recent = e.nearby_name and now - e.nearby_name[0] <= self.stranger_after
            if recent and age >= self.stranger_after:
                self._set(e, UNSURE, e.nearby_name[1], now)
                continue
            if e.ever_tagged:
                continue
            scores = self._medians(e)
            me = [s for s in e.me_samples if s is not None]
            clearly_other = bool(scores) and max(scores.values()) < low and (not me or median(me) < low)
            if age >= self.stranger_after and (t.unlit or clearly_other or age >= self.stranger_after + self.grace):
                self._set(e, STRANGER, None, now)

    def dango_spot(self, panel: bool) -> Rect | None:
        return self._spots.get(panel)

    def shift(self, seconds: float) -> None:
        """暂停期间没看见的时间不算证据持续时间。呼喊 at 是标识，不挪。"""
        for e in self._evidence.values():
            e.last += seconds
            e.tag_start += seconds
            e.tag_last += seconds
            e.panel_at += seconds
            e.verdict = replace(e.verdict, since=e.verdict.since + seconds)
            if e.nearby_name:
                e.nearby_name = (e.nearby_name[0] + seconds, e.nearby_name[1])
        if self._challenger:
            self._challenger = (self._challenger[0], self._challenger[1] + seconds)
        if self._weak_since is not None:
            self._weak_since += seconds

    def _dango(self, frame: FrameIn):
        candidates = []
        by_id = {t.id: t for t in frame.tracks}
        spot = self._spots.get(frame.panel)
        cued = set()
        for t in frame.tracks:
            e = self._evidence[t.id]
            parts = []
            if spot is not None and not frame.sliding:
                dist = abs(t.box.x + t.box.w / 2 - spot.x - spot.w / 2) / max(spot.w, 1)
                pos = 0.5 if frame.moved else max(0.0, 1 - dist / 1.5)
                parts.append(('pos', self.cfg.dango_w_pos, pos))
            if t.scores is not None and t.scores.me is not None:
                # 好友底库还空着时也要够像自己才有分，不然谁都满分
                base = max(max(t.scores.dino.values(), default=0.0), self.acfg.dango_match - LOOK_SPAN)
                e.look = min(1.0, max(0.0, (t.scores.me - base) / LOOK_SPAN))
            if e.look is not None:
                parts.append(('look', self.cfg.dango_w_look, e.look))
            if t.self_score is not None:
                parts.append(('self', self.cfg.dango_w_self, t.self_score))
            if t.with_bg is not None:
                parts.append(('pivot', self.cfg.dango_w_pivot, float(not t.with_bg)))
            parts = [(kind, w, s) for kind, w, s in parts if w > 0]
            if any(k != 'pos' for k, _, _ in parts):
                cued.add(t.id)
            if parts:
                raw = sum(w * s for _, w, s in parts) / sum(w for _, w, _ in parts)
                e.smooth = raw if e.smooth is None else 0.7 * e.smooth + 0.3 * raw
            tagged = e.key is not None and frame.now - e.tag_last <= self.cfg.tag_hold + 1e-9  # 头上有标签的是别人
            if (not e.stable and not t.unlit and not tagged and t.id in cued
                    and e.smooth >= self.cfg.dango_min):
                candidates.append((e.smooth, t.id))
        current = self._evidence.get(self.dango_id)
        here = self.dango_id in by_id
        # 漏检一两帧（track_buffer 内证据还在）不换人，照常走下面的换人滞回
        if current is None or (here and (current.stable or by_id[self.dango_id].unlit)):
            self.dango_id = None
            self._weak_since = None
        if not frame.sliding:
            if self.dango_id is not None:
                if current.smooth is not None and current.smooth < self.cfg.dango_min:
                    if self._weak_since is None:
                        self._weak_since = frame.now
                    if frame.now - self._weak_since >= self.cfg.dango_switch:
                        self.dango_id = None
                else:
                    self._weak_since = None
            best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
            if self.dango_id is None:
                self.dango_id = best[1] if best else None
                self._challenger = None
            elif best and best[1] != self.dango_id and best[0] > (current.smooth or 0.0):
                if self._challenger is None or self._challenger[0] != best[1]:
                    self._challenger = (best[1], frame.now)
                elif frame.now - self._challenger[1] >= self.cfg.dango_switch:
                    self.dango_id = best[1]
                    self._challenger = None
            else:
                self._challenger = None
        if self.dango_id in by_id:
            t = by_id[self.dango_id]
            e = self._evidence[t.id]
            self._set(e, DANGO, None, frame.now, e.smooth or 0.0)
            # 只靠位置分的帧不挪位置，不然认错的人走到哪位置跟到哪、永远满分
            if not frame.sliding and t.id in cued:
                self._spots[frame.panel] = t.box
