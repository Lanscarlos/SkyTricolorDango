"""身份板：只吃证据，不碰画面、模型和墙钟，可按帧重放。"""

from dataclasses import dataclass, field
from statistics import median

from skydango.config import AppearanceConfig, IdentityConfig
from .bubbles import Rect

FRIEND_TAG = "friend_tag"
FRIEND_LOOK = "friend_look"
DANGO = "dango"
UNSURE = "unsure"
STRANGER = "stranger"
PENDING = "pending"


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
                        t.id, c.key, c) for t in frame.tracks for c in t.tags)
        assigned: dict[int, TagCand] = {}
        used = set()
        for _, id, key, cand in pairs:
            if id not in assigned and key not in used:
                assigned[id] = cand
                used.add(key)
        self._previous = {c.key: id for id, c in assigned.items()}
        for id, cand in assigned.items():
            e = self._evidence[id]
            if e.key != cand.key or now - e.tag_last > self.cfg.tag_hold + 1e-9:
                e.key, e.tag_start = cand.key, now
            e.tag_last = now
            if now - e.tag_start + 1e-9 >= self.cfg.tag_hold:
                for other_id, other in self._evidence.items():
                    if other_id != id and other.stable and other.stable.key == cand.key:
                        name = other.stable.name
                        other.stable = None
                        self._set(other, UNSURE if name else PENDING, name, now)
                e.stable = cand
        for t in frame.tracks:
            e = self._evidence[t.id]
            if e.stable:
                self._set(e, FRIEND_TAG, e.stable.name, now)
        self._friends(frame)
        return StepOut({id: c.key for id, c in assigned.items()},
                       {t.id: self._evidence[t.id].verdict for t in frame.tracks}, {})

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
            if e.stable:
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
