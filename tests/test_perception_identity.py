import numpy as np

from skydango.config import AppearanceConfig, EnvConfig, IdentityConfig, PerceptionConfig
from skydango.vision.appearance import AppearanceBook, ColorEmbedder
from skydango.vision.bubbles import Rect
from skydango.vision.identity import CallEnd, FrameIn, IdentityBoard
from skydango.vision.perception import PerceptionWatcher
from skydango.vision.track import Track
from test_perception import FRIENDS, FakeDetector, FakeOcr, player, tag
from test_perception_appearance import PINK, WHITE, OCR, XIAOMING, FANQIE, FakeHard, FakeDino, paint, run, players


def make(identity=False, **kw):
    acfg = AppearanceConfig(enabled=True, every=1, min_samples=3)
    cfg = PerceptionConfig(stranger_after=1.0, keep=5.0, stranger_min_height=0.0, **kw)
    emb = ColorEmbedder()
    det = FakeDetector()
    board = IdentityBoard(IdentityConfig(enabled=True), acfg, stranger_after=1.0, grace=1.5,
        stranger_min_height=0.0, track_buffer=cfg.track_buffer, call_window=6.0) if identity else None
    w = PerceptionWatcher(det, FakeOcr(OCR), cfg, EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0, 0, 0.335, 0.855], background=False,
        appearance=AppearanceBook(acfg, emb.key, keep=5), embedder=emb, appearance_cfg=acfg,
        identity=board)
    return w, det


def track(id, cls, box, score=0.9, **kw):
    return Track(id, cls, box, score, 0.0, 0.0, **kw)


def test_identity_frame_collects_evidence():
    w, _ = make(identity=True)
    p = track(1, 'player', player(1000).box)
    p2 = track(2, 'player', player(1400).box)
    s = track(3, 'self', p2.box, 0.87)
    label = track(4, 'name_tag', tag(990, 110).box, data={'name': XIAOMING})
    img = paint((player(1000), PINK), (player(1400), WHITE))
    fresh = w._appearance_features(img, [p, p2], [s], [], 0.0, 1920, 1080, False)
    f = w._identity_frame([p, p2], [s], [label], fresh, 0.0, 1920, 1080, False)
    assert f.tracks[0].tags[0].name == XIAOMING
    assert f.tracks[0].scores is not None
    assert f.tracks[1].self_score == 0.87


def test_identity_frame_pivot():
    w, _ = make(identity=True)
    w.last_shift = (200.0, 0.0)
    p = track(1, 'player', Rect(1000, 400, 90, 220), drift=(200, 0))
    q = track(2, 'player', Rect(1400, 400, 90, 220), drift=(0, 0))
    f = w._identity_frame([p, q], [], [], {}, 0, 1920, 1080, False)
    assert [t.with_bg for t in f.tracks] == [True, False]


def test_identity_frame_consumes_call_end_once():
    from skydango.vision.people import CallSeen
    w, _ = make(identity=True)
    w._call = CallSeen(0, {}, ended=True)
    w._call.tracks = {1: XIAOMING}
    f = w._identity_frame([], [], [], {}, 6.1, 1920, 1080, False)
    assert f.call_end == CallEnd(0, {1: XIAOMING})
    assert w._identity_frame([], [], [], {}, 6.2, 1920, 1080, False).call_end is None
