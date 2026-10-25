import numpy as np

from skydango.config import AppearanceConfig, EnvConfig, IdentityConfig, PerceptionConfig
from skydango.vision.appearance import AppearanceBook, ColorEmbedder
from skydango.vision.bubbles import Rect
from skydango.vision.identity import CallEnd, FrameIn, IdentityBoard
from skydango.vision.detect import Detection
from skydango.vision.perception import PerceptionWatcher
from skydango.vision.track import Track
from test_perception import FRIENDS, FakeDetector, FakeOcr, player, tag
from test_perception_appearance import PINK, WHITE, GREEN, OCR, XIAOMING, FANQIE, FakeHard, FakeDino, paint, run, players


def test_flipping_tag_between_crowded_friends_makes_no_stranger():
    w, det = make(identity=True)
    a, b = player(1000), player(1060)
    for i in range(40):
        t = round(i * 0.1, 3)
        tg = tag(1000 if i % 2 else 1060, 110)
        run(w, det, [(a, PINK), (b, GREEN), (tg, None)], t, t)
    assert w.strangers(t) == 0


def test_dango_found_without_self_box():
    w, det = make(identity=True)
    me = player(915)
    run(w, det, [(me, WHITE), (Detection('self', me.box, 0.9), None)], 0.0, 0.3)
    t = run(w, det, [(me, WHITE), (player(1400), GREEN)], 0.4, 3.0)
    (d,) = [p for p in w.last_tracks if p.data.get('dango')]
    assert d.box.x == 915 and w.strangers(t) == 1
    assert w.identity.dango_id == d.id


def test_self_only_detection_is_evidence_for_board():
    w, det = make(identity=True)
    me = Detection('self', player(915).box, 0.9)
    run(w, det, [(me, WHITE)], 0, 0.3)
    assert w.identity.dango_id is not None
    assert w.appearance.me is not None


def test_legacy_path_unchanged_when_identity_off():
    w, _ = make()
    assert w.identity is None


def test_board_errors_fall_back_to_legacy(monkeypatch):
    w, det = make(identity=True)
    monkeypatch.setattr(w.identity, 'step', lambda fi: (_ for _ in ()).throw(RuntimeError('坏了')))
    run(w, det, [(player(1000), PINK)], 0, 1.2)
    assert w.identity is None and w.identity_off == '连续出错'


def test_identity_log_records_frames():
    w, det = make(identity=True)
    w.identity_log = []
    run(w, det, [(player(1000), PINK)], 0, 0.3)
    assert len(w.identity_log) == 4 and all(isinstance(f, FrameIn) for f in w.identity_log)


def test_pause_does_not_count_towards_unsure_timeout():
    from skydango.vision.identity import Scores, TrackIn
    w, det = make(identity=True)
    run(w, det, [(player(1000), PINK)], 0, 0.2)
    p = players(w)[0]
    for t in (0.3, 0.4, 0.5):
        w.identity.step(FrameIn(t, 1920, 1080, False, False, False,
            (TrackIn(p.id, p.box, p.first, scores=Scores({XIAOMING: 0.85}, {}, None)),)))
    now = [0.5]
    w.clock = lambda: now[0]
    w.hold('panel')
    now[0] = 20.5
    w.release('panel')
    run(w, det, [(player(1000), PINK)], 20.6, 20.6)
    assert players(w)[0].data.get('unsure', (None,))[0] == XIAOMING


def test_saver_excludes_unconfirmed_tag_frames():
    from test_perception_appearance import FakeSaver
    w, det = make(identity=True)
    w.saver = FakeSaver()
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0, 0.4)
    assert w.saver.calls == []
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.5, 0.7)
    assert w.saver.calls and all(c[1] == XIAOMING for c in w.saver.calls)


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


# ---- 整分支评审（10-25）补的 ----

def test_board_error_keeps_call_result_for_next_frame(monkeypatch):
    from skydango.vision.people import CallSeen
    w, det = make(identity=True)
    w._call = CallSeen(0, {}, ended=True)
    w._call.tracks = {}
    real, seen = w.identity.step, []

    def flaky(fi):
        seen.append(fi.call_end)
        if len(seen) == 1:
            raise RuntimeError('坏了一帧')
        return real(fi)
    monkeypatch.setattr(w.identity, 'step', flaky)
    run(w, det, [(player(1000), PINK)], 6.1, 6.2)
    assert seen[0] is not None and seen[1] == seen[0]


def test_board_error_in_apply_counts_and_restores(monkeypatch):
    w, det = make(identity=True)
    monkeypatch.setattr(w, '_learn', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('坏了')))
    t = 0.0
    while w.identity is not None and t < 3.0:  # 退回旧流程后旧流程也会调 _learn，到这儿就停
        run(w, det, [(player(1000), PINK), (tag(990, 110), None)], t, t)
        t = round(t + 0.1, 3)
    assert w.identity is None and w.identity_off == '连续出错'


def test_rejected_self_box_is_not_dango_downstream():
    from skydango.vision.perception import dango_self
    w, det = make(identity=True)
    me, friend = player(915), player(1400)
    items = [(me, WHITE), (Detection('self', me.box, 0.9), None), (Detection('self', friend.box, 0.5), None)]
    run(w, det, items, 0, 0.3)
    selfs = [t for t in w.last_tracks if t.cls == 'self']
    assert len(selfs) == 2
    good = [t for t in selfs if dango_self(t)]
    assert [t.box.x for t in good] == [915]
    assert w._self_now(0.3, False).x == 915
    assert w._ref_height(1080) == me.box.h
    assert [o['x'] for o in w.overlay(0.3) if o['kind'] == 'self'] == [915]

