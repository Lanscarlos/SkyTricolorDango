"""身份板接进感知层（spec 2026-10-22-identity-board-design.md §2.2、§8）：假检测器 + 合成画面。"""

from skydango.config import AppearanceConfig, EnvConfig, IdentityConfig, PerceptionConfig
from skydango.vision.appearance import AppearanceBook, ColorEmbedder
from skydango.vision.detect import Detection
from skydango.vision.identity import FrameIn, IdentityBoard
from skydango.vision.perception import PerceptionWatcher

from test_perception import FRIENDS, FakeDetector, FakeOcr, player, tag
from test_perception_appearance import GREEN, OCR, PINK, WHITE, XIAOMING, paint, run


def make(identity=False, **cfg):
    """同 test_perception_appearance.make（颜色特征、every = 1），identity = True 时挂上身份板。"""
    cfg.setdefault("stranger_after", 1.0)
    cfg.setdefault("keep", 5.0)
    det = FakeDetector()
    acfg = AppearanceConfig(enabled=True, every=1, min_samples=3, outfit_change=True)
    pcfg = PerceptionConfig(**cfg)
    emb = ColorEmbedder()
    board = None
    if identity:
        board = IdentityBoard(IdentityConfig(enabled=True), acfg, stranger_after=1.0, grace=1.5, stranger_min_height=0.0,
                              track_buffer=pcfg.track_buffer, call_window=6.0)
    w = PerceptionWatcher(
        det, FakeOcr(OCR), pcfg, EnvConfig(), lambda: list(FRIENDS), log_roi=[0.0, 0.0, 0.335, 0.855], background=False,
        appearance=AppearanceBook(acfg, emb.key, keep=cfg["keep"]), embedder=emb, appearance_cfg=acfg, identity=board,
    )
    return w, det


def _split(w):
    tracks = w.last_tracks
    players = [t for t in tracks if t.cls in ("player", "player_unlit")]
    return players, [t for t in tracks if t.cls == "self"], [t for t in tracks if t.cls == "name_tag"]


# ── Task 6：一帧的证据拼成 FrameIn ──


def test_identity_frame_collects_evidence():
    w, det = make(identity=True)
    a, b = player(1000), player(1400)
    items = [(a, PINK), (tag(990, 110), None), (b, WHITE), (Detection("self", b.box, 0.8), None)]
    run(w, det, items, 0.0, 0.0)
    players, selfs, tags = _split(w)
    img = paint(*[(d, c) for d, c in items if c is not None])
    fresh = w._appearance_features(img, players, selfs, [], 0.0, 1920, 1080, False)
    fi = w._identity_frame(players, selfs, tags, fresh, 0.0, 1920, 1080, False)
    assert isinstance(fi, FrameIn) and fi.call_end is None
    by = {t.box.x: t for t in fi.tracks}
    assert by[1000].tags[0].name == XIAOMING and by[1000].tags[0].key == XIAOMING
    assert by[1000].self_score is None and by[1000].scores is not None
    assert by[1400].self_score == 0.8 and by[1400].tags == ()
    assert by[1000].with_bg is None  # 镜头没转


def test_identity_frame_pivot():
    w, det = make(identity=True)
    run(w, det, [(player(1000), PINK), (player(1400), GREEN)], 0.0, 0.0)
    players, selfs, tags = _split(w)
    follow, still = sorted(players, key=lambda t: t.box.x)
    follow.drift, follow.data["drift_prev"] = (200.0, 0.0), 0.0
    still.drift, still.data["drift_prev"] = (0.0, 0.0), 0.0
    w.last_shift = (200.0, 0.0)
    fi = w._identity_frame(players, selfs, tags, {}, 0.0, 1920, 1080, False)
    by = {t.id: t for t in fi.tracks}
    assert by[follow.id].with_bg is True and by[still.id].with_bg is False
    assert follow.data["drift_prev"] == 200.0
