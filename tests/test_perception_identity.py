"""身份板接进感知层（spec 2026-10-22-identity-board-design.md §2.2、§8）：假检测器 + 合成画面。"""

from skydango.config import AppearanceConfig, EnvConfig, IdentityConfig, PerceptionConfig
from skydango.vision.appearance import AppearanceBook, ColorEmbedder
from skydango.vision.detect import Detection
from skydango.vision.identity import FrameIn, IdentityBoard
from skydango.vision.perception import PerceptionWatcher

from test_perception import FRIENDS, FakeDetector, FakeOcr, player, tag
from test_perception_appearance import GREEN, OCR, PINK, WHITE, XIAOMING, paint, run


def make(identity=False, id_kw=None, **cfg):
    """同 test_perception_appearance.make（颜色特征、every = 1），identity = True 时挂上身份板。"""
    cfg.setdefault("stranger_after", 1.0)
    cfg.setdefault("keep", 5.0)
    det = FakeDetector()
    acfg = AppearanceConfig(enabled=True, every=1, min_samples=3, outfit_change=True)
    pcfg = PerceptionConfig(**cfg)
    emb = ColorEmbedder()
    board = None
    if identity:
        board = IdentityBoard(IdentityConfig(enabled=True, **(id_kw or {})), acfg, stranger_after=1.0, grace=1.5, stranger_min_height=0.0,
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


# ── Task 7：process() 接上身份板 ──


def test_flipping_tag_between_crowded_friends_makes_no_stranger():
    w, det = make(identity=True)
    a, b = player(1000), player(1060)
    for i in range(40):  # 标签在 a、b 头上来回跳
        t = round(i * 0.1, 3)
        tg = tag(1000 if i % 2 else 1060, 110)
        run(w, det, [(a, PINK), (b, GREEN), (tg, None)], t, t)
    assert w.strangers(t) == 0


def test_dango_found_without_self_box():
    w, det = make(identity=True)
    me = player(915)
    run(w, det, [(me, WHITE), (Detection("self", me.box, 0.9), None)], 0.0, 0.3)
    t = run(w, det, [(me, WHITE), (player(1400), GREEN)], 0.4, 3.0)
    (d,) = [p for p in w.last_tracks if p.data.get("dango")]
    assert d.box.x == 915 and w.strangers(t) == 1


def test_legacy_path_unchanged_when_identity_off():
    w, det = make()
    assert w.identity is None  # make() 默认不挂身份板：test_perception_appearance 全部照旧


def test_board_errors_fall_back_to_legacy(monkeypatch):
    w, det = make(identity=True)
    monkeypatch.setattr(w.identity, "step", lambda fi: (_ for _ in ()).throw(RuntimeError("坏了")))
    run(w, det, [(player(1000), PINK)], 0.0, 1.2)
    assert w.identity is None and w.identity_off == "连续出错"


def test_identity_log_records_frames():
    w, det = make(identity=True)
    w.identity_log = []
    run(w, det, [(player(1000), PINK)], 0.0, 0.3)
    assert len(w.identity_log) == 4 and all(isinstance(f, FrameIn) for f in w.identity_log)


def test_tagged_friend_gets_name_and_learns():
    w, det = make(identity=True)
    a = player(1000)
    t = run(w, det, [(a, PINK), (tag(990, 110), None)], 0.0, 1.0)
    (p,) = [p for p in w.last_tracks if p.cls == "player"]
    assert p.data["name"] == XIAOMING and p.data["tagged"] and not p.data.get("stranger")
    assert w.appearance.friends[XIAOMING].n >= 3  # 稳定挂着名字的帧学进底库
    assert XIAOMING in w.nearby(t)


def test_unsure_call_not_shown_becomes_stranger():
    """拿不准（标签被旁边的人占着、后来看不到了）→ 身体喊一声 → 窗口里没亮在他头上 → 陌生人（CallEnd 经 FrameIn 交给身份板）。"""
    w, det = make(identity=True)
    a, b = player(1000), player(1060)
    for i in range(16):
        t = round(i * 0.1, 3)
        run(w, det, [(a, PINK), (b, GREEN), (tag(1000 if i % 2 else 1060, 110), None)], t, t)
    t = run(w, det, [(a, PINK), (b, GREEN)], 1.6, 4.0)
    unsure = w.unsure(t)
    assert len(unsure) == 2 and {u[1] for u in unsure} == {XIAOMING} and w.strangers(t) == 0
    w.called(t)
    w.mark_unsure_called([u[0] for u in unsure], t)
    mid = run(w, det, [(a, PINK), (b, GREEN)], t + 0.1, t + 5.9)
    assert w.strangers(mid) == 0  # 那一声还在途中
    end = run(w, det, [(a, PINK), (b, GREEN)], mid + 0.1, t + 6.4)  # 窗口里谁头上都没亮
    assert w.strangers(end) == 2
    assert all(not p.data.get("unsure") for p in w.last_tracks if p.cls == "player")


def test_conflict_between_look_and_tag_is_reported():
    from test_perception_appearance import FANQIE, FakeHard

    hard = FakeHard()
    w, det = make(identity=True, id_kw={"friend_feature": "color"})  # 没有 DINOv2：好友外观分用颜色
    w.hardcases = hard
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 1.0)  # 小明挂着名字，学进底库
    run(w, det, [], 1.1, 2.4)  # 看不到了（轨迹断开）
    body = player(400)
    run(w, det, [(body, PINK)], 2.5, 3.2)
    (p,) = [p for p in w.last_tracks if p.cls == "player"]
    assert p.data["maybe"] == XIAOMING and not p.data.get("stranger")
    run(w, det, [(body, PINK), (tag(390, 120), None)], 3.3, 4.0)
    (p,) = [p for p in w.last_tracks if p.cls == "player"]
    assert p.data["name"] == FANQIE and "maybe" not in p.data
    assert [r for r, _ in hard.reports] == ["identity_conflict"]
