"""身份板（vision/identity.py）：合成证据序列，不碰画面、不调模型。"""

from __future__ import annotations

from skydango.config import AppearanceConfig, IdentityConfig
from skydango.vision.bubbles import Rect
from skydango.vision.identity import (
    DANGO,
    FRIEND_LOOK,
    FRIEND_TAG,
    PENDING,
    STRANGER,
    UNSURE,
    CallEnd,
    FrameIn,
    IdentityBoard,
    Scores,
    TagCand,
    TrackIn,
    Verdict,
)


def board(**kw) -> IdentityBoard:
    cfg = kw.pop("cfg", IdentityConfig())
    acfg = kw.pop("acfg", AppearanceConfig())
    opts = dict(stranger_after=1.0, grace=1.5, stranger_min_height=0.05, track_buffer=1.0, call_window=6.0)
    opts.update(kw)
    return IdentityBoard(cfg, acfg, **opts)


def frame(now: float, *tracks: TrackIn, **kw) -> FrameIn:
    opts = dict(width=1920, height=1080, panel=False, sliding=False, moved=False)
    opts.update(kw)
    return FrameIn(now=now, tracks=tuple(tracks), **opts)


def tr(id: int, x: int, tags=(), **kw) -> TrackIn:
    box = kw.pop("box", Rect(x, 400, 90, 220))
    first = kw.pop("first", 0.0)
    return TrackIn(id=id, box=box, first=first, tags=tuple(tags), **kw)


def sc(me=None, **friends) -> Scores:
    return Scores(dino=dict(friends), color=dict(friends), me=me)


# ── Task 1：数据类型、带记性的标签分配 ──


def test_tag_goes_to_closest_and_becomes_friend_after_hold():
    b = board()
    for t in (0.0, 0.2, 0.4):
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 5.0),)), tr(2, 1100, tags=(TagCand("小明", "小明", 30.0),))))
        assert out.tags == {1: "小明"} and out.verdicts[1].kind == PENDING
    out = b.step(frame(0.5, tr(1, 1000, tags=(TagCand("小明", "小明", 5.0),)), tr(2, 1100, tags=(TagCand("小明", "小明", 30.0),))))
    assert out.verdicts[1] == Verdict(FRIEND_TAG, "小明", out.verdicts[1].score, 0.5)


def test_sticky_tag_resists_a_slightly_better_neighbour():
    b = board()
    for t in (0.0, 0.3, 0.6):  # 轨迹 1 稳定挂着小明
        b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 10.0),))))
    out = b.step(frame(0.7, tr(1, 1000, tags=(TagCand("小明", "小明", 10.0),)), tr(2, 1050, tags=(TagCand("小明", "小明", 7.0),))))
    assert out.tags == {1: "小明"}  # 10 × tag_stick 0.5 = 5 < 7


def test_name_moves_only_after_tag_hold():
    b = board()
    for t in (0.0, 0.3, 0.6):
        b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 10.0),)), tr(2, 1300)))
    for t in (0.7, 0.9, 1.1):  # 标签只挨着轨迹 2 了
        out = b.step(frame(t, tr(1, 1000), tr(2, 1300, tags=(TagCand("小明", "小明", 4.0),))))
        assert out.verdicts[1].kind == FRIEND_TAG and out.verdicts[1].name == "小明"
    out = b.step(frame(1.2, tr(1, 1000), tr(2, 1300, tags=(TagCand("小明", "小明", 4.0),))))
    assert out.verdicts[2].kind == FRIEND_TAG and out.verdicts[1].kind == UNSURE and out.verdicts[1].name == "小明"


def test_unread_tag_marks_tagged_without_name():
    b = board()
    for t in (0.0, 0.3, 0.6):
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("#77", None, 3.0),))))
    assert out.verdicts[1] == Verdict(FRIEND_TAG, None, out.verdicts[1].score, out.verdicts[1].since)


def test_identity_config_defaults():
    c = IdentityConfig()
    assert (c.enabled, c.tag_hold, c.window, c.friend_feature, c.dango_min) == (False, 0.5, 5, "dino", 0.5)


def test_absent_track_forgotten_after_buffer():
    b = board()
    for t in (0.0, 0.3, 0.6):
        b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 3.0),))))
    b.step(frame(1.7, tr(2, 1300)))  # 轨迹 1 断了 1.1 秒 > track_buffer：忘掉，名字空出来
    out = b.step(frame(1.8, tr(1, 1000)))
    assert out.verdicts[1].kind == PENDING


# ── Task 2：外观判好友 ──


def test_look_alike_after_min_samples():
    b = board()
    for i, t in enumerate((0.0, 0.1, 0.2)):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.93, 番茄=0.70))))
    assert out.verdicts[1].kind == FRIEND_LOOK and out.verdicts[1].name == "小明"


def test_one_outlier_sample_does_not_flip_look():
    b = board()
    for t, s in ((0.0, 0.93), (0.1, 0.93), (0.2, 0.93), (0.3, 0.40), (0.4, 0.93)):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=s, 番茄=0.7))))
        assert t < 0.2 or out.verdicts[1].kind == FRIEND_LOOK


def test_margin_too_small_is_unsure_not_look():
    b = board()
    for t in (0.0, 0.1, 0.2):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.90, 番茄=0.88))))
    assert out.verdicts[1].kind == UNSURE and out.verdicts[1].name == "小明"


def test_name_given_to_one_track_only():
    b = board()
    for t in (0.0, 0.1, 0.2):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.95)), tr(2, 1400, scores=sc(小明=0.91))))
    assert out.verdicts[1].kind == FRIEND_LOOK and out.verdicts[2].kind != FRIEND_LOOK


def test_stable_tag_beats_look():
    b = board()
    for t in (0.0, 0.3, 0.6):
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("番茄", "番茄", 3.0),), scores=sc(小明=0.95, 番茄=0.5))))
    assert out.verdicts[1].kind == FRIEND_TAG and out.verdicts[1].name == "番茄"


def test_flipping_tag_never_makes_a_stranger():  # 10-22 轨迹 6
    b = board()
    for i in range(30):
        t = i * 0.1
        a, c = (3.0, 5.0) if i % 2 else (5.0, 3.0)  # 5 × tag_stick 0.5 = 2.5 < 3：黏在先拿到的那条上
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("饶总", "饶总", a),)), tr(2, 1080, tags=(TagCand("饶总", "饶总", c),))))
        assert all(v.kind != STRANGER for v in out.verdicts.values()) and out.learn == {}


def test_relink_counts_as_look_unless_appearance_disagrees():
    b = board()
    out = b.step(frame(0.0, tr(1, 1000, relink="小明")))
    assert out.verdicts[1].kind == FRIEND_LOOK
    for t in (0.1, 0.2, 0.3):
        out = b.step(frame(t, tr(1, 1000, relink="小明", scores=sc(小明=0.50, 番茄=0.6))))
    assert out.verdicts[1].kind != FRIEND_LOOK


def test_look_hysteresis_drop():
    b = board()  # match 0.88，drop 0.03：已经像的到 0.86 还算像，新来的不算
    for t in (0.0, 0.1, 0.2):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.95, 番茄=0.5))))
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.86, 番茄=0.5))))
    assert out.verdicts[1].kind == FRIEND_LOOK
    for t in (0.8, 0.9, 1.0):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.84, 番茄=0.5))))
    assert out.verdicts[1].kind == UNSURE


def test_friend_feature_picks_scores():
    both = Scores(dino={"小明": 0.95}, color={"小明": 0.75}, me=None)  # 平均 0.85：拿不准
    for feat, kind in (("dino", FRIEND_LOOK), ("color", PENDING), ("both", UNSURE)):
        b = board(cfg=IdentityConfig(friend_feature=feat))
        for t in (0.0, 0.1, 0.2):
            out = b.step(frame(t, tr(1, 1000, scores=both)))
        assert out.verdicts[1].kind == kind, feat
