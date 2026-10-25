from skydango.config import AppearanceConfig, IdentityConfig
from skydango.vision.bubbles import Rect
from skydango.vision.identity import (
    IdentityBoard, TagCand, Scores, TrackIn, CallEnd, FrameIn, Verdict,
    FRIEND_TAG, FRIEND_LOOK, DANGO, UNSURE, STRANGER, PENDING,
)


def board(**kw):
    cfg = kw.pop("cfg", IdentityConfig())
    acfg = kw.pop("acfg", AppearanceConfig())
    return IdentityBoard(cfg, acfg, **(dict(stranger_after=1.0, grace=1.5,
        stranger_min_height=0.05, track_buffer=1.0, call_window=6.0) | kw))


def frame(now, *tracks, **kw):
    return FrameIn(now=now, tracks=tracks, **(dict(width=1920, height=1080,
        panel=False, sliding=False, moved=False) | kw))


def tr(id, x, tags=(), **kw):
    return TrackIn(id=id, tags=tags, **(dict(box=Rect(x, 400, 90, 220), first=0.0) | kw))


def sc(me=None, **friends):
    return Scores(dino=friends, color=friends, me=me)


def test_friend_feature_picks_scores():
    for feature, want in [('dino', '小明'), ('color', '番茄'), ('both', '小明')]:
        b = board(cfg=IdentityConfig(friend_feature=feature))
        scores = Scores({'小明': 1.0, '番茄': 0.5}, {'小明': 0.9, '番茄': 1.0}, None)
        for t in (0, 0.1, 0.2):
            out = b.step(frame(t, tr(1, 1000, scores=scores)))
        assert out.verdicts[1].name == want and out.verdicts[1].kind == FRIEND_LOOK


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


def test_clearly_nobody_becomes_stranger():
    b = board()
    for t in (0.0, 0.5, 1.0):
        out = b.step(frame(t, tr(1, 1000, scores=sc(me=0.4, 小明=0.5, 番茄=0.6))))
    assert out.verdicts[1].kind == STRANGER

def test_empty_galleries_wait_like_before():
    b = board(stranger_after=1.0, grace=1.5)
    out = b.step(frame(2.0, tr(1, 1000, first=0.0, scores=sc())))
    assert out.verdicts[1].kind == PENDING
    out = b.step(frame(2.6, tr(1, 1000, first=0.0, scores=sc())))
    assert out.verdicts[1].kind == STRANGER

def test_once_tagged_never_stranger_by_looks_only():
    b = board()
    for t in (0.0, 0.3, 0.6):
        b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 5.0),)), tr(2, 1300)))
    for t in (0.7, 0.9, 1.1, 1.3):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.3)), tr(2, 1300, tags=(TagCand("小明", "小明", 4.0),))))
    out = b.step(frame(1.4, tr(1, 1000, scores=sc(小明=0.3)), tr(2, 1300, tags=(TagCand("小明", "小明", 4.0),))))
    assert out.verdicts[1].kind == UNSURE

def test_unsure_call_shown_confirms_and_not_shown_rejects():
    b = board()
    for t in (0.0, 0.1, 0.2):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.86, 番茄=0.5)), tr(2, 1400, scores=sc(番茄=0.86, 小明=0.5))))
    out = b.step(frame(3.0, tr(1, 1000, called_at=2.5), tr(2, 1400, called_at=2.5), call_end=CallEnd(2.5, {1: "小明"})))
    assert out.verdicts[1].kind == FRIEND_TAG and out.verdicts[2].kind == STRANGER

def test_unsure_without_call_times_out():
    b = board(call_window=6.0)  # unsure_wait 2.0
    for t in (0.0, 0.1, 0.2):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.86, 番茄=0.5))))
    assert b.step(frame(8.0, tr(1, 1000))).verdicts[1].kind == UNSURE
    assert b.step(frame(8.3, tr(1, 1000))).verdicts[1].kind == STRANGER

def test_tag_held_by_neighbour_means_unsure():  # 10-22 轨迹 6：标签一直被旁边的人占着
    b = board()
    for i in range(30):
        t = i * 0.1
        a, c = (3.0, 5.0) if i % 2 else (5.0, 3.0)
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("饶总", "饶总", a),)), tr(2, 1080, tags=(TagCand("饶总", "饶总", c),))))
    assert out.verdicts[1].kind == UNSURE and out.verdicts[1].name == "饶总"

def test_unlit_and_panel_and_small():
    b = board()
    out = b.step(frame(1.2, tr(1, 1000, unlit=True), tr(2, 300, under_panel=True), tr(3, 1500, box=Rect(1500, 400, 20, 40))))
    assert out.verdicts[1].kind == STRANGER and out.verdicts[2].kind == PENDING and out.verdicts[3].kind == PENDING


def test_dango_without_self_box_by_position_and_look():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=0.9)))  # 记下位置
    for t in (0.1, 0.2, 0.3):
        out = b.step(frame(t, tr(1, 960, scores=sc(me=0.9, 小明=0.5)), tr(2, 1300, scores=sc(me=0.5, 小明=0.5))))
    assert out.verdicts[1].kind == DANGO and b.dango_id == 1

def test_no_dango_evidence_no_dango():
    b = board()
    out = b.step(frame(0.0, tr(1, 960), tr(2, 1300)))
    assert all(v.kind != DANGO for v in out.verdicts.values()) and b.dango_id is None

def test_self_box_jumping_onto_tagged_friend_does_not_steal():
    b = board()
    for t in (0.0, 0.1):
        b.step(frame(t, tr(1, 960, self_score=0.9), tr(2, 1040, tags=(TagCand("小明", "小明", 3.0),))))
    for t in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        out = b.step(frame(t, tr(1, 960), tr(2, 1040, self_score=0.9, tags=(TagCand("小明", "小明", 3.0),))))
    assert b.dango_id == 1 and out.verdicts[2].kind == FRIEND_TAG

def test_pivot_separates_dango_from_hugging_friend():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=0.9)))
    for t in (0.1, 0.2, 0.3, 0.4):
        out = b.step(frame(t, tr(1, 960, with_bg=False), tr(2, 990, with_bg=True)))
    assert b.dango_id == 1

def test_panel_slide_keeps_dango():
    b = board()
    for t in (0.0, 0.1):
        b.step(frame(t, tr(1, 960, self_score=0.9), tr(2, 1400)))
    for t in (0.2, 0.6, 1.0, 1.4, 1.8):  # 面板横移：团子滑到 1360，另一个人滑到原来团子的位置
        out = b.step(frame(t, tr(1, 1360), tr(2, 960), sliding=True))
    assert b.dango_id == 1 and out.verdicts[2].kind != DANGO

def test_challenger_needs_dango_switch_seconds():
    b = board(cfg=IdentityConfig(dango_w_pos=0))  # 单独测换人延迟，不让位置分压过 self 分。
    b.step(frame(0.0, tr(1, 960, self_score=0.6), tr(2, 1300)))
    for t in (0.1, 0.5, 0.9):
        b.step(frame(t, tr(1, 960, self_score=0.6), tr(2, 1300, self_score=1.0)))
    assert b.dango_id == 1
    for t in (1.3, 1.7, 2.1):
        b.step(frame(t, tr(1, 960, self_score=0.6), tr(2, 1300, self_score=1.0)))
    assert b.dango_id == 2


def test_learn_only_stable_tag_frames_with_samples():
    b = board()
    s = sc(小明=0.9)
    outs = [b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 3.0),), scores=s))) for t in (0.0, 0.3, 0.6)]
    assert outs[0].learn == {} and outs[1].learn == {} and outs[2].learn == {1: ("friend", "小明")}
    out = b.step(frame(0.7, tr(1, 1000, scores=s)))  # 标签被挡：不学
    assert out.learn == {}

def test_learn_dango_and_stranger():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=1.0)))
    out = b.step(frame(0.1, tr(1, 960, self_score=1.0, scores=sc(me=0.9))))
    assert out.learn == {1: ("me", "")}
    for t in (0.0, 0.5, 1.0):
        out = b.step(frame(t, tr(5, 1500, sid="陌生人A", scores=sc(me=0.2, 小明=0.3))))
    assert out.learn[5] == ("stranger", "陌生人A")

def test_look_alike_and_unsure_never_learn():
    b = board()
    for t in (0.0, 0.1, 0.2, 0.3):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.95)), tr(2, 1300, scores=sc(番茄=0.85, 小明=0.5))))
    assert out.learn == {}




# ---- 整分支评审（10-25）补的 ----

def test_marked_unsure_without_call_result_still_times_out():  # 标了"喊过"其实没喊成（dry-run、被拦）
    b = board(call_window=6.0)
    for t in (0.0, 0.1, 0.2):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.86, 番茄=0.5))))
    assert b.step(frame(9.0, tr(1, 1000, called_at=2.5))).verdicts[1].kind == UNSURE  # 窗口还可能没结束
    assert b.step(frame(11.0, tr(1, 1000, called_at=2.5))).verdicts[1].kind == STRANGER

def test_alternating_tag_never_stabilises_or_learns():  # 误差 1 / 100 来回换，黏性压不住
    b = board()
    for i in range(30):
        a, c = (1.0, 100.0) if i % 2 else (100.0, 1.0)
        out = b.step(frame(i * 0.1, tr(1, 1000, tags=(TagCand("小明", "小明", a),), scores=sc(小明=0.4)),
                           tr(2, 1080, tags=(TagCand("小明", "小明", c),), scores=sc(小明=0.4))))
        assert out.learn == {} and all(v.kind != STRANGER for v in out.verdicts.values())
        assert all(v.kind != FRIEND_TAG for v in out.verdicts.values())

def test_shared_call_resolves_unsure_not_called_for():  # 在场确认那一声也算数
    b = board()
    for t in (0.0, 0.1, 0.2):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.86, 番茄=0.5))))
    out = b.step(frame(3.0, tr(1, 1000), call_end=CallEnd(2.5, {})))
    assert out.verdicts[1].kind == STRANGER

def test_unlit_judged_behind_panel_and_far():
    b = board()
    out = b.step(frame(1.2, tr(1, 300, unlit=True, under_panel=True), tr(2, 1500, unlit=True, box=Rect(1500, 400, 20, 40))))
    assert out.verdicts[1].kind == STRANGER and out.verdicts[2].kind == STRANGER

def test_dango_missing_one_frame_keeps_dango():
    b = board(cfg=IdentityConfig(dango_w_pos=0))
    for t in (0.0, 0.1, 0.2):
        b.step(frame(t, tr(1, 960, self_score=0.9), tr(2, 1300, self_score=0.8)))
    b.step(frame(0.3, tr(2, 1300, self_score=0.8)))  # 团子漏检一帧
    out = b.step(frame(0.4, tr(1, 960), tr(2, 1300, self_score=0.8)))
    assert b.dango_id == 1 and out.verdicts[1].kind == DANGO and out.verdicts[2].kind != DANGO

def test_look_needs_absolute_score_without_friend_galleries():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=0.9)))
    for t in (1.5, 1.6, 1.7, 1.8):  # 团子轨迹没了；好友底库还空着，一个陌生人有点像团子
        out = b.step(frame(t, tr(2, 1000, scores=sc(me=0.6))))
    assert b.dango_id is None and out.verdicts[2].kind != DANGO

def test_tagged_friend_with_first_self_hint_is_not_dango():
    b = board()
    for i in range(10):
        out = b.step(frame(i * 0.15, tr(1, 1000, tags=(TagCand("小明", "小明", 3.0),),
                                         self_score=0.7 if i < 2 else None, scores=sc(小明=0.9))))
    assert b.dango_id is None and out.verdicts[1].kind == FRIEND_TAG

def test_wrong_dango_drifting_away_from_spot_is_dropped():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=0.9)))
    for i in range(1, 30):  # 被认成团子的人只剩位置分，慢慢走开
        b.step(frame(i * 0.1, tr(1, 960 + 20 * i)))
    assert b.dango_id is None

def test_dango_does_not_claim_friend_look_name():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=0.9)))
    for t in (0.1, 0.2, 0.3, 0.4):
        out = b.step(frame(t, tr(1, 960, scores=sc(me=0.97, 小明=0.95)), tr(2, 1300, scores=sc(小明=0.9))))
    assert out.verdicts[1].kind == DANGO and out.verdicts[2].kind == FRIEND_LOOK and out.verdicts[2].name == "小明"
