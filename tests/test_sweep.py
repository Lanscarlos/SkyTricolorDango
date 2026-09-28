import pytest

from skydango.vision.bubbles import Rect
from skydango.vision.sweep import (
    STRANGER_WHO, UNKNOWN_WHO, UNLIT_WHO, SelfFound, Sighting, SweepEntry, SweepResult, bearing, direction, distance,
    find_self, merge,
)


def test_bearing_and_direction():
    assert bearing(0.0, 960, 1920, 2.0, 90) == 0.0
    assert bearing(0.5, 960, 1920, 2.0, 90) == 90.0
    assert bearing(0.0, 1920, 1920, 2.0, 90) == 45.0
    assert bearing(0.0, 0, 1920, 2.0, 90) == 315.0
    assert bearing(2.5, 960, 1920, 2.0, 90) == 90.0  # 第二圈取模
    assert [direction(d) for d in (0, 22, 23, 90, 180, 200, 270, 337.4, 338, 359)] == \
        ["前", "前", "右前", "右", "后", "后", "左", "左前", "前", "前"]


def test_friend_sightings_merge_by_name():
    s = [Sighting(80, "懒洋洋大王", i, 200) for i in range(3)] + [Sighting(100, "懒洋洋大王", 3, 220)]
    [e] = merge(s, 30, ref_h=200, near=0.8, far=0.4)
    assert (e.who, e.direction, e.frames, e.distance) == ("懒洋洋大王", "右", 4, "近")
    assert e.degrees == pytest.approx(85, abs=1)


def test_strangers_apart_are_not_merged_and_wraparound_is_merged():
    far_apart = merge([Sighting(10, STRANGER_WHO, 0), Sighting(60, STRANGER_WHO, 1)], 30, 200, 0.8, 0.4)
    assert len(far_apart) == 2
    [e] = merge([Sighting(355, STRANGER_WHO, 0), Sighting(5, STRANGER_WHO, 1)], 30, 200, 0.8, 0.4)
    assert e.direction == "前" and (e.degrees < 3 or e.degrees > 357) and e.frames == 2 and e.distance is None


def test_cluster_takes_majority_class():
    s = [Sighting(180, UNLIT_WHO, i) for i in range(3)] + [Sighting(185, STRANGER_WHO, 3)]
    [e] = merge(s, 30, 200, 0.8, 0.4)
    assert e.who == UNLIT_WHO and e.frames == 4


def test_unknown_tags_cluster_apart_from_strangers():
    s = [Sighting(90, UNKNOWN_WHO, 0), Sighting(92, STRANGER_WHO, 1)]
    assert sorted(e.who for e in merge(s, 30, 200, 0.8, 0.4)) == sorted([UNKNOWN_WHO, STRANGER_WHO])


def test_beside_friend_is_not_spread_over_directions():
    s = [Sighting(i * 45.0, "卡洛", i, 300, beside=True) for i in range(8)]
    [e] = merge(s, 30, 200, 0.8, 0.4)
    assert e.direction == "身边" and e.frames == 8


def test_result_text():
    r = SweepResult([SweepEntry("前", 0, "懒洋洋大王", 5, None), SweepEntry("右后", 140, STRANGER_WHO, 3, None),
                     SweepEntry("右后", 150, UNLIT_WHO, 4, None)], None, 30, 2.0)
    assert r.text() == "正前方：懒洋洋大王；右后方：2 个陌生人（1 个没点火）"
    assert SweepResult([], None, 30, 2.0).text() == "转了一圈，身边没看到别人"


def test_result_text_order_distance_and_unknown():
    r = SweepResult([SweepEntry("左", 270, UNKNOWN_WHO, 2, None), SweepEntry("身边", 0, "卡洛", 20, "近"),
                     SweepEntry("左", 265, "番茄", 3, "远")], None, 30, 2.0)
    assert r.text() == "身边：卡洛（近）；左边：番茄（远）、1 个没认出名字的人"


def sweeping(n=20):  # 中间一个不动的框 + 一个从右往左横扫的框
    return [[Rect(900, 500, 120, 240), Rect(1800 - i * 90, 450, 100, 220)] for i in range(n)]


def test_find_self_picks_the_still_box_in_the_middle():
    f = find_self(sweeping(), 1920, 0.03)
    assert f.box == Rect(900, 500, 120, 240) and len(f.per_frame) == 20 and all(j == 0 for _, j in f.static)
    assert len(f.static) == 20


def test_find_self_takes_the_median_box():
    frames = [[Rect(900 + (i % 3), 500, 120 + (i % 2) * 4, 240)] for i in range(20)]
    assert find_self(frames, 1920, 0.03).box == Rect(901, 500, 122, 240)


def test_find_self_gives_up_when_holding_hands():
    frames = [[Rect(900, 500, 120, 240), Rect(1000, 500, 110, 230)] for _ in range(20)]
    f = find_self(frames, 1920, 0.03)
    assert f.box is None and f.per_frame == {} and len(f.static) == 40


def test_find_self_needs_enough_frames():
    assert find_self(sweeping(3), 1920, 0.03) == SelfFound(None, set(), {})


def test_find_self_ignores_boxes_seen_in_few_frames():
    frames = [[Rect(900, 500, 120, 240)] if i % 2 else [] for i in range(20)]
    assert find_self(frames, 1920, 0.03).box is None


def test_find_self_ignores_moving_box():
    frames = [[Rect(900 + i * 20, 500, 120, 240)] for i in range(20)]  # 一直在挪（每帧 20 px，IoU 接得上）
    assert find_self(frames, 1920, 0.03).box is None


def test_distance_bands():
    assert [distance(h, 200, 0.8, 0.4) for h in (160, 159, 80, 79)] == ["近", "中", "中", "远"]


# ---- 评审修正 ----
def test_beside_unknown_sightings_stay_beside():
    s = [Sighting(i * 14.0, "卡洛", i, 300, beside=True) for i in range(26)]
    s += [Sighting(i * 90.0, UNKNOWN_WHO, 26 + i, 300, beside=True) for i in range(4)]
    assert [(e.who, e.direction) for e in merge(s, 30, 200, 0.8, 0.4)] == [("卡洛", "身边")]
    only_unknown = [Sighting(i * 45.0, UNKNOWN_WHO, i, 300, beside=True) for i in range(8)]
    assert [(e.who, e.direction) for e in merge(only_unknown, 30, 200, 0.8, 0.4)] == [(UNKNOWN_WHO, "身边")]


def test_generic_clusters_next_to_a_friend_are_the_friend_flickering():
    s = [Sighting(95, "懒洋洋大王", i, 200) for i in range(4)]
    s += [Sighting(100, STRANGER_WHO, 4), Sighting(90, UNKNOWN_WHO, 5), Sighting(180, STRANGER_WHO, 6)]
    got = sorted((e.who, e.direction) for e in merge(s, 30, 200, 0.8, 0.4))
    assert got == sorted([("懒洋洋大王", "右"), (STRANGER_WHO, "后")])
    unlit_next_to = merge([Sighting(95, "懒洋洋大王", 0), Sighting(100, UNLIT_WHO, 1)], 30, 200, 0.8, 0.4)
    assert len(unlit_next_to) == 2  # 黑影一定不是好友
