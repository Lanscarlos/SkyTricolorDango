"""同一个人同时出 player + player_unlit 两个框（docs/progress/2026-10-03-plan.md ① 第 2 项）。

检测器按类别各自做 NMS，两个几乎重合的框都留下：黑影框新开一条轨迹、当场判陌生人（gesture-bow-1 20 s：好友鞠躬，player 0.51 / player_unlit 0.38）。
合成一个框（留分高的），但真黑影也常是 player 分数更高（candle 录像 p0.61 / u0.32），所以被压掉的黑影框记在留下的轨迹上（data["unlit_at"]），
点亮陌生人那边能用；黑影也和点过火的陌生人一样等 stranger_after，挂过名字标签的轨迹不判陌生人。
"""

from test_perception import FakeDetector, FakeOcr, frame, player, tag, watcher

from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.perception import UNLIT, merge_people

BODY = Rect(1000, 400, 90, 220)


def unlit(box=BODY, score=0.9):
    return Detection(UNLIT, box, score)


def lit(box=BODY, score=0.9):
    return Detection("player", box, score)


def run(w, det, frames, start=0.0, step=0.25):
    t = start
    for dets in frames:
        det.frames = [dets]
        w.process(frame(), t, False)
        t += step
    return t - step


def test_merge_keeps_higher_score_and_flags_suppressed_unlit():
    near = Rect(1002, 402, 90, 218)
    kept, flagged = merge_people([lit(score=0.51), unlit(near, 0.38), tag(990, 110)])
    assert [d.cls for d in kept] == ["player", "name_tag"]
    assert flagged == {0}  # 留下的 player 压掉过一个黑影框
    kept, flagged = merge_people([lit(score=0.32), unlit(near, 0.72)])
    assert [d.cls for d in kept] == [UNLIT] and flagged == set()


def test_merge_leaves_two_people_standing_close():
    beside = Rect(1050, 400, 90, 220)  # IoU ≈ 0.29：挨着的两个人
    kept, _ = merge_people([lit(), unlit(beside)])
    assert len(kept) == 2


def test_friend_bowing_with_duplicate_unlit_box_is_not_a_stranger():
    det = FakeDetector()
    w = watcher(det, ocr=FakeOcr({110: "懒洋洋大王"}))
    t = run(w, det, [[lit(), tag(990, 110)]] * 6)
    t = run(w, det, [[lit(score=0.51), unlit(Rect(1002, 402, 90, 218), 0.38)]] * 2, start=t + 0.25)
    assert w.strangers(t) == 0
    assert w.nearby(t) == ["懒洋洋大王"]


def test_unlit_waits_stranger_after_like_everyone_else():
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[unlit()]] * 3)  # 0.5 秒
    assert w.strangers(t) == 0 and w.unlit(t) == 0
    t = run(w, det, [[unlit()]] * 3, start=t + 0.25)  # 1.25 秒
    assert w.strangers(t) == 1 and w.unlit(t) == 1


def test_tagged_track_flipping_to_unlit_is_not_a_stranger():
    """远处的好友偶尔被认成黑影（walkaway-1002-1）：挂过名字标签的轨迹不判陌生人。"""
    det = FakeDetector()
    w = watcher(det, ocr=FakeOcr({110: "懒洋洋大王"}))
    t = run(w, det, [[lit(), tag(990, 110)]] * 6)
    t = run(w, det, [[unlit()]] * 8, start=t + 0.25)
    assert w.strangers(t) == 0


def test_tagged_track_keeps_its_name_while_its_tag_is_visible_and_it_flips_unlit():
    """gesture-bow-1 23 s：好友鞠躬时这一帧只认成黑影，头顶名字标签照样清清楚楚。黑影挂不上标签，
    不能当成"他的标签在别处"把名字摘掉，然后判成陌生人。"""
    det = FakeDetector()
    w = watcher(det, ocr=FakeOcr({110: "懒洋洋大王"}))
    t = run(w, det, [[lit(), tag(990, 110)]] * 6)
    t = run(w, det, [[unlit(score=0.6), tag(990, 110)]] * 2, start=t + 0.25)
    assert w.strangers(t) == 0
    (track,) = [x for x in w.last_tracks if x.cls in ("player", UNLIT)]
    assert track.data.get("name") == "懒洋洋大王"


def test_suppressed_unlit_is_remembered_on_the_track():
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[lit(score=0.61), unlit(Rect(1002, 402, 90, 218), 0.32)]])
    (track,) = [x for x in w.last_tracks if x.cls in ("player", UNLIT)]
    assert track.cls == "player" and track.data["unlit_at"] == t
