"""感知层的追踪升级（spec 2026-10-01-tracking-relink-motion）：低分框续轨迹、画面平移、镜头事件、续命、接回、运动方向。"""

import numpy as np
import pytest

from skydango.config import EnvConfig, PerceptionConfig
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.perception import PerceptionWatcher, detector_conf

from test_perception import FRIENDS, FakeDetector, FakeOcr, frame

NAME = FRIENDS[0]  # 懒洋洋大王：标签宽 70
OTHER = FRIENDS[1]  # 番茄炒蛋盖饭：标签宽 80
OFF = dict(sticky_names=False, track_low=False, track_predict=False, track_pan=False, relink=False, motion=False)


def player(x, y=400, w=90, h=220, s=0.9, cls="player"):
    return Detection(cls, Rect(x, y, w, h), s)


def tag(px, w=70, y=330, h=44):
    """头顶在 player(px) 正上方的名字标签。"""
    return Detection("name_tag", Rect(px + 45 - w // 2, y, w, h), 0.9)


def make(clock=None, **cfg):
    cfg.setdefault("stranger_after", 1.0)
    cfg.setdefault("far_crops", 0)
    det = FakeDetector()
    w = PerceptionWatcher(
        det, FakeOcr({70: NAME, 80: OTHER}), PerceptionConfig(**cfg), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, **({"clock": clock} if clock else {}),
    )
    return w, det


def run(w, det, dets, start, end, step=0.1, img=None):
    """[start, end) 每 step 秒一帧，检测器一直返回 dets（可以是 t → dets 的函数）。返回最后的时间。"""
    t = start
    while t < end - 1e-9:
        det.frames = [dets(t) if callable(dets) else dets]
        w.process(img(t) if callable(img) else (img if img is not None else frame()), round(t, 3), False)
        t = round(t + step, 3)
    return t


def body(w):
    """画面里唯一一条人物轨迹。"""
    (p,) = [t for t in w.last_tracks if t.cls in ("player", "player_unlit")]
    return p


def fade_scene(w, det):
    """好友带标签站着 1 秒 → 标签消失、分数掉到 0.3 两秒 → 分数回到 0.9（还是没标签）两秒。"""
    run(w, det, [player(800), tag(800)], 0.0, 1.0)
    first = body(w).id
    run(w, det, [player(800, s=0.3)], 1.0, 3.0)
    run(w, det, [player(800)], 3.0, 5.0)
    return first


# ---- Task 3：接线 ----
def test_all_switches_off_is_unchanged():
    w, det = make(**OFF)
    first = fade_scene(w, det)
    p = body(w)
    assert p.id != first  # 低分框不进追踪：轨迹断了
    assert p.data.get("stranger") is True  # 新轨迹没挂过标签：判陌生人
    assert w.last_shift is None


def test_low_conf_box_continues_track_in_watcher():
    w, det = make()
    first = fade_scene(w, det)
    p = body(w)
    assert p.id == first and p.weak_hits > 0
    assert not p.data.get("stranger") and p.data.get("name") == NAME


def test_detector_conf_low_when_track_low():
    assert detector_conf(PerceptionConfig(hardcases=False, track_low=True)) == 0.25
    assert detector_conf(PerceptionConfig(hardcases=False, track_low=False)) == 0.35


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)


def test_pan_feeds_tracker():
    base = _texture()
    for on in (True, False):
        w, det = make(track_pan=on, track_predict=False)
        run(w, det, [player(300)], 0.0, 0.3, img=base)
        first = body(w).id
        run(w, det, [player(460)], 0.3, 0.4, img=np.roll(base, 160, axis=1))
        assert (body(w).id == first) is on
        if on:
            assert w.last_shift[0] == pytest.approx(160, abs=8)


def test_far_tags_second_update_keeps_dropped():
    w, det = make(far_crops=3)
    det.frames = [[player(300)]]
    w.process(frame(), 0.0, False)
    gone = body(w)
    small = player(1000, w=25, h=60)
    det.frames = [[small], [Detection("name_tag", Rect(20, 60, 30, 12), 0.9)]]  # 原图一次、远处裁剪一次
    w.process(frame(), 1.2, False)
    assert w.far_runs == 1 and any(t.cls == "name_tag" for t in w.last_tracks)
    assert gone in w.tracker.dropped


def walk(t):
    return [player(round(800 - 100 * t))]


def test_camera_moved_zoom_clears_hist_and_calms():
    w, det = make()
    t = run(w, det, walk, 0.0, 1.0)
    assert body(w).data["hist"] and body(w).vx != 0
    w.camera_moved(t, "zoom")
    t = run(w, det, walk, t, t + 0.4)
    p = body(w)
    assert not p.data.get("hist") and not p.data.get("motion_hist") and p.vx == 0
    t = run(w, det, walk, t, t + 0.5)  # settle（0.6）过了：照常
    assert body(w).data["hist"] and body(w).vx != 0
    w.camera_moved(t, "turn")
    run(w, det, walk, t, t + 0.2)
    assert len(body(w).data["hist"]) > 2  # 转镜头不清


def test_resume_zeroes_velocity():
    now = [0.0]
    w, det = make(clock=lambda: now[0])
    base = _texture(3)

    def go(a, b):
        t = a
        while t < b - 1e-9:
            now[0] = t
            det.frames = [walk(t)]
            w.process(base, t, False)
            t = round(t + 0.1, 3)

    go(0.0, 1.0)
    assert body(w).vx != 0 and w.last_shift is not None
    w.hold("test")
    now[0] = 3.0
    w.release("test")
    assert all(t.vx == t.vy == t.vh == 0 for t in w.tracker.tracks.values())
    go(3.0, 3.1)
    assert w.last_shift is None  # 暂停前的缩略图作废


# ---- Task 4：续命 ----
def test_sticky_keeps_friend_nearby_while_track_alive():
    w, det = make()
    run(w, det, [player(800), tag(800)], 0.0, 1.0)
    t = run(w, det, [player(800)], 1.0, 11.0)
    assert NAME in w.nearby(t)


def test_sticky_off_drops_after_keep():
    w, det = make(sticky_names=False)
    run(w, det, [player(800), tag(800)], 0.0, 1.0)
    t = run(w, det, [player(800)], 1.0, 7.0)
    assert NAME not in w.nearby(t)


def test_sticky_ends_after_track_dropped():
    w, det = make(relink=False)
    run(w, det, [player(800), tag(800)], 0.0, 1.0)
    run(w, det, [player(800)], 1.0, 3.0)  # 最后看到 2.9
    run(w, det, [], 3.0, 7.5)
    assert NAME in w.nearby(7.5)  # 轨迹 4.0 删掉，keep 从 2.9 算
    run(w, det, [], 7.5, 8.5)
    assert NAME not in w.nearby(8.5)


def test_sticky_ignores_maybe():
    w, det = make(relink=False)
    run(w, det, [player(800)], 0.0, 0.5)
    body(w).data["maybe"] = NAME  # 只有 maybe、没挂过标签：不靠续命
    t = run(w, det, [player(800)], 0.5, 1.0)
    assert NAME not in w.nearby(t)


# ---- Task 5：失踪好友接回 ----
def lose(w, det, x=800, name_w=70):
    """好友带标签站 1 秒（最后看到 0.9）→ 1.3 秒什么都看不到（2.0 时轨迹被删）。"""
    run(w, det, [player(x), tag(x, name_w)], 0.0, 1.0)
    run(w, det, [], 1.0, 2.2)


def test_relink_within_gate():
    w, det = make()
    lose(w, det)
    t = run(w, det, [player(830)], 2.2, 4.0)
    p = body(w)
    assert p.data["maybe"] == NAME and p.data["maybe_by"] == "relink"
    assert not p.data.get("stranger") and not p.data.get("name")
    assert [(q.name, q.sure) for q in w.people(t)] == [(NAME, False)]
    t = run(w, det, [player(830)], 4.0, 8.0)
    assert NAME in w.nearby(t)  # 一直没冒"走开了"
    assert w.strangers(t) == 0


@pytest.mark.parametrize("dets,start", [
    ([player(1600)], 2.2),  # 离得太远
    ([player(830)], 6.2),  # 超过 keep（最后看到 0.9）才冒出来
    ([player(830, cls="player_unlit")], 2.2),  # 黑影不接
])
def test_relink_rejects_far_or_late_or_unlit(dets, start):
    w, det = make()
    lose(w, det)
    if start > 2.2:
        run(w, det, [], 2.2, start)
    run(w, det, dets, start, start + 1.5)
    assert not body(w).data.get("maybe")


def test_relink_cancelled_when_tag_shown_elsewhere():
    w, det = make()
    lose(w, det)
    run(w, det, [player(830), player(300), tag(300)], 2.2, 3.0)
    by_x = {t.box.x: t for t in w.last_tracks if t.cls == "player"}
    assert by_x[300].data.get("name") == NAME
    assert not by_x[830].data.get("maybe")
    assert NAME not in w._lost


def test_relink_ambiguous_two_lost_one_candidate():
    w, det = make()
    run(w, det, [player(800), tag(800), player(1100), tag(1100, 80)], 0.0, 1.0)
    run(w, det, [], 1.0, 2.2)
    run(w, det, [player(950)], 2.2, 3.5)  # 两个人中间：对得上两条记录
    assert not body(w).data.get("maybe")


def test_relink_ambiguous_two_candidates():
    w, det = make()
    lose(w, det)
    run(w, det, [player(830), player(770)], 2.2, 3.5)  # 一样近：不知道是哪个
    assert not any(t.data.get("maybe") for t in w.last_tracks)


@pytest.mark.parametrize("name_w,expect", [(70, NAME), (80, OTHER)])
def test_relinked_track_confirmed_or_renamed_by_tag(name_w, expect):
    w, det = make()
    lose(w, det)
    run(w, det, [player(830)], 2.2, 3.0)
    assert body(w).data["maybe"] == NAME
    run(w, det, [player(830), tag(830, name_w)], 3.0, 3.5)
    d = body(w).data
    assert d["name"] == expect and "maybe" not in d and "maybe_by" not in d


def test_relinked_track_can_relink_again():
    w, det = make()
    lose(w, det)
    run(w, det, [player(830)], 2.2, 3.0)  # 最后看到 2.9
    run(w, det, [], 3.0, 4.2)  # 4.0 删掉
    run(w, det, [player(860)], 4.2, 5.0)
    assert body(w).data.get("maybe") == NAME


def test_relink_survives_pause():
    now = [0.0]
    w, det = make(clock=lambda: now[0])

    def go(dets, a, b):
        t = a
        while t < b - 1e-9:
            now[0] = t
            det.frames = [dets]
            w.process(frame(), t, False)
            t = round(t + 0.1, 3)

    go([player(800), tag(800)], 0.0, 1.0)
    go([], 1.0, 2.2)
    w.hold("blackout")
    now[0] = 12.2
    w.release("blackout")  # 暂停了 10 秒：失踪记录跟着往后挪
    go([player(830)], 12.2, 13.0)
    assert body(w).data.get("maybe") == NAME


def test_relink_off():
    w, det = make(relink=False)
    lose(w, det)
    run(w, det, [player(830)], 2.2, 4.0)
    assert not body(w).data.get("maybe") and body(w).data.get("stranger")


def test_appearance_does_not_drop_relinked():
    import test_perception_appearance as ta

    w, det = ta.make()
    ta.friend_then_gone(w, det)  # 小明（粉）在 1000，最后看到 1.0
    ta.run(w, det, [], 1.1, 2.4)
    body_ = ta.player(1020)
    ta.run(w, det, [(body_, ta.PINK)], 2.5, 3.0)
    (p,) = ta.players(w)
    assert p.data["maybe"] == ta.XIAOMING and p.data["maybe_by"] == "relink"
    ta.run(w, det, [(body_, ta.GREEN)], 3.1, 5.0)  # 颜色全变了：位置连续性说了算，不摘
    assert ta.players(w)[0].data.get("maybe") == ta.XIAOMING
