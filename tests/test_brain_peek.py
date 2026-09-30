"""look_person 换角度（peek）：纯决策逻辑。画面 1920×1080，团子在中间。"""

from skydango.brain.peek import Done, Obs, PeekPlanner, Turn, Zoom, occluded, overlap_x, pick_self
from skydango.config import Config
from skydango.vision.bubbles import Rect

W, H = 1920, 1080
ME = Rect(860, 400, 200, 400)  # 团子：画面正中、屏高 37%


def planner(**kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg.peek, k, v)
    return PeekPlanner(cfg.peek, cfg.track, W, H)


def tag_at(cx, y=380):
    return Rect(cx - 60, y, 120, 36)


# ---- 挑团子框 ----

def test_pick_self_takes_box_nearest_center():
    far = Rect(100, 400, 200, 400)
    assert pick_self([far, ME], W, 0.15) == ME


def test_pick_self_ignores_boxes_off_center():  # 认到旁边去了（v5 以前把白头发好友认成团子）不采信
    assert pick_self([Rect(100, 400, 200, 400)], W, 0.15) is None
    assert pick_self([], W, 0.15) is None


# ---- 判断被挡住 ----

def test_tag_over_self_is_occluded():
    assert occluded(tag_at(960, y=360), ME)
    assert occluded(tag_at(1070, y=450), ME)  # 标签压在团子上半身、稍微偏右


def test_tag_beside_self_is_not_occluded():
    assert not occluded(tag_at(1500), ME)


def test_tag_far_above_self_is_not_occluded():  # 标签在团子头顶很高的地方：人在后面远处的山坡上，没被挡
    assert not occluded(tag_at(960, y=0), ME)


def test_overlap_x_is_share_of_first_box():
    assert overlap_x(Rect(0, 0, 100, 10), Rect(50, 0, 100, 10)) == 0.5
    assert overlap_x(Rect(0, 0, 100, 10), Rect(200, 0, 100, 10)) == 0.0


# ---- 闭环：露出来 ----

def test_turns_away_from_self_on_the_tag_side():
    p = planner()
    # 标签在团子中心左边：按右（远处的东西往左移，团子是支点不动）
    a = p.next(Obs(tag=tag_at(930), body=None, me=ME))
    assert isinstance(a, Turn) and a.direction == "right"
    p = planner()
    a = p.next(Obs(tag=tag_at(990), body=None, me=ME))
    assert isinstance(a, Turn) and a.direction == "left"


def test_turn_length_follows_remaining_gap():
    near = planner().next(Obs(tag=tag_at(1080), body=None, me=ME))  # 快出来了
    far = planner().next(Obs(tag=tag_at(965), body=None, me=ME))  # 还在正后方
    assert isinstance(near, Turn) and isinstance(far, Turn)
    assert near.seconds < far.seconds
    cfg = Config().track
    assert cfg.nudge_min <= near.seconds <= far.seconds <= cfg.nudge_max


def test_direction_follows_current_side_each_step():  # 转过头、标签跑到另一边：下一下反过来
    p = planner()
    assert p.next(Obs(tag=tag_at(930), body=None, me=ME)).direction == "right"
    assert p.next(Obs(tag=tag_at(1000), body=None, me=ME)).direction == "left"


def test_body_clear_of_self_is_revealed():
    p = planner(too_small=0.0)  # 只看露没露出来
    body = Rect(1150, 420, 160, 380)  # 和团子框不重叠
    a = p.next(Obs(tag=tag_at(1230), body=body, me=ME))
    assert a == Done("revealed")


def test_body_still_mostly_behind_self_keeps_turning():
    p = planner()
    body = Rect(900, 420, 160, 380)  # 大半压在团子框里
    assert isinstance(p.next(Obs(tag=tag_at(980), body=body, me=ME)), Turn)


def test_stuck_after_same_direction_presses_without_progress():
    p = planner(max_zoom_out=0)
    cfg = Config().track
    obs = Obs(tag=tag_at(930), body=None, me=ME)
    for _ in range(cfg.stall_nudges):
        assert isinstance(p.next(obs), Turn)
    assert p.next(obs) == Done("stuck")


def test_stuck_zooms_out_first_when_allowed():
    p = planner(max_zoom_out=1)
    obs = Obs(tag=tag_at(930), body=None, me=ME)
    for _ in range(Config().track.stall_nudges):
        p.next(obs)
    assert p.next(obs) == Zoom("out")
    assert isinstance(p.next(obs), Turn)  # 拉远之后重新计数接着转


def test_progress_resets_stall_count():
    p = planner(max_zoom_out=0)
    x = 930
    for _ in range(5):  # 每一下都往外挪 30 px：一直在进展，不算转不动
        assert isinstance(p.next(Obs(tag=tag_at(x), body=None, me=ME)), Turn)
        x -= 30


def test_tag_clear_of_self_without_body_box_counts_as_revealed():  # 人出来了、YOLO 还没框上身体：按标签估
    assert planner().next(Obs(tag=tag_at(700), body=None, me=ME)) == Done("revealed")


# ---- 缩放：贴太近先拉远 ----

def test_self_too_big_zooms_out_before_turning():
    big = Rect(660, 100, 600, 900)  # 屏高 83%
    p = planner(too_close=0.5, max_zoom_out=2)
    assert p.next(Obs(tag=tag_at(930, y=90), body=None, me=big)) == Zoom("out")
    assert p.next(Obs(tag=tag_at(930, y=90), body=None, me=big)) == Zoom("out")
    assert isinstance(p.next(Obs(tag=tag_at(930, y=90), body=None, me=big)), Turn)  # 拉远次数用完：转


def test_without_self_box_only_turns_and_never_zooms():
    p = planner()
    a = p.next(Obs(tag=tag_at(930), body=None, me=None))
    assert isinstance(a, Turn)


# ---- 缩放：露出来但太小就拉近 ----

def test_small_revealed_body_zooms_in():
    p = planner(too_small=0.25)
    small = Rect(1300, 500, 60, 150)  # 屏高 14%
    assert p.next(Obs(tag=tag_at(1330, y=460), body=small, me=ME)) == Zoom("in")
    bigger = Rect(1350, 450, 120, 300)  # 拉近后 28%：够了
    assert p.next(Obs(tag=tag_at(1410, y=410), body=bigger, me=ME)) == Done("revealed")


def test_zoom_in_that_hides_friend_again_is_undone():
    p = planner(too_small=0.25)
    small = Rect(1150, 500, 60, 150)
    assert p.next(Obs(tag=tag_at(1180, y=460), body=small, me=ME)) == Zoom("in")
    behind = Rect(950, 450, 120, 300)  # 拉近后又压到团子身上
    assert p.next(Obs(tag=tag_at(1010, y=410), body=behind, me=ME)) == Zoom("out")
    assert p.next(Obs(tag=tag_at(1180, y=460), body=small, me=ME)) == Done("revealed")


def test_zoom_in_that_pushes_friend_off_screen_is_undone():
    p = planner(too_small=0.25)
    small = Rect(1700, 500, 60, 150)
    assert p.next(Obs(tag=tag_at(1730, y=460), body=small, me=ME)) == Zoom("in")
    edge = Rect(1850, 450, 70, 300)  # 贴到画面右边了
    assert p.next(Obs(tag=tag_at(1885, y=410), body=edge, me=ME)) == Zoom("out")


def test_zoom_in_capped():
    p = planner(too_small=0.9, max_zoom_in=2)
    small = Rect(1300, 500, 60, 150)
    obs = Obs(tag=tag_at(1330, y=460), body=small, me=ME)
    assert p.next(obs) == Zoom("in")
    assert p.next(obs) == Zoom("in")
    assert p.next(obs) == Done("revealed")


def test_lost_target_is_done():
    assert planner().next(Obs(tag=None, body=None, me=ME)) == Done("lost")


# ---- 评审修正 ----

def test_zoom_in_that_loses_friend_entirely_is_undone():  # 拉近后人整个出了画面（看不到）：退一步，不直接放弃
    p = planner(too_small=0.25)
    small = Rect(1700, 500, 60, 150)
    assert p.next(Obs(tag=tag_at(1730, y=460), body=small, me=ME)) == Zoom("in")
    assert p.next(None) == Zoom("out")
    assert p.next(None) == Done("lost")


def test_missing_obs_before_reveal_is_lost():
    assert planner().next(None) == Done("lost")


def test_flip_flopping_direction_counts_as_stuck():  # 方向假设不对时标签来回越过中线：别抖到预算用完
    p = planner()
    assert p.next(Obs(tag=tag_at(930), body=None, me=ME)).direction == "right"
    assert p.next(Obs(tag=tag_at(990), body=None, me=ME)).direction == "left"
    assert p.next(Obs(tag=tag_at(930), body=None, me=ME)).direction == "right"
    assert p.next(Obs(tag=tag_at(990), body=None, me=ME)) == Done("stuck")
