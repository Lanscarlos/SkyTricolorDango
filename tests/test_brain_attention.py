"""空闲注意力：纯决策（plan 2026-09-30-idle-attention Task 3 / 4）。画面宽 1920，中线 960，中间带 40% = 576~1344。"""

import random

import pytest

from skydango.brain.attention import Attention, Target
from skydango.brain.peek import Turn
from skydango.config import Config

W = 1920


def attn(now=0.0, seed=1, **kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg.attention, k, v)
    a = Attention(cfg.attention, cfg.track, random.Random(seed), now)
    a.width = W
    return a


def tgt(key, kind, x, who=None, fresh=None):
    return Target(key, kind, x, who, fresh)


def test_friend_talking_beats_stranger_talking():
    a = attn()
    th = a.think([tgt("t:1", "talk_stranger", 300), tgt("t:2", "talk_friend", 1600, "小明", 0.0)], 0.0)
    assert th.current.key == "t:2" and th.action.direction == "right"  # 目标在右边：按右把它拉向中间（同 track）


def test_target_on_left_turns_left():
    th = attn().think([tgt("t:1", "talk_friend", 200, "小明", 0.0)], 0.0)
    assert th.action.direction == "left"


def test_centered_target_needs_no_turn():
    th = attn().think([tgt("t:1", "talk_friend", 1000, "小明", 0.0)], 0.0)
    assert th.centered and th.action is None


def test_turn_seconds_scale_with_offset_and_clamp():
    track = Config().track
    far = attn().think([tgt("t:1", "talk_friend", 1910, "小明")], 0.0).action.seconds
    near = attn().think([tgt("t:1", "talk_friend", 1400, "小明")], 0.0).action.seconds
    assert track.nudge_min <= near < far <= track.nudge_max


def test_no_second_press_before_settle():
    a = attn()
    targets = [tgt("t:1", "talk_friend", 1600, "小明")]
    th = a.think(targets, 0.0)
    a.pressed(th.action, 0.0)
    assert a.think(targets, 0.3).action is None
    assert a.think(targets, 0.7).action is not None


def test_bored_of_centered_target_then_switch():
    a = attn()
    talk = tgt("t:1", "talk_friend", 960, "小明", 0.0)
    stand = tgt("t:2", "friend_present", 1600, "小红")
    keys = [a.think([talk, stand], i * 0.5).current.key for i in range(13)]  # 0 ~ 6 秒
    assert keys[0] == "t:1" and keys[-1] == "t:2"  # 盯着说话的人 5 秒看腻了，换去看站着的


def test_new_bubble_resets_boredom():
    a = attn()
    stand = tgt("t:2", "friend_present", 1600, "小红")
    for i in range(13):
        th = a.think([tgt("t:1", "talk_friend", 960, "小明", 0.0), stand], i * 0.5)
    assert th.current.key == "t:2"
    th = a.think([tgt("t:1", "talk_friend", 960, "小明", 6.5), stand], 9.0)  # 他又说了一句（新气泡）
    assert th.current.key == "t:1"


def test_switch_margin_and_hold_prevent_flip_flop():
    a = attn()
    switches, last = 0, None
    for i in range(20):  # 两个好友同时说话，谁都不在中间：分数一样，不来回换
        th = a.think([tgt("t:1", "talk_friend", 300, "小明", 0.0), tgt("t:2", "talk_friend", 1600, "小红", 0.0)], i * 0.5)
        switches += th.current.key != last
        last = th.current.key
    assert switches == 1


def test_switch_needs_margin():
    a = attn(act_on_me=0.8)
    a.think([tgt("t:1", "approach", 300, "小明", 0.0)], 0.0)
    th = a.think([tgt("t:1", "approach", 300, "小明", 0.0), tgt("t:2", "act_on_me", 1600, "小红", 3.0)], 3.0)
    assert th.current.key == "t:1"  # 0.8 没比 0.7 高出 0.2


def test_stall_adds_boredom_and_moves_on():
    a = attn()
    targets = [tgt("t:1", "talk_friend", 1600, "小明", 0.0), tgt("t:2", "friend_present", 300, "小红")]
    t, keys = 0.0, []
    for _ in range(12):  # 偏差一直没变：转不动，每连按 stall_nudges 下加一截看腻，腻了就换
        th = a.think(targets, t)
        keys.append(th.current.key)
        if th.action is not None:
            a.pressed(th.action, t)
        t += 0.7
    assert keys[0] == "t:1" and "t:2" in keys


def test_stall_uses_positions_after_settle():
    a = attn(max_step=1.0)
    t0 = tgt("t:1", "talk_friend", 1600, "小明", 0.0)
    th = a.think([t0], 0.0)
    a.pressed(th.action, 0.0)
    for i in range(1, 6):  # 每下之后等过 settle 才看，偏差每次缩小 60 px：有进展，不算转不动
        th = a.think([tgt("t:1", "talk_friend", 1600 - 60 * i, "小明", 0.0)], i * 0.7)
        if th.action is None:
            break
        a.pressed(th.action, i * 0.7)
    assert a.think([tgt("t:1", "talk_friend", 1600 - 60 * i, "小明", 0.0)], i * 0.7 + 0.7).current.key == "t:1"


def test_stall_streak_resets_after_long_gap():
    a = attn()
    targets = [tgt("t:1", "talk_friend", 1600, "小明", 0.0), tgt("t:2", "friend_present", 300, "小红")]
    for t in (0.0, 0.7):
        a.pressed(a.think(targets, t).action, t)
    th = a.think(targets, 3.5)  # 隔了 2.8 秒（> 3 × settle）：别人可能动过镜头，重新计数
    a.pressed(th.action, 3.5)
    th = a.think(targets, 4.2)
    assert th.current.key == "t:1" and th.action is not None


def test_target_gone_stops_turning():
    a = attn()
    th = a.think([tgt("t:1", "talk_friend", 1600, "小明", 0.0)], 0.0)
    a.pressed(th.action, 0.0)
    th = a.think([], 0.7)
    assert th.current is None and th.action is None


def test_time_jump_is_clamped():
    a = attn()
    a.think([tgt("t:1", "talk_friend", 960, "小明", 0.0)], 0.0)
    th = a.think([tgt("t:1", "talk_friend", 960, "小明", 0.0), tgt("t:2", "friend_present", 1600)], 36000.0)
    assert th.current.key == "t:1"  # 只算 max_step（1 秒）的看腻：0.8 > 0.2，还看着他


def test_same_person_multiple_kinds_keeps_highest():
    th = attn().think([tgt("t:1", "friend_present", 1600, "小明"), tgt("t:1", "talk_friend", 1600, "小明", 0.0)], 0.0)
    assert th.current.kind == "talk_friend"


def test_look_first_only_for_fresh_talk_or_approach():
    assert attn().think([tgt("t:1", "talk_friend", 1600, "小明", 0.0)], 0.5).look_first
    assert attn().think([tgt("t:1", "approach", 1600, "小明", 0.0)], 0.5).look_first
    assert not attn().think([tgt("t:1", "talk_friend", 1600, "小明", 0.0)], 5.0).look_first  # 老气泡
    assert not attn().think([tgt("t:1", "friend_present", 1600, "小明")], 0.5).look_first


def test_below_min_interest_is_not_a_target():
    th = attn(friend_present=0.1).think([tgt("t:1", "friend_present", 1600, "小明")], 0.0)
    assert th.current is None and th.action is None


def test_describe_lines():
    a = attn()
    assert a.describe() == "没在看什么"
    a.think([tgt("t:1", "talk_friend", 1600, "小明", 0.0)], 0.0)
    assert a.describe() == "在看：小明（在说话）"
    a.think([tgt("a:1600", "approach", 1600, None, 1.0)], 1.0)
    assert a.describe() == "在看：陌生人（朝你走过来）"
