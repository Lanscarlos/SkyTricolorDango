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


# ---- Task 4：随意看、模式、focus ----

def run_idle(a, t0, t1, step=0.7, targets=()):
    """空转：每圈 think，给了按键就当真按了；返回 [(时间, Thought)]。"""
    out, t = [], t0
    while t <= t1:
        th = a.think(list(targets), t)
        if th.action is not None:
            a.pressed(th.action, t)
        out.append((t, th))
        t += step
    return out


def test_wander_after_random_quiet_interval():
    a = attn()
    log = run_idle(a, 0.0, 40.0)
    starts = [t for (t, th), (_, prev) in zip(log[1:], log) if th.wandering and not prev.wandering]
    assert starts and 8.0 <= starts[0] <= 20.0 + 0.7
    first = [th for t, th in log if th.wandering][:6]
    assert all(th.action is None or th.action.seconds == Config().track.nudge_max for th in first)
    presses = 0
    for (t, th), (_, nxt) in zip(log, log[1:]):
        presses += th.action is not None
        if th.wandering and not nxt.wandering:
            break
    assert 2 <= presses <= 4
    assert a.describe() in ("闲着随意看", "没在看什么")


def test_wander_never_same_side_more_than_limit():
    a = attn(wander_min=1.0, wander_max=1.0, wander_presses=[1, 1])
    sides = [th.action.direction for t, th in run_idle(a, 0.0, 200.0) if th.wandering and th.action is not None]
    assert len(sides) >= 10
    run = longest = 1
    for x, y in zip(sides, sides[1:]):
        run = run + 1 if x == y else 1
        longest = max(longest, run)
    assert longest <= 2


def test_target_interrupts_wander():
    a = attn(wander_min=1.0, wander_max=1.0, wander_presses=[4, 4])
    log = run_idle(a, 0.0, 3.0)
    assert any(th.wandering for t, th in log)
    th = a.think([tgt("t:1", "talk_friend", 1600, "小明", 3.5)], 3.7)
    assert not th.wandering and th.current.key == "t:1"


def first_wander(a):
    return next(t for t, th in run_idle(a, 0.0, 200.0, step=0.5) if th.wandering)


def test_wander_scale_and_curious_mode_shorten_interval():
    base = first_wander(attn(seed=3))
    slow = attn(seed=3)
    assert next(t for t, th in [(t, slow.think([], t, wander_scale=2.0)) for t in [i * 0.5 for i in range(400)]] if th.wandering) > base
    curious = attn(seed=3)
    curious.set_mode("好奇", None)
    assert first_wander(curious) < base


def test_focused_mode_ignores_low_interest_and_no_wander():
    a = attn(wander_min=1.0, wander_max=1.0)
    a.set_mode("专心", None)
    log = run_idle(a, 0.0, 10.0, targets=[tgt("t:1", "friend_present", 1600, "小红")])
    assert all(th.current is None and th.action is None and not th.wandering for t, th in log)
    th = a.think([tgt("t:2", "talk_friend", 1600, "小明", 11.0)], 11.0)
    assert th.current.key == "t:2"


def test_still_mode_never_moves_nor_looks_first():
    a = attn(wander_min=1.0, wander_max=1.0)
    a.set_mode("别动", None)
    for t, th in run_idle(a, 0.0, 10.0, targets=[tgt("t:1", "talk_friend", 1600, "小明", 0.0)]):
        assert th.action is None and not th.look_first
    assert a.describe().startswith("注意力：别动；在看：小明")


def test_curious_raises_stranger_talk():
    a = attn()
    a.set_mode("好奇", None)
    th = a.think([tgt("t:1", "talk_stranger", 1600), tgt("n:小红", "approach", 300, "小红", 0.0)], 0.0)
    assert a.interest(th.current) == pytest.approx(0.7)


def test_focus_boosts_person_and_slows_boredom():
    a = attn()
    a.set_mode("随意", "懒洋洋大王")
    th = a.think([tgt("t:1", "friend_present", 960, "懒洋洋大玉"), tgt("a:300", "approach", 300, None, 0.0)], 0.0)  # OCR 错一个字也认
    assert th.current.key == "t:1"
    for i in range(1, 9):  # 看了 4 秒：普通人已经腻到 0.8，关注的人只腻一半
        a.think([tgt("t:1", "friend_present", 960, "懒洋洋大玉")], i * 0.5)
    assert a._bored["t:1"] == pytest.approx(0.4, abs=0.05)


def test_set_mode_rejects_unknown():
    with pytest.raises(ValueError):
        attn().set_mode("发呆", None)


def test_describe_with_mode_and_focus():
    a = attn()
    a.set_mode("专心", "小明")
    a.think([tgt("t:1", "friend_present", 960, "小明")], 0.0)
    assert a.describe() == "注意力：专心，关注小明；在看：小明（站在那）"
    a.set_mode("随意", "")
    assert a.focus is None and a.describe() == "在看：小明（站在那）"
