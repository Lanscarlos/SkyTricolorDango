"""大脑工具 find（plan 2026-10-03-attention-search Task 6）：技能层里转镜头找人，找到 / 没找到发 task_done / task_failed。"""

import pytest
from test_brain_attention_body import attn_body, steps

from skydango.brain.body import ToolError
from skydango.vision.bubbles import Rect
from skydango.vision.people import Person

XIAOMING = Person(9, "friend", "小明", Rect(1500, 400, 100, 300), "右边", "近")


def find_body(clock, **kw):
    b, dev, reader, events, env, cam = attn_body(clock, **kw)
    b.cfg.panel.idle_peek = 1e9
    b.cfg.attention.scan_after = 1e9  # 找完后注意力不接着环顾：这里只数 find 自己按的键
    b.cfg.call.enabled = False  # 不喊：只转（喊一声的路线在 search 单测里）
    return b, events, env, cam


def done_events(events):
    return [(e.kind, e.text) for e in events.drain() if e.kind in ("task_done", "task_failed")]


def test_find_refuses_stranger_names_and_dry_run(clock):
    b, events, env, cam = find_body(clock)
    with pytest.raises(ToolError, match="好友名单里没有"):
        b.find("阿白")
    dry, _, _, _ = find_body(clock, live=False)
    with pytest.raises(ToolError, match="dry-run"):
        dry.find("小明")


def test_find_target_already_in_view(clock):
    b, events, env, cam = find_body(clock)
    env.people_list = [XIAOMING]
    assert b.find("小明") == "小明就在画面里（右边·近），不用找"
    assert b.skills.active is None


def test_find_turns_full_circle_then_fails(clock):
    b, events, env, cam = find_body(clock)
    out = b.find("小明", 60)
    assert out.startswith("开始找小明了")
    steps(b, clock, 80)
    assert len(cam.nudges) == 6 * b.cfg.attention.seg_presses and len({d for d, s in cam.nudges}) == 1
    assert done_events(events) == [("task_failed", "找小明没做成：转了一圈没找到小明")]


def test_find_found_mid_way(clock):
    b, events, env, cam = find_body(clock)
    b.find("小明", 60)
    steps(b, clock, 5)
    env.people_list = [XIAOMING]
    steps(b, clock, 2)
    assert done_events(events) == [("task_done", "找小明：找到了小明（右边·近）")]
    assert b.skills.active is None
