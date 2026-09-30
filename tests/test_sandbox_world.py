"""沙盒世界的部件 + 真 Body（brain-sandbox 计划 Task 3）：不起大脑，单步 body.step()。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from skydango.brain.body import Body, ToolError
from skydango.brain.events import EventQueue
from skydango.brain.world import World
from skydango.config import Config
from skydango.inner.ledger import Ledger
from skydango.inner.store import InnerStore
from skydango.sandbox.clock import SimClock
from skydango.sandbox.transcript import Transcript
from skydango.sandbox.world import (
    GRAY,
    SandboxDevice,
    SandboxEmotes,
    SandboxLocomotion,
    SandboxReader,
    Scene,
    emote_names,
    sandbox_world,
)

WALL = 1_790_000_000.0
FRIENDS = ["小明", "阿花"]


class Mono:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make_cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.reply.dry_run = False  # 沙盒总是 live
    cfg.reply.disclosure_prefix = ""
    cfg.wheel.library_dir = str(tmp_path / "没有图标库")
    cfg.sandbox.emotes = ["鞠躬", "挥手"]
    return cfg


@pytest.fixture
def sb(tmp_path):
    cfg = make_cfg(tmp_path)
    mono = Mono()
    sim = SimClock(real_wall=lambda: WALL, real_mono=mono)
    transcript = Transcript(sim)
    scene = Scene()
    world = sandbox_world(cfg, sim, transcript, scene)
    events = EventQueue(clock=world.clock)
    ledger = Ledger(cfg.inner, lambda: FRIENDS, WALL, store=InnerStore(tmp_path / "inner"), persist=False)
    body = Body(
        cfg, world.device, world.reader, world.sender, world.self_filter, events,
        env=world.env, social=world.social, emotes=world.emotes, camera=world.camera, locomotion=world.locomotion,
        friend_checker=world.friend_checker, panels=world.panels, panel_ops=world.panel_ops, panel=world.panel,
        ledger=ledger, clock=world.clock, wall=world.wall, sleep=lambda s: None,
    )
    body.friend_names = lambda: FRIENDS
    body.panel.start(world.clock())
    return SimpleNamespace(
        body=body, reader=world.reader, env=world.env, events=events, transcript=transcript, ledger=ledger,
        world=world, sim=sim, mono=mono, scene=scene, cfg=cfg,
    )


def test_heard_becomes_chat_event(sb):
    sb.reader.say("小明", "在吗")
    sb.body.step()
    assert any(e.kind == "chat" and "在吗" in e.text for e in sb.events.drain())


def test_heard_keeps_speaker(sb):
    sb.reader.say("小明", "在吗")
    sb.body.step()
    assert list(sb.body.chat)[-1][1:] == ("小明", "在吗")


def test_nearby_list_drives_arrive_and_ledger(sb):
    sb.env.friends = ["小明"]
    sb.body.step()
    assert any(e.kind == "arrive" for e in sb.events.drain())
    assert sb.ledger.card("小明").visits == 1


def test_leave_after_removed(sb):
    sb.env.friends = ["小明"]
    sb.body.step()
    sb.events.drain()
    sb.env.friends = []
    sb.mono.t += 1
    sb.body.step()
    assert any(e.kind == "leave" and e.who == "小明" for e in sb.events.drain())


def test_strangers_count(sb):
    sb.env.stranger_count = 2
    sb.body.step()
    assert any(e.kind == "stranger" for e in sb.events.drain())
    assert "身边的陌生人：2 个" in sb.body.status()


def test_place_change_is_notice(sb):
    sb.env.friends = ["小明"]  # 有熟人才发 notice
    sb.env.place = "云野"
    sb.body.step()
    sb.events.drain()
    sb.env.place = "雨林"
    sb.mono.t += 1
    sb.body.step()
    assert any(e.kind == "notice" and "雨林" in e.text for e in sb.events.drain())


def test_say_and_emote_go_to_transcript(sb):
    sb.body.say("你好呀", live=True)
    sb.body.emote("鞠躬", live=True)
    kinds = [(r["kind"], r["text"]) for r in sb.transcript.since(0)]
    assert ("said", "你好呀") in kinds and ("act", "（团子做了 鞠躬）") in kinds


def test_said_row_has_sim_time(sb):
    sb.sim.skip(3600)
    sb.body.say("晚上好", live=True)
    row = sb.transcript.since(0)[-1]
    assert row["t"] == WALL + 3600 and row["who"] == "团子"


def test_emote_not_on_wheel_refused(sb):
    with pytest.raises(ToolError):
        sb.body.emote("跳舞", live=True)


def test_move_goes_to_transcript(sb):
    result = sb.body.move("forward", 2, live=True)
    assert "2 步" in result
    assert sb.transcript.since(0)[-1]["text"] == "（团子往前走了 2 步）"


def test_no_camera_tools(sb):
    with pytest.raises(ToolError, match="没有视角控制"):
        sb.body.camera_move("left", 1, live=True)
    with pytest.raises(ToolError, match="没有视角控制"):
        sb.body.camera_reset(live=True)


def test_scene_text_or_blind():
    s = Scene()
    assert s.describe([]) == "看不清，眼前什么也看不出来"
    s.text = "云野，篝火旁"
    assert s.describe([]) == "云野，篝火旁"
    s.text = "   "
    assert s.describe([]) == "看不清，眼前什么也看不出来"


def test_gray_frame_is_not_blackout(sb):
    sb.body.step()
    assert sb.body.blackout is False


def test_device_frame_is_gray_copy():
    d = SandboxDevice()
    a = d.screenshot()
    assert a.shape == (1080, 1920, 3) and a.dtype == np.uint8 and int(a.mean()) == 128
    a[:] = 0
    assert int(GRAY.mean()) == 128 and d.ime_shown() is False
    d.tap(1, 2)
    d.hw_key(46)
    d.hw_key_hold(17, 0.1)
    d.hw_key_down(17)
    d.hw_key_up(17)
    d.key(4)
    d.input_text("x")
    d.swipe(0, 0, 1, 1)


def test_reader_queue():
    r = SandboxReader()
    r.say("小明", "一")
    r.say("阿花", "二")
    got = r.read(None, 5.0)
    assert [(m.speaker, m.text, m.seen_at) for m in got] == [("小明", "一", 5.0), ("阿花", "二", 5.0)]
    assert r.read(None, 6.0) == [] and r.panel_visible(None) is True and r.panel_closed_since is None


def test_sender_bubble(sb):
    s = sb.world.sender
    assert s.open() is True and s.opened is True
    s.cancel()
    assert s.opened is False
    s.open()
    s.send("嗯嗯")
    assert s.opened is False
    rows = [(r["kind"], r["text"]) for r in sb.transcript.since(0)]
    assert rows[0] == ("act", "（团子头顶冒出输入气泡）") and rows[-1] == ("said", "嗯嗯")


def test_emotes_reflex_and_interval():
    sim = SimClock(real_wall=lambda: WALL, real_mono=lambda: 100.0)
    t = Transcript(sim)
    e = SandboxEmotes(["鞠躬", "挥手"], t, sim.clock, min_interval=20.0)
    assert e.on_wheel() == ["鞠躬", "挥手"] and e.available() == ["鞠躬", "挥手"]
    e.perform("挥手", reflex=True)
    assert t.since(0)[-1]["text"] == "（团子下意识地 挥手）"
    assert e.available() == ["鞠躬", "挥手"]  # 反射不占大脑的冷却
    assert e.last_any == 100.0 and e.last_emote == float("-inf")
    assert e.perform("鞠躬") == 1
    assert e.available() == [] and e.available(True) == ["鞠躬", "挥手"]
    e.pretend("鞠躬")
    assert t.since(0)[-1]["text"] == "（团子做了 鞠躬）"
    e.restore()


def test_locomotion_directions():
    sim = SimClock(real_wall=lambda: WALL)
    t = Transcript(sim)
    loc = SandboxLocomotion(t)
    loc.move("W", 1)
    loc.move("left", 9, max_steps=3)
    assert [r["text"] for r in t.since(0)] == ["（团子往前走了 1 步）", "（团子往左走了 3 步）"]
    with pytest.raises(ValueError):
        loc.move("up")


def test_transcript_limit_and_since():
    sim = SimClock(real_wall=lambda: WALL)
    t = Transcript(sim, limit=3)
    for i in range(5):
        t.add("event", str(i))
    assert t.version == 5 and [r["text"] for r in t.since(0)] == ["2", "3", "4"]
    assert [r["seq"] for r in t.since(4)] == [5]
    with pytest.raises(ValueError):
        t.add("啥", "x")


def test_world_shape(sb):
    w = sb.world
    assert isinstance(w, World) and w.name == "sandbox"
    assert w.camera is None and w.friend_checker is None and w.panels is None and w.panel_ops is None and w.social is None
    assert w.wall() == WALL and w.describe([]) == "看不清，眼前什么也看不出来"
    assert w.panel.cfg.mode == "always"
    w.close()


def test_emote_names_fallback(tmp_path):
    cfg = make_cfg(tmp_path)
    assert emote_names(cfg) == ["鞠躬", "挥手"]
