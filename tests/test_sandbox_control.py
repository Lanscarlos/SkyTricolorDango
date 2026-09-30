"""沙盒操作和状态（brain-sandbox 计划 Task 4）：真 Body + 沙盒世界 + 假大脑，身体线程真的跑。"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from conftest import FakeLlm

from skydango.brain.body import Body, ToolError
from skydango.brain.events import EventQueue
from skydango.brain.world import BrainParts
from skydango.config import Config
from skydango.inner.ledger import Ledger
from skydango.inner.log import MindLog
from skydango.inner.mind import Mind
from skydango.inner.reflect import Reflector
from skydango.inner.store import InnerStore
from skydango.sandbox.clock import SimClock
from skydango.sandbox.control import SandboxControl, clock_text
from skydango.sandbox.transcript import Transcript
from skydango.sandbox.world import Scene, sandbox_world

WALL = time.mktime((2026, 9, 30, 23, 0, 0, 0, 0, -1))  # 沙盒时间：9 月 30 日 23:00
FRIENDS = ["小明", "阿花"]


class FakeBrain:
    def __init__(self) -> None:
        self.in_turn = False
        self.left = 0.0

    def limit_left(self, now: float) -> float:
        return self.left


class Mono:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def build(tmp_path, reply="{}", wait=None, start=True):
    cfg = Config()
    cfg.reply.dry_run = False
    cfg.reply.disclosure_prefix = ""
    cfg.wheel.library_dir = str(tmp_path / "没有图标库")
    cfg.sandbox.emotes = ["鞠躬"]
    cfg.vision.poll_interval = 0.05
    sim = SimClock(offset=WALL - time.time())
    transcript = Transcript(sim)
    scene = Scene()
    world = sandbox_world(cfg, sim, transcript, scene)
    events = EventQueue(clock=world.clock)
    ledger = Ledger(cfg.inner, lambda: FRIENDS, sim.wall(), store=InnerStore(tmp_path / "inner"), persist=False)
    llm = FakeLlm(reply, wait=wait)
    reflector = Reflector(cfg.inner, llm, sim.clock, threaded=True)
    mind_log = MindLog(None, persist=False)
    body = Body(
        cfg, world.device, world.reader, world.sender, world.self_filter, events,
        env=world.env, social=world.social, emotes=world.emotes, camera=world.camera, locomotion=world.locomotion,
        friend_checker=world.friend_checker, panels=world.panels, panel_ops=world.panel_ops, panel=world.panel,
        ledger=ledger, mind=Mind(), reflector=reflector, mind_log=mind_log, clock=world.clock, wall=world.wall,
    )
    body.friend_names = lambda: FRIENDS
    brain = FakeBrain()
    parts = BrainParts(body=body, eyes=None, events=events, brain=brain, trace=None, reflector=reflector, ledger=ledger,
                       store=None, mind_log=mind_log)
    mono = Mono()
    control = SandboxControl(parts, world, sim, transcript, scene, mono=mono)
    stop = threading.Event()
    thread = threading.Thread(target=body.run, args=(0.0, stop), daemon=True)
    if start:
        thread.start()
    return SimpleNamespace(cfg=cfg, sim=sim, transcript=transcript, scene=scene, world=world, events=events, body=body,
                           brain=brain, control=control, stop=stop, thread=thread, mind_log=mind_log, llm=llm, mono=mono,
                           reflector=reflector)


@pytest.fixture
def sb(tmp_path):
    s = build(tmp_path)
    yield s
    s.stop.set()
    s.thread.join(3)


def wait_for(cond, seconds=3.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def rows(sb, kind=None):
    return [r for r in sb.transcript.since(0) if kind is None or r["kind"] == kind]


def test_say_goes_to_transcript_and_becomes_chat(sb):
    out = sb.control.op({"op": "say", "who": "小明", "text": "在吗"})
    assert out["ok"]
    assert [(r["who"], r["text"]) for r in rows(sb, "heard")] == [("小明", "在吗")]
    assert wait_for(lambda: any(e.kind == "chat" and "在吗" in e.text for e in sb.events.recent(20)))


def test_come_and_leave(sb):
    assert sb.control.op({"op": "come", "who": "小明"})["ok"]
    assert sb.world.env.friends == ["小明"]
    assert wait_for(lambda: any(e.kind == "arrive" for e in sb.events.recent(20)))
    assert ("event", "── 小明来了 ──") in [(r["kind"], r["text"]) for r in rows(sb)]
    assert sb.control.op({"op": "come", "who": "小明"})["ok"] is False  # 已经在了
    assert sb.control.op({"op": "leave", "who": "小明"})["ok"]
    assert sb.world.env.friends == [] and rows(sb)[-1]["text"] == "── 小明走了 ──"
    assert sb.control.op({"op": "leave", "who": "小明"})["ok"] is False


def test_strangers_place_scene(sb):
    assert sb.control.op({"op": "strangers", "n": 2})["ok"] and sb.world.env.stranger_count == 2
    assert sb.control.op({"op": "place", "name": "云野"})["ok"] and sb.world.env.place == "云野"
    assert sb.control.op({"op": "scene", "text": "篝火旁"})["ok"] and sb.scene.text == "篝火旁"
    assert [r["kind"] for r in rows(sb)] == ["event"] * 3
    with pytest.raises(ValueError):
        sb.control.op({"op": "strangers", "n": 21})
    with pytest.raises(ValueError):
        sb.control.op({"op": "strangers", "n": True})


def test_skip_moves_clock_and_logs_energy(sb):
    before = sb.sim.offset
    energy = len([r for r in sb.mind_log.recent() if r["kind"] == "energy"])
    out = sb.control.op({"op": "skip", "seconds": 3600})
    assert out["ok"] and sb.sim.offset - before == 3600
    after = [r for r in sb.mind_log.recent() if r["kind"] == "energy"]
    assert len(after) >= energy + 2
    assert rows(sb)[-1]["text"] == "── 快进 1 小时 ──"
    for bad in (0, -5, float("nan"), float("inf"), "abc", None):
        with pytest.raises(ValueError):
            sb.control.op({"op": "skip", "seconds": bad})


def test_time_forward_and_refuses_backwards(sb):
    out = sb.control.op({"op": "time", "at": "08:00"})
    assert out["ok"]
    t = time.localtime(sb.sim.wall())
    assert (t.tm_mon, t.tm_mday, t.tm_hour) == (10, 1, 8)  # 23:00 → 明天 8 点
    assert "拨到" in rows(sb)[-1]["text"]
    out = sb.control.op({"op": "time", "at": "2026-09-30 22:00"})
    assert out["ok"] is False and "不能往回拨" in out["text"]


def test_reflect_twice_says_already_running(tmp_path):
    gate = threading.Event()
    s = build(tmp_path, wait=gate)
    try:
        assert s.control.op({"op": "reflect"})["ok"]
        out = s.control.op({"op": "reflect"})
        assert out == {"ok": False, "text": "正在反思"}
    finally:
        gate.set()
        s.stop.set()
        s.thread.join(3)


def test_reflection_changes_go_to_transcript(tmp_path):
    s = build(tmp_path, reply='{"mood": {"level": "开心", "text": "有人来聊天"}}')
    try:
        assert s.control.op({"op": "reflect"})["ok"]
        assert wait_for(lambda: any(r["text"].startswith("── 反思：") for r in rows(s, "event")))
        line = next(r["text"] for r in rows(s, "event") if r["text"].startswith("── 反思："))
        assert "心情 平常→开心" in line and line.endswith(" ──")
    finally:
        s.stop.set()
        s.thread.join(3)


def test_notice_without_friends_is_blocked_with_reason(sb):
    out = sb.control.op({"op": "notice", "text": "天黑了"})
    assert out["ok"] is False
    (row,) = rows(sb, "blocked")
    assert "天黑了" in row["text"] and "身边没有好友" in row["why"]
    assert not any(e.kind == "notice" for e in sb.events.recent(20))


def test_notice_with_friend_becomes_event(sb):
    sb.control.op({"op": "come", "who": "小明"})
    out = sb.control.op({"op": "notice", "text": "天黑了"})
    assert out["ok"]
    assert any(e.kind == "notice" and "天黑了" in e.text for e in sb.events.recent(20))


def test_blocked_say_goes_to_transcript(sb):
    with pytest.raises(ToolError):
        sb.body.call(lambda: sb.body.say("你好呀"))  # 没熟人：主动开口被拦
    (row,) = rows(sb, "blocked")
    assert row["text"] == "你好呀" and "身边没有好友" in row["why"]


def test_bad_requests(sb):
    for req in ({"op": "飞"}, {}, {"op": "say", "text": "在吗"}, {"op": "say", "who": "小明"}, {"op": "come"},
                {"op": "time"}, "不是字典"):
        with pytest.raises(ValueError):
            sb.control.op(req)


def test_idle(tmp_path):
    s = build(tmp_path, start=False)
    c = s.control
    s.events.put("chat", "小明：在吗")
    assert c.idle() is False
    s.events.drain()
    s.brain.in_turn = True
    assert c.idle() is False
    s.brain.in_turn = False
    assert c.idle() is False  # 刚安静下来
    s.mono.t += 1.0
    assert c.idle() is False
    s.mono.t += 1.5
    assert c.idle() is True
    s.world.reader.say("小明", "在吗")  # 身体还没读的冒充发言也算有事
    assert c.idle() is False


def test_state_long_poll_times_out_without_new_rows(sb):
    sb.control.op({"op": "come", "who": "小明"})
    v = sb.transcript.version
    started = time.monotonic()
    st = sb.control.state(after=v, timeout=0.1)
    assert 0.08 <= time.monotonic() - started < 1.0
    assert st["lines"] == [] and st["version"] == v
    assert st["friends"] == ["小明"] and st["clock_text"].startswith("9月30日 周三 23:")
    assert st["limit"] == "" and set(st["energy"]) >= {"level", "score", "note"}


def test_state_returns_new_rows_and_limit(sb):
    sb.brain.left = 125
    st = sb.control.state(after=0, timeout=0.0)
    assert st["limit"] == "额度用完，约 3 分钟后再试"
    sb.control.op({"op": "say", "who": "小明", "text": "在吗"})
    st = sb.control.state(after=0, timeout=0.0)
    assert [r["text"] for r in st["lines"]] == ["在吗"]


def test_clock_text():
    assert clock_text(time.mktime((2026, 9, 30, 23, 30, 0, 0, 0, -1))) == "9月30日 周三 23:30"
