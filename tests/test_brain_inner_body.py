"""身体接上内心账本（spec 2026-09-30-inner-phase1 §5）：在场、听到、说话记账；arrive 带交情、status 的身边好友带交情。"""

import json
from types import SimpleNamespace

from test_brain_body import FakeEnv, body, msg

from skydango.chat.memory import Turn
from skydango.config import Config
from skydango.inner import open_ledger
from skydango.inner.ledger import Ledger, Session
from skydango.inner.store import InnerStore

FRIENDS = ["懒洋洋大王", "阿花"]
WALL0 = 1_790_000_000.0


def make(clock, tmp_path, live=False, **kw):
    env = FakeEnv()
    led = Ledger(Config().inner, lambda: FRIENDS, WALL0, store=InnerStore(tmp_path), persist=live)
    b, dev, reader, events = body(clock, live=live, env=env, ledger=led, **kw)
    b.friend_names = lambda: FRIENDS
    return b, env, reader, events, led


def test_arrive_carries_note_return_does_not(clock, tmp_path):
    b, env, _, events, _ = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]
    b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == ["懒洋洋大王 来到身边（第一次在身边见到）"]
    env.near = []
    clock.advance(10)
    b.step()
    events.drain()  # 攒着的“走开”和“回来”会互相抵消：先取走
    env.near = ["懒洋洋大王"]
    clock.advance(10)
    b.step()
    assert [e.text for e in events.drain() if e.kind in ("arrive", "return")] == ["懒洋洋大王 回来了"]


def test_status_uses_status_line(clock, tmp_path):
    b, env, *_ = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]
    b.step()
    assert "身边的好友：懒洋洋大王（今天刚来·今天刚认识）" in b.status()


def test_heard_and_said_counted(clock, tmp_path):
    b, env, reader, _, led = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]
    reader.batches = [[msg("团子在吗")]]
    b.step()
    b.say("在")
    c = led.card("懒洋洋大王")
    assert (c.lines, c.to_me, led.session.said, led.session.heard) == (1, 1, 1, 1)


def test_present_during_track(clock, tmp_path):  # 跟踪中不发人来人走，但账照记
    b, env, _, events, led = make(clock, tmp_path)
    b.skills.active = SimpleNamespace(quiet_people=True, goal="盯着")
    env.near = ["懒洋洋大王"]
    b._watch_people(clock())  # 直接调：不经过 skills.tick
    assert led.card("懒洋洋大王").visits == 1
    assert not [e for e in events.drain() if e.kind == "arrive"]


def test_live_saves_each_loop_throttled(clock, tmp_path):
    b, env, *_ = make(clock, tmp_path, live=True)
    env.near = ["懒洋洋大王"]
    b.step()
    assert (tmp_path / "current.json").exists() and (tmp_path / "people.json").exists()


def test_dry_run_disk_untouched(tmp_path):  # Review Focus 1
    (tmp_path / "current.json").write_text(json.dumps(Session(start=WALL0 - 99, saved=WALL0 - 9).to_dict()), encoding="utf-8")
    turns = [Turn(WALL0 - 999, "新的聊天消息：\n懒洋洋大王：「嗨」", "嗨")]
    led = open_ledger(Config().inner, tmp_path, lambda: ["懒洋洋大王"], lambda: turns, persist=False, now=WALL0)
    led.present(["懒洋洋大王"], WALL0)
    led.save(WALL0)
    led.close("", WALL0 + 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["current.json"]
    assert led.card("懒洋洋大王").lines == 1 and led.history[-1].ended == "crash"
    assert [s.ended for s in led.history] == ["backfill", "crash"]


def test_bad_people_json_no_backfill_keeps_running(tmp_path):  # Review Focus 5
    (tmp_path / "people.json").write_text("{坏", encoding="utf-8")
    turns = [Turn(WALL0 - 999, "新的聊天消息：\n懒洋洋大王：「嗨」", "嗨")]
    led = open_ledger(Config().inner, tmp_path, lambda: ["懒洋洋大王"], lambda: turns, persist=True, now=WALL0)
    assert led.card("懒洋洋大王") is None
    led.present(["懒洋洋大王"], WALL0)
    assert led.card("懒洋洋大王").visits == 1


def test_ledger_errors_do_not_break_body(clock, tmp_path):
    b, env, _, events, led = make(clock, tmp_path)

    def boom(*a):
        raise RuntimeError("坏了")

    led.present = boom
    led.save = boom
    env.near = ["懒洋洋大王"]
    b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == ["懒洋洋大王 来到身边"]
    assert "身边的好友：懒洋洋大王" in b.status()


def test_status_falls_back_when_status_line_breaks(clock, tmp_path):
    b, env, _, _, led = make(clock, tmp_path)
    led.status_line = lambda *a: (_ for _ in ()).throw(RuntimeError("坏了"))
    env.near = ["懒洋洋大王", "阿花"]
    assert "身边的好友：懒洋洋大王、阿花" in b.status()


def test_no_ledger_unchanged(clock):
    b, *_ = body(clock, env=FakeEnv())
    assert b.ledger is None and "身边的好友：没看到" in b.status()
