"""身体接上性格（spec 2026-09-30-inner-phase3 §2 §4）：收着点、反思套性格档案、arrive 带老梗、材料带性格。"""

import time

from conftest import FakeLlm
from test_brain_body import FakeEnv, body, msg

from skydango.config import Config
from skydango.inner.ledger import Card, Ledger
from skydango.inner.mind import Mind
from skydango.inner.persona import Persona, Trait
from skydango.inner.reflect import Reflector
from skydango.inner.store import InnerStore

FRIENDS = ["懒洋洋大王", "阿花"]
WALL = time.mktime((2026, 9, 30, 15, 0, 0, 0, 0, -1))


def make(clock, tmp_path, live=False, reflect_reply="{}", **kw):
    env = FakeEnv()
    cfg = Config().inner
    cfg.persona = True
    led = Ledger(cfg, lambda: FRIENDS, WALL, store=InnerStore(tmp_path), persist=live,
                 cards={"懒洋洋大王": Card(first_met=0, days=["d1", "d2", "d3"])})
    reflector = Reflector(cfg, FakeLlm(reflect_reply), clock, threaded=False)
    b, dev, reader, events = body(clock, live=live, env=env, ledger=led, mind=Mind(), reflector=reflector,
                                  persona=Persona(), wall=lambda: WALL, **kw)
    b.friend_names = lambda: FRIENDS
    return b, env, reader, events


def test_upset_friend_gets_soft_in_status(clock, tmp_path):
    b, env, reader, _ = make(clock, tmp_path)
    env.near = ["阿花"]
    said = "今天真的很难过" + "呜" * 20
    reader.batches = [[msg(said, speaker="阿花")]]
    b.step()
    assert f"对阿花收着点（他刚说「{said[:20]}」）" in b.status()  # 原话截 20 字
    assert b.soft_names(WALL) == {"阿花"}
    b.wall = lambda: WALL + 31 * 60
    assert "收着点" not in b.status() and b.soft_names(WALL + 31 * 60) == set()


def test_soft_several_friends(clock, tmp_path):
    b, env, reader, _ = make(clock, tmp_path)
    reader.batches = [[msg("我好难过"), msg("我也不开心", speaker="阿花")]]
    b.step()
    assert "对懒洋洋大王、阿花收着点（他们刚说难过）" in b.status()


def test_stranger_upset_not_soft(clock, tmp_path):
    b, env, reader, _ = make(clock, tmp_path)
    reader.batches = [[msg("我好难过", speaker="路人甲")]]
    b.step()
    assert b.soft_names(WALL) == set() and "收着点" not in b.status()


def test_arrive_carries_jokes(clock, tmp_path):
    b, env, _, events = make(clock, tmp_path)
    b.persona.jokes = [Trait("路痴带路", who="懒洋洋大王")]
    env.near = ["懒洋洋大王"]
    b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == ["懒洋洋大王 来到身边（第一次在身边见到）。你们的老梗：路痴带路"]


def test_reflection_applies_persona_live_only(clock, tmp_path):
    reply = '{"persona_add": {"catchphrases": ["害，懒得动"]}}'
    for live in (False, True):
        d = tmp_path / str(live)
        b, env, reader, _ = make(clock, d, live=live, reflect_reply=reply)
        reader.batches = [[msg("嗨")]]
        b.step()
        clock.advance(1300)
        b.step()
        b.step()
        assert [t.text for t in b.persona.catchphrases] == ["害，懒得动"]
        assert (d / "persona.json").exists() is live


def test_soft_friend_passed_to_persona(clock, tmp_path):
    reply = '{"persona_add": {"jokes": [{"who": "懒洋洋大王", "text": "路痴"}]}}'
    b, env, reader, _ = make(clock, tmp_path, reflect_reply=reply)
    reader.batches = [[msg("我好难过")]]
    b.step()
    clock.advance(1300)
    b.step()
    b.step()
    assert b.persona.jokes == []


def test_materials_have_traits(clock, tmp_path):
    b, *_ = make(clock, tmp_path)
    assert "你攒下的性格：\n（还没有）" in b.reflect_materials(False)
    b.persona.catchphrases = [Trait("害，懒得动")]
    assert "口头禅：害，懒得动" in b.reflect_materials(False)


def test_persona_off_no_soft(clock, tmp_path):
    b, env, reader, _ = make(clock, tmp_path)
    b.persona = None
    reader.batches = [[msg("我好难过")]]
    b.step()
    assert "收着点" not in b.status()
    assert "你攒下的性格" not in b.reflect_materials(False)


def test_persona_errors_only_logged(clock, tmp_path):
    b, env, _, events = make(clock, tmp_path)

    def boom(*a, **k):
        raise RuntimeError("坏了")

    b.persona.joke_note = boom
    b.persona.section = boom
    env.near = ["懒洋洋大王"]
    b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == ["懒洋洋大王 来到身边（第一次在身边见到）"]
    text = b.reflect_materials(False)  # 拼性格出错：这一段不写，材料照拼
    assert "精力：" in text and "你攒下的性格" not in text


def test_no_persona_unchanged(clock):
    b, *_ = body(clock, env=FakeEnv())
    b.step()
    assert b.persona is None and b.soft_names(0) == set()


def test_arrive_skips_jokes_for_soft_friend(clock, tmp_path):  # 终审 I3b
    b, env, reader, events = make(clock, tmp_path)
    b.persona.jokes = [Trait("路痴带路", who="懒洋洋大王")]
    reader.batches = [[msg("今天好难过")]]
    b.step()
    events.drain()
    env.near = ["懒洋洋大王"]
    b.step()
    arrive = [e.text for e in events.drain() if e.kind == "arrive"]
    assert arrive and "老梗" not in arrive[0]


def test_soft_names_this_session_keeps_expired(clock, tmp_path):  # 终审 I3a
    b, env, reader, _ = make(clock, tmp_path)
    reader.batches = [[msg("今天好难过")]]
    b.step()
    assert b.soft_names_this_session() == {"懒洋洋大王"}
    assert b.soft_names(WALL + 3600) == set() and b.soft_names_this_session() == {"懒洋洋大王"}
