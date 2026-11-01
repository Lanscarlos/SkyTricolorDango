"""身体接上对每个人的感觉（spec 2026-11-01-friend-feelings §2 §4）：下线反思的材料和上下文、arrive、status、introspect。"""

from test_brain_body import FakeEnv, body, msg

from skydango.config import Config
from skydango.inner.ledger import Ledger
from skydango.inner.mind import Mind
from skydango.inner.store import InnerStore

FRIENDS = ["懒洋洋大王", "阿花"]
WALL0 = 1_790_000_000.0
FEEL = {"text": "嘴贫但靠谱", "close": "熟了", "trend": "平稳", "since": "2026-09-20", "past": []}


def make(clock, tmp_path, feelings=True, **kw):
    env = FakeEnv()
    led = Ledger(Config().inner, lambda: FRIENDS, WALL0, store=InnerStore(tmp_path), persist=False)
    b, dev, reader, events = body(clock, env=env, ledger=led, mind=Mind(), **kw)
    b.cfg.inner.feelings = feelings
    b.friend_names = lambda: FRIENDS
    return b, env, reader, events, led


def test_final_materials_have_feelings_block(clock, tmp_path):
    b, env, reader, _, led = make(clock, tmp_path)
    reader.batches = [[msg("团子在吗")]]
    env.near = ["懒洋洋大王"]
    b.step()
    assert "你对他们的感觉" in b.reflect_materials(True)
    assert "懒洋洋大王：还没有印象" in b.reflect_materials(True)
    assert "你对他们的感觉" not in b.reflect_materials(False)


def test_final_materials_unchanged_when_off(clock, tmp_path):
    b, env, reader, _, _ = make(clock, tmp_path, feelings=False)
    reader.batches = [[msg("团子在吗")]]
    env.near = ["懒洋洋大王"]
    b.step()
    assert "你对他们的感觉" not in b.reflect_materials(True)
    assert b.feeling_context() is None


def _arrive(b, env, events):
    env.near = ["懒洋洋大王"]
    b.step()
    return [e.text for e in events.drain() if e.kind == "arrive"]


def test_arrive_carries_feeling(clock, tmp_path):  # spec §4
    b, env, _, events, led = make(clock, tmp_path)
    led.present(["懒洋洋大王"], WALL0 - 86400 * 2)
    led.cards["懒洋洋大王"].feeling = dict(FEEL)
    led.cards["懒洋洋大王"].last_seen = WALL0 - 86400 * 2
    b.mind.want_note = lambda name: "。你惦记着：考试"
    (text,) = _arrive(b, env, events)
    assert "。你对他：嘴贫但靠谱（熟了，最近平稳）" in text
    assert text.index("你对他") < text.index("你惦记着")


def test_arrive_soft_only_close(clock, tmp_path):
    b, env, _, events, led = make(clock, tmp_path)
    led.present(["懒洋洋大王"], WALL0 - 86400 * 2)
    led.cards["懒洋洋大王"].feeling = dict(FEEL)
    led.cards["懒洋洋大王"].last_seen = WALL0 - 86400 * 2
    b._soft_until["懒洋洋大王"] = (WALL0 + 600, "难过")
    (text,) = _arrive(b, env, events)
    assert "。你对他：（熟了）" in text and "嘴贫但靠谱" not in text


def test_arrive_unchanged_when_off(clock, tmp_path):
    b, env, _, events, led = make(clock, tmp_path, feelings=False)
    led.present(["懒洋洋大王"], WALL0 - 86400 * 2)
    led.cards["懒洋洋大王"].feeling = dict(FEEL)
    led.cards["懒洋洋大王"].last_seen = WALL0 - 86400 * 2
    (text,) = _arrive(b, env, events)
    assert "你对他" not in text


def test_status_feel_bit(clock, tmp_path):
    b, env, _, events, led = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]
    b.step()
    led.cards["懒洋洋大王"].feeling = dict(FEEL)
    assert "·熟了）" in b.status()
    b.cfg.inner.feelings = False
    assert "熟了" not in b.status()


def test_introspect_relations(clock, tmp_path):
    b, env, _, events, led = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]
    b.step()
    led.cards["懒洋洋大王"].feeling = dict(FEEL)
    assert "嘴贫但靠谱" in b.introspect("关系")
    b.cfg.inner.feelings = False
    assert "没开" in b.introspect("关系")


def test_feeling_context_met_and_soft(clock, tmp_path):
    b, env, reader, _, _ = make(clock, tmp_path)
    reader.batches = [[msg("我今天好难过", speaker="阿花")]]
    env.near = ["懒洋洋大王"]
    b.step()
    ctx = b.feeling_context()
    assert ctx["met"] >= {"懒洋洋大王", "阿花"} and "阿花" in ctx["soft"] and ctx["today"]
