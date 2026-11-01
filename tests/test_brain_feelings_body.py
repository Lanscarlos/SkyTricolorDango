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


def test_feeling_context_met_and_soft(clock, tmp_path):
    b, env, reader, _, _ = make(clock, tmp_path)
    reader.batches = [[msg("我今天好难过", speaker="阿花")]]
    env.near = ["懒洋洋大王"]
    b.step()
    ctx = b.feeling_context()
    assert ctx["met"] >= {"懒洋洋大王", "阿花"} and "阿花" in ctx["soft"] and ctx["today"]
