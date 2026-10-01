"""身体接上冷场（spec 2026-10-01-lull-musing §1~§3）：节点事件、走开 / 回来、附注、心里想的、状态、反思材料。"""

from conftest import FakeLlm
from test_brain_body import FakeEnv, body, msg

from skydango.brain.events import BACKGROUND
from skydango.config import Config
from skydango.inner.log import MindLog
from skydango.inner.mind import Mind
from skydango.inner.reflect import Reflector

FRIENDS = ["懒洋洋大王", "阿花"]


def lb(clock, lull=True, **kw):
    env = kw.pop("env", FakeEnv())
    env.near = ["懒洋洋大王"]
    b, dev, reader, events = body(clock, env=env, wall=clock, lull=lull, **kw)
    b.friend_names = lambda: FRIENDS
    b.step()  # 懒洋洋大王来了（arrive）：先消化掉
    events.drain()
    return b, env, reader, events


def chat_then_quiet(b, reader, events, clock, quiet=65):
    reader.batches = [[msg("在吗")]]
    b.step()
    b.say("在呢")
    events.drain()
    clock.advance(quiet)
    b.step()


def lulls(events):
    return [e for e in events.drain() if e.kind == "lull"]


def test_stage_event_is_urgent(clock):
    b, env, reader, events = lb(clock)
    chat_then_quiet(b, reader, events, clock)
    got = lulls(events)
    assert len(got) == 1 and got[0].text.startswith("冷场  懒洋洋大王 1 分钟没说话了。")
    assert "lull" not in BACKGROUND


def test_chat_after_lull_has_note(clock):
    b, env, reader, events = lb(clock)
    chat_then_quiet(b, reader, events, clock)
    b.mused("不说：等\n心里：他是不是去忙了")
    reader.batches = [[msg("刚去倒水")]]
    b.step()
    chats = [e.text for e in events.drain() if e.kind == "chat"]
    assert chats == ["聊天  懒洋洋大王：「刚去倒水」（冷场了 1 分钟，你刚才在想：他是不是去忙了）"]


def test_mused_outside_lull_ignored(clock):
    b, env, reader, events = lb(clock)
    got = []
    b.on_musing = got.append
    b.mused("心里：嗯")
    assert got == []


def test_on_musing_and_mind_log(clock):
    log = MindLog(None, False)
    b, env, reader, events = lb(clock, mind_log=log)
    got = []
    b.on_musing = got.append
    chat_then_quiet(b, reader, events, clock)
    b.mused("心里：他是不是去忙了")
    assert got == ["他是不是去忙了"]
    assert log.recent()[-1]["kind"] == "musing" and log.recent()[-1]["text"] == "他是不是去忙了"


def test_leave_becomes_lull(clock):
    b, env, reader, events = lb(clock)
    reader.batches = [[msg("我去拿个东西")]]
    b.step()
    events.drain()
    env.near = []
    b.step()
    assert [e.kind for e in events.drain()] == ["leave"]  # 先等一会儿：可能只是名字标签闪了一下
    clock.advance(15)
    b.step()
    assert [e.kind for e in events.drain()] == ["lull"]


def test_leave_flicker_stays_background(clock):  # 评审 #4
    b, env, reader, events = lb(clock)
    reader.batches = [[msg("我去拿个东西")]]
    b.step()
    events.drain()
    env.near = []
    b.step()
    clock.advance(5)
    env.near = ["懒洋洋大王"]
    b.step()
    assert events.drain() == []  # 走开和回来在队列里互相抵消


def test_musing_from_turn_before_lull_dropped(clock):  # 评审 #3
    b, env, reader, events = lb(clock)
    began = clock()
    chat_then_quiet(b, reader, events, clock)
    got = []
    b.on_musing = got.append
    b.mused("心里：嗯", began)
    assert got == []
    b.mused("心里：嗯", clock())
    assert got == ["嗯"]


def test_quiet_friend_leave_stays_background(clock):
    b, env, reader, events = lb(clock)
    env.near = []
    b.step()
    assert [e.kind for e in events.drain()] == ["leave"]


def test_return_becomes_lull(clock):
    b, env, reader, events = lb(clock)
    reader.batches = [[msg("我去拿个东西")]]
    b.step()
    env.near = []
    b.step()
    clock.advance(15)
    b.step()
    events.drain()
    clock.advance(30)
    env.near = ["懒洋洋大王"]
    b.step()
    got = lulls(events)
    assert len(got) == 1 and got[0].text.startswith("懒洋洋大王 回来了（走开了")


def test_status_has_lull(clock):
    b, env, reader, events = lb(clock)
    chat_then_quiet(b, reader, events, clock)
    assert "冷场：懒洋洋大王" in b.status()


def test_reflect_chat_gets_summary(clock):
    b, env, reader, events = lb(clock, mind=Mind())
    chat_then_quiet(b, reader, events, clock)
    reader.batches = [[msg("刚去倒水")]]
    b.step()
    assert any(who == "（冷场）" for _, who, _ in b._reflect_chat)
    assert any(who == "（冷场）" for _, who, _ in b._session_chat)


class StirReflector(Reflector):
    def __init__(self, clock):
        super().__init__(Config().inner, FakeLlm("{}"), clock, threaded=False)
        self.stirs = 0

    def stirred(self, now):
        self.stirs += 1
        super().stirred(now)


def test_final_stage_stirs_reflector(clock):
    reflector = StirReflector(clock)
    b, env, reader, events = lb(clock, mind=Mind(), reflector=reflector)
    chat_then_quiet(b, reader, events, clock, quiet=185)  # say() 自己也会 stirred：之后才开始数
    reflector.stirs = 0
    clock.advance(10)
    b.step()
    assert reflector.stirs == 0  # 不是最后一个节点
    clock.advance(170)
    b.step()
    assert reflector.stirs == 1


def test_disabled_unchanged(clock):
    b, env, reader, events = lb(clock, lull=False)
    assert b.lulls is None
    chat_then_quiet(b, reader, events, clock)
    assert lulls(events) == [] and "冷场" not in b.status()
    env.near = []
    b.step()
    assert [e.kind for e in events.drain()] == ["leave"]


def test_inner_snapshot_musing(clock):
    b, env, reader, events = lb(clock, mind=Mind())
    chat_then_quiet(b, reader, events, clock)
    b.mused("心里：他是不是去忙了")
    assert b.inner_snapshot()["musing"][0]["musings"][0]["text"] == "他是不是去忙了"


def test_final_reflection_flushes(clock):
    b, env, reader, events = lb(clock, mind=Mind())
    chat_then_quiet(b, reader, events, clock)
    b.reflect_materials(True)
    assert b.lulls.active() == []
    assert any(who == "（冷场）" and "到下线还没结束" in text for _, who, text in b._session_chat)
