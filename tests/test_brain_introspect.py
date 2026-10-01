"""幕后的 introspect 工具（spec 2026-10-01-backstage §2）。"""

import asyncio

from test_brain_body import body
from test_brain_inner_log_body import WALL, make
from test_brain_mcp import talk
from test_brain_tools import FakeBody, FakeEyes

from skydango.brain.mcp_server import SkyServer
from skydango.brain.tools import TOOL_NAMES, ToolBox
from skydango.config import Config
from skydango.inner.persona import Persona, Trait
from skydango.inner.reflect import Reflector


class IntrospectBody(FakeBody):
    def introspect(self, topic):
        self.calls.append(("introspect", topic))
        return f"查了{topic}"


def _names(tb):
    server = SkyServer(tb)
    server.start()
    try:
        names, _ = asyncio.run(talk(server.url, []))
    finally:
        server.stop()
    return names


def test_introspect_hidden_when_off():
    assert _names(ToolBox(FakeBody(), FakeEyes())) == TOOL_NAMES


def test_introspect_listed_when_on():
    tb = ToolBox(IntrospectBody(), FakeEyes(), backstage=True)
    server = SkyServer(tb)
    server.start()
    try:
        names, (res,) = asyncio.run(talk(server.url, [("introspect", {"topic": "精力"})]))
    finally:
        server.stop()
    assert names == TOOL_NAMES + ["introspect"]
    assert not res.is_error and res.content[0].text == "查了精力"


def test_introspect_off_rejects():
    out, err = ToolBox(IntrospectBody(), FakeEyes()).run("introspect", {"topic": "精力"})
    assert err


def test_introspect_bad_topic():
    out, err = ToolBox(IntrospectBody(), FakeEyes(), backstage=True).run("introspect", {"topic": "代码"})
    assert err and "精力 / 反思 / 性格 / 日记 / 眼睛" in out


def test_introspect_eyes():
    eyes = FakeEyes()
    tb = ToolBox(IntrospectBody(), eyes, backstage=True)
    out, err = tb.run("introspect", {"topic": "眼睛"})
    assert not err and "还没看过" in out
    eyes.latest = ("一棵树", 70.0)  # FakeBody.clock() == 100
    out, _ = tb.run("introspect", {"topic": "眼睛"})
    assert "30 秒前" in out and "一棵树" in out and "haiku" in out


def test_introspect_not_action():
    tb = ToolBox(IntrospectBody(), FakeEyes(), backstage=True)
    tb.run("introspect", {"topic": "精力"})
    assert tb.acted is False and tb.body.calls == [("introspect", "精力")]


def test_introspect_inner_off(clock):
    b, *_ = body(clock)
    assert "没开" in b.introspect("反思") and "没开" in b.introspect("性格") and "没开" in b.introspect("日记")
    assert "= " in b.introspect("精力") and "基础" in b.introspect("精力")


def test_introspect_reflect_rows(clock, tmp_path):
    b = make(clock, tmp_path)
    b.apply_reflection({"mood": {"level": "开心", "text": "有人来聊天"}, "grudge": "keep"})
    b.apply_reflection({"mood": {"level": "低落", "text": "没人理"}, "grudge": "keep"})
    out = b.introspect("反思")
    assert "心情 平常→开心（有人来聊天）" in out and "没人理" in out
    assert out.index("有人来聊天") < out.index("没人理")


def test_introspect_reflect_none_yet(clock, tmp_path):
    assert "还没反思过" in make(clock, tmp_path).introspect("反思")


def test_introspect_persona(clock, tmp_path):
    p = Persona(catchphrases=[Trait("害", since=WALL, last_used=WALL, hits=3)], jokes=[Trait("路痴", who="小明", since=WALL)])
    out = make(clock, tmp_path, persona=p).introspect("性格")
    assert "害" in out and "3 次" in out and "小明" in out and "路痴" in out


def test_introspect_persona_empty(clock, tmp_path):
    assert "还没攒下" in make(clock, tmp_path).introspect("性格")


def test_introspect_diary_truncated(clock, tmp_path):
    b = make(clock, tmp_path)
    assert "还没写过" in b.introspect("日记")
    b.ledger.store.append_diary("好" * 1000, WALL)
    out = b.introspect("日记")
    assert out.count("好") == 600


def test_reflector_next_in(clock):
    cfg = Config().inner
    r = Reflector(cfg, None, clock, threaded=False)
    r._last_run = 0.0
    assert r.next_in(200.0) == cfg.reflect_every - 200
    r._final = True
    assert r.next_in(200.0) is None
