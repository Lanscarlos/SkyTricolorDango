"""按 Q 喊一声：大脑工具 call、提示词、手动"喊一声"、接线（spec 2026-10-01-q-call §3）。"""

import asyncio

from test_brain_body import body
from test_brain_manual import here, manual_events
from test_brain_mcp import talk
from test_brain_tools import FakeBody, FakeEyes

from skydango import cli
from skydango.brain.calling import CallResult
from skydango.brain.manual import ManualControl
from skydango.brain.mcp_server import SkyServer
from skydango.brain.prompt import CALL_RULE, brain_prompt
from skydango.brain.tools import ACTIONS, TOOL_NAMES, ToolBox
from skydango.config import Config, ReplyConfig
from skydango.vision.people import CallSeen, Seen


class Env:
    def __init__(self, ready_after=1):
        self.polls = 0
        self.ready_after = ready_after

    def call_result(self, at):
        self.polls += 1
        if self.ready_after is not None and self.polls >= self.ready_after:
            return CallSeen(at, {"小明": Seen("右边", "远")}, ended=True)
        return None


class CallBody(FakeBody):
    def __init__(self, result=None, env=None):
        super().__init__()
        self.result = result or CallResult(100.0, "brain")
        self.env = env or Env()
        self.last_call = None

    def call_out(self, reason, *, live=False):
        self.calls.append(("call_out", reason, live))
        return self.result


def toolbox(b, **kw):
    tb = ToolBox(b, FakeEyes(), call=True, **kw)
    tb.sleep = lambda s: None
    return tb


def _names(tb):
    server = SkyServer(tb)
    server.start()
    try:
        names, _ = asyncio.run(talk(server.url, []))
    finally:
        server.stop()
    return names


def test_call_tool_registered_only_when_enabled():
    assert _names(ToolBox(FakeBody(), FakeEyes())) == TOOL_NAMES
    names = _names(toolbox(CallBody()))
    i = names.index("look_around")
    assert names[i + 1] == "call" and names[:i + 1] + names[i + 2:] == TOOL_NAMES
    assert "call" in ACTIONS


def test_call_tool_refused_is_error():
    b = CallBody(CallResult(100.0, "brain", refused="20 秒前刚喊过，等一会儿再喊"))
    assert toolbox(b).run("call", {}) == ("20 秒前刚喊过，等一会儿再喊", True)
    assert ToolBox(b, FakeEyes()).run("call", {})[1] is True  # 没开：没有这个工具


def test_call_tool_waits_for_window():
    b = CallBody(env=Env(ready_after=3))
    out, err = toolbox(b).run("call", {})
    assert not err and out == "喊了一声：认出 小明（右边·远）。" and b.env.polls == 3
    assert b.calls == [("call_out", "brain", False)]


def test_call_tool_owner_note_and_dry():
    out, _ = toolbox(CallBody(CallResult(100.0, "brain-owner"))).run("call", {})
    assert out.endswith("（主人命令模式）")
    out, err = toolbox(CallBody(CallResult(100.0, "brain", dry=True))).run("call", {})
    assert (out, err) == ("dry-run：没喊", False)


def test_call_tool_gives_up():
    out, err = toolbox(CallBody(env=Env(ready_after=None))).run("call", {})
    assert not err and "没等到结果" in out


def test_call_in_sandbox_missing():
    out, err = toolbox(CallBody(), sandbox=True).run("call", {})
    assert err and "沙盒里没有这个" in out


def test_prompt_call_rule_only_when_enabled():
    old = brain_prompt(ReplyConfig(), None)
    assert brain_prompt(ReplyConfig(), None, call=False) == old and CALL_RULE not in old
    new = brain_prompt(ReplyConfig(), None, call=True)
    assert CALL_RULE in new and new.index("look_around") < new.index(CALL_RULE) < new.index("## 移动")


def test_manual_call(clock):
    b, device, _, events = body(clock)
    b.call_out = lambda reason, live=False: CallResult(clock(), reason)
    b.env = Env()
    ctl = ManualControl(here(b), events=events)
    ctl.sleep = lambda s: None
    out = ctl.run("call", {})
    assert out == {"ok": True, "text": "喊了一声：认出 小明（右边·远）。"}
    [e] = manual_events(events)
    assert "喊了一声" in e.text


def test_manual_call_refused(clock):
    b, device, _, events = body(clock)
    b.call_out = lambda reason, live=False: CallResult(clock(), reason, refused="画面黑着")
    out = ManualControl(here(b), events=events).run("call", {})
    assert out == {"ok": False, "text": "画面黑着"} and manual_events(events) == []


def test_manual_options_call(clock):
    b, _, _, _ = body(clock)
    assert ManualControl(b).options()["call"] is False  # 假 env 没有感知层
    b.env = type("P", (), {"unnamed": lambda self, now: 0, "call_result": lambda self, at: None})()
    assert ManualControl(b).options()["call"] is True
    b.cfg.call.enabled = False
    assert ManualControl(b).options()["call"] is False


def test_console_page_has_call_button():  # viewer 网页删了（spec 2026-10-04-console-attach），按钮在管理面板「真机团子」页
    import importlib.resources

    static = importlib.resources.files("skydango.console") / "static"
    assert 'id="lc-call"' in (static / "console.html").read_text(encoding="utf-8")
    assert 'send("call"' in (static / "livectl.js").read_text(encoding="utf-8")


def test_call_enabled_needs_perception():
    cfg = Config()

    class Perception:
        def unnamed(self, now):
            return 0

    assert cli._call_enabled(cfg, Perception()) is True
    assert cli._call_enabled(cfg, object()) is False  # 整图 OCR / 沙盒：收不到呼喊窗口
    cfg.call.enabled = False
    assert cli._call_enabled(cfg, Perception()) is False
