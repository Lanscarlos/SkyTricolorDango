import json
from types import SimpleNamespace

import pytest

from skydango.brain.toolloop import BLIND_NOTE, ToolLoopBrain
from skydango.models.errors import ModelError


def msg(content=None, tool_calls=None, usage=None):
    """usage=(prompt, completion)：FakeClient 把它放到响应的 .usage 上；不给就没有 usage 属性。"""
    return SimpleNamespace(content=content, tool_calls=tool_calls, fake_usage=usage)


def tool_call(name, arguments):
    return SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=arguments))


def ok(content="好"):
    return msg(content=content)


def _args(a):
    return a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)


def calls_then_ok(calls):
    return [msg(content=None, tool_calls=[tool_call(name, _args(a))]) for (name, a) in calls] + [ok("好")]


class FakeClient:
    def __init__(self, script):
        self.script = list(script)

    def create(self, **kw):
        item = self.script.pop(0) if self.script else ok()
        if isinstance(item, Exception):
            raise item
        resp = SimpleNamespace(choices=[SimpleNamespace(message=item)])
        if getattr(item, "fake_usage", None) is not None:
            resp.usage = SimpleNamespace(prompt_tokens=item.fake_usage[0], completion_tokens=item.fake_usage[1])
        return resp


def client(script):
    if hasattr(script, "chat"):
        return script
    if isinstance(script, Exception) or not isinstance(script, (list, tuple)):
        script = [script]
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=FakeClient(script).create)))


def always_calls():
    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=msg(content=None, tool_calls=[tool_call("say", '{"text":"你好"}')]))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def toolbox(ret=None):
    calls = []

    def run(name, args):
        calls.append((name, args))
        return ret if ret is not None else ("好", False)

    return SimpleNamespace(calls=calls, run=run)


def say_schema():
    return {"type": "function", "function": {"name": "say", "description": "说一句",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}


def test_no_tool_calls_returns_text():
    brain = ToolLoopBrain(client(ok("好")), "sys", toolbox(), [], model="deepseek-chat", temperature=0.8, max_tokens=4096)
    assert brain.send("在吗") == {"result": "好", "subtype": "success", "num_turns": 1, "usage": None,
                                   "provider": "deepseek", "model": "deepseek-chat"}


def test_single_tool_call_then_text():
    tb = toolbox()
    brain = ToolLoopBrain(client(calls_then_ok([("say", {"text": "你好"})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    r = brain.send("在吗")
    assert r["result"] == "好" and tb.calls == [("say", {"text": "你好"})]


def test_iteration_cap():   # 模型不停调工具：到 max_steps+2 轮停
    brain = ToolLoopBrain(client(always_calls()), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, max_steps=2)
    r = brain.send("在吗")
    assert r["subtype"] == "success"


def test_bad_json_arguments_fed_back_as_error():  # Review Focus 1
    tb = toolbox()
    brain = ToolLoopBrain(client(calls_then_ok([("say", "not json")])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    assert brain.send("在吗")["subtype"] == "success"  # 不崩，把解析错误喂回模型


def test_unknown_tool_not_executed():  # Review Focus 2：模型硬调 look
    tb = toolbox()
    brain = ToolLoopBrain(client(calls_then_ok([("look", {"image": False})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    brain.send("在吗")
    assert tb.calls == []  # 没执行，回"没有这个工具"


def test_non_string_result_serialized():  # Review Focus 3
    tb = toolbox(ret=([{"type": "image", "source": {}}], False))
    brain = ToolLoopBrain(client(calls_then_ok([("say", {})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    brain.send("在吗")  # 不崩；tool 消息 content 被 json.dumps 成字符串


def test_api_error_raises_not_limit():
    brain = ToolLoopBrain(client(Exception("500")), "sys", toolbox(), [], model="deepseek-chat", temperature=0.8, max_tokens=4096)
    with pytest.raises(ModelError) as e:
        brain.send("在吗")
    assert e.value.down is None and e.value.provider == "deepseek"


def test_turn_timeout():
    # deadline 已经是过去（turn_timeout=0）：循环开头就该抛超时，不等 create
    brain = ToolLoopBrain(client(always_calls()), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, turn_timeout=0.0, clock=lambda: 0.0)
    with pytest.raises(ModelError):
        brain.send("在吗")


def test_fallback_note_nonempty():
    assert "看不到画面" in BLIND_NOTE


def test_on_message_records_tool_calls():
    msgs = []
    tb = toolbox()
    brain = ToolLoopBrain(client(calls_then_ok([("say", {"text": "你好"})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, on_message=msgs.append)
    brain.send("在吗")
    assert [m["type"] for m in msgs] == ["assistant", "user"]
    tool_use = msgs[0]["message"]["content"][0]
    assert tool_use["type"] == "tool_use" and tool_use["name"] == "mcp__sky__say" and tool_use["input"] == {"text": "你好"}
    tool_result = msgs[1]["message"]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["content"] == "好" and tool_result["is_error"] is False


def test_on_message_reports_unknown_tool_as_error():
    msgs = []
    tb = toolbox()
    brain = ToolLoopBrain(client(calls_then_ok([("look", {"image": False})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, on_message=msgs.append)
    brain.send("在吗")
    tool_result = msgs[1]["message"]["content"][0]
    assert tool_result["content"] == "没有这个工具：look" and tool_result["is_error"] is True


def test_text_alongside_tool_calls_goes_to_timeline():
    msgs = []
    script = [msg(content="我想想", tool_calls=[tool_call("say", '{"text":"你好"}')]), ok("好")]
    brain = ToolLoopBrain(client(script), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, on_message=msgs.append)
    brain.send("在吗")
    assert [m["type"] for m in msgs] == ["assistant", "assistant", "user"]
    assert msgs[0]["message"]["content"] == [{"type": "text", "text": "我想想"}]
    assert msgs[1]["message"]["content"][0]["type"] == "tool_use"


def test_usage_is_summed_over_requests():
    script = [msg(content=None, tool_calls=[tool_call("say", '{"text":"你好"}')], usage=(100, 10)), msg(content="好", usage=(120, 5))]
    brain = ToolLoopBrain(client(script), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    r = brain.send("在吗")
    assert r["usage"] == {"input_tokens": 220, "output_tokens": 15}
    assert r["num_turns"] == 2 and r["subtype"] == "success" and r["result"] == "好"


def test_usage_none_when_client_gives_none():
    brain = ToolLoopBrain(client([ok("好")]), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    r = brain.send("在吗")
    assert r["usage"] is None and r["num_turns"] == 1


# ---- 短期记忆（10-03 晚：每轮只发 [system, 这一轮]，同一句话跨运行说 4 遍、隔 8 秒说两遍）----
class Recording:
    """记下每次请求的 messages；按 script 依次回。"""

    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **kw):
        self.requests.append([dict(m) for m in kw["messages"]])
        item = self.script.pop(0) if self.script else ok("")
        return SimpleNamespace(choices=[SimpleNamespace(message=item)])

    def client(self):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))


def emote_schema():
    return {"type": "function", "function": {"name": "emote", "description": "做动作",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}


def brain_with(rec, tb=None, **kw):
    return ToolLoopBrain(rec.client(), "sys", tb or toolbox(), [say_schema(), emote_schema()],
                         model="deepseek-chat", temperature=0.8, max_tokens=4096, **kw)


def test_next_turn_sees_what_it_said_last_turn():
    rec = Recording(calls_then_ok([("say", {"text": "卡洛回来啦"}), ("emote", {"name": "鞠躬"})]) + [ok("好")])
    brain = brain_with(rec, history=8)
    brain.send("卡洛：我回来了")
    brain.send("卡洛：嗯")
    second = rec.requests[-1]
    assert second[0] == {"role": "system", "content": "sys"} and second[-1] == {"role": "user", "content": "卡洛：嗯"}
    assert second[1] == {"role": "user", "content": "卡洛：我回来了"}
    assert second[2]["role"] == "assistant" and "说了「卡洛回来啦」" in second[2]["content"] and "鞠躬" in second[2]["content"]
    assert len(second) == 4


def test_history_keeps_last_n_turns():
    rec = Recording([])
    brain = brain_with(rec, history=2)
    for k in range(4):
        brain.send(f"第{k}轮")
    last = rec.requests[-1]
    assert [m["content"] for m in last if m["role"] == "user"] == ["第1轮", "第2轮", "第3轮"]


def test_history_truncates_long_wake_messages_and_respects_budget():
    from skydango.brain.toolloop import HISTORY_CHARS, HISTORY_TEXT

    rec = Recording([])
    brain = brain_with(rec, history=50)
    for k in range(40):
        brain.send(f"{k}:" + "状" * 3000)
    last = rec.requests[-1]
    past = last[1:-1]
    assert all(len(m["content"]) <= HISTORY_TEXT + 20 for m in past if m["role"] == "user")
    assert sum(len(m["content"]) for m in past) <= HISTORY_CHARS
    assert past and past[-2]["content"].startswith("38:")  # 留下的是最近的


def test_history_zero_is_old_behaviour():
    rec = Recording([])
    brain = brain_with(rec, history=0)
    brain.send("一")
    brain.send("二")
    assert rec.requests[-1] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "二"}]


def test_blocked_say_and_other_tools_are_summarised():
    tb = SimpleNamespace(calls=[], run=lambda name, args: ("你刚说过「晚安」，换个说法或者不说", True) if name == "say"
                         else ("没找到相关的聊天", False))
    recall = {"type": "function", "function": {"name": "recall", "description": "翻",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}
    rec = Recording(calls_then_ok([("say", {"text": "晚安"}), ("recall", {"query": "上周日"})]) + [ok("")])
    brain = ToolLoopBrain(rec.client(), "sys", tb, [say_schema(), recall], model="m", temperature=0.8, max_tokens=4096, history=8)
    brain.send("a")
    brain.send("b")
    summary = rec.requests[-1][2]["content"]
    assert "想说「晚安」被拦下" in summary and "recall" in summary and "没找到" in summary


def test_failed_turn_is_not_remembered():
    rec = Recording([])
    calls = {"n": 0}

    def create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("网络断了")
        return rec.create(**kw)

    brain = ToolLoopBrain(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "sys", toolbox(),
                          [say_schema()], model="m", temperature=0.8, max_tokens=4096, history=8)
    with pytest.raises(ModelError):
        brain.send("一")
    brain.send("二")
    assert rec.requests[-1] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "二"}]


def test_fallback_note_rules_from_10_03_night():
    assert "查不到" in BLIND_NOTE and "emote" in BLIND_NOTE and "刚说过" in BLIND_NOTE


def test_aside_note():
    from skydango.brain.toolloop import ASIDE_NOTE

    assert "跟别人说" in ASIDE_NOTE and ASIDE_NOTE == "\n- 标着“跟别人说”的话默认不接"


def _status_error(status, code=None):
    import openai

    try:  # openai 3.x 换成了 httpx2
        import httpx2 as httpx
    except ImportError:
        import httpx
    body = {"error": {"code": code}} if code else None
    return openai.APIStatusError("boom", response=httpx.Response(status, request=httpx.Request("POST", "http://x")), body=body)


def test_sdk_402_is_limit_down():
    brain = ToolLoopBrain(client(_status_error(402)), "sys", toolbox(), [], model="deepseek-chat", temperature=0.8,
                          max_tokens=4096, provider="deepseek")
    with pytest.raises(ModelError) as e:
        brain.send("在吗")
    assert e.value.down == "limit" and e.value.provider == "deepseek"


def test_result_has_provider_and_model():
    out = ToolLoopBrain(client(ok("好")), "sys", toolbox(), [], model="deepseek-chat", temperature=0.8, max_tokens=4096,
                        provider="deepseek").send("在吗")
    assert out["provider"] == "deepseek" and out["model"] == "deepseek-chat"


def _full_toolbox():
    """make_session 要给 openai_tools 用：有 body.env、calling、backstage。"""
    tb = toolbox()
    tb.body, tb.calling, tb.backstage = SimpleNamespace(env=object()), False, False
    return tb


def test_blind_note_in_system(tmp_path):
    from skydango.brain.sessions import make_session
    from skydango.config import BrainConfig, Config
    from skydango.models.config import ModelRef, resolve
    from skydango.models.gate import ProviderGates
    from skydango.models.registry import Registry

    reg = Registry(resolve(Config()), ProviderGates(), tmp_path, environ={"DEEPSEEK_API_KEY": "k"})
    s = make_session(reg, ModelRef("deepseek", "deepseek-chat"), prompt="prompt", toolbox=_full_toolbox(), mcp_url="http://x",
                     workdir=tmp_path, cfg=BrainConfig(), addressee=False, on_message=None)
    assert isinstance(s, ToolLoopBrain) and s.provider == "deepseek" and s.model == "deepseek-chat"
    assert s.system.startswith("prompt\n\n" + BLIND_NOTE.splitlines()[0]) and "Claude 额度不足" not in s.system
    assert s.max_tokens == 4096 and s.history == BrainConfig().history


def test_make_session_claude_code(tmp_path, monkeypatch):
    from skydango.brain.session import BrainSession
    from skydango.brain.sessions import make_session
    from skydango.config import BrainConfig, Config
    from skydango.models import claude_code
    from skydango.models.config import ModelRef, resolve
    from skydango.models.gate import ProviderGates
    from skydango.models.registry import Registry

    monkeypatch.setattr(claude_code, "resolve_claude", lambda path: ["claude"])
    reg = Registry(resolve(Config()), ProviderGates(), tmp_path, environ={"SKYDANGO_CLAUDE_TOKEN": "t"})
    s = make_session(reg, ModelRef("claude", "sonnet"), prompt="prompt", toolbox=_full_toolbox(), mcp_url="http://x",
                     workdir=tmp_path, cfg=BrainConfig(), addressee=False, on_message=None)
    assert isinstance(s, BrainSession) and s.provider == "claude" and s.model == "sonnet"
