import json
import sys
import types
from types import SimpleNamespace

import pytest

from skydango.brain.claude import ClaudeError
from skydango.brain.deepseek import FALLBACK_NOTE, DeepSeekBrain, build_client
from skydango.config import LlmConfig


def msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


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
        return SimpleNamespace(choices=[SimpleNamespace(message=item)])


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
    brain = DeepSeekBrain(client(ok("好")), "sys", toolbox(), [], model="deepseek-chat", temperature=0.8, max_tokens=4096)
    assert brain.send("在吗") == {"result": "好", "subtype": "success"}


def test_single_tool_call_then_text():
    tb = toolbox()
    brain = DeepSeekBrain(client(calls_then_ok([("say", {"text": "你好"})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    r = brain.send("在吗")
    assert r["result"] == "好" and tb.calls == [("say", {"text": "你好"})]


def test_iteration_cap():   # 模型不停调工具：到 max_steps+2 轮停
    brain = DeepSeekBrain(client(always_calls()), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, max_steps=2)
    r = brain.send("在吗")
    assert r["subtype"] == "success"


def test_bad_json_arguments_fed_back_as_error():  # Review Focus 1
    tb = toolbox()
    brain = DeepSeekBrain(client(calls_then_ok([("say", "not json")])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    assert brain.send("在吗")["subtype"] == "success"  # 不崩，把解析错误喂回模型


def test_unknown_tool_not_executed():  # Review Focus 2：模型硬调 look
    tb = toolbox()
    brain = DeepSeekBrain(client(calls_then_ok([("look", {"image": False})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    brain.send("在吗")
    assert tb.calls == []  # 没执行，回"没有这个工具"


def test_non_string_result_serialized():  # Review Focus 3
    tb = toolbox(ret=([{"type": "image", "source": {}}], False))
    brain = DeepSeekBrain(client(calls_then_ok([("say", {})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    brain.send("在吗")  # 不崩；tool 消息 content 被 json.dumps 成字符串


def test_api_error_raises_not_limit():
    brain = DeepSeekBrain(client(Exception("500")), "sys", toolbox(), [], model="deepseek-chat", temperature=0.8, max_tokens=4096)
    with pytest.raises(ClaudeError) as e:
        brain.send("在吗")
    assert e.value.limit is False


def test_turn_timeout():
    # deadline 已经是过去（turn_timeout=0）：循环开头就该抛超时，不等 create
    brain = DeepSeekBrain(client(always_calls()), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, turn_timeout=0.0, clock=lambda: 0.0)
    with pytest.raises(ClaudeError):
        brain.send("在吗")


def test_fallback_note_nonempty():
    assert "看不到画面" in FALLBACK_NOTE


def test_build_client(monkeypatch):
    made = {}
    mod = types.ModuleType("openai")
    mod.OpenAI = lambda **kw: made.update(kw) or object()
    monkeypatch.setitem(sys.modules, "openai", mod)
    monkeypatch.setenv("SKYDANGO_TEST_KEY", "k")
    build_client(LlmConfig(base_url="https://api.deepseek.com", api_key_env="SKYDANGO_TEST_KEY"))
    assert made["base_url"] == "https://api.deepseek.com" and made["max_retries"] == 0


def test_on_message_records_tool_calls():
    msgs = []
    tb = toolbox()
    brain = DeepSeekBrain(client(calls_then_ok([("say", {"text": "你好"})])), "sys", tb, tools=[say_schema()],
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
    brain = DeepSeekBrain(client(calls_then_ok([("look", {"image": False})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, on_message=msgs.append)
    brain.send("在吗")
    tool_result = msgs[1]["message"]["content"][0]
    assert tool_result["content"] == "没有这个工具：look" and tool_result["is_error"] is True
