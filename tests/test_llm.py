import sys
import types

import pytest

from skydango.chat.llm import AnthropicClient, make_llm, read_key
from skydango.config import LlmConfig


class FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="好呀")])


@pytest.fixture
def fake_anthropic(monkeypatch):
    messages = FakeMessages()
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda **kw: types.SimpleNamespace(messages=messages)
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("SKYDANGO_TEST_KEY", "k")
    return messages


def llm_cfg(model):
    return LlmConfig(provider="anthropic", base_url="", model=model, api_key_env="SKYDANGO_TEST_KEY")


def test_sonnet_5_gets_no_temperature_and_thinking_off(fake_anthropic):
    reply = AnthropicClient(llm_cfg("claude-sonnet-5")).complete("sys", [{"role": "user", "content": "在吗"}])
    assert reply == "好呀"
    kw = fake_anthropic.kwargs
    assert "temperature" not in kw  # Sonnet 5 传了 temperature 返回 400
    assert kw["thinking"] == {"type": "disabled"}  # 不关的话默认思考，200 的 max_tokens 会被吃光
    assert kw["model"] == "claude-sonnet-5" and kw["max_tokens"] == 200


def test_older_models_keep_temperature(fake_anthropic):
    AnthropicClient(llm_cfg("claude-haiku-4-5")).complete("sys", [{"role": "user", "content": "在吗"}])
    kw = fake_anthropic.kwargs
    assert kw["temperature"] == 0.8 and "thinking" not in kw


def test_read_key_reports_missing_variable(monkeypatch):
    monkeypatch.delenv("SKYDANGO_NO_SUCH_KEY", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    with pytest.raises(RuntimeError, match="SKYDANGO_NO_SUCH_KEY"):
        read_key("SKYDANGO_NO_SUCH_KEY")


def test_max_retries_is_passed_to_sdk(monkeypatch):
    made = {}
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda **kw: made.update(kw) or types.SimpleNamespace(messages=FakeMessages())
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("SKYDANGO_TEST_KEY", "k")
    cfg = llm_cfg("claude-sonnet-5")
    cfg.max_retries = 0
    AnthropicClient(cfg)
    assert made["max_retries"] == 0


def test_openai_with_timeout_returns_shorter_copy(monkeypatch):  # 终审 M1：下线反思改走 DeepSeek 时缩短超时、不重试
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    client = make_llm(LlmConfig(timeout=90.0, max_retries=2), api_key="sk-given")
    short = client.with_timeout(35.0, max_retries=0)
    assert short is not client and short.cfg is client.cfg
    assert (short._client.timeout, short._client.max_retries) == (35.0, 0)
    assert (client._client.timeout, client._client.max_retries) == (90.0, 2)  # 原来的不动
    assert client.with_timeout(20.0)._client.max_retries == 2  # 不给就沿用


def test_anthropic_with_timeout(monkeypatch):
    made = []

    def build(**kw):
        c = types.SimpleNamespace(messages=FakeMessages(), kw=kw)
        c.with_options = lambda **o: types.SimpleNamespace(messages=c.messages, kw={**kw, **o})
        made.append(c)
        return c

    mod = types.ModuleType("anthropic")
    mod.Anthropic = build
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("SKYDANGO_TEST_KEY", "k")
    client = AnthropicClient(llm_cfg("claude-sonnet-5"))
    short = client.with_timeout(35.0, max_retries=0)
    assert short._client.kw["timeout"] == 35.0 and short._client.kw["max_retries"] == 0
    assert client._client is made[0]


def test_make_llm_uses_given_key(monkeypatch):
    # 管理面板「测试大模型」用页面上还没保存的 Key：不经过环境变量
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    client = make_llm(LlmConfig(), api_key="sk-given")
    assert client._client.api_key == "sk-given"
