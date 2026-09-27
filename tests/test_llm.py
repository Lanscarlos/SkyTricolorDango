import sys
import types

import pytest

from skydango.chat.llm import AnthropicClient, read_key
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
