import sys
import types

from skydango.brain.client import BrainClient
from skydango.config import BrainConfig


def test_create_sends_adaptive_thinking_effort_and_auto_cache(monkeypatch):
    calls = {}

    class Resp:
        def to_dict(self):
            return {"content": [{"type": "text", "text": "嗯"}], "stop_reason": "end_turn", "usage": {}}

    class Messages:
        def create(self, **kw):
            calls.update(kw)
            return Resp()

    made = {}
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda **kw: made.update(kw) or types.SimpleNamespace(messages=Messages())
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    out = BrainClient(BrainConfig()).create(
        [{"type": "text", "text": "s"}], [{"role": "user", "content": "hi"}], tools=[{"name": "look"}]
    )
    assert out["stop_reason"] == "end_turn"
    assert made["api_key"] == "k" and "base_url" not in made
    assert calls["model"] == "claude-sonnet-5" and calls["max_tokens"] == 4000
    assert calls["thinking"] == {"type": "adaptive"} and calls["output_config"] == {"effort": "low"}
    assert calls["cache_control"] == {"type": "ephemeral"}
    assert calls["tools"] == [{"name": "look"}] and "temperature" not in calls


def test_no_tools_key_when_none(monkeypatch):
    calls = {}
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda **kw: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: calls.update(kw) or types.SimpleNamespace(to_dict=lambda: {}))
    )
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    BrainClient(BrainConfig()).create([], [{"role": "user", "content": "hi"}], max_tokens=500)
    assert "tools" not in calls and calls["max_tokens"] == 500
