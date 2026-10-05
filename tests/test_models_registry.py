import pytest

from skydango.config import Config
from skydango.models.config import ModelRef, resolve
from skydango.models.errors import ModelError, ModelUnavailable
from skydango.models.gate import ProviderGates
from skydango.models.registry import Registry


class FakeBackend:
    def __init__(self, key):
        self.key, self.calls, self.exc = key, [], None

    def message(self, system, content, *, history=None, max_tokens=None, timeout=None):
        self.calls.append({"system": system, "content": content, "history": history, "max_tokens": max_tokens, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return {"result": f"{self.key[0]}:{content}", "usage": {}, "provider": self.key[0], "model": self.key[1]}


def _registry(tmp_path, cfg=None, environ=None):
    fakes: dict = {}

    def backends(provider, model, usecfg, cwd):
        return fakes.setdefault((provider.id, model), FakeBackend((provider.id, model)))

    reg = Registry(resolve(cfg or Config()), ProviderGates(), tmp_path, environ=environ or {}, backends=backends)
    return reg, fakes


def test_main_ok(tmp_path):
    reg, fakes = _registry(tmp_path)
    assert reg.call("brain").message("S", "x")["provider"] == "deepseek"
    assert ("claude", "sonnet") not in fakes


def test_main_down_switches_and_trips(tmp_path):
    reg, fakes = _registry(tmp_path)
    call = reg.call("memory")
    call.text("S", "x")
    fakes[("deepseek", "deepseek-chat")].exc = ModelError("402", down="limit", provider="deepseek")
    assert call.text("S", "y") == "claude:y"
    assert reg.gates.ok("deepseek") is False
    assert call.text("S", "z") == "claude:z"
    assert len(fakes[("deepseek", "deepseek-chat")].calls) == 2


def test_main_other_error_raises(tmp_path):
    reg, fakes = _registry(tmp_path)
    call = reg.call("memory")
    call.text("S", "x")
    fakes[("deepseek", "deepseek-chat")].exc = ModelError("超时")
    with pytest.raises(ModelError):
        call.text("S", "y")
    assert reg.gates.ok("deepseek") and ("claude", "sonnet") not in fakes


def test_no_backup_raises_unavailable_after_trip(tmp_path):
    reg, fakes = _registry(tmp_path)
    call = reg.call("reply")
    call.text("S", "x")
    err = ModelError("401", down="auth", provider="deepseek")
    fakes[("deepseek", "deepseek-chat")].exc = err
    with pytest.raises(ModelError) as e:
        call.text("S", "y")
    assert e.value is err
    with pytest.raises(ModelUnavailable):
        call.text("S", "z")


def test_disabled_use(tmp_path):
    cfg = Config()
    cfg.models = {"eyes": {"main": "deepseek/deepseek-chat"}}
    reg, _ = _registry(tmp_path, cfg)
    call = reg.call("eyes")
    assert call.available() is False
    with pytest.raises(ModelUnavailable):
        call.text("S", "x")


def test_complete_passes_history_and_default_max_tokens(tmp_path):
    reg, fakes = _registry(tmp_path)
    reg.call("memory").complete("S", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                      {"role": "user", "content": "c"}])
    got = fakes[("deepseek", "deepseek-chat")].calls[0]
    assert got["content"] == "c" and len(got["history"]) == 2 and got["max_tokens"] == 4096


def test_timeout_setter(tmp_path):
    reg, fakes = _registry(tmp_path)
    call = reg.call("reflect", timeout=60)
    call.timeout = 7
    call.text("S", "x")
    assert fakes[("deepseek", "deepseek-chat")].calls[0]["timeout"] == 7


def test_describe(tmp_path):
    reg, _ = _registry(tmp_path)
    assert reg.describe("brain") == "deepseek/deepseek-chat（备 claude/sonnet）"
    reg.gates.trip("deepseek", "limit", "402")
    assert reg.current("brain") == ModelRef("claude", "sonnet")


def test_requirements(tmp_path, monkeypatch):
    from skydango.chat import llm as chat_llm
    from skydango.models import claude_code

    monkeypatch.setattr(chat_llm, "_user_env", lambda name: "")
    monkeypatch.setattr(claude_code, "_user_env", lambda name: "")
    reg, _ = _registry(tmp_path, environ={})
    probs = reg.requirements(["brain"])
    assert {p.provider for p in probs} == {"deepseek", "claude"}
    reg, _ = _registry(tmp_path, environ={"DEEPSEEK_API_KEY": "k"})
    assert {p.provider for p in reg.requirements(["brain"])} == {"claude"}
