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
    fakes[("deepseek", "deepseek-flash")].exc = ModelError("402", down="limit", provider="deepseek")
    assert call.text("S", "y") == "claude:y"
    assert reg.gates.ok("deepseek") is False
    assert call.text("S", "z") == "claude:z"
    assert len(fakes[("deepseek", "deepseek-flash")].calls) == 2


def test_main_other_error_raises(tmp_path):
    reg, fakes = _registry(tmp_path)
    call = reg.call("memory")
    call.text("S", "x")
    fakes[("deepseek", "deepseek-flash")].exc = ModelError("超时")
    with pytest.raises(ModelError):
        call.text("S", "y")
    assert reg.gates.ok("deepseek") and ("claude", "sonnet") not in fakes


def test_no_backup_raises_unavailable_after_trip(tmp_path):
    reg, fakes = _registry(tmp_path)
    call = reg.call("reply")
    call.text("S", "x")
    err = ModelError("401", down="auth", provider="deepseek")
    fakes[("deepseek", "deepseek-flash")].exc = err
    with pytest.raises(ModelError) as e:
        call.text("S", "y")
    assert e.value is err
    with pytest.raises(ModelUnavailable):
        call.text("S", "z")


def test_disabled_use(tmp_path):
    cfg = Config()
    cfg.models = {"eyes": {"main": "deepseek/deepseek-chat", "backup": ""}}
    reg, _ = _registry(tmp_path, cfg)
    call = reg.call("eyes")
    assert call.available() is False
    with pytest.raises(ModelUnavailable):
        call.text("S", "x")


def test_complete_passes_history_and_default_max_tokens(tmp_path):
    reg, fakes = _registry(tmp_path)
    reg.call("memory").complete("S", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                      {"role": "user", "content": "c"}])
    got = fakes[("deepseek", "deepseek-flash")].calls[0]
    assert got["content"] == "c" and len(got["history"]) == 2 and got["max_tokens"] == 4096


def test_timeout_setter(tmp_path):
    reg, fakes = _registry(tmp_path)
    call = reg.call("reflect", timeout=60)
    call.timeout = 7
    call.text("S", "x")
    assert fakes[("deepseek", "deepseek-flash")].calls[0]["timeout"] == 7


def test_describe(tmp_path):
    reg, _ = _registry(tmp_path)
    assert reg.describe("brain") == "deepseek/deepseek-flash（备 claude/sonnet）"
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


def test_shortened_timeout_means_no_retries_on_main_too(tmp_path, monkeypatch):  # 终审 I2：下线反思主和备都不重试
    from types import SimpleNamespace

    from skydango.models import openai_compat

    made = []

    def build_client(provider, *, api_key=None, timeout=None, max_retries=None, environ=None):
        made.append(max_retries)

        def create(**kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="好"))], usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client.with_options = lambda **o: client
        return client

    monkeypatch.setattr(openai_compat, "build_client", build_client)
    reg = Registry(resolve(Config()), ProviderGates(), tmp_path, environ={"DEEPSEEK_API_KEY": "k"})
    call = reg.call("reflect", timeout=600)
    call.timeout = 35
    assert call.text("S", "x") == "好" and made == [0]


# ---- 用量（spec 2026-10-06-model-usage §2.2）----
from skydango.models.usage import UsageMeter  # noqa: E402


def _metered(tmp_path):
    reg, fakes = _registry(tmp_path)
    reg.meter = UsageMeter(reg.setup, reg.gates, source="live")
    return reg, fakes


def _rows(reg):
    return {(r["use"], r["model"], r["backup"]): r for r in reg.meter.snapshot()["run"]["rows"]}


def test_gated_call_records_main_and_backup(tmp_path):
    reg, fakes = _metered(tmp_path)
    call = reg.call("memory")
    fakes.setdefault(("deepseek", "deepseek-flash"), FakeBackend(("deepseek", "deepseek-flash")))
    orig = fakes[("deepseek", "deepseek-flash")].message

    def with_usage(*a, **kw):
        out = orig(*a, **kw)
        return {**out, "usage": {"input_tokens": 10, "output_tokens": 2}}

    fakes[("deepseek", "deepseek-flash")].message = with_usage
    call.message("S", "x")
    r = _rows(reg)[("memory", "deepseek/deepseek-flash", False)]
    assert (r["calls"], r["input"], r["output"]) == (1, 10, 2)
    fakes[("deepseek", "deepseek-flash")].exc = ModelError("402", down="limit", provider="deepseek")
    call.message("S", "y")
    r = _rows(reg)
    assert r[("memory", "deepseek/deepseek-flash", False)]["fails"] == 1
    assert r[("memory", "claude/sonnet", True)]["calls"] == 1


def test_gated_call_meter_error_ignored(tmp_path):
    reg, _ = _metered(tmp_path)

    def boom(*a, **kw):
        raise RuntimeError("坏了")

    reg.meter.record = boom
    assert reg.call("memory").message("S", "x")["result"] == "deepseek:x"


def test_backend_passes_rate_limit_to_meter(tmp_path, monkeypatch):
    from skydango.models import claude_code

    reg = Registry(resolve(Config()), ProviderGates(), tmp_path, environ={})
    reg.meter = UsageMeter(reg.setup, reg.gates, source="live")
    monkeypatch.setattr(claude_code, "claude_base", lambda provider, environ=None: (["claude"], {}))
    b = reg.backend(ModelRef("claude", "haiku"), "eyes", tmp_path)
    b.on_event({"status": "rejected"})
    assert next(p for p in reg.meter.snapshot()["providers"] if p["id"] == "claude")["rate"]["status"] == "rejected"


def test_log_summary_skips_follows(tmp_path, caplog):   # spec 2026-10-06-brain-compact §5：recap 跟着大脑，不单独报
    import logging

    reg, _ = _registry(tmp_path)
    with caplog.at_level(logging.INFO):
        reg.log_summary()
    assert "模型：brain" in caplog.text and "模型：recap" not in caplog.text
