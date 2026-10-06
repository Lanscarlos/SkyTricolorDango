"""cli 改从 Registry 拿模型（spec 2026-10-05-model-providers §2.5）。"""

import logging
from types import SimpleNamespace

import pytest

from skydango import cli
from skydango.brain.toolloop import ToolLoopBrain
from skydango.models import claude_code, openai_compat
from skydango.models.errors import ModelError
from test_cli_brain import fake_brain_run


def _no_claude(monkeypatch):
    def refuse(provider, environ=None):
        raise ModelError("没有 Claude 令牌：先运行 claude setup-token", down="auth", provider=provider.id)

    monkeypatch.setattr(claude_code, "claude_base", refuse)


def _fake_openai(monkeypatch):
    def create(**kw):
        msg = SimpleNamespace(content="好", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), with_options=lambda **o: client)
    monkeypatch.setattr(openai_compat, "build_client", lambda provider, **kw: client)
    return client


def _default_models(cfg):
    cfg.models = {}
    cfg.sources = {}
    return cfg


def test_run_brain_all_deepseek_without_claude_token(tmp_path, monkeypatch, caplog):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    _default_models(cfg)
    _no_claude(monkeypatch)
    _fake_openai(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    seen = []
    with caplog.at_level(logging.WARNING):
        cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=seen.append)
    assert isinstance(seen[0].brain.session, ToolLoopBrain) and seen[0].brain.session.provider == "deepseek"
    assert any("eyes" in r.getMessage() and "Claude 令牌" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_run_brain_no_model_raises(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    _default_models(cfg)
    _no_claude(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    with pytest.raises(RuntimeError, match="大脑没有能用的模型"):
        cli._run_brain(cfg, run, no_emotes=True, duration=1.0)


def test_echo_flag_uses_echo_reply(monkeypatch, capsys):
    from skydango.config import Config

    lines = iter(["小明：在吗", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    cli.cmd_chat(Config(), SimpleNamespace(echo=True, emotes=""))
    assert "收到：在吗" in capsys.readouterr().out


def _record_calls(monkeypatch):
    from skydango.models.registry import Registry

    uses = []

    class FakeCall:
        def __init__(self, use):
            self.use = use

        def available(self):
            return True

        registry = SimpleNamespace(current=lambda use: None)

        def message(self, system, content, **kw):
            raise SystemExit("记下用处就停")

    def call(self, use, **kw):
        uses.append(use)
        return FakeCall(use)

    monkeypatch.setattr(Registry, "call", call)
    return uses


def test_addressee_label_uses_text_label(tmp_path, monkeypatch):
    from skydango.chat import addressee_eval as ae

    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    uses = _record_calls(monkeypatch)
    seen = {}

    def label(lines, out, run_fn, **kw):
        seen["run"] = run_fn
        return {}

    monkeypatch.setattr(ae, "claude_labels", lambda items, lines, run_fn, cache: label(lines, cache, run_fn))
    monkeypatch.setattr(cli, "_addressee_setup", lambda cfg: (["小明", "阿花"], {}, ["团子"], 30.0))
    monkeypatch.setattr(ae, "read_log", lambda path, name: [])
    monkeypatch.setattr(ae, "pick", lambda lines, names: [])
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "agent.log").write_text("", encoding="utf-8")
    try:
        cli.main(["addressee", "label", str(tmp_path / "r"), "--out", str(tmp_path / "o")])
    except SystemExit:
        pass
    assert uses == ["text_label"]


def test_gesture_label_uses_image_label(tmp_path, monkeypatch):
    from model_seams import fake_claude

    fake_claude(monkeypatch, ["claude"])
    uses = _record_calls(monkeypatch)
    import cv2
    import numpy as np

    ds = tmp_path / "ds" / "_unlabeled" / "clip1"
    ds.mkdir(parents=True)
    for i in range(16):
        cv2.imwrite(str(ds / f"{i:02d}.jpg"), np.zeros((32, 32, 3), np.uint8))
    from skydango.config import Config

    cfg = Config()
    cfg.gesture.dataset = str(tmp_path / "ds")
    try:
        cli.cmd_perception(cfg, SimpleNamespace(action="gesture-label", source=None, recheck=False, blind=False))
    except SystemExit:
        pass
    assert uses == ["image_label"]



def test_registry_offline_meter(tmp_path, monkeypatch):   # spec 2026-10-06-model-usage §2.3 §4
    from skydango.config import Config

    monkeypatch.delenv("SKYDANGO_USAGE_LEDGER")
    monkeypatch.setattr("atexit.register", lambda fn: None)
    cfg = Config()
    cfg.usage.ledger = str(tmp_path / "usage.json")
    reg = cli._registry(cfg, tmp_path / "w", environ={})
    assert reg.meter.source == "offline" and reg.meter.ledger.path == tmp_path / "usage.json"
    reg2 = cli._registry(cfg, tmp_path / "w", environ={}, source="live")
    assert reg2.meter.source == "live"



def test_registry_no_ledger_in_tests(tmp_path):
    from skydango.config import Config

    assert cli._registry(Config(), tmp_path, environ={}).meter.ledger is None


def test_start_balance_starts_watcher(monkeypatch, tmp_path):   # spec 2026-10-06-model-usage §5.1

    from skydango.config import Config
    from skydango.models import balance as bal

    monkeypatch.undo()  # conftest 把 _start_balance 换成了空的：这里测真的
    started = []
    monkeypatch.setattr(bal.BalanceWatcher, "start", lambda self: started.append(self.every))
    reg = cli._registry(Config(), tmp_path, environ={})
    w = cli._start_balance(Config(), reg)
    assert started == [300.0] and w.meter is reg.meter
