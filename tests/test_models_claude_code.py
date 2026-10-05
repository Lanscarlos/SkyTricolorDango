import json
import sys
from pathlib import Path

import pytest

from skydango.models import claude_code
from skydango.models.claude_code import ClaudeCodeBackend, claude_base, claude_command, claude_env
from skydango.models.config import ProviderConfig
from skydango.models.errors import ModelError

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]
PROVIDER = ProviderConfig("claude", "claude-code", models=("haiku",), vision=("haiku",))


def _backend(tmp_path, mode="ok"):
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE=mode, FAKE_CLAUDE_LOG=str(tmp_path / "log.jsonl"))
    return ClaudeCodeBackend(PROVIDER, "haiku", tmp_path / "w", base=FAKE, env=env)


def _sent(tmp_path):
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    return [l["message"] for l in lines if "message" in l]


def test_command_has_model_and_system():
    cmd = claude_command(["claude"], "haiku", "S")
    assert cmd[cmd.index("--model") + 1] == "haiku" and cmd[cmd.index("--system-prompt") + 1] == "S"
    assert cmd[cmd.index("--effort") + 1] == "low" and "--no-session-persistence" not in cmd
    cmd = claude_command(["claude"], "sonnet", "S", persist=False)
    assert "--no-session-persistence" in cmd and "--effort" not in cmd


def test_message_returns_result_and_usage(tmp_path):
    out = _backend(tmp_path).message("S", "看")
    assert out == {"result": "收到：看", "usage": {"input_tokens": 10, "output_tokens": 2}, "provider": "claude", "model": "haiku"}


def test_history_joined(tmp_path):
    _backend(tmp_path).message("S", "b", history=[{"role": "user", "content": "a"}])
    assert _sent(tmp_path) == ["a\n\nb"]


def test_limit_error_carries_provider(tmp_path):
    with pytest.raises(ModelError) as e:
        _backend(tmp_path, "limit").message("S", "看")
    assert e.value.down == "limit" and e.value.provider == "claude"


def test_missing_token_is_auth_down(monkeypatch):
    monkeypatch.setattr(claude_code, "_user_env", lambda name: "")
    with pytest.raises(ModelError) as e:
        claude_base(PROVIDER, environ={})
    assert e.value.down == "auth" and e.value.provider == "claude" and "setup-token" in str(e.value)
