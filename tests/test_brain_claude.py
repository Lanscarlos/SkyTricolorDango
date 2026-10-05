import json
import sys
import threading
from pathlib import Path

import pytest

from skydango.models.gate import ProviderGates
from skydango.brain.claude import (ClaudeError, StreamProcess, check_result, claude_down,
                                   claude_env, one_shot, resolve_claude)

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]
IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}


def fake_env(tmp_path, mode="ok"):
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE=mode, FAKE_CLAUDE_LOG=str(tmp_path / "log.jsonl"))
    return env


def log_lines(tmp_path):
    return [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_claude_env_isolates_from_user_login(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-oat01-bad")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    env = claude_env("tok", tmp_path)
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok" and env["CLAUDE_CONFIG_DIR"] == str(tmp_path.resolve())


def test_claude_env_drops_endpoint_and_model_overrides(monkeypatch, tmp_path):
    # 10-03 晚：起团子的会话把 ANTHROPIC_BASE_URL 指到 DeepSeek，一次性 claude -p 拿着 Claude 令牌去打 DeepSeek，全部 401
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
              "ANTHROPIC_CUSTOM_HEADERS", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("SKYDANGO_KEEP_ME", "1")
    env = claude_env("tok", tmp_path)
    assert not [k for k in env if k.upper().startswith("ANTHROPIC_")]
    assert "CLAUDE_CODE_USE_BEDROCK" not in env and "CLAUDE_CODE_USE_VERTEX" not in env
    assert env["SKYDANGO_KEEP_ME"] == "1" and env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok"


def test_resolve_claude_reports_missing():
    with pytest.raises(RuntimeError, match="找不到 Claude Code"):
        resolve_claude("definitely-not-a-command-xyz")


def test_round_trip_text_and_image(tmp_path):
    p = StreamProcess(FAKE, fake_env(tmp_path), tmp_path / "work")
    seen = []
    p.send("你好")
    assert check_result(p.until_result(10, seen.append)) == "收到：你好"
    assert [m["type"] for m in seen] == ["system", "assistant", "result"]
    p.send([{"type": "text", "text": "看图"}, IMG])
    assert check_result(p.until_result(10)) == "收到：看图"
    p.close()
    assert not p.alive()
    lines = log_lines(tmp_path)
    assert lines[0]["token"] == "tok" and lines[0]["api_key"] is None
    assert lines[2] == {"message": "看图", "images": 1}


def test_timeout_and_death(tmp_path):
    hang = StreamProcess(FAKE, fake_env(tmp_path, "hang"), tmp_path / "w1")
    hang.send("x")
    with pytest.raises(ClaudeError, match="没有结果"):
        hang.until_result(0.5)
    hang.kill()
    assert not hang.alive()
    die = StreamProcess(FAKE, fake_env(tmp_path, "die"), tmp_path / "w2")
    die.send("x")
    with pytest.raises(ClaudeError, match="退出了"):
        die.until_result(10)


def test_limit_is_flagged(tmp_path):
    p = StreamProcess(FAKE, fake_env(tmp_path, "limit"), tmp_path / "w")
    p.send("x")
    with pytest.raises(ClaudeError) as err:
        check_result(p.until_result(10))
    assert err.value.limit
    p.close()


def test_one_shot(tmp_path):
    assert one_shot(FAKE, fake_env(tmp_path), tmp_path / "w", [{"type": "text", "text": "描述"}, IMG], 10) == "收到：描述"


def test_one_shot_message_returns_full_result(tmp_path):
    from skydango.brain.claude import one_shot_message

    m = one_shot_message(FAKE, fake_env(tmp_path), tmp_path / "w", [{"type": "text", "text": "描述"}], 10)
    assert m["result"] == "收到：描述" and m["usage"]["input_tokens"] == 10


# ---- 记忆整理用的 Claude（和 DeepSeek 的 LlmClient 同一个接口） ----
def test_check_result_recognises_auth_errors():
    for m in ({"subtype": "error", "is_error": True, "result": "x", "api_error_status": 401},
              {"subtype": "success", "is_error": True, "result": "401 Authentication Fails, Your api key: ****gwAA is invalid"},
              {"subtype": "success", "is_error": True, "result": "OAuth token has expired"}):
        with pytest.raises(ClaudeError) as e:
            check_result(m)
        assert e.value.auth and not e.value.limit and claude_down(e.value) == "auth"


def test_check_result_limit_and_other():
    with pytest.raises(ClaudeError) as e:
        check_result({"subtype": "error", "is_error": True, "result": "x", "api_error_status": 429})
    assert claude_down(e.value) == "limit"
    assert claude_down(ClaudeError("超时")) is None and claude_down(RuntimeError("x")) is None


def test_gate_trips_once_and_stays_closed(caplog):
    g = ProviderGates()
    assert g.ok("claude") and g.reason("claude") is None
    assert g.trip("claude", "auth", "401 bad key") is True
    assert g.trip("claude", "limit", "429") is False
    assert not g.ok("claude") and g.reason("claude").startswith("认证失败")
    assert sum("claude 不能用了" in r.getMessage() for r in caplog.records) == 1


def test_gate_trip_from_many_threads_once(caplog):
    g = ProviderGates()
    ts = [threading.Thread(target=g.trip, args=("claude", "auth", "401")) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not g.ok("claude") and sum("claude 不能用了" in r.getMessage() for r in caplog.records) == 1

