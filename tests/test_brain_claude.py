import json
import sys
import threading
from pathlib import Path

import pytest

from skydango.brain.claude import (ClaudeError, ClaudeGate, GatedLlm, StreamProcess, check_result, claude_down,
                                   claude_env, gated_describe, one_shot, resolve_claude)

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
def test_claude_llm_complete(tmp_path):
    from skydango.brain.claude import ClaudeLlm

    llm = ClaudeLlm(FAKE, fake_env(tmp_path), "sonnet", tmp_path / "w", timeout=10)
    assert llm.complete("你负责记笔记", [{"role": "user", "content": "整理一下"}], max_tokens=200) == "收到：整理一下"
    args = log_lines(tmp_path)[0]["args"]
    assert args[args.index("--system-prompt") + 1] == "你负责记笔记"
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--tools") + 1] == ""  # 不给任何内置工具


def test_claude_llm_raises_on_limit(tmp_path):
    from skydango.brain.claude import ClaudeLlm

    llm = ClaudeLlm(FAKE, fake_env(tmp_path, "limit"), "sonnet", tmp_path / "w", timeout=10)
    with pytest.raises(ClaudeError) as err:
        llm.complete("s", [{"role": "user", "content": "x"}])
    assert err.value.limit


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
    g = ClaudeGate()
    assert g.ok() and g.reason is None
    assert g.trip("auth", "401 bad key") is True
    assert g.trip("limit", "429") is False
    assert not g.ok() and g.reason.startswith("认证失败")
    assert sum("Claude 不能用了" in r.getMessage() for r in caplog.records) == 1


def test_gate_trip_from_many_threads_once(caplog):
    g = ClaudeGate()
    ts = [threading.Thread(target=g.trip, args=("auth", "401")) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not g.ok() and sum("Claude 不能用了" in r.getMessage() for r in caplog.records) == 1


class _Fake:
    """假的 LLM：按脚本返回或抛，记调用次数。"""

    def __init__(self, result="ok", exc=None):
        self.result, self.exc, self.calls = result, exc, 0

    def complete(self, system, messages, max_tokens=None):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result


def test_gated_llm_uses_claude_while_open():
    claude, backup, gate = _Fake("来自claude"), _Fake("来自backup"), ClaudeGate()
    llm = GatedLlm(claude, backup, gate)
    assert llm.complete("s", [{"role": "user", "content": "x"}]) == "来自claude"
    assert (claude.calls, backup.calls) == (1, 0)
    assert (llm.claude, llm.backup, llm.gate) == (claude, backup, gate)


def test_gated_llm_auth_error_trips_and_uses_backup():
    claude, backup, gate = _Fake(exc=ClaudeError("bad token", auth=True)), _Fake("来自backup"), ClaudeGate()
    llm = GatedLlm(claude, backup, gate)
    assert llm.complete("s", []) == "来自backup"
    assert not gate.ok()
    assert (claude.calls, backup.calls) == (1, 1)


def test_gated_llm_closed_gate_skips_claude():
    claude, backup, gate = _Fake("来自claude"), _Fake("来自backup"), ClaudeGate()
    gate.trip("limit", "")
    assert GatedLlm(claude, backup, gate).complete("s", []) == "来自backup"
    assert (claude.calls, backup.calls) == (0, 1)


def test_gated_llm_timeout_does_not_trip():
    claude, backup, gate = _Fake(exc=ClaudeError("超时")), _Fake("来自backup"), ClaudeGate()
    with pytest.raises(ClaudeError):
        GatedLlm(claude, backup, gate).complete("s", [])
    assert gate.ok()
    assert backup.calls == 0


def test_gated_llm_no_backup_raises_without_calling_claude():
    claude, gate = _Fake("来自claude"), ClaudeGate()
    gate.trip("limit", "")
    with pytest.raises(ClaudeError):
        GatedLlm(claude, None, gate).complete("s", [])
    assert claude.calls == 0


def test_gated_describe_skips_and_trips():
    calls = []

    def describe(content):
        calls.append(content)
        raise ClaudeError("额度", limit=True)

    gate = ClaudeGate()
    wrapped = gated_describe(describe, gate)
    with pytest.raises(ClaudeError):
        wrapped("img")
    assert not gate.ok() and calls == ["img"]
    with pytest.raises(ClaudeError) as info:
        wrapped("img2")
    assert info.value.limit and calls == ["img"]  # 闸关了不再调


def test_gated_describe_passes_through_while_open():
    assert gated_describe(lambda c: f"看到{c}", ClaudeGate())("x") == "看到x"
