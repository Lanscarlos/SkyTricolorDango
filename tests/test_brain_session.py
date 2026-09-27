import json
import sys
from pathlib import Path

import pytest

from skydango.brain.claude import ClaudeError, claude_env
from skydango.brain.session import BrainSession

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]


def session(tmp_path, mode="ok", timeout=10.0):
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE=mode, FAKE_CLAUDE_LOG=str(tmp_path / "log.jsonl"))
    return BrainSession(FAKE, env, tmp_path / "brain", "http://127.0.0.1:1/mcp", "规则", "sonnet", "low", timeout)


def starts(tmp_path):
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    return [l["args"] for l in lines if "args" in l]


def test_command_locks_tools_and_writes_config(tmp_path):
    s = session(tmp_path)
    s.send("你好")
    args = starts(tmp_path)[0]
    assert args[args.index("--tools") + 1] == "" and args[args.index("--allowedTools") + 1] == "mcp__sky"
    assert args[args.index("--permission-mode") + 1] == "dontAsk" and args[args.index("--model") + 1] == "sonnet"
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--verbose", "--append-system-prompt-file"):
        assert flag in args
    assert "--resume" not in args
    cfg = json.loads((tmp_path / "brain" / "mcp.json").read_text(encoding="utf-8"))
    assert cfg == {"mcpServers": {"sky": {"type": "http", "url": "http://127.0.0.1:1/mcp"}}}
    assert (tmp_path / "brain" / "prompt.md").read_text(encoding="utf-8") == "规则"
    s.close()


def test_one_process_many_turns(tmp_path):
    s = session(tmp_path)
    assert s.send("一")["result"] == "收到：一"
    assert s.send("二")["result"] == "收到：二"
    assert len(starts(tmp_path)) == 1 and s.session_id == "fake-session-1"
    s.close()


def test_resumes_same_session_after_crash(tmp_path):
    s = session(tmp_path, "die_unless_resume")
    with pytest.raises(ClaudeError):
        s.send("一")
    assert s.send("二")["result"] == "收到：二"
    second = starts(tmp_path)[1]
    assert second[second.index("--resume") + 1] == "fake-session-1"
    s.close()


def test_timeout_kills_process_and_limit_is_flagged(tmp_path):
    s = session(tmp_path, "hang", timeout=0.5)
    with pytest.raises(ClaudeError, match="没有结果"):
        s.send("一")
    assert not s._proc.alive()
    limited = session(tmp_path, "limit")
    with pytest.raises(ClaudeError) as err:
        limited.send("一")
    assert err.value.limit
    limited.close()


def test_missing_mcp_is_an_error_and_messages_are_forwarded(tmp_path):
    seen = []
    s = session(tmp_path, "nomcp")
    s.on_message = seen.append
    with pytest.raises(ClaudeError, match="MCP"):
        s.send("一")
    assert seen[0]["type"] == "system" and seen[-1]["type"] == "result"
    s.close()
