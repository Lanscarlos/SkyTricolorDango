import json
import sys
from pathlib import Path

import pytest
from conftest import FakeDevice, scene
from test_brain_body import FakeReader

from skydango import cli
from skydango.brain.claude import claude_env
from skydango.chat.tracker import SelfFilter
from skydango.config import Config
from skydango.runlog import RunDir

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]


def test_run_brain_wires_everything(tmp_path, monkeypatch):
    log = tmp_path / "claude.jsonl"
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE="ok", FAKE_CLAUDE_LOG=str(log))
    monkeypatch.setattr(cli, "_brain_env", lambda cfg: (FAKE, env))
    monkeypatch.setattr(cli, "_device", lambda cfg: FakeDevice([scene()]))
    monkeypatch.setattr(cli, "_build_reader", lambda cfg: (FakeReader(), SelfFilter(60, 0.8, "")))
    cfg = Config()
    cfg.run.dir = str(tmp_path / "runs")
    cfg.llm.provider = "echo"
    cfg.env.enabled = False
    cfg.reply.memory_dir = ""
    run = RunDir.create(cfg, "dry-brain")
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    starts = [l["args"] for l in lines if "args" in l]
    messages = [l["message"] for l in lines if "message" in l]
    assert any("--append-system-prompt-file" in a for a in starts)  # 大脑进程起来了
    assert any("没有新事件" in m for m in messages)  # 上线后马上醒了一次
    assert (run.path / "brain.jsonl").exists()
    mcp = json.loads((run.path / "brain" / "session" / "mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["sky"]["url"].startswith("http://127.0.0.1:")


def test_brain_env_needs_token(monkeypatch):
    monkeypatch.delenv("SKYDANGO_CLAUDE_TOKEN", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    with pytest.raises(RuntimeError, match="setup-token"):
        cli._brain_env(Config())


def test_run_and_look_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "cmd_run", lambda cfg, args: seen.update(brain=args.brain))
    monkeypatch.setattr(cli, "cmd_look", lambda cfg, args: seen.update(prompt=args.prompt))
    cli.main(["run", "--brain"])
    cli.main(["look", "--prompt", "q.txt"])
    assert seen == {"brain": True, "prompt": "q.txt"}
