"""沙盒端到端（brain-sandbox 计划 Task 5）：fake_claude 当大脑，冒充小明说"在吗"，团子回话进聊天记录；
真的 memory/ 一个字节都不动（Review Focus 5）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from model_seams import old_world
from skydango import cli
from skydango.brain.claude import claude_env
from skydango.config import Config

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    headers = {"X-Skydango": "1", "Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET", headers=headers)
    with NO_PROXY.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def sandbox_setup(tmp_path, monkeypatch, mode="say"):
    log = tmp_path / "claude.jsonl"
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE=mode, FAKE_CLAUDE_LOG=str(log))
    memory = tmp_path / "memory"
    (memory / "inner").mkdir(parents=True)
    (memory / "friends.md").write_text("## 小明\n老朋友\n", encoding="utf-8")
    (memory / "history.jsonl").write_text(
        json.dumps({"t": time.time() - 86400, "user": "小明：「明天见」", "reply": "好呀"}, ensure_ascii=False) + "\n", encoding="utf-8")
    (memory / "inner" / "days.jsonl").write_text(
        json.dumps({"start": time.time() - 90000, "end": time.time() - 86000, "ended": "normal"}) + "\n", encoding="utf-8")
    cfg = Config()
    cfg.reply.memory_dir = str(memory)
    cfg.sandbox.dir = str(tmp_path / "sandbox")
    cfg.sandbox.emotes = ["鞠躬"]
    cfg.wheel.library_dir = str(tmp_path / "没有图标库")
    cfg.run.dir = str(tmp_path / "runs")
    old_world(monkeypatch, cfg, FAKE, env)
    port = free_port()
    args = argparse.Namespace(port=port, no_browser=True, parent_pid=None, start="resume", duration=90.0)
    return cfg, args, memory, f"http://127.0.0.1:{port}/"


def test_sandbox_end_to_end_keeps_memory_untouched(tmp_path, monkeypatch):
    cfg, args, memory, base = sandbox_setup(tmp_path, monkeypatch)
    before = hashes(memory)
    result: dict = {}

    def drive():
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if call(base + "status", timeout=2)["ready"]:
                        break
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(0.1)
            result["op"] = call(base + "op", {"op": "say", "who": "小明", "text": "在吗"})
            seen, after = [], 0
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not any(r["kind"] == "said" for r in seen):
                st = call(base + f"state?after={after}&wait=1")
                seen += st["lines"]
                after = st["version"]
            result["lines"] = seen
            deadline = time.monotonic() + 15  # 等团子这一轮做完（say 还在写记录时下线会打断它）
            while time.monotonic() < deadline and not call(base + f"state?after={after}&wait=0")["idle"]:
                time.sleep(0.2)
        except Exception as exc:  # noqa: BLE001 —— 出错也要让主线程收尾
            result["error"] = repr(exc)
        finally:
            try:
                call(base + "shutdown", {})
            except Exception as exc:  # noqa: BLE001
                result["shutdown"] = repr(exc)

    worker = threading.Thread(target=drive, daemon=True)
    worker.start()
    cli.cmd_sandbox(cfg, args)  # 主线程：/shutdown → interrupt_main → 走 Ctrl+C 的收尾
    worker.join(10)
    assert "error" not in result, result
    assert result["op"]["ok"]
    kinds = [(r["kind"], r["who"], r["text"]) for r in result["lines"]]
    assert ("heard", "小明", "在吗") in kinds
    assert any(k == "said" and "在呢" in t for k, _, t in kinds)
    sandbox = Path(cfg.sandbox.dir)
    days = (sandbox / "memory" / "inner" / "days.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(days) == 1 and json.loads(days[0])["ended"] == "normal"
    assert (sandbox / "clock.json").is_file()
    assert hashes(memory) == before  # Review Focus 5：真的 memory/ 一个字节都没变
    assert list((tmp_path / "runs").glob("*-sandbox"))


def test_sandbox_rejects_bad_wake_hour(tmp_path, monkeypatch):
    cfg, args, _, _ = sandbox_setup(tmp_path, monkeypatch)
    cfg.sandbox.wake_hour = 24
    with pytest.raises(SystemExit, match="wake_hour"):
        cli.cmd_sandbox(cfg, args)


def test_sandbox_rejects_start_before_floor(tmp_path, monkeypatch):
    cfg, args, _, _ = sandbox_setup(tmp_path, monkeypatch)
    from skydango.sandbox.clock import save

    save(Path(cfg.sandbox.dir) / "clock.json", time.time() + 86400 * 3)
    args.start = "2020-01-01 09:00"
    with pytest.raises(SystemExit, match="不能往回拨"):
        cli.cmd_sandbox(cfg, args)


def test_sandbox_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "cmd_sandbox", lambda cfg, args: seen.update(vars(args)))
    cli.main(["sandbox", "--port", "19400", "--no-browser", "--parent-pid", "42", "--start", "sleep"])
    assert (seen["port"], seen["no_browser"], seen["parent_pid"], seen["start"]) == (19400, True, 42, "sleep")
    cli.main(["sandbox"])
    assert seen["port"] is None and seen["start"] == "resume"
