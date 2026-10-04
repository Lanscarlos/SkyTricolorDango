"""大脑沙盒审查意见的回归测试：管理面板：孤儿团子、沙盒目录和真 memory 重合、重置拿锁（审查 1 / 2 / 6）。"""


from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from test_brain_tools import FakeBody
from test_console_sandbox import make, post
from test_console_server import no_registry, request, upstream  # noqa: F401

from skydango.console.sandbox_view import reset
from skydango.sandbox.clock import save


# ---- 1. 起沙盒也要探团子的端口：终端起的团子占着 19391 时不许起（两个大脑同时在线）----
def test_sandbox_start_refused_while_terminal_dango_runs(tmp_path, upstream):  # noqa: F811
    s = make(tmp_path, upstream)
    try:
        s.port_free = lambda port: False
        s.probe = lambda port: {"pid": 1, "run_dir": "r", "live": True, "brain": True, "emotes": True, "duration": 0.0,
                                "started": 1.0, "console": False}
        status, res = post(s, "api/sandbox/start", {"start": "resume"})
        assert status == 409 and any("先停团子" in p["text"] for p in res["problems"])
        assert s.runner.started == [] and s.runner.state == "running"  # 接上了
    finally:
        s.stop()


# ---- 2. 沙盒目录和真 memory 重合：重置 / 启动都拒绝，真的 memory/ 一个文件都不能少 ----
def _real_memory(root: Path) -> Path:
    mem = root / "memory"
    (mem / "inner").mkdir(parents=True)
    (mem / "friends.md").write_text("## 小明\n", encoding="utf-8")
    (mem / "inner" / "days.jsonl").write_text('{"start": 1, "end": 2}\n', encoding="utf-8")
    return mem


@pytest.mark.parametrize("sandbox, memory", [(".", "memory"), ("memory/沙盒", "memory"), ("sb", "sb/memory/真的")])
def test_reset_refuses_overlap_with_real_memory(tmp_path, sandbox, memory):
    mem = tmp_path / memory
    (mem / "inner").mkdir(parents=True)
    (mem / "friends.md").write_text("## 小明\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重合"):
        reset(tmp_path / sandbox, mem)
    assert (mem / "friends.md").is_file() and (mem / "inner").is_dir()


def test_reset_same_memory_via_relative_paths(tmp_path, monkeypatch):  # 审查者的复现：cwd 下 reset(".", "memory")
    mem = _real_memory(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="重合"):
        reset(Path("."), Path("memory"))
    assert (mem / "friends.md").is_file()


def test_reset_api_refuses_overlap(tmp_path, upstream):  # noqa: F811
    s = make(tmp_path, upstream)
    try:
        mem = _real_memory(tmp_path)
        s.sandbox_dir = lambda: tmp_path  # 沙盒 memory = 真 memory
        status, res = post(s, "api/sandbox/reset", {})
        assert status == 409 and "重合" in res["text"] and (mem / "friends.md").is_file()
    finally:
        s.stop()


def test_cmd_sandbox_refuses_overlap(tmp_path, monkeypatch):
    from test_cli_sandbox import hashes, sandbox_setup

    from skydango import cli

    cfg, args, memory, _ = sandbox_setup(tmp_path, monkeypatch)
    before = hashes(memory)
    cfg.sandbox.dir = str(tmp_path)  # tmp_path/memory 就是真 memory
    with pytest.raises(SystemExit, match="重合"):
        cli.cmd_sandbox(cfg, args)
    assert hashes(memory) == before and not (tmp_path / "clock.json").exists()


# ---- 6. 重置记忆拿设备锁（和起沙盒互斥）----
def test_reset_takes_the_lock(tmp_path, upstream):  # noqa: F811
    class SpyLock:
        def __init__(self):
            self.entered = 0
            self._lock = threading.Lock()

        def __enter__(self):
            self.entered += 1
            self._lock.acquire()
            return self

        def __exit__(self, *exc):
            self._lock.release()

    s = make(tmp_path, upstream)
    try:
        s._device_lock = SpyLock()
        assert post(s, "api/sandbox/reset", {})[0] == 200
        assert s._device_lock.entered == 1  # 检查状态和删目录都在锁里（和起沙盒互斥）
    finally:
        s.stop()
