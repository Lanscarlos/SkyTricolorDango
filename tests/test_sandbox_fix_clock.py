"""大脑沙盒审查意见的回归测试：沙盒时钟漏网的地方：recall、--duration、随手记的“今天是”（审查 3 / 4 / 7）。"""


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


# ---- 3. recall 的天数按身体的墙上时间（沙盒快进之后）----
def test_recall_uses_body_wall(tmp_path):
    from skydango.brain.tools import ToolBox
    from skydango.chat.memory import MemoryStore

    future = time.time() + 100 * 86400  # 沙盒时间比现在快 100 天
    store = MemoryStore(tmp_path)
    store.history.append("小明：「上个月去了暴风眼」", "好厉害", future - 20 * 86400)  # 沙盒里是 20 天前
    body = FakeBody()
    body.wall = lambda: future
    out, err = ToolBox(body, memory=store).run("recall", {"query": "暴风眼", "days": 14})
    assert not err and "没找到" in out and "上个月去了暴风眼" not in out  # 沙盒里是 20 天前，超过 14 天，不该翻到


# ---- 4. --duration 用真实时间：快进不会提前下线 ----
def test_body_duration_uses_real_time(clock):
    from test_brain_body import body

    b, *_ = body(clock)
    steps = []
    real_step = b.step

    def step():
        steps.append(1)
        clock.advance(3600)  # 沙盒快进一小时
        real_step()

    b.step = step
    b.sleep = lambda s: time.sleep(0.01)
    started = time.monotonic()
    b.run(duration=0.3)
    assert time.monotonic() - started >= 0.25 and len(steps) >= 3


# ---- 7. 随手记整理的"今天是"用墙上时间（沙盒传模拟时钟）----
def test_notes_keeper_uses_wall(tmp_path):
    from skydango.chat.memory import MemoryStore, NotesKeeper, format_date

    seen = []

    class Llm:
        def complete(self, system, messages, max_tokens=None):
            seen.append(messages[0]["content"])
            return "## 小明\n- 爱跑图"

    store = MemoryStore(tmp_path)
    store.history.append("小明：「在吗」", "在", 1.0)
    future = time.time() + 100 * 86400
    keeper = NotesKeeper(Llm(), store, "团子", every=1, background=False, wall=lambda: future)
    keeper.update_now()
    assert seen and f"今天是 {format_date(future)}" in seen[-1]


def test_run_brain_gives_notes_the_world_wall(tmp_path, monkeypatch):
    from test_cli_brain import fake_brain_run

    from skydango import cli

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "mem")
    cfg.reply.dry_run = False
    world = cli._game_world(cfg, run, True)
    world.wall = lambda: 1_790_000_000.0
    seen = []
    cli._run_brain(cfg, run, world, 1.0, on_ready=seen.append)
    assert seen[0].body.notes.wall is world.wall
