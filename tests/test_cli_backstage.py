"""幕后接进启动（spec 2026-10-01-backstage §3、§5）：_backstage_prompt 拼「幕后」、live 才写标记。"""

import json

from skydango.chat.memory import MemoryStore
from skydango.cli import _backstage_prompt, _recent_changes
from skydango.config import Config

HEAD = "a" * 40


def fake_runner(calls):
    def run(args):
        calls.append(args)
        if args[0] == "rev-parse":
            return HEAD + "\n"
        if args[0] == "log":
            return f"{HEAD}\x1f{int(NOW - 60)}\x1ffeat(brain): 会查自己了\n"
        return ""
    return run


NOW = 1_790_000_000.0


def _cfg(enabled=True, dry_run=False):
    cfg = Config()
    cfg.backstage.enabled = enabled
    cfg.reply.dry_run = dry_run
    cfg.brain.owner_name = "卡洛"
    return cfg


def test_backstage_prompt_off(tmp_path):
    calls = []
    assert _backstage_prompt(_cfg(enabled=False), MemoryStore(tmp_path), NOW, run=fake_runner(calls)) == ""
    assert calls == []


def test_backstage_prompt_live_writes_seen(tmp_path):
    out = _backstage_prompt(_cfg(), MemoryStore(tmp_path), NOW, run=fake_runner([]))
    assert out.startswith("## 幕后") and "做出你的人是卡洛" in out and "- 今天 feat(brain): 会查自己了" in out
    assert json.loads((tmp_path / "inner" / "backstage.json").read_text("utf-8")) == {"seen": HEAD}


def test_backstage_prompt_dry_run_does_not_write(tmp_path):
    out = _backstage_prompt(_cfg(dry_run=True), MemoryStore(tmp_path), NOW, run=fake_runner([]))
    assert "会查自己了" in out and not (tmp_path / "inner" / "backstage.json").exists()


def test_backstage_prompt_reads_seen(tmp_path):
    (tmp_path / "inner").mkdir()
    (tmp_path / "inner" / "backstage.json").write_text(json.dumps({"seen": "b" * 40}), encoding="utf-8")
    calls = []
    _backstage_prompt(_cfg(dry_run=True), MemoryStore(tmp_path), NOW, run=fake_runner(calls))
    assert ["merge-base", "--is-ancestor", "b" * 40, "HEAD"] in calls


def test_backstage_prompt_no_store():
    out = _backstage_prompt(_cfg(), None, NOW, run=fake_runner([]))
    assert out.startswith("## 幕后")


def test_backstage_prompt_uses_real_time_not_world_clock(tmp_path, monkeypatch):
    """沙盒的 wall() 会比真实时间超前：更新记录要按真实时间算（spec §3）。"""
    monkeypatch.setattr("skydango.cli.time.time", lambda: NOW)
    out = _backstage_prompt(_cfg(dry_run=True), MemoryStore(tmp_path), run=fake_runner([]))
    assert "- 今天 feat(brain): 会查自己了" in out


def test_recent_changes_ignores_seen(tmp_path):  # introspect(改动)：不管告诉过没有，按最近几天查
    (tmp_path / "inner").mkdir()
    (tmp_path / "inner" / "backstage.json").write_text(json.dumps({"seen": HEAD}), encoding="utf-8")
    calls = []
    lines = _recent_changes(_cfg(), run=fake_runner(calls), now=lambda: NOW)()
    assert lines == ["- 今天 feat(brain): 会查自己了"]
    assert not any(c[0] == "merge-base" for c in calls)


def test_recent_changes_off():
    assert _recent_changes(_cfg(enabled=False), run=fake_runner([])) is None
