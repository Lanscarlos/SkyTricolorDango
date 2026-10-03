import _thread
import json
import sys
import time
from pathlib import Path

import pytest
from conftest import FakeDevice, FakeOcr, scene
from test_brain_body import FakeReader

from skydango import cli
from skydango.brain.claude import claude_env
from skydango.brain.manual import ManualControl
from skydango.brain.trace import BrainTrace
from skydango.chat.tracker import SelfFilter
from skydango.config import Config, ViewerConfig
from skydango.runlog import RunDir

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]


def fake_brain_run(tmp_path, monkeypatch):
    """_run_brain 的准备：假 claude 进程（fake_claude.py）、假设备、假读聊天。返回 (cfg, run, 假 claude 的日志)。"""
    log = tmp_path / "claude.jsonl"
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE="ok", FAKE_CLAUDE_LOG=str(log))
    monkeypatch.setattr(cli, "_brain_env", lambda cfg: (FAKE, env))
    monkeypatch.setattr(cli, "_device", lambda cfg: FakeDevice([scene()]))
    monkeypatch.setattr(cli, "_build_reader", lambda cfg: (FakeReader(), SelfFilter(60, 0.8, "")))
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda engine, threads=8: FakeOcr())
    cfg = Config()
    cfg.panels.cards_dir = str(Path(__file__).resolve().parents[1] / "assets" / "panels")
    cfg.run.dir = str(tmp_path / "runs")
    cfg.llm.provider = "echo"
    cfg.env.enabled = False
    cfg.reply.memory_dir = ""
    return cfg, RunDir.create(cfg, "dry-brain"), log


class FakeViewer:
    brain = None
    control = None
    chat = None

    def update(self, *args, **kwargs):
        return True


def test_run_brain_with_viewer_attaches_control(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    v = FakeViewer()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert isinstance(v.control, ManualControl)


def test_run_brain_on_lan_has_no_control(tmp_path, monkeypatch, caplog):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.viewer.host = "0.0.0.0"
    v = FakeViewer()
    with caplog.at_level("WARNING"):
        cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert v.control is None and "局域网模式下关掉了手动控制" in caplog.text


def test_run_brain_with_viewer_attaches_chat(tmp_path, monkeypatch):
    from skydango.brain.transcript import Transcript

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    v = FakeViewer()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert isinstance(v.chat, Transcript)


def test_run_brain_on_lan_has_no_chat(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.viewer.host = "0.0.0.0"
    v = FakeViewer()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert v.chat is None  # 聊天原话只给本机看


def test_run_brain_with_viewer_records_turns(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    v = FakeViewer()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert isinstance(v.brain, BrainTrace)
    out = v.brain.since(0, 0.0)
    first = out["turns"][0]
    assert first["reason"] == "heartbeat" and "没有新事件" in first["prompt"]
    assert first["end"] is not None and first["error"] is None
    assert out["state"]["model"] == cfg.brain.model


def test_run_brain_wires_everything(tmp_path, monkeypatch):
    cfg, run, log = fake_brain_run(tmp_path, monkeypatch)
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    starts = [l["args"] for l in lines if "args" in l]
    messages = [l["message"] for l in lines if "message" in l]
    assert any("--append-system-prompt-file" in a for a in starts)  # 大脑进程起来了
    assert any("没有新事件" in m for m in messages)  # 上线后马上醒了一次
    assert (run.path / "brain.jsonl").exists()
    mcp = json.loads((run.path / "brain" / "session" / "mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["sky"]["url"].startswith("http://127.0.0.1:")


def test_run_brain_memory_uses_claude(tmp_path, monkeypatch):
    import skydango.chat.memory as memory

    cfg, run, log = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False  # live 才整理记忆
    seen = []
    real = memory.NotesKeeper
    monkeypatch.setattr(memory, "NotesKeeper", lambda llm, *a, **k: seen.append(llm) or real(llm, *a, **k))
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert seen[0].complete("你负责记笔记", [{"role": "user", "content": "整理"}]) == "收到：整理"
    args = [json.loads(l)["args"] for l in log.read_text(encoding="utf-8").splitlines() if "args" in l][-1]
    assert args[args.index("--model") + 1] == cfg.brain.memory_model
    assert args[args.index("--system-prompt") + 1] == "你负责记笔记"


def test_run_brain_gives_recall_the_memory_even_in_dry_run(tmp_path, monkeypatch):
    import skydango.brain.tools as tools
    from skydango.chat.memory import MemoryStore

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    assert cfg.reply.dry_run
    seen = []
    real = tools.ToolBox
    monkeypatch.setattr(tools, "ToolBox", lambda *a, **k: seen.append(k.get("memory")) or real(*a, **k))
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert isinstance(seen[0], MemoryStore) and seen[0].dir == MemoryStore(cfg.reply.memory_dir).dir


def test_run_brain_prompt_carries_recent_turns(tmp_path, monkeypatch):
    import time

    from skydango.chat.memory import MemoryStore

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    MemoryStore(cfg.reply.memory_dir).history.append("新的聊天消息：\n懒洋洋大王：「明天跑暴风眼」", "行啊", time.time() - 3600)
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    prompt = (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")
    assert "## 上次聊到哪" in prompt and "懒洋洋大王：「明天跑暴风眼」 → 我：行啊" in prompt


def test_run_brain_wires_proactive(tmp_path, monkeypatch):
    import skydango.brain.eyes as eyes_mod

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    seen = {}
    real = eyes_mod.Eyes

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(eyes_mod, "Eyes", spy)
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0)
    assert seen["proactive"] is cfg.proactive and seen["busy"](0.0) is False  # 没开 env：好友永远不在身边
    body = seen["on_news"].__self__
    assert seen["on_news"].__func__.__name__ == "news" and body.friend_names.__name__ == "names"  # cli 的 _friend_names


def test_profile_template_has_likes(tmp_path):
    import argparse

    assert "## 喜好和看法" in cli.PROFILE_TEMPLATE and "樱花发型天下第一" in cli.PROFILE_TEMPLATE
    cfg = Config()
    cfg.reply.memory_dir = str(tmp_path)
    cli.cmd_memory(cfg, argparse.Namespace(action="init"))
    assert "## 喜好和看法" in (tmp_path / "profile.md").read_text(encoding="utf-8")


def test_memory_update_uses_claude(tmp_path, monkeypatch):
    import argparse

    from skydango.chat.memory import MemoryStore

    monkeypatch.chdir(tmp_path)
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE="ok", FAKE_CLAUDE_LOG=str(tmp_path / "claude.jsonl"))
    monkeypatch.setattr(cli, "_claude_base", lambda cfg, hint="": (FAKE, env))
    cfg = Config()
    cfg.llm.provider = "nope"  # 用到 DeepSeek（make_llm）就会报错
    cfg.reply.memory_dir = str(tmp_path / "memory")
    MemoryStore(cfg.reply.memory_dir).history.append("懒洋洋大王：「我加班」", "辛苦啦", 1.0)
    cli.cmd_memory(cfg, argparse.Namespace(action="update"))
    assert "我加班" in MemoryStore(cfg.reply.memory_dir).notes()  # 假 claude 把收到的内容原样写回


def test_brain_env_needs_token(monkeypatch):
    monkeypatch.delenv("SKYDANGO_CLAUDE_TOKEN", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    with pytest.raises(RuntimeError, match="setup-token"):
        cli._brain_env(Config())


def test_run_and_look_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "cmd_run", lambda cfg, args: seen.update(brain=args.brain, no_brain=args.no_brain))
    monkeypatch.setattr(cli, "cmd_look", lambda cfg, args: seen.update(prompt=args.prompt))
    cli.main(["run", "--brain"])
    cli.main(["look", "--prompt", "q.txt"])
    assert seen == {"brain": True, "no_brain": False, "prompt": "q.txt"}


@pytest.mark.parametrize("argv, mode", [(["run"], "brain"), (["run", "--brain"], "brain"), (["run", "--no-brain"], "agent")])
def test_run_defaults_to_brain(tmp_path, monkeypatch, argv, mode):
    seen = []
    monkeypatch.setattr(cli, "_run_brain", lambda *a, **k: seen.append("brain"))
    monkeypatch.setattr(cli, "_run_agent", lambda *a, **k: seen.append("agent"))
    monkeypatch.chdir(tmp_path)  # runs/ 建在临时目录
    cli.main(argv)
    assert seen == [mode]


def test_brain_env_error_mentions_no_brain(monkeypatch):
    monkeypatch.delenv("SKYDANGO_CLAUDE_TOKEN", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    with pytest.raises(RuntimeError, match="--no-brain"):
        cli._brain_env(Config())


def test_viewer_serves_brain_before_browser_opens(monkeypatch):
    # run --brain --view：浏览器一打开就请求 /brain；这时还没挂上 trace 会拿到 404，页面就不再请求
    import urllib.request
    import webbrowser

    seen = {}

    def fake_open(url):
        with urllib.request.urlopen(url + "brain?after=0", timeout=5) as r:
            seen["status"] = r.status

    monkeypatch.setattr(webbrowser, "open", fake_open)
    cfg = Config()
    cfg.viewer.port = 0
    viewer = cli._viewer(cfg, brain=True)
    try:
        assert seen == {"status": 200} and isinstance(viewer.brain, BrainTrace)
    finally:
        viewer.stop()


def test_run_brain_reuses_the_viewers_trace(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    v = FakeViewer()
    v.brain = trace = BrainTrace()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert v.brain is trace and trace.since(0, 0.0)["turns"]


def spy_body(monkeypatch):
    import skydango.brain.body as body_module

    seen = {}
    real = body_module.Body

    def make(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(body_module, "Body", make)
    return seen


def test_run_brain_wires_panels(tmp_path, monkeypatch):
    from skydango.game.panels import PanelOps
    from skydango.vision.panels import PanelWatcher

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    seen = spy_body(monkeypatch)
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0)
    assert isinstance(seen["panels"], PanelWatcher) and isinstance(seen["panel_ops"], PanelOps)
    assert set(seen["panels"].cards) >= {"chat_log", "shared_invite"}


def test_run_brain_panels_disabled(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.panels.enabled = False
    seen = spy_body(monkeypatch)
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0)
    assert seen["panels"] is None and seen["panel_ops"] is None



def test_run_brain_wires_locomotion(tmp_path, monkeypatch):
    from skydango.brain.locomotion import Locomotion

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.brain.move_step = 0.2
    seen = spy_body(monkeypatch)
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0)
    assert isinstance(seen["locomotion"], Locomotion) and seen["locomotion"].step == 0.2

def test_brain_env_without_mcp_mentions_no_brain(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "mcp" else real(name, *a))
    with pytest.raises(RuntimeError, match="mcp.*--no-brain"):
        cli._brain_env(Config())


# ---- 管理面板起子进程用的参数 ----
@pytest.mark.parametrize("argv, dry", [(["run", "--dry-run"], True), (["run", "--live"], False)])
def test_run_dry_run_flag_beats_config(tmp_path, monkeypatch, argv, dry):
    (tmp_path / "config.toml").write_text("[reply]\ndry_run = false\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "_run_brain", lambda cfg, *a, **k: seen.append(cfg.reply.dry_run))
    monkeypatch.chdir(tmp_path)
    cli.main(argv)
    assert seen == [dry]


def test_run_live_and_dry_run_conflict():
    with pytest.raises(SystemExit):
        cli.main(["run", "--live", "--dry-run"])


def test_run_viewer_port_forces_local_and_no_browser(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_viewer", lambda cfg, open_browser=True, brain=False, on_shutdown=None: seen.update(
        host=cfg.viewer.host, port=cfg.viewer.port, open=open_browser))
    monkeypatch.setattr(cli, "_run_brain", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text('[viewer]\nhost = "0.0.0.0"\n', encoding="utf-8")
    cli.main(["run", "--view", "--viewer-port", "19391", "--no-browser"])
    assert seen == {"host": "127.0.0.1", "port": 19391, "open": False}


def test_run_parent_pid_starts_watchdog(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("skydango.console.watchdog.watch_parent", lambda pid, interrupt, **k: seen.append((pid, interrupt)))
    monkeypatch.setattr(cli, "_run_brain", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_viewer", lambda cfg, open_browser=True, brain=False, on_shutdown=None: seen.append(on_shutdown))
    cli.main(["run", "--view", "--parent-pid", "4321"])
    pid, watch_hook = seen[0]
    assert pid == 4321 and seen[1] is watch_hook  # 看门狗和 /shutdown 共用同一个只触发一次的中断


def test_viewer_hooks_shutdown_to_interrupt_main(monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: None)
    calls = []
    monkeypatch.setattr(_thread, "interrupt_main", lambda: calls.append(1))
    v = cli._viewer(Config(viewer=ViewerConfig(port=0)), open_browser=False)
    try:
        v.on_shutdown()
        v.on_shutdown()
        assert calls == [1]
    finally:
        v.stop()


def test_lan_viewer_has_no_shutdown(monkeypatch):  # 终审：局域网模式别人也能停团子
    v = cli._viewer(Config(viewer=ViewerConfig(port=0, host="0.0.0.0")), open_browser=False)
    try:
        assert v.on_shutdown is None
    finally:
        v.stop()


def test_run_brain_live_writes_inner_ledger(tmp_path, monkeypatch):
    from skydango.inner.store import InnerStore

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False
    cfg.inner.reflect = False  # 第 1 期的路径：大脑写经过（反思开着时由最终反思写，见下面几条）
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    (day,) = InnerStore(tmp_path / "memory" / "inner").days()
    assert day.ended == "normal" and day.summary.startswith("收到：")  # 假大脑把退出前的经过原样回过来
    assert not (tmp_path / "memory" / "inner" / "current.json").exists()
    prompt = (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")
    assert "## 日子" in prompt and "这是你第一次上线" in prompt and "别报数字" in prompt


def test_run_brain_dry_run_inner_prompt_but_no_files(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert not (tmp_path / "memory" / "inner").exists()
    assert "## 日子" in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")


def test_run_brain_inner_disabled(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False
    cfg.inner.enabled = False
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert not (tmp_path / "memory" / "inner").exists()
    assert "## 日子" not in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")


def test_memory_show_prints_inner(tmp_path, capsys):
    import argparse

    cfg = Config()
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cli.cmd_memory(cfg, argparse.Namespace(action="show"))
    out = capsys.readouterr().out
    assert "===== 关系卡 =====" in out and "===== 最近 10 次上线 =====" in out


def _claude_log(log):
    return [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]


def test_run_brain_live_final_reflection_replaces_farewell(tmp_path, monkeypatch):
    from skydango.brain.prompt import MIND_RULES, SUMMARY_REQUEST
    from skydango.inner.reflect import REFLECT_SYSTEM
    from skydango.inner.store import InnerStore

    cfg, run, log = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    lines = _claude_log(log)
    systems = [l["args"][l["args"].index("--system-prompt") + 1] for l in lines if "args" in l and "--system-prompt" in l["args"]]
    assert any(s.startswith(REFLECT_SYSTEM) for s in systems)  # 下线前的最终反思（第 3 期默认后面接 PERSONA_SYSTEM）
    assert not any(SUMMARY_REQUEST in l.get("message", "") for l in lines)  # 不再让大脑写经过
    (day,) = InnerStore(tmp_path / "memory" / "inner").days()
    assert day.ended == "normal"  # 假模型回的不是 JSON：summary 空，照样 close
    assert MIND_RULES in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")


def test_run_brain_reflect_off_uses_farewell(tmp_path, monkeypatch):
    from skydango.brain.prompt import MIND_RULES, SUMMARY_REQUEST

    cfg, run, log = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False
    cfg.inner.reflect = False
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert any(SUMMARY_REQUEST in l.get("message", "") for l in _claude_log(log))
    assert MIND_RULES not in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")


def test_run_brain_dry_run_reflect_writes_nothing(tmp_path, monkeypatch):
    from skydango.brain.prompt import MIND_RULES

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert not (tmp_path / "memory" / "inner").exists()
    assert MIND_RULES in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")


def test_final_reflection_timeout_fits_console_stop():  # 终审 I7
    cfg = Config()
    assert cli._final_timeout(cfg) == 35.0  # stop_timeout 60 − 25（身体收尾、等大脑线程）
    cfg.console.stop_timeout = 20
    assert cli._final_timeout(cfg) == 10.0
    cfg.console.stop_timeout = 600
    assert cli._final_timeout(cfg) == cfg.inner.reflect_timeout


def test_profile_template_has_temper():
    for s in ("## 脾气", "毛病", "执念", "雷点"):
        assert s in cli.PROFILE_TEMPLATE
    assert cli.PROFILE_TEMPLATE.index("## 喜好和看法") < cli.PROFILE_TEMPLATE.index("## 脾气")



# ---- 内心层第 3 期：性格 ----
def _reflect_systems(log):
    return [l["args"][l["args"].index("--system-prompt") + 1] for l in _claude_log(log)
            if "args" in l and "--system-prompt" in l["args"]]


def test_run_brain_live_persona_on(tmp_path, monkeypatch):
    from skydango.brain.prompt import TEMPER_RULES
    from skydango.inner.persona import Persona, Trait
    from skydango.inner.reflect import PERSONA_SYSTEM, REFLECT_SYSTEM
    from skydango.inner.store import InnerStore

    cfg, run, log = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False
    InnerStore(tmp_path / "memory" / "inner").write_persona(Persona(catchphrases=[Trait("害，懒得动", since=time.time())]))
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    prompt = (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")
    assert TEMPER_RULES in prompt and "## 你攒下的性格" in prompt and "口头禅：害，懒得动" in prompt
    assert prompt.index("## 你攒下的性格") < prompt.index("## 日子")
    systems = [s for s in _reflect_systems(log) if s.startswith(REFLECT_SYSTEM)]
    assert systems and all(PERSONA_SYSTEM in s for s in systems)


def test_run_brain_persona_off_is_phase2(tmp_path, monkeypatch):
    from skydango.brain.prompt import TEMPER_RULES
    from skydango.inner.reflect import REFLECT_SYSTEM

    cfg, run, log = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False
    cfg.inner.persona = False
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert TEMPER_RULES not in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")
    assert REFLECT_SYSTEM in _reflect_systems(log)


def test_run_brain_dry_run_persona_reads_but_writes_nothing(tmp_path, monkeypatch):
    from skydango.brain.prompt import TEMPER_RULES

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert not (tmp_path / "memory" / "inner").exists()
    assert TEMPER_RULES in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")


def test_dry_run_bad_persona_not_renamed(tmp_path, monkeypatch):  # Review Focus 4
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    inner = tmp_path / "memory" / "inner"
    inner.mkdir(parents=True)
    (inner / "persona.json").write_text("坏", encoding="utf-8")
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    assert (inner / "persona.json").exists() and not list(inner.glob("persona.json.bad-*"))


def test_memory_show_prints_persona(tmp_path, capsys):
    import argparse

    cfg = Config()
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cli.cmd_memory(cfg, argparse.Namespace(action="show"))
    out = capsys.readouterr().out
    assert "===== 性格档案 =====" in out and not (tmp_path / "memory" / "inner").exists()
    cfg.inner.persona = False
    cli.cmd_memory(cfg, argparse.Namespace(action="show"))
    assert "===== 性格档案 =====" not in capsys.readouterr().out


# ---- 沙盒计划 Task 2：_run_brain 拆出 World ----
def test_run_brain_takes_time_from_world(tmp_path, monkeypatch):
    from skydango.brain.body import Body
    from skydango.brain.world import BrainParts
    from skydango.inner.store import InnerStore

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "mem")
    cfg.reply.dry_run = False
    world = cli._game_world(cfg, run, True)
    fixed = 1_790_000_000.0
    world.wall = lambda: fixed
    world.clock = lambda: time.monotonic()  # 换一个认得出的钟
    seen = []
    cli._run_brain(cfg, run, world, 3.0, on_ready=seen.append)
    (day,) = InnerStore(tmp_path / "mem" / "inner").days()
    assert day.start == fixed and day.end == fixed  # 账本用的是 world 的墙上时间
    assert len(seen) == 1 and isinstance(seen[0], BrainParts) and isinstance(seen[0].body, Body)
    parts = seen[0]
    assert parts.body.clock is world.clock and parts.body.wall is world.wall
    assert parts.events.clock is world.clock and parts.eyes.clock is world.clock
    assert parts.brain.clock is world.clock and parts.brain.wall is world.wall  # 大脑和事件队列同一个钟
    assert parts.reflector.clock is world.clock and parts.ledger is not None and parts.trace is None


def test_run_brain_still_builds_the_game_world(tmp_path, monkeypatch):
    seen = []
    real = cli._game_world
    monkeypatch.setattr(cli, "_game_world", lambda cfg, run, no_emotes: seen.append(no_emotes) or real(cfg, run, no_emotes))
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0)
    assert seen == [True]


def test_cmd_run_builds_world_after_token_check(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_run_brain", lambda cfg, run, world=None, *a, **k: seen.update(world=world, k=k))
    monkeypatch.chdir(tmp_path)
    cli.main(["run", "--no-emotes"])
    assert seen["world"] is None and seen["k"].get("no_emotes") is True  # 世界（连设备）在 _run_brain 里、令牌检查之后建


class ImeDevice(FakeDevice):
    """会切输入法的假设备：切换记进 order（和身体收尾排先后）。"""

    def __init__(self, order, current="com.sohu.inputmethod.sogou/.SogouIME"):
        super().__init__([scene()])
        self.order, self.current = order, current

    def current_ime(self):
        return self.current

    def list_imes(self):
        return ["com.sohu.inputmethod.sogou/.SogouIME", "com.android.adbkeyboard/.AdbIME"]

    def set_ime(self, ime_id):
        self.order.append(ime_id)
        self.current = ime_id


@pytest.mark.parametrize("switch", [True, False])
def test_run_brain_switches_ime_and_back_after_body_shutdown(tmp_path, monkeypatch, switch):
    from skydango.brain.body import Body

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.device.switch_ime = switch
    order = []
    dev = ImeDevice(order)
    monkeypatch.setattr(cli, "_device", lambda cfg: dev)
    shutdown = Body.shutdown
    monkeypatch.setattr(Body, "shutdown", lambda self: (order.append("shutdown"), shutdown(self))[1])
    during = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: during.append(dev.current))
    if switch:
        assert during == ["com.android.adbkeyboard/.AdbIME"]
        assert order == ["com.android.adbkeyboard/.AdbIME", "shutdown", "com.sohu.inputmethod.sogou/.SogouIME"]
    else:
        assert order == ["shutdown"]


# ---- DeepSeek 备用大脑（spec 2026-10-03-deepseek-fallback-brain） ----

def _fake_tb():
    from types import SimpleNamespace

    return SimpleNamespace(body=SimpleNamespace(env=None), calling=False, backstage=False)


def test_fallback_brain_off_when_disabled(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.brain.fallback = False
    assert cli._fallback_brain(cfg, object(), "p") is None


def test_fallback_brain_none_when_no_key(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    monkeypatch.setattr("skydango.brain.deepseek.read_key", lambda name: (_ for _ in ()).throw(RuntimeError("没 Key")))
    assert cli._fallback_brain(cfg, object(), "p") is None


def test_fallback_brain_none_when_openai_missing(tmp_path, monkeypatch):
    import skydango.brain.deepseek as ds

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    monkeypatch.setattr(ds, "build_client", lambda llm, key=None: (_ for _ in ()).throw(ImportError("No module named 'openai'")))
    assert cli._fallback_brain(cfg, _fake_tb(), "p") is None


def test_run_brain_survives_missing_openai(tmp_path, monkeypatch):
    import skydango.brain.deepseek as ds

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    monkeypatch.setattr(ds, "build_client", lambda llm, key=None: (_ for _ in ()).throw(ImportError("No module named 'openai'")))
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: seen.append(parts.brain))
    assert seen[0].fallback_session is None and seen[0].on_fallback is False


def test_fallback_brain_built(tmp_path, monkeypatch):
    import skydango.brain.deepseek as ds

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    made = {}
    monkeypatch.setattr(ds, "build_client", lambda llm, key=None: made.update(key=key) or object())
    brain = cli._fallback_brain(cfg, _fake_tb(), "prompt")
    assert isinstance(brain, ds.DeepSeekBrain) and brain.system == "prompt\n\n" + ds.FALLBACK_NOTE
    assert brain.model == cfg.llm.model and brain.max_tokens == cfg.brain.fallback_max_tokens


def test_run_brain_passes_fallback_session(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    sentinel = type("FB", (), {"send": lambda self, t: {"result": "好", "subtype": "success"}})()
    monkeypatch.setattr(cli, "_fallback_brain", lambda cfg, tb, p, on_message=None: sentinel)
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: seen.append(parts.brain))
    assert seen[0].fallback_session is sentinel


def test_force_fallback_swaps_session(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.brain.force_fallback = True
    sentinel = type("FB", (), {"send": lambda self, t: {"result": "好", "subtype": "success"}})()
    monkeypatch.setattr(cli, "_fallback_brain", lambda cfg, tb, p, on_message=None: sentinel)
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: seen.append(parts.brain))
    assert seen[0].session is sentinel and seen[0].on_fallback is True


def test_force_fallback_without_session_is_noop(tmp_path, monkeypatch):  # Review Focus 5
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.brain.force_fallback = True
    monkeypatch.setattr(cli, "_fallback_brain", lambda cfg, tb, p, on_message=None: None)
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: seen.append(parts.brain))
    assert seen[0].session is not None and seen[0].on_fallback is False
