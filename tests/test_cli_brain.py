import _thread
import json
import sys
import time
from pathlib import Path

import pytest
from conftest import FakeDevice, FakeOcr, scene
from test_brain_body import FakeReader

from model_seams import old_world
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
    monkeypatch.setattr(cli, "_device", lambda cfg: FakeDevice([scene()]))
    monkeypatch.setattr(cli, "_build_reader", lambda cfg: (FakeReader(), SelfFilter(60, 0.8, "")))
    monkeypatch.setattr(cli, "_chat_reader", lambda cfg, dev, save=None: (FakeReader(), SelfFilter(60, 0.8, "")))
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda engine, threads=8: FakeOcr())
    cfg = Config()
    cfg.panels.cards_dir = str(Path(__file__).resolve().parents[1] / "assets" / "panels")
    cfg.run.dir = str(tmp_path / "runs")
    old_world(monkeypatch, cfg, FAKE, env)
    cfg.env.enabled = False
    cfg.reply.memory_dir = ""
    return cfg, RunDir.create(cfg, "dry-brain"), log


@pytest.fixture(autouse=True)
def api_port_not_19391(monkeypatch):
    """run 总在 [viewer] port（默认 19391）开接口：测试里换成随机端口，免得撞上本机正在跑的团子 / 管理面板。"""
    real = cli._viewer

    def viewer(cfg, *a, **k):
        if cfg.viewer.port == 19391:
            cfg.viewer.port = 0
        return real(cfg, *a, **k)

    monkeypatch.setattr(cli, "_viewer", viewer)


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


def test_run_brain_with_viewer_attaches_chat(tmp_path, monkeypatch):
    from skydango.brain.transcript import Transcript

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    v = FakeViewer()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert isinstance(v.chat, Transcript)


def test_run_brain_with_viewer_records_turns(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    v = FakeViewer()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert isinstance(v.brain, BrainTrace)
    out = v.brain.since(0, 0.0)
    first = out["turns"][0]
    assert first["reason"] == "heartbeat" and "没有新事件" in first["prompt"]
    assert first["end"] is not None and first["error"] is None
    assert out["state"]["model"] == f"claude/{cfg.brain.model}"


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


def test_memory_update_uses_memory_model(tmp_path, monkeypatch):
    import argparse

    from skydango.chat.memory import MemoryStore

    monkeypatch.chdir(tmp_path)
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE="ok", FAKE_CLAUDE_LOG=str(tmp_path / "claude.jsonl"))
    cfg = old_world(monkeypatch, Config(), FAKE, env)  # [models.memory] = claude/sonnet（假 claude 进程）
    cfg.reply.memory_dir = str(tmp_path / "memory")
    MemoryStore(cfg.reply.memory_dir).history.append("懒洋洋大王：「我加班」", "辛苦啦", 1.0)
    cli.cmd_memory(cfg, argparse.Namespace(action="update"))
    assert "我加班" in MemoryStore(cfg.reply.memory_dir).notes()  # 假 claude 把收到的内容原样写回


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


def test_no_brain_model_error_mentions_no_brain(tmp_path, monkeypatch):
    from skydango.models.errors import ModelError

    def refuse(provider, environ=None):
        raise ModelError("没有 Claude 令牌：先运行 claude setup-token", down="auth", provider=provider.id)

    monkeypatch.setattr("skydango.models.claude_code.claude_base", refuse)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="--no-brain") as e:
        cli._check_brain_models(cli._registry(Config(), tmp_path))
    assert "setup-token" in str(e.value) and "DEEPSEEK_API_KEY" in str(e.value)


def test_viewer_has_brain_trace_from_the_start(monkeypatch):
    # 管理面板一连上就请求 /brain：trace 要在接口起来之前挂好
    cfg = Config()
    cfg.viewer.port = 0
    viewer = cli._viewer(cfg, True, lambda: None, {"pid": 1})
    try:
        assert isinstance(viewer.brain, BrainTrace) and viewer.run_info == {"pid": 1}
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


def test_old_viewer_flags_are_gone():  # spec 2026-10-04-console-attach §1
    with pytest.raises(SystemExit) as err:  # 只有管理面板传过；--no-browser 留着（旧会话常和 --view 一起写）
        cli.main(["run", "--viewer-port", "1"])
    assert err.value.code == 2


def test_view_flag_is_ignored_with_notice(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli, "_run_brain", lambda *a, **k: seen.append(1))
    monkeypatch.chdir(tmp_path)
    cli.main(["run", "--view"])
    assert seen == [1] and "--view 已经不用了" in capsys.readouterr().out


def test_run_always_opens_api_with_run_info(tmp_path, monkeypatch):
    import os

    seen = {}

    def fake_brain(cfg, run, world=None, duration=0.0, viewer=None, **k):
        seen.update(viewer=viewer, info=dict(viewer.run_info), run_dir=str(run.path.resolve()))

    monkeypatch.setattr(cli, "_run_brain", fake_brain)
    monkeypatch.setattr("skydango.console.watchdog.watch_parent", lambda pid, interrupt, **k: None)
    monkeypatch.chdir(tmp_path)
    cli.main(["run", "--dry-run", "--no-emotes", "--duration", "5", "--parent-pid", "4321"])
    info = seen["info"]
    assert seen["viewer"] is not None and seen["viewer"].on_shutdown is not None
    assert info["pid"] == os.getpid() and info["run_dir"] == seen["run_dir"] and info["console"] is True
    assert info["live"] is False and info["brain"] is True and info["emotes"] is False and info["duration"] == 5.0
    assert abs(info["started"] - time.time()) < 60


def test_run_refuses_when_port_taken(tmp_path, monkeypatch):
    import socket

    seen = []
    monkeypatch.setattr(cli, "_run_brain", lambda *a, **k: seen.append("brain"))
    monkeypatch.setattr(cli, "_device", lambda cfg: pytest.fail("端口被占时不该碰设备"))
    monkeypatch.chdir(tmp_path)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        (tmp_path / "config.toml").write_text(f"[viewer]\nport = {port}\n", encoding="utf-8")
        with pytest.raises(SystemExit) as err:
            cli.main(["run"])
    assert f"{port} 端口上已经有一个团子在跑" in str(err.value.code) and seen == []


def test_run_parent_pid_starts_watchdog(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("skydango.console.watchdog.watch_parent", lambda pid, interrupt, **k: seen.append((pid, interrupt)))
    monkeypatch.setattr(cli, "_run_brain", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_viewer", lambda cfg, brain, on_shutdown, run_info: seen.append(on_shutdown))
    cli.main(["run", "--parent-pid", "4321"])
    pid, watch_hook = seen[0]
    assert pid == 4321 and seen[1] is watch_hook  # 看门狗和 /shutdown 共用同一个只触发一次的中断


def test_viewer_hooks_shutdown_to_interrupt_main(monkeypatch):
    calls = []
    monkeypatch.setattr(_thread, "interrupt_main", lambda: calls.append(1))
    from skydango.console.watchdog import once

    v = cli._viewer(Config(viewer=ViewerConfig(port=0)), False, once(_thread.interrupt_main), {})
    try:
        v.on_shutdown()
        v.on_shutdown()
        assert calls == [1]
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


def _final_parts(tmp_path, monkeypatch, reply="{}", claude_exc=None):
    """_final_reflection 的准备：live 记忆目录 + 真的 _inner_mind（反思主 claude/sonnet、备 deepseek）+ 假身体。
    返回 (cfg, store, ledger, reflector, body, registry, seen)：seen 记 claude 每次的超时、deepseek 换超时时的超时 / 重试。"""
    from types import SimpleNamespace

    from model_seams import fake_claude
    from skydango.chat.memory import MemoryStore
    from skydango.inner.mind import Mind

    seen = {"claude": [], "deepseek": []}
    fake_claude(monkeypatch, ["claude"])

    def one_shot_message(cmd, env, cwd, content, timeout, on_message=None):
        seen["claude"].append(timeout)
        if claude_exc is not None:
            raise claude_exc
        return {"result": reply}

    def build_client(provider, *, api_key=None, timeout=None, max_retries=None, environ=None):
        def create(**kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))], usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client.with_options = lambda **o: seen["deepseek"].append((o.get("timeout"), max_retries)) or client
        return client

    monkeypatch.setattr("skydango.models.claude_code.one_shot_message", one_shot_message)
    monkeypatch.setattr("skydango.models.openai_compat.build_client", build_client)
    cfg = Config()
    cfg.models = {"reflect": {"main": "claude/sonnet", "backup": "deepseek/deepseek-chat"}}
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False
    store = MemoryStore(cfg.reply.memory_dir)
    ledger = cli._inner_ledger(cfg, store, time.time())
    registry = cli._registry(cfg, tmp_path / "run")
    _, reflector = cli._inner_mind(cfg, ledger, registry)
    body = SimpleNamespace(
        reflect_materials=lambda final: "材料：今天和阿花玩了", mind=Mind(), _safe_friends=lambda: [], persona=None,
        soft_names_this_session=lambda: set(), mind_log=None, _energy=None,
    )
    return cfg, store, ledger, reflector, body, registry, seen


def test_final_reflection_shortens_timeout(tmp_path, monkeypatch):  # 终审 I1
    cfg, store, ledger, reflector, body, _, seen = _final_parts(tmp_path, monkeypatch)
    assert reflector.llm.timeout == cfg.inner.reflect_timeout > cli._final_timeout(cfg)
    cli._final_reflection(cfg, body, reflector, ledger, store)
    assert seen["claude"] == [cli._final_timeout(cfg)]  # 管理面板 stop_timeout 到之前就得写完日记


@pytest.mark.parametrize("how", ["closed", "auth"])
def test_final_reflection_writes_diary_via_backup(tmp_path, monkeypatch, how):  # 终审 M1：闸关着 / Claude 一直 401，日记照样写
    from skydango.brain.claude import ClaudeError

    reply = json.dumps({"mood": {"level": "开心", "text": "今天和阿花玩了"}, "diary": "今天和阿花去了雨林。", "memos": []},
                       ensure_ascii=False)
    cfg, store, ledger, reflector, body, registry, seen = _final_parts(
        tmp_path, monkeypatch, reply, claude_exc=ClaudeError("Claude Code 这一轮失败：401 Invalid bearer token", auth=True))
    if how == "closed":
        registry.gates.trip("claude", "limit", "429")
    cli._final_reflection(cfg, body, reflector, ledger, store)
    assert len(seen["claude"]) == (0 if how == "closed" else 1) and not registry.gates.ok("claude")
    assert seen["deepseek"] == [(cli._final_timeout(cfg), 0)]  # 备用也按下线的预算：缩短、不重试
    assert ledger.store.last_diaries(1) and "今天和阿花去了雨林。" in ledger.store.last_diaries(1)[0]


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


# ---- 大脑的主 / 备用和按供应商的闸（spec 2026-10-05-model-providers §2.2 §2.3）----
def _fake_deepseek(monkeypatch):
    from types import SimpleNamespace

    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="好", tool_calls=None))], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **o: client
    monkeypatch.setattr("skydango.models.openai_compat.build_client", lambda provider, **kw: client)


def test_run_brain_backup_session_is_lazy(tmp_path, monkeypatch):
    from skydango.brain.session import BrainSession
    from skydango.brain.toolloop import ToolLoopBrain

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.models["brain"] = {"main": "claude/sonnet", "backup": "deepseek/deepseek-chat"}
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    _fake_deepseek(monkeypatch)
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: seen.append(parts.brain))
    brain = seen[0]
    assert isinstance(brain.session, BrainSession) and brain.session.provider == "claude"
    backup = brain.fallback()
    assert isinstance(backup, ToolLoopBrain) and backup.provider == "deepseek" and backup.model == "deepseek-chat"


def test_run_brain_shares_one_registry(tmp_path, monkeypatch):
    from skydango.models.errors import ModelUnavailable
    from skydango.models.registry import GatedCall

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.reply.memory_dir = str(tmp_path / "memory")
    cfg.reply.dry_run = False  # live 才有 NotesKeeper
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=seen.append)
    parts = seen[0]
    gates = parts.brain.gates
    notes_llm, reflect_llm = parts.body.notes.llm, parts.reflector.llm
    assert isinstance(notes_llm, GatedCall) and isinstance(reflect_llm, GatedCall)
    assert notes_llm.registry.gates is gates and reflect_llm.registry.gates is gates  # 一次运行一套闸
    assert parts.eyes.available() is True
    assert "模型：大脑 claude/sonnet" in parts.body.status()
    gates.trip("claude", "auth", "401")
    assert parts.eyes.available() is False  # 眼睛没备用：闸关了就不看
    assert "眼睛用不了" in parts.body.status()
    with pytest.raises(ModelUnavailable):
        parts.eyes.describe([{"type": "text", "text": "看"}])  # 不起 claude 进程，直接抛
    assert notes_llm.complete("你负责记笔记", [{"role": "user", "content": "整理"}]) == "收到：整理"  # 改走备用（echo）


def test_run_brain_sandbox_scene_describe_not_gated(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    world = cli._game_world(cfg, run, True)

    def scene_text(content):
        return "沙盒场景"

    world.describe = scene_text  # 沙盒自己的场景描述：不走 Claude，闸关了也照样看
    seen = []
    cli._run_brain(cfg, run, world, 1.0, on_ready=seen.append)
    assert seen[0].eyes.describe is scene_text and seen[0].eyes.available is None


def test_wardrobe_describe_follows_models(tmp_path):
    from types import SimpleNamespace

    from skydango.models.errors import ModelUnavailable

    cfg = Config()
    env = SimpleNamespace(appearance=SimpleNamespace(load_cards=lambda cards: None), on_described=lambda *a: None)
    registry = cli._registry(cfg, tmp_path)
    w = cli._wardrobe(cfg, env, None, SimpleNamespace(name="game"), registry, time.monotonic)
    assert w is not None and w.available() is True
    registry.gates.trip("deepseek", "limit", "402")  # 主 deepseek/deepseek-flash、备 claude/haiku：两家都关才停
    assert w.available() is True
    registry.gates.trip("claude", "limit", "429")
    assert w.available() is False
    with pytest.raises(ModelUnavailable):
        w.describe([{"type": "text", "text": "看"}])


# ---- 读聊天走无障碍节点（spec 2026-10-04-a11y-chat-reader §5~§6）----
class FakeA11yClient:
    made = []

    def __init__(self, adb_path, serial, **kwargs):
        self.args = (adb_path, serial)
        self.starts = self.stops = 0
        self.alive = True
        self.error = ""
        FakeA11yClient.made.append(self)

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1

    def latest(self, max_age=None):
        return None

    def others_running(self):
        return False


def chat_cfg(monkeypatch, source="a11y", mode="log"):
    FakeA11yClient.made = []
    monkeypatch.setattr("skydango.device.a11y.A11yReader", FakeA11yClient)
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda engine, threads=8: FakeOcr())
    cfg = Config()
    cfg.vision.mode, cfg.vision.source = mode, source
    cfg.reply.memory_dir = ""
    return cfg


def test_chat_reader_a11y_source_starts_client(monkeypatch):
    from skydango.chat.a11yreader import A11yChatReader, FallbackReader

    cfg = chat_cfg(monkeypatch)
    reader, sf = cli._chat_reader(cfg, FakeDevice([scene()]))
    assert isinstance(reader, FallbackReader) and isinstance(reader.a11y, A11yChatReader)
    assert isinstance(sf, SelfFilter) and reader.a11y.self_filter is sf and reader.ocr.self_filter is sf
    (client,) = FakeA11yClient.made
    assert client.args == (cfg.device.adb_path, cfg.device.serial) and client.starts == 1
    assert reader.describe() == "无障碍"
    cli._stop_reader(reader)
    assert client.stops == 1


def test_chat_reader_ocr_source_is_plain_chatreader(monkeypatch):
    from skydango.chat.reader import ChatReader

    cfg = chat_cfg(monkeypatch, source="ocr")
    reader, sf = cli._chat_reader(cfg, FakeDevice([scene()]))
    assert type(reader) is ChatReader and FakeA11yClient.made == []


def test_chat_reader_needs_log_mode_and_a_device(monkeypatch):
    from skydango.chat.reader import ChatReader

    cfg = chat_cfg(monkeypatch, mode="bubble")
    assert type(cli._chat_reader(cfg, FakeDevice([scene()]))[0]) is ChatReader
    cfg = chat_cfg(monkeypatch)
    assert type(cli._chat_reader(cfg, None)[0]) is ChatReader  # view --images 回放：没有设备
    assert FakeA11yClient.made == []


def test_chat_reader_rejects_unknown_source(monkeypatch):
    cfg = chat_cfg(monkeypatch, source="ocr2")
    with pytest.raises(ValueError, match="vision.source"):
        cli._chat_reader(cfg, FakeDevice([scene()]))


def test_stop_reader_swallows_errors():
    class Boom:
        def stop(self):
            raise RuntimeError("adb 断了")

    cli._stop_reader(Boom())
    cli._stop_reader(FakeReader())  # 没有 stop 也行


def test_game_world_close_stops_reader(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    stopped = []

    class StoppingReader(FakeReader):
        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(cli, "_chat_reader", lambda cfg, dev, save=None: (StoppingReader(), SelfFilter(60, 0.8, "")))
    world = cli._game_world(cfg, run, True)
    world.close()
    assert stopped == [True]


class StoppingReader(FakeReader):
    def __init__(self, stopped):
        super().__init__()
        self.stopped = stopped

    def stop(self):
        self.stopped.append(True)


def _raise(exc):
    def f(*args, **kwargs):
        raise exc

    return f


def test_game_world_build_interrupted_stops_reader(tmp_path, monkeypatch):  # Ctrl+C / 面板点停止时还在建
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(cli, "_chat_reader", lambda cfg, dev, save=None: (StoppingReader(stopped), SelfFilter(60, 0.8, "")))
    monkeypatch.setattr(cli, "_camera", _raise(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        cli._game_world(cfg, run, True)
    assert stopped == [True]


def test_game_world_close_stops_reader_even_if_panels_fail(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(cli, "_chat_reader", lambda cfg, dev, save=None: (StoppingReader(stopped), SelfFilter(60, 0.8, "")))
    monkeypatch.setattr(cli, "_stop_scene", _raise(RuntimeError("env 停不下来")))
    world = cli._game_world(cfg, run, True)
    with pytest.raises(RuntimeError):
        world.close()
    assert stopped == [True]


def test_run_agent_build_failure_stops_reader(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(cli, "_chat_reader", lambda cfg, dev, save=None: (StoppingReader(stopped), SelfFilter(60, 0.8, "")))
    monkeypatch.setattr(cli, "_camera", _raise(RuntimeError("镜头建不起来")))
    with pytest.raises(RuntimeError, match="镜头"):
        cli._run_agent(cfg, run, no_emotes=True)
    assert stopped == [True]


def test_old_view_command_line_still_runs(tmp_path, monkeypatch, capsys):  # 终审：旧会话写的 run --live --view --no-browser
    seen = []
    monkeypatch.setattr(cli, "_run_brain", lambda *a, **k: seen.append(1))
    monkeypatch.chdir(tmp_path)
    cli.main(["run", "--view", "--no-browser"])
    assert seen == [1] and "--view 已经不用了" in capsys.readouterr().out


def test_chat_reader_saves_raw_snapshots(monkeypatch, tmp_path):
    """跑团子时把原始快照存进运行目录（a11y.jsonl）：出了问题能拿真机录像回放查（10-04 晚）。"""
    cfg = chat_cfg(monkeypatch)
    FakeA11yClient.kwargs = None
    orig_init = FakeA11yClient.__init__

    def init(self, adb_path, serial, **kwargs):
        orig_init(self, adb_path, serial, **kwargs)
        FakeA11yClient.kwargs = kwargs

    monkeypatch.setattr(FakeA11yClient, "__init__", init)
    out = tmp_path / "a11y.jsonl"
    reader, _ = cli._chat_reader(cfg, FakeDevice([scene()]), out)
    on_line = FakeA11yClient.kwargs["on_line"]
    on_line(b'{"t":1}')
    on_line(b'{"t":2}')
    cli._stop_reader(reader)
    assert out.read_bytes() == b'{"t":1}\n{"t":2}\n'



def test_run_brain_exposes_usage(tmp_path, monkeypatch):   # spec 2026-10-06-model-usage §6.1
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=seen.append)
    parts = seen[0]
    snap = parts.usage()
    assert snap["source"] == "live" and parts.brain.meter is not None
    assert parts.usage.__self__ is parts.brain.meter
