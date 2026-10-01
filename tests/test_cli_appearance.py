"""认装扮的接线（spec 2026-10-01-appearance，Task 10）：_scene_watcher / _perception 挂记忆簿、_run_brain 起描述器。"""

import argparse
import threading
import time

from conftest import FakeOcr
from test_cli_brain import fake_brain_run

from skydango import cli
from skydango.brain.prompt import APPEARANCE_RULES
from skydango.config import Config
from skydango.runlog import RunDir
from skydango.vision.appearance import AppearanceBook, CropSaver


class EmptyDetector:
    providers = ["Fake"]
    imgsz = 960

    def detect(self, img):
        return []


def fake_models(monkeypatch):
    monkeypatch.setattr("skydango.vision.detect.make_detector", lambda *a, **k: EmptyDetector())
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda *a, **k: FakeOcr())


def test_scene_watcher_attaches_appearance_book(monkeypatch, tmp_path):
    fake_models(monkeypatch)
    cfg = Config()
    cfg.run.dir = str(tmp_path / "runs")
    cfg.perception.enabled = True
    cfg.perception.keep = 7.0
    cfg.appearance.enabled = True
    run = RunDir.create(cfg, "dry")
    w = cli._scene_watcher(cfg, run=run)
    assert isinstance(w.appearance, AppearanceBook) and w.appearance.key == "color-v1" and w.appearance.keep == 7.0
    assert w.embedder is not None and w.appearance_cfg is cfg.appearance
    assert isinstance(w.saver, CropSaver) and w.saver.folder == run.path / "appearance"
    assert w.saver.save_every == cfg.appearance.save_every and w.saver.save_max == cfg.appearance.save_max
    assert w.wardrobe is None  # 描述器由 _run_brain 挂
    assert cli._scene_watcher(cfg).saver is None  # 没有运行目录：不存
    cfg.appearance.save = False
    assert cli._scene_watcher(cfg, run=run).saver is None
    cfg.appearance.enabled = False
    w = cli._scene_watcher(cfg, run=run)
    assert w.appearance is None and w.saver is None


def test_appearance_without_perception_warns(monkeypatch, caplog):
    fake_models(monkeypatch)
    cfg = Config()
    cfg.appearance.enabled = True
    with caplog.at_level("WARNING"):
        w = cli._scene_watcher(cfg, background=False)
    assert "[appearance] 要配合 [perception] 用，现在没打开，不认装扮" in caplog.text
    assert getattr(w, "appearance", None) is None


def test_perception_command_gets_book_but_no_saver(monkeypatch):
    fake_models(monkeypatch)
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    cfg = Config()
    cfg.appearance.enabled = True
    args = argparse.Namespace(model=None, device=None, imgsz=None, far_crops=None)
    _, w = cli._perception(cfg, args)
    assert isinstance(w.appearance, AppearanceBook) and w.saver is None and w.wardrobe is None
    cfg.appearance.enabled = False
    _, w = cli._perception(cfg, args)
    assert w.appearance is None


def test_inner_ledger_keeps_configured_outfits(tmp_path):
    from skydango.chat.memory import MemoryStore

    cfg = Config()
    cfg.appearance.outfit_keep = 5
    ledger = cli._inner_ledger(cfg, MemoryStore(tmp_path / "mem"), 1_790_000_000.0)
    assert ledger is not None and ledger.outfit_keep == 5


def brain_with_perception(tmp_path, monkeypatch):
    """_run_brain + 真的 PerceptionWatcher（假检测器什么都认不出）+ 记忆簿；返回 (cfg, run, world, 记录)。"""
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    fake_models(monkeypatch)
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    cfg.env.enabled = True
    cfg.perception.enabled = True
    cfg.appearance.enabled = True
    cfg.reply.memory_dir = str(tmp_path / "mem")
    seen = {"cards": [], "runs": []}
    outfits = {"小明": [{"desc": "红斗篷", "key": "color-v1", "feat": [1.0, 0.0]}]}
    monkeypatch.setattr("skydango.inner.ledger.Ledger.all_outfits", lambda self: outfits)
    real_load = AppearanceBook.load_cards
    monkeypatch.setattr(AppearanceBook, "load_cards", lambda self, o: seen["cards"].append(o) or real_load(self, o))

    def fake_run(self, stop):
        seen["runs"].append((threading.current_thread().name, threading.current_thread().daemon, stop))
        stop.wait()

    monkeypatch.setattr("skydango.vision.wardrobe.Wardrobe.run", fake_run)
    world = cli._game_world(cfg, run, True)
    world.clock = lambda: time.monotonic()  # 换一个认得出的钟
    return cfg, run, world, seen


def test_run_brain_loads_cards_and_starts_wardrobe(tmp_path, monkeypatch):
    cfg, run, world, seen = brain_with_perception(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr("skydango.brain.claude.one_shot", lambda *a: calls.append(a) or "描述")
    cli._run_brain(cfg, run, world, 2.0)
    env = world.env
    assert seen["cards"] == [{"小明": [{"desc": "红斗篷", "key": "color-v1", "feat": [1.0, 0.0]}]}]
    wardrobe = env.wardrobe
    assert wardrobe is not None and wardrobe.clock is world.clock  # 和感知层的帧时间同一个钟
    assert wardrobe.on_done == env.on_described and wardrobe.cfg is cfg.appearance
    [(name, daemon, stop)] = seen["runs"]
    assert name == "wardrobe" and daemon and stop.is_set()  # 和眼睛同一个 stop，下线时一起停
    calls.clear()  # 眼睛可能也调过
    assert wardrobe.describe(["图"]) == "描述"
    [(cmd, _env, cwd, content, timeout)] = calls
    assert cmd[cmd.index("--model") + 1] == cfg.appearance.describe_model
    assert cwd == run.path / "brain" / "wardrobe" and content == ["图"] and timeout == cfg.appearance.describe_timeout
    prompt = (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")
    assert APPEARANCE_RULES.splitlines()[0] in prompt


def test_run_brain_describe_off_has_no_wardrobe(tmp_path, monkeypatch):
    cfg, run, world, seen = brain_with_perception(tmp_path, monkeypatch)
    cfg.appearance.describe = False
    cli._run_brain(cfg, run, world, 1.0)
    assert world.env.wardrobe is None and seen["runs"] == []
    assert len(seen["cards"]) == 1  # 关系卡里的旧外观照样载入
    assert APPEARANCE_RULES.splitlines()[0] in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")


def test_run_brain_sandbox_never_describes(tmp_path, monkeypatch):
    cfg, run, world, seen = brain_with_perception(tmp_path, monkeypatch)
    world.name = "sandbox"
    cli._run_brain(cfg, run, world, 1.0)
    assert world.env.wardrobe is None and seen["runs"] == []


def test_run_brain_without_ledger_skips_cards(tmp_path, monkeypatch):
    cfg, run, world, seen = brain_with_perception(tmp_path, monkeypatch)
    cfg.inner.enabled = False
    cli._run_brain(cfg, run, world, 1.0)
    assert seen["cards"] == [] and world.env.wardrobe is not None


def test_run_brain_without_appearance_prompt_unchanged(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)  # env 关着
    cfg.appearance.enabled = True
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0)
    assert APPEARANCE_RULES.splitlines()[0] not in (run.path / "brain" / "session" / "prompt.md").read_text(encoding="utf-8")


def test_run_brain_bad_cards_only_logged(tmp_path, monkeypatch, caplog):
    cfg, run, world, seen = brain_with_perception(tmp_path, monkeypatch)

    def boom(self, outfits):
        raise ValueError("坏了")

    monkeypatch.setattr(AppearanceBook, "load_cards", boom)
    with caplog.at_level("ERROR"):
        cli._run_brain(cfg, run, world, 1.0)
    assert "关系卡里的装扮载入出错" in caplog.text and world.env.wardrobe is not None
