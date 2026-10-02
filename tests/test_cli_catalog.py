"""图鉴收集的接线和离线工具（spec 2026-10-02-catalog-collect §4、§6）。"""

import time
from pathlib import Path

from test_cli_appearance import fake_models

from skydango import cli
from skydango.config import Config
from skydango.runlog import RunDir
from skydango.vision.catalog import CatalogCollector


def cfg_for(tmp_path):
    cfg = Config()
    cfg.run.dir = str(tmp_path / "runs")
    cfg.catalog.dir = str(tmp_path / "catalog")
    cfg.perception.enabled = True
    return cfg


def test_scene_watcher_attaches_collector_only_with_run(monkeypatch, tmp_path):
    fake_models(monkeypatch)
    cfg = cfg_for(tmp_path)
    run = RunDir.create(cfg, "dry")
    w = cli._scene_watcher(cfg, run=run)
    assert isinstance(w.catalog, CatalogCollector)
    assert w.catalog.folder == Path(cfg.catalog.dir) / "inbox" / time.strftime("%Y-%m-%d") / run.path.name
    assert cli._scene_watcher(cfg).catalog is None  # 没有运行目录（view 等）：不收
    cfg.catalog.enabled = False
    assert cli._scene_watcher(cfg, run=run).catalog is None
    w.stop()


def test_stop_scene_reports_saved(capsys, tmp_path):
    class Env:
        hardcases = None

        def __init__(self, catalog):
            self.catalog = catalog

        def stop(self):
            pass

    class Saved:
        saved = 3
        folder = tmp_path / "x"

    cli._stop_scene(Env(Saved()))
    assert "图鉴收集：存了 3 张" in capsys.readouterr().out
    cli._stop_scene(Env(None))  # 没收集器：不报错


def test_cli_catalog_collect_on_recording(tmp_path, monkeypatch):
    import json

    import numpy as np
    from conftest import FakeOcr

    from skydango.imageio import imwrite
    from skydango.vision.bubbles import Rect
    from skydango.vision.detect import Detection

    class BigStranger:
        providers = ["Fake"]
        imgsz = 960

        def detect(self, img):
            return [Detection("player", Rect(800, 300, 200, 400), 0.9)]

    rng = np.random.default_rng(0)
    src = tmp_path / "rec-1"
    src.mkdir()
    for i in range(20):
        imwrite(src / f"{i:04d}_{i * 0.5:06.2f}s.jpg", rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8))
    monkeypatch.setattr("skydango.vision.detect.make_detector", lambda *a, **k: BigStranger())
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda *a, **k: FakeOcr())
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: []))
    monkeypatch.setattr(cli, "_person_attrs", lambda cfg: None)
    out = tmp_path / "out"
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "catalog", "collect", str(src), "--fps", "2", "-o", str(out)])
    folder = out / "inbox" / "offline" / "rec-1"
    index = [json.loads(x) for x in (folder / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert index and all(r["who"].startswith("陌生人-t") for r in index)
    cands = [json.loads(x) for x in (out / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cands) >= len(index) and {"sharp", "height", "fail"} <= set(cands[0])
    assert (out / "sheet.jpg").exists()
