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
