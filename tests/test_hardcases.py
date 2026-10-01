import json

import numpy as np

from skydango.config import Config, EnvConfig, PerceptionConfig
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.hardcases import HardCaseCollector
from skydango.vision.ocr import OcrLine
from skydango.vision.track import Track

FRIENDS = ["懒洋洋大王", "番茄炒蛋盖饭"]


class FakeOcr:
    def __init__(self, texts):
        self.texts = texts

    def recognize(self, img):
        return [OcrLine(t, 0.99, Rect(700, 300, 120, 40)) for t in self.texts]


def img(v=100):
    out = np.full((1080, 1920, 3), v, np.uint8)
    out[:, : v * 5 % 1920] = 255 - v  # 每个 v 画面都不一样，缩略图差异够大
    return out


def collector(tmp_path, ocr=None, **cfg):
    cfg.setdefault("audit_interval", 0.0)
    return HardCaseCollector(
        tmp_path / "hard", PerceptionConfig(**cfg), EnvConfig(), [0.0, 0.0, 0.335, 0.855], lambda: list(FRIENDS),
        ocr=ocr, background=False, wall=lambda: 1_790_000_000.0,
    )


def track(tid, cls="player", x=1000, hits=1, flips=0, first=0.0, last=0.0):
    t = Track(tid, cls, Rect(x, 400, 90, 220), 0.9, first, last, hits)
    t.flips = flips
    return t


def saved(tmp_path):
    path = tmp_path / "hard.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_low_conf_track_saved_after_half_second(tmp_path):
    c = collector(tmp_path)
    low = [Detection("player", Rect(1500, 400, 90, 220), 0.3)]
    c.check(img(), 0.0, [], low, set(), False)
    c.check(img(), 0.3, [], low, set(), False)
    assert c.saved == 0
    c.check(img(), 0.6, [], low, set(), False)
    rows = saved(tmp_path)
    assert c.saved == 1 and rows[0]["reason"] == "low_conf"
    assert (tmp_path / "hard" / rows[0]["file"]).exists() and ":" not in rows[0]["file"]


def test_flicker_saved_on_third_birth(tmp_path):
    c = collector(tmp_path)
    c.check(img(), 0.0, [track(1)], [], set(), False)
    c.check(img(), 1.1, [track(2)], [], set(), False)
    assert c.saved == 0
    c.check(img(), 2.2, [track(3)], [], set(), False)
    assert [r["reason"] for r in saved(tmp_path)] == ["flicker"]


def test_unlit_vs_player_saved(tmp_path):
    c = collector(tmp_path)
    c.check(img(), 0.0, [track(1, hits=5, flips=1)], [], set(), False)
    assert c.saved == 0
    c.check(img(), 0.1, [track(1, hits=6, flips=2)], [], set(), False)
    rows = saved(tmp_path)
    assert [r["reason"] for r in rows] == ["unlit_vs_player"] and rows[0]["boxes"][0]["cls"] == "player"


def test_audit_ocr_only_and_yolo_only(tmp_path):
    c = collector(tmp_path / "a", FakeOcr(["懒洋洋大王"]), audit_interval=30.0)
    c.check(img(), 0.0, [], [], set(), False)
    assert [r["reason"] for r in saved(tmp_path / "a")] == ["ocr_only"]
    c = collector(tmp_path / "b", FakeOcr([]), audit_interval=30.0)
    c.check(img(), 0.0, [], [], {"番茄炒蛋盖饭"}, False)
    assert [r["reason"] for r in saved(tmp_path / "b")] == ["yolo_only"]
    c.check(img(50), 10.0, [], [], {"番茄炒蛋盖饭"}, False)  # 没到 audit_interval：不核对
    assert c.saved == 1


def test_limits_interval_dedupe_and_max(tmp_path):
    c = collector(tmp_path, hardcase_max=2)
    c.check(img(100), 0.0, [track(1, flips=2)], [], set(), False)
    c.check(img(50), 3.0, [track(2, flips=2)], [], set(), False)  # 5 s 内
    assert c.saved == 1
    c.check(img(100), 6.0, [track(3, flips=2)], [], set(), False)  # 和上一张一样
    assert c.saved == 1
    c.check(img(50), 12.0, [track(4, flips=2)], [], set(), False)
    c.check(img(150), 18.0, [track(5, flips=2)], [], set(), False)  # 到上限了
    assert c.saved == 2


def test_report_saves_with_reason_and_limits(tmp_path):
    c = collector(tmp_path)
    assert c.report(img(100), 0.0, "appearance", "按外观认成 懒洋洋大王，名字标签是 番茄炒蛋盖饭", [track(1)]) is True
    assert c.report(img(50), 1.0, "appearance", "又一次", [track(1)]) is False  # 5 s 内
    (entry,) = saved(tmp_path)
    assert entry["reason"] == "appearance" and entry["detail"].startswith("按外观认成")
    assert entry["file"].endswith("_appearance.jpg") and entry["boxes"][0]["cls"] == "player"


class CountingCollector:
    def __init__(self):
        self.calls = 0

    def check(self, *args):
        self.calls += 1


def test_no_collection_while_held():
    from test_perception import FakeDetector, frame, player, watcher

    det = FakeDetector()
    det.frames = [[player(1000)]]
    w = watcher(det)
    w.hardcases = CountingCollector()
    w.process(frame(), 0.0, panel_visible=False)
    w.hold("camera")
    w.process(frame(), 0.1, panel_visible=False)
    assert w.hardcases.calls == 1


def test_scene_watcher_builds_collector_only_with_run_dir(tmp_path, monkeypatch):
    from skydango import cli
    from skydango.runlog import RunDir
    import skydango.vision.detect as detect
    import skydango.vision.ocr as ocr

    monkeypatch.setattr(detect, "make_detector", lambda *a, **k: object())
    monkeypatch.setattr(ocr, "make_ocr", lambda *a, **k: FakeOcr([]))
    cfg = Config()
    cfg.perception.enabled = True
    cfg.run.dir = str(tmp_path / "runs")
    assert cli._scene_watcher(cfg).hardcases is None
    run = RunDir.create(cfg, "dry")
    w = cli._scene_watcher(cfg, run=run)
    assert w.hardcases is not None and w.hardcases.folder == run.hard
    cfg.perception.hardcases = False
    assert cli._scene_watcher(cfg, run=run).hardcases is None
