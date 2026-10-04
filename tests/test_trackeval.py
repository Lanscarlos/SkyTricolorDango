"""perception track-eval（spec 2026-10-01-tracking-relink-motion §6）：基线 vs 当前配置。"""

import numpy as np

from skydango.config import PerceptionConfig
from skydango.vision.trackeval import SWITCHES, baseline, evaluate, report_md, subsample

from test_perception_tracking import NAME, make, player, tag

STEP = 0.1


def scene(t):
    """好友带标签 → 标签消失、分数掉到 0.3 → 分数回 0.9（没标签）→ 标签再亮。"""
    if t < 1.0:
        return [player(800), tag(800)]
    if t < 3.0:
        return [player(800, s=0.3)]
    if t < 5.0:
        return [player(800)]
    return [player(800), tag(800)]


class TimedDetector:
    """帧左上角像素存着第几帧：按时间出框。"""

    def detect(self, img):
        return scene(round(int(img[0, 0, 0]) * STEP, 3))


def items(n=60):
    for i in range(n):
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[0, 0, 0] = i
        yield round(i * STEP, 3), f"{i:04d}.jpg", img, False


def watcher(**cfg):
    w, _ = make(**cfg)
    w.detector = TimedDetector()
    return w


def test_baseline_turns_switches_off():
    cfg = baseline(PerceptionConfig(conf=0.4))
    assert cfg.conf == 0.4 and not any(getattr(cfg, k) for k in SWITCHES)


def test_baseline_reports_break_and_wronged_current_does_not():
    off = {k: False for k in SWITCHES}
    base = evaluate(items(), watcher(**off), keep=5.0)
    cur = evaluate(items(), watcher(), keep=5.0)
    b, c = base["friends"][NAME], cur["friends"][NAME]
    assert b["breaks"].get("低分框", 0) >= 1 and b["wronged"] == 1
    assert sum(c["breaks"].values()) == 0 and c["wronged"] == 0
    md = report_md(base, cur, {"source": "合成", "frames": 60, "fps": 10})
    assert "基线" in md and "当前配置" in md and NAME in md


def test_relink_verdicts():
    def lost_then_back(t):
        if t < 1.0:
            return [player(800), tag(800)]
        if t < 2.2:
            return []
        if t < 3.0:
            return [player(830)]
        return [player(830), tag(830)]

    w = watcher()
    w.detector.detect = lambda img: lost_then_back(round(int(img[0, 0, 0]) * STEP, 3))
    out = evaluate(items(40), w, keep=5.0)
    assert out["friends"][NAME]["relinks"] == {"对": 1, "错": 0, "未证实": 0}


def test_subsample_by_recording_time():
    kept = [t for t, *_ in subsample(items(20), fps=2)]
    assert kept == [0.0, 0.5, 1.0, 1.5]


def test_cli_track_eval_writes_report(tmp_path, monkeypatch):
    import json

    from conftest import FakeOcr

    from skydango import cli
    from skydango.imageio import imwrite

    class OnePlayer:
        providers = ["Fake"]
        imgsz = 960

        def detect(self, img):
            return [player(800)]

    src = tmp_path / "rec"
    src.mkdir()
    for i in range(20):
        imwrite(src / f"{i:04d}_{i * 0.1:06.2f}s.jpg", np.full((1080, 1920, 3), 90, np.uint8))
    monkeypatch.setattr("skydango.vision.detect.make_detector", lambda *a, **k: OnePlayer())
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda *a, **k: FakeOcr())
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: []))
    built = []  # 每次建感知层时 [attrs] 开没开：基线不接第二层
    monkeypatch.setattr(cli, "_person_attrs", lambda cfg, *a: built.append(cfg.attrs.enabled))
    out = tmp_path / "out"
    (tmp_path / "config.toml").write_text("[attrs]\nenabled = true\n", encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "track-eval", str(src), "--fps", "5", "-o", str(out)])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline"]["frames"] == summary["current"]["frames"] == 10
    assert "# 追踪和接回" in (out / "report.md").read_text(encoding="utf-8")
    assert built == [True, False]  # 先建当前配置、再建基线


def test_breaks_not_recounted_on_frames_without_update():
    """被挡住提前返回的帧（没跑 tracker.update）不能把上一帧的 dropped 再算一遍。"""
    w = watcher(**{k: False for k in SWITCHES})
    orig = w.process
    w.process = lambda f, t, p: orig(f, t, p) if not 2.05 <= t < 2.6 else None
    out = evaluate(items(40), w, keep=5.0)
    assert sum(out["friends"][NAME]["breaks"].values()) == 1


def test_relink_refuted_by_tag_elsewhere_counts_wrong():
    def scene2(t):
        if t < 1.0:
            return [player(800), tag(800)]
        if t < 2.2:
            return []
        if t < 3.0:
            return [player(830)]
        return [player(830), player(300), tag(300)]  # 他本人的标签在别处亮出来

    w = watcher()
    w.detector.detect = lambda img: scene2(round(int(img[0, 0, 0]) * STEP, 3))
    out = evaluate(items(40), w, keep=5.0)
    assert out["friends"][NAME]["relinks"]["错"] == 1
