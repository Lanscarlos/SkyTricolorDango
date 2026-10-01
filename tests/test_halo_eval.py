"""perception halo-eval：录像上画头顶亮度曲线、给建议的 halo_rise（spec 2026-10-01-q-call §5）。"""

from types import SimpleNamespace

import numpy as np

from skydango.vision.bubbles import Rect
from skydango.vision.halo import head_region
from skydango.vision.halo_eval import Curves, plot, report_md, segments, suggest

W, H = 1920, 1080
A = Rect(940, 500, 60, 150)
B = Rect(300, 500, 60, 150)


def synth():
    rng = np.random.default_rng(0)
    c = Curves()
    tracks = [SimpleNamespace(id=1, cls="self", box=A), SimpleNamespace(id=2, cls="player", box=B),
              SimpleNamespace(id=3, cls="name_tag", box=Rect(0, 0, 10, 10))]
    for i in range(30):
        f = np.full((H, W, 3), 80, np.uint8)
        ra, rb = head_region(A, W, H), head_region(B, W, H)
        if 10 <= i <= 15:
            f[ra.y:ra.y2, ra.x:ra.x2] = 140  # 团子按了 Q
        f[rb.y:rb.y2, rb.x:rb.x2] = 80 + int(rng.integers(-3, 4))  # 旁边的人：噪声
        c.add(f, i / 30, tracks)
    return c


def test_rises_and_segments():
    rises = synth().rises()
    assert set(rises) == {1, 2}  # 只看人物和团子
    assert max(v for _, v in rises[1]) > 55
    assert max(abs(v) for _, v in rises[2]) < 8
    (seg,) = segments(rises[1], 25.0)
    assert 10 / 30 - 1e-6 <= seg[0] < seg[1] <= 15 / 30 + 1e-6 and seg[2] > 55
    assert segments(rises[2], 25.0) == []


def test_suggest_between_noise_and_peak():
    s = suggest(synth().rises())
    assert 3 <= s["noise"] < 8 and s["peak"] > 55
    assert s["noise"] < s["halo_rise"] < s["peak"]
    assert suggest({1: [(0.0, 2.0), (0.1, 3.0)]})["halo_rise"] is None  # 没有像样的峰：分不开


def test_report_and_plot(tmp_path):
    rises = synth().rises()
    summary = {"source": "tmp/record/x", "frames": 30, "suggest": suggest(rises), "current": 25.0,
               "tracks": {str(k): {"peak": max(v for _, v in pts), "segments": segments(pts, 25.0)} for k, pts in rises.items()}}
    md = report_md(summary)
    assert "建议" in md and "halo_rise" in md
    out = tmp_path / "curves.png"
    plot(rises, out)
    assert out.stat().st_size > 0


def test_cli_halo_eval_runs_on_a_recording(tmp_path, monkeypatch):
    import json

    from conftest import FakeOcr

    from skydango import cli
    from skydango.imageio import imwrite
    from skydango.vision.detect import Detection

    class OnePlayer:
        providers = ["Fake"]
        imgsz = 960

        def detect(self, img):
            return [Detection("player", A, 0.9)]

    src = tmp_path / "rec"
    src.mkdir()
    r = head_region(A, W, H)
    for i in range(20):
        f = np.full((H, W, 3), 90, np.uint8)
        if 8 <= i <= 10:
            f[r.y:r.y2, r.x:r.x2] = 160
        imwrite(src / f"{i:04d}_{i * 0.1:06.2f}s.png", f)
    monkeypatch.setattr("skydango.vision.detect.make_detector", lambda *a, **k: OnePlayer())
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda *a, **k: FakeOcr())
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: []))
    out = tmp_path / "out"
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "halo-eval", str(src), "-o", str(out)])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 20 and summary["suggest"]["peak"] > 50
    assert (out / "curves.png").exists() and "建议 halo_rise" in (out / "report.md").read_text(encoding="utf-8")
