"""感知层三期的命令行：perception unknown-names、places、perception clips / gesture-eval。"""

import json

from skydango import cli


def test_perception_unknown_names_lists_by_count(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "20260928-100000-dry" / "unknown_names"
    folder.mkdir(parents=True)
    rows = [
        {"name": "甲", "count": 1, "first": "2026-09-28 10:00:01", "last": "2026-09-28 10:00:01", "image": "001.jpg"},
        {"name": "丙", "count": 5, "first": "2026-09-28 10:00:01", "last": "2026-09-28 10:03:00", "image": "002.jpg"},
        {"name": "乙", "count": 9, "first": "2026-09-28 10:00:01", "last": "2026-09-28 10:03:00", "image": "003.jpg"},
    ]
    (folder / "names.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: ["乙"]))
    cli.main(["perception", "unknown-names", "--runs", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.index("丙") < out.index("甲") and "乙" not in out.split("\n", 1)[1]
    assert str(folder / "002.jpg") in out and "friends.md" in out


def test_perception_unknown_names_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: []))
    cli.main(["perception", "unknown-names", "--runs", str(tmp_path)])
    assert "没有" in capsys.readouterr().out


class FarDetector:
    imgsz = 960
    providers = ["Fake"]

    def detect(self, img):
        from skydango.vision.bubbles import Rect
        from skydango.vision.detect import Detection

        if img.shape[:2] == (1080, 1920):
            return [Detection("player", Rect(1000, 500, 24, 60), 0.9)]
        return []


def fake_models(monkeypatch):
    from skydango.vision import detect, ocr

    monkeypatch.setattr(detect, "make_detector", lambda *a, **k: FarDetector())
    monkeypatch.setattr(ocr, "make_ocr", lambda *a, **k: type("O", (), {"read_line": lambda self, img: None})())


def test_perception_bench_far_crops_switch(tmp_path, monkeypatch, capsys):
    import numpy as np

    from skydango.imageio import imwrite

    imwrite(tmp_path / "a.jpg", np.zeros((1080, 1920, 3), np.uint8))
    fake_models(monkeypatch)
    cli.main(["perception", "bench", "--images", str(tmp_path), "-n", "3"])
    assert "远处二次检测 " in capsys.readouterr().out
    cli.main(["perception", "bench", "--images", str(tmp_path), "-n", "3", "--far-crops", "0"])
    assert "远处二次检测 0 次" in capsys.readouterr().out
