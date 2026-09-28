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


def _scene(color, seed):
    import numpy as np

    ramp = np.linspace(0.3, 1.0, 1080)[:, None, None]
    img = (ramp * np.array(color, float)[None, None, :]).repeat(1920, axis=1)
    img = img + np.random.default_rng(seed).normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_places_add_test_and_bench(tmp_path, monkeypatch, capsys):
    from skydango.imageio import imwrite

    monkeypatch.chdir(tmp_path)
    shots = tmp_path / "shots"
    shots.mkdir()
    for i in range(3):
        imwrite(shots / f"yun{i}.jpg", _scene((40, 200, 60), i))
        imwrite(shots / f"yu{i}.jpg", _scene((200, 80, 40), i))
    for i in range(2):
        cli.main(["places", "add", "云野", "--image", str(shots / f"yun{i}.jpg"), "--model", "thumb"])
        cli.main(["places", "add", "雨林", "--image", str(shots / f"yu{i}.jpg"), "--model", "thumb"])
    out = capsys.readouterr().out
    assert "云野" in out and len(list((tmp_path / "places" / "云野").glob("*.jpg"))) == 2
    probe = tmp_path / "probe"
    probe.mkdir()
    imwrite(probe / "a.jpg", _scene((40, 200, 60), 7))
    cli.main(["places", "test", str(probe), "--model", "thumb"])
    out = capsys.readouterr().out
    assert "a.jpg" in out and "云野" in out
    cli.main(["places", "bench", "--model", "thumb"])
    out = capsys.readouterr().out
    assert "thumb" in out and "认对 4/4" in out


def test_places_test_with_empty_library_explains(tmp_path, monkeypatch):
    import pytest

    monkeypatch.chdir(tmp_path)
    (tmp_path / "p").mkdir()
    with pytest.raises(SystemExit):
        cli.main(["places", "test", str(tmp_path / "p"), "--model", "thumb"])
