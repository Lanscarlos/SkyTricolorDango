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


class WalkDetector:
    imgsz = 960
    providers = ["Fake"]

    def detect(self, img):
        from skydango.vision.bubbles import Rect
        from skydango.vision.detect import Detection

        return [Detection("player", Rect(900, 400, 90, 220), 0.9)]


def test_perception_clips_from_record_dir(tmp_path, monkeypatch, capsys):
    import numpy as np

    from skydango.imageio import imwrite
    from skydango.vision import detect

    rec = tmp_path / "rec"
    rec.mkdir()
    for i in range(20):
        imwrite(rec / f"{i:04d}_{i * 0.125:06.2f}s.jpg", np.zeros((1080, 1920, 3), np.uint8))
    monkeypatch.setattr(detect, "make_detector", lambda *a, **k: WalkDetector())
    out = tmp_path / "clips"
    cli.main(["perception", "clips", str(rec), "-o", str(out)])
    assert len([p for p in out.iterdir() if p.is_dir()]) == 1
    assert "1 段" in capsys.readouterr().out


def test_perception_gesture_eval_prints_verdict(tmp_path, monkeypatch, capsys):
    import numpy as np

    from skydango.imageio import imwrite
    from skydango.vision import gesture

    for label, value in (("wave", 100), ("none", 0)):
        clip = tmp_path / "data" / label / "c0"
        clip.mkdir(parents=True)
        for i in range(16):
            imwrite(clip / f"{i:02d}.png", np.full((32, 32, 3), value, np.uint8))

    class Fake:
        def __init__(self, *a, **k):
            pass

        def classify(self, clip):
            return ("wave", 0.95) if clip[0].mean() > 50 else ("none", 0.99)

    monkeypatch.setattr(gesture, "OnnxGestureClassifier", Fake)
    cli.main(["perception", "gesture-eval", str(tmp_path / "data"), "--model", "x.onnx"])
    out = capsys.readouterr().out
    assert "wave" in out and "100%" in out and "达标" in out


def test_places_add_works_before_any_feature_model(tmp_path, monkeypatch, capsys):
    from skydango.imageio import imwrite

    monkeypatch.chdir(tmp_path)
    imwrite(tmp_path / "a.jpg", _scene((40, 200, 60), 0))
    cli.main(["places", "add", "云野", "--image", str(tmp_path / "a.jpg")])  # 默认 models/places.onnx 不存在
    assert len(list((tmp_path / "places" / "云野").glob("*.jpg"))) == 1


def test_places_test_builds_the_masking_detector_once(tmp_path, monkeypatch, capsys):
    from skydango.imageio import imwrite
    from skydango.vision import detect

    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "sky-yolo.onnx").write_bytes(b"x")  # 模型文件在：会建检测器遮人
    built = []

    def make(*a, **k):
        built.append(1)
        return FarDetector()

    monkeypatch.setattr(detect, "make_detector", make)
    shots = tmp_path / "shots"
    shots.mkdir()
    imwrite(shots / "a.jpg", _scene((40, 200, 60), 0))
    cli.main(["places", "add", "云野", "--image", str(shots / "a.jpg")])
    for i in range(3):
        imwrite(shots / f"p{i}.jpg", _scene((40, 200, 60), i + 1))
    built.clear()
    cli.main(["places", "test", str(shots), "--model", "thumb"])
    assert len(built) == 1
