import json

from conftest import FakeDevice, scene
from test_brain_body import FakeReader

from skydango import cli
from skydango.chat.tracker import SelfFilter


def fake_env(monkeypatch):
    dev = FakeDevice([scene()])
    monkeypatch.setattr(cli, "_device", lambda cfg: dev)
    monkeypatch.setattr(cli, "_build_reader", lambda cfg: (FakeReader(), SelfFilter(60, 0.8, "")))
    return dev


def test_camera_spin_saves_frames(tmp_path, monkeypatch, capsys):
    dev = fake_env(monkeypatch)
    out = tmp_path / "s"
    cli.main(["camera", "spin", "--seconds", "0.2", "-o", str(out)])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["turns"] == 1 and summary["frames"] >= 1
    assert ("hw_down", 106) in dev.calls and ("hw_up", 106) in dev.calls
    printed = capsys.readouterr().out
    assert "张" in printed and str(out) in printed


def test_camera_spin_turns_clamped(tmp_path, monkeypatch):
    fake_env(monkeypatch)
    out = tmp_path / "s"
    cli.main(["camera", "spin", "--turns", "5", "--seconds", "0.1", "-o", str(out)])
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["turns"] == 2


# ---- perception label --spin ----
import pytest  # noqa: E402

from skydango.imageio import imwrite  # noqa: E402
from skydango.vision.bubbles import Rect  # noqa: E402
from skydango.vision.detect import Detection  # noqa: E402
from skydango.vision.weaklabel import with_self  # noqa: E402


def test_with_self_replaces_overlapping_player():
    boxes = [("player", Rect(900, 500, 90, 220)), ("player", Rect(100, 500, 90, 220)), ("name_tag", Rect(900, 440, 90, 40))]
    out = with_self(boxes, Rect(902, 502, 90, 220))
    assert ("self", Rect(902, 502, 90, 220)) in out and ("player", Rect(900, 500, 90, 220)) not in out and len(out) == 3


def test_label_spin_requires_model(tmp_path):
    with pytest.raises(SystemExit, match="--model"):
        cli.main(["perception", "label", str(tmp_path), "--spin", "-o", str(tmp_path / "ds")])


def test_label_spin_adds_self_boxes(tmp_path, monkeypatch):
    import numpy as np

    src = tmp_path / "spin"
    src.mkdir()
    img = np.zeros((1080, 1920, 3), np.uint8)
    for name in ["before", "after"] + [f"{i:03d}_{i * 0.07:.2f}s" for i in range(8)]:
        imwrite(src / f"{name}.jpg", img)

    class Det:
        def detect(self, frame):  # 团子每帧都在中间；另一个人只在一帧里
            return [Detection("player", Rect(900, 500, 90, 220), 0.9)]

    class Ocr:
        def recognize(self, frame):
            return []

    monkeypatch.setattr("skydango.vision.detect.make_detector", lambda *a, **k: Det())
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda *a, **k: Ocr())
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    out = tmp_path / "ds"
    cli.main(["perception", "label", str(src), "--spin", "--model", "m.onnx", "-o", str(out), "--val", "0"])
    labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert len(labels) == 8  # before / after 不要
    for f in labels:
        [line] = f.read_text(encoding="utf-8").splitlines()
        assert line.startswith("3 ")  # self 的类别号


def test_look_describes_given_image(tmp_path, monkeypatch, capsys):
    import numpy as np

    img = tmp_path / "a.jpg"
    imwrite(img, np.full((1080, 1920, 3), 80, np.uint8))

    class Env:
        labels = {}

        def observe(self, frame, now, panel_visible):
            pass

    got = []
    monkeypatch.setattr(cli, "_brain_env", lambda cfg: (["claude"], {}))
    monkeypatch.setattr(cli, "_device", lambda cfg: pytest.fail("给了图片就不截屏"))
    monkeypatch.setattr(cli, "_scene_watcher", lambda cfg, **kw: Env())
    monkeypatch.setattr("skydango.brain.claude.one_shot", lambda *a: got.append(a) or "描述")
    monkeypatch.chdir(tmp_path)
    cli.main(["look", str(img)])
    out = capsys.readouterr().out
    assert "图里没认出好友的名字" in out and "描述" in out and len(got) == 1
