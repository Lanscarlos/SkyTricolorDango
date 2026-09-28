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
