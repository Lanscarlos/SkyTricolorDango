import argparse

from skydango import cli
from skydango.config import Config, load_config
from skydango.console.settings import FIELDS
from skydango.vision.identity import IdentityBoard
from test_cli_appearance import fake_models
from test_perception_appearance import FakeDino


def test_identity_board_requirements():
    c = Config()
    assert cli._identity_board(c, None) == (None, '')
    c.identity.enabled = c.perception.enabled = True
    assert '要先开认装扮' in cli._identity_board(c, object())[1]
    c.appearance.enabled = True
    assert 'DINOv2 没加载' in cli._identity_board(c, None)[1]
    board, note = cli._identity_board(c, object())
    assert isinstance(board, IdentityBoard) and '外观为主' in note
    c.perception.enabled = False
    assert cli._identity_board(c, object())[0] is None


def test_both_watcher_constructors_attach_identity(monkeypatch):
    fake_models(monkeypatch)
    monkeypatch.setattr(cli, '_icon_classifier', lambda cfg: None)
    monkeypatch.setattr(cli, '_dino_embedder', lambda cfg: FakeDino())
    c = Config()
    c.identity.enabled = c.perception.enabled = c.appearance.enabled = True
    c.call.window = 4.5
    a = argparse.Namespace(model=None, device=None, imgsz=None, far_crops=None)
    for w in (cli._scene_watcher(c, background=False), cli._perception(c, a)[1]):
        assert isinstance(w.identity, IdentityBoard)
        assert w.identity.call_window == w.call_window == 4.5


def test_identity_setting_loads_from_overlay(tmp_path):
    overlay = tmp_path / 'console.toml'
    overlay.write_text('[identity]\nenabled = true\n', encoding='utf-8')
    c = load_config(None, overlay)
    assert c.identity.enabled
    field = next(f for f in FIELDS if f.key == 'identity.enabled')
    assert field.kind == 'bool'
