import numpy as np
import pytest
from conftest import FakeDevice
from test_wheel import icon

from skydango import cli
from skydango.config import Config
from skydango.game.emotes import EmotePlayer
from skydango.game.wheel import WheelError
from skydango.imageio import imwrite


class PanelReader:
    def panel_visible(self, frame):
        return True


@pytest.fixture
def cfg(tmp_path):
    cfg = Config()
    cfg.vision.mode = "log"
    imwrite(tmp_path / "鞠躬.png", icon("circle"))
    cfg.wheel.library_dir = str(tmp_path)
    return cfg


def device():
    return FakeDevice([np.zeros((1080, 1920, 3), np.uint8)])


def build(cfg, no_emotes):
    dev = device()
    return cli._build_emotes(cfg, dev, cli._panel(cfg, dev, PanelReader()), no_emotes=no_emotes)


def test_no_emotes_flag_or_disabled(cfg):
    assert build(cfg, no_emotes=True) is None
    cfg.emotes.enabled = False
    assert build(cfg, no_emotes=False) is None


def test_empty_library(cfg, tmp_path):
    cfg.wheel.library_dir = str(tmp_path / "没有")
    assert build(cfg, no_emotes=False) is None


def test_start_failure_disables_emotes(cfg, monkeypatch):
    def boom(self):
        raise WheelError("没能打开轮盘编辑界面")

    monkeypatch.setattr(EmotePlayer, "start", boom)
    assert build(cfg, no_emotes=False) is None


def test_builds_player(cfg, monkeypatch):
    monkeypatch.setattr(EmotePlayer, "start", lambda self: None)
    player = build(cfg, no_emotes=False)
    assert isinstance(player, EmotePlayer)
    assert player.panel is not None and player.panel.active
