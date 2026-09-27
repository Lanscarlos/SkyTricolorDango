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


def test_no_emotes_flag_or_disabled(cfg):
    assert cli._build_emotes(cfg, device(), PanelReader(), no_emotes=True) is None
    cfg.emotes.enabled = False
    assert cli._build_emotes(cfg, device(), PanelReader(), no_emotes=False) is None


def test_empty_library(cfg, tmp_path):
    cfg.wheel.library_dir = str(tmp_path / "没有")
    assert cli._build_emotes(cfg, device(), PanelReader(), no_emotes=False) is None


def test_start_failure_disables_emotes(cfg, monkeypatch):
    def boom(self):
        raise WheelError("没能打开轮盘编辑界面")

    monkeypatch.setattr(EmotePlayer, "start", boom)
    assert cli._build_emotes(cfg, device(), PanelReader(), no_emotes=False) is None


def test_builds_player(cfg, monkeypatch):
    monkeypatch.setattr(EmotePlayer, "start", lambda self: None)
    player = cli._build_emotes(cfg, device(), PanelReader(), no_emotes=False)
    assert isinstance(player, EmotePlayer)
    assert player.panel_key == cfg.vision.log_open_key and player.panel_visible() is True
