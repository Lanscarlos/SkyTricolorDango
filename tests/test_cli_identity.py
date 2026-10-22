"""[identity] 开关接进 run / 离线工具和设置页（spec 2026-10-22-identity-board §2.3、§8）。"""

import logging

from skydango import cli
from skydango.config import Config
from skydango.console.settings import FIELDS
from skydango.vision.identity import IdentityBoard


def _cfg(identity=True, appearance=True, perception=True):
    cfg = Config()
    cfg.identity.enabled = identity
    cfg.appearance.enabled = appearance
    cfg.perception.enabled = perception
    return cfg


def test_board_built_when_all_on():
    board, line = cli._identity_board(_cfg(), object())
    assert isinstance(board, IdentityBoard) and "外观为主" in line and "dino" in line


def test_board_needs_appearance():
    board, line = cli._identity_board(_cfg(appearance=False), object())
    assert board is None and "要先开认装扮" in line


def test_board_needs_dino():
    board, line = cli._identity_board(_cfg(), None)
    assert board is None and "DINOv2 没加载" in line


def test_board_disabled_is_silent():
    assert cli._identity_board(_cfg(identity=False), object()) == (None, "")


def test_board_needs_perception():
    board, line = cli._identity_board(_cfg(perception=False), object())
    assert board is None and line


def test_settings_has_identity_field():
    names = [f.key for f in FIELDS]
    assert "identity.enabled" in names
    assert names.index("identity.enabled") == names.index("appearance.enabled") + 1


def test_appearance_parts_include_board_and_log(caplog):
    caplog.set_level(logging.INFO, logger="skydango")
    cfg = _cfg()
    board, line = cli._identity_board(cfg, object())
    assert board is not None
    parts = cli._identity_parts(cfg, object())
    assert isinstance(parts["identity"], IdentityBoard)
    assert "身份板：外观为主" in caplog.text
    assert cli._identity_parts(_cfg(identity=False), object()) == {}
