from pathlib import Path

import pytest

from skydango.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads():
    cfg = load_config(ROOT / "config.example.toml")
    assert cfg.reply.dry_run is True
    assert cfg.device.serial.startswith("127.0.0.1")


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[reply]\nmax_char = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reply.max_char"):
        load_config(p)


def test_nested_override(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[vision.bubble]\nmin_value = 200\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.vision.bubble.min_value == 200
    assert cfg.vision.bubble.max_saturation == 45


def test_emotes_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[emotes]\nextra = ["拥抱"]\nswap_slots = [7]\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.emotes.extra == ["拥抱"] and cfg.emotes.swap_slots == [7]
    assert cfg.emotes.enabled is True and cfg.emotes.min_interval == 20.0


def test_brain_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[brain]\nenabled = true\nheartbeat = [30, 60]\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.brain.enabled is True and cfg.brain.heartbeat == [30, 60]
    assert cfg.brain.model == "claude-sonnet-5" and cfg.brain.effort == "low"
    assert cfg.brain.image_size == [1280, 720] and cfg.brain.max_steps == 6
