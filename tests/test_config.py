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
    assert cfg.brain.model == "sonnet" and cfg.brain.eyes_model == "haiku" and cfg.brain.effort == "low"
    assert cfg.brain.token_env == "SKYDANGO_CLAUDE_TOKEN" and cfg.brain.config_dir == ".brain-claude"
    assert cfg.brain.turn_timeout == 120 and cfg.brain.limit_retry == 600
    assert cfg.brain.image_size == [1280, 720] and cfg.brain.max_steps == 6
    assert cfg.brain.move_step == 0.3 and cfg.brain.move_min_interval == 3.0
    assert cfg.brain.owner_name == "" and cfg.brain.owner_window == 30.0


def test_spin_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[spin]\nhfov = 75.0\n", encoding="utf-8")
    cfg = load_config(p)
    s = cfg.spin
    assert s.hfov == 75.0
    assert (s.seconds_per_turn, s.fps, s.max_turns, s.min_interval, s.merge_deg, s.self_motion) == (2.0, 15.0, 2, 10.0, 30.0, 0.03)
