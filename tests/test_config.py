from pathlib import Path

import pytest

from skydango.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads():
    cfg = load_config(ROOT / "config.example.toml")
    assert cfg.reply.dry_run is True
    assert cfg.panel.mode == "always" and cfg.panel.idle_peek == 30.0  # 真机验收前默认常开
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


def test_perception_far_crop_defaults(tmp_path):
    cfg = load_config(None)
    assert cfg.perception.far_height == 0.08 and cfg.perception.far_crops == 3
    p = tmp_path / "c.toml"
    p.write_text("[perception]\nfar_crops = 0\n", encoding="utf-8")
    assert load_config(p).perception.far_crops == 0
    assert load_config(ROOT / "config.example.toml").perception.far_crops == 3


def test_places_section(tmp_path):
    cfg = load_config(None)
    pl = cfg.places
    assert (pl.enabled, pl.dir, pl.model, pl.size, pl.norm, pl.device) == (False, "places", "models/places.onnx", 224, "imagenet", "cpu")
    assert (pl.place_min, pl.place_margin, pl.place_interval) == (0.8, 0.05, 30.0)
    p = tmp_path / "c.toml"
    p.write_text("[places]\nmodel = \"thumb\"\n", encoding="utf-8")
    assert load_config(p).places.model == "thumb"
    assert load_config(ROOT / "config.example.toml").places.enabled is False


def test_places_folder_is_gitignored():
    assert "places/" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()


def test_gesture_section(tmp_path):
    g = load_config(None).gesture
    assert (g.enabled, g.model, g.labels) == (False, "models/gesture.onnx", ["none", "wave", "bow"])
    assert g.names == {"wave": "挥手", "bow": "鞠躬"}
    assert (g.frames, g.fps, g.size, g.interval, g.min_prob, g.cooldown) == (16, 8.0, 112, 2.0, 0.9, 30.0)
    p = tmp_path / "c.toml"
    p.write_text("[gesture]\nlabels = [\"none\", \"wave\"]\nnames = {wave = \"招手\"}\n", encoding="utf-8")
    g2 = load_config(p).gesture
    assert g2.labels == ["none", "wave"] and g2.names == {"wave": "招手"}
    assert load_config(ROOT / "config.example.toml").gesture.enabled is False


def test_panel_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[panel]\nmode = "auto"\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.panel.mode == "auto"
    assert (cfg.panel.idle_peek, cfg.panel.quiet_close, cfg.panel.peek_cooldown) == (30.0, 45.0, 5.0)
    assert (cfg.panel.bubble_wait, cfg.panel.bubble_gone, cfg.panel.bubble_strangers, cfg.panel.open_timeout) == (15.0, 3.0, False, 1.5)
    assert load_config(None).panel.mode == "always"
