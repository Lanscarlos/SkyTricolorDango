from pathlib import Path

import pytest

from skydango.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads():
    cfg = load_config(ROOT / "config.example.toml")
    assert cfg.reply.dry_run is True
    assert cfg.panel.mode == "auto" and cfg.panel.idle_peek == 30.0  # 2026-09-30 真机验收后默认按需打开
    assert cfg.device.serial.startswith("127.0.0.1")
    assert cfg.proactive.enabled is True and cfg.proactive.quota_quiet == 2
    assert cfg.lull.enabled is True and cfg.lull.stages == [60.0, 180.0, 360.0]


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


def test_proactive_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[proactive]\nquota_busy = 6\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.proactive.quota_busy == 6 and cfg.proactive.enabled is True and cfg.proactive.min_gap == 60.0
    assert cfg.proactive.self_names == ["团子", "三彩"] and cfg.proactive.cold_after == 3


def test_track_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[track]\ngain = 0.2\n", encoding="utf-8")
    t = load_config(p).track
    assert t.gain == 0.2
    assert (t.deadband, t.nudge_min, t.nudge_max, t.settle, t.lost_after, t.max_age) == (0.15, 0.02, 0.1, 0.6, 3.0, 0.5)
    assert (t.stall_nudges, t.stall_px, t.max_seconds, t.chase_max) == (3, 20.0, 60, 0.15)
    example = load_config(ROOT / "config.example.toml").track
    assert (example.deadband, example.gain, example.settle, example.max_seconds) == (0.15, 0.15, 0.6, 60)
    assert (example.nudge_max, example.chase_max) == (0.1, 0.15)


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
    assert (g.enabled, g.model, g.labels) == (False, "models/gesture.onnx", ["none", "wave", "bow", "cheer", "shy"])
    assert g.names == {"wave": "挥手", "bow": "鞠躬", "cheer": "欢呼", "shy": "害羞"}
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
    assert load_config(None).panel.mode == "auto"


# ---- 管理面板：console.toml 叠加、secrets.toml ----
def test_console_defaults():
    from skydango.config import Config

    c = Config().console
    assert (c.port, c.child_port, c.stop_timeout, c.log_lines) == (19390, 19391, 60.0, 500)
    assert (c.brain, c.live, c.emotes, c.duration) == (True, False, True, 0.0)


def test_console_toml_overrides_config_toml(tmp_path):
    from skydango.config import load_all

    (tmp_path / "config.toml").write_text('[device]\nserial = "a"\nadb_path = "x"\n', encoding="utf-8")
    (tmp_path / "console.toml").write_text('[device]\nserial = "b"\n[console]\nlive = true\n', encoding="utf-8")
    loaded = load_all(tmp_path / "config.toml", environ={})
    assert loaded.cfg.device.serial == "b" and loaded.cfg.device.adb_path == "x" and loaded.cfg.console.live is True
    assert loaded.overridden == ["device.serial"]


def test_console_toml_without_config_toml(tmp_path):
    from skydango.config import load_all

    (tmp_path / "console.toml").write_text('[llm]\nmodel = "m"\n', encoding="utf-8")
    assert load_all(tmp_path / "config.toml", environ={}).cfg.llm.model == "m"


def test_console_toml_unknown_key_names_the_file(tmp_path):
    from skydango.config import load_all

    (tmp_path / "console.toml").write_text("[device]\nserail = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="console.toml.*device.serail"):
        load_all(tmp_path / "config.toml", environ={})


def test_secrets_override_environment(tmp_path):
    from skydango.config import load_all

    (tmp_path / "secrets.toml").write_text('[env]\nDEEPSEEK_API_KEY = "sk-new"\n', encoding="utf-8")
    env = {"DEEPSEEK_API_KEY": "sk-old"}
    loaded = load_all(tmp_path / "config.toml", environ=env)
    assert env["DEEPSEEK_API_KEY"] == "sk-new" and loaded.secrets == ["DEEPSEEK_API_KEY"]


def test_secret_must_be_string(tmp_path):
    from skydango.config import load_all

    (tmp_path / "secrets.toml").write_text("[env]\nX = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="X"):
        load_all(tmp_path / "config.toml", environ={})


def test_perception_object_classes_appended():
    from skydango.config import PerceptionConfig

    p = PerceptionConfig()
    assert p.classes[:6] == ["player", "name_tag", "social_ring", "self", "player_unlit", "typing"]  # 旧编号不变
    assert p.classes[6:] == ["bench", "bonfire", "instrument", "spirit"]
    assert (p.object_min_hits, p.object_near, p.object_far) == (3, 0.85, 0.65)
    assert load_config(ROOT / "config.example.toml").perception.object_min_hits == 3


def test_example_config_has_persona_keys():
    import tomllib

    from skydango.config import InnerConfig

    raw = tomllib.loads((ROOT / "config.example.toml").read_text(encoding="utf-8"))["inner"]
    keys = ("persona", "fade_days", "catchphrases_max", "jokes_per_friend", "jokes_max", "opinions_max", "soft_minutes")
    assert {k: raw[k] for k in keys} == {k: getattr(InnerConfig(), k) for k in keys}


def test_example_config_has_appearance():
    import dataclasses
    import tomllib

    from skydango.config import AppearanceConfig

    raw = tomllib.loads((ROOT / "config.example.toml").read_text(encoding="utf-8"))["appearance"]
    d = AppearanceConfig()
    assert raw == {f.name: getattr(d, f.name) for f in dataclasses.fields(d)}
    assert load_config(ROOT / "config.example.toml").appearance.enabled is False


def test_example_config_has_attrs():
    import dataclasses
    import tomllib

    from skydango.config import AttrsConfig

    raw = tomllib.loads((ROOT / "config.example.toml").read_text(encoding="utf-8"))["attrs"]
    d = AttrsConfig()
    assert raw == {f.name: getattr(d, f.name) for f in dataclasses.fields(d)}
    assert load_config(ROOT / "config.example.toml").attrs.enabled is False


def test_example_config_has_sandbox():
    import tomllib

    from skydango.config import SandboxConfig

    raw = tomllib.loads((ROOT / "config.example.toml").read_text(encoding="utf-8"))["sandbox"]
    d = SandboxConfig()
    assert raw == {"dir": d.dir, "port": d.port, "step_timeout": d.step_timeout, "wake_hour": d.wake_hour, "emotes": d.emotes}
    assert (d.dir, d.port, d.step_timeout, d.wake_hour, d.emotes) == ("sandbox", 19392, 180.0, 9, [])
    assert load_config(ROOT / "config.example.toml").sandbox.port == 19392


def test_backstage_config(tmp_path):
    from skydango.config import Config, load_config

    c = Config().backstage
    assert (c.enabled, c.changelog_max, c.changelog_days) == (False, 10, 7)
    path = tmp_path / "c.toml"
    path.write_text("[backstage]\nenabled = true\n", encoding="utf-8")
    assert load_config(path).backstage.enabled is True


def test_call_config_defaults():
    from skydango.config import Config

    c = Config()
    assert (c.call.enabled, c.call.auto, c.call.min_gap, c.call.window, c.call.burst) == (True, True, 20.0, 6.0, 1.0)
    assert (c.call.auto_quota, c.call.auto_window, c.call.auto_after_leave) == (3, 60.0, 30.0)
    assert (c.call.halo, c.call.halo_rise, c.call.halo_center) == (False, 25.0, 0.35)
    assert c.perception.sticky_names is True and c.perception.edge_band == 0.06


def test_call_section_loads(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[call]\nauto = false\nmin_gap = 30\n[perception]\nedge_band = 0\n", encoding="utf-8")
    c = load_config(p)
    assert c.call.auto is False and c.call.min_gap == 30 and c.perception.edge_band == 0


def test_q_key_and_panel_action():
    from skydango.device.base import LINUX_KEY_Q
    from skydango.vision.panels import ACTIONS

    assert LINUX_KEY_Q == 16 and "call" in ACTIONS


def test_catalog_defaults_and_example():
    from skydango.config import CatalogConfig

    c = CatalogConfig()
    assert c.enabled is True and c.dir == "catalog"
    assert (c.every, c.min_height, c.edge, c.pad, c.sharp_min) == (0.5, 0.25, 8, 0.1, 50.0)
    assert (c.per_who, c.gap, c.flush_every, c.max_per_run) == (6, 3.0, 300.0, 300)
    assert c.dark_max == 0.5
    ex = load_config(ROOT / "config.example.toml").catalog
    assert ex == CatalogConfig()
