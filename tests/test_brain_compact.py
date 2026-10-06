"""OpenAI 兼容大脑的历史压缩（spec 2026-10-06-brain-compact）：配置、compact.py 的纯函数和 InboxWatch。"""


def test_compact_config_defaults_and_toml(tmp_path):
    from skydango.config import CompactConfig, Config, load_config

    assert Config().brain.compact == CompactConfig(enabled=True, budget=64000, keep_turns=6, recap_max=1500, retry=300.0)
    p = tmp_path / "c.toml"
    p.write_text("[brain.compact]\nenabled = false\nbudget = 32000\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.brain.compact.enabled is False and cfg.brain.compact.budget == 32000 and cfg.brain.compact.keep_turns == 6
    assert cfg.sources["brain.compact.budget"] == "config"


def test_compact_settings_fields():
    from skydango.console.settings import KNOWN

    assert {"brain.compact.enabled", "brain.compact.budget", "brain.compact.keep_turns"} <= set(KNOWN)
    assert KNOWN["brain.compact.enabled"].kind == "bool" and KNOWN["brain.compact.budget"].kind == "int"
    assert KNOWN["brain.compact.keep_turns"].kind == "int"
