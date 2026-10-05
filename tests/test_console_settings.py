import json
import tomllib

import pytest

from skydango.console.settings import FIELDS, SettingsStore, mask


def store(tmp_path, config="", console="", secrets="", environ=None):
    for name, text in (("config.toml", config), ("console.toml", console), ("secrets.toml", secrets)):
        if text:
            (tmp_path / name).write_text(text, encoding="utf-8")
    return SettingsStore(tmp_path / "config.toml", environ={} if environ is None else environ)


def field(view, key):
    return next(f for f in view["fields"] if f["key"] == key)


@pytest.fixture(autouse=True)
def no_registry(monkeypatch):
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")


def test_every_spec_field_is_listed():
    assert [f.key for f in FIELDS] == [
        "console.brain", "console.live", "console.emotes", "console.duration",
        "device.adb_path", "device.serial", "device.switch_ime", "device.capture", "llm.provider", "llm.base_url", "llm.model", "secret.llm",
        "secret.claude", "brain.claude_path", "brain.model", "brain.eyes_model", "brain.memory_model",
        "proactive.enabled", "proactive.quota_busy", "proactive.quota_quiet", "proactive.min_gap", "proactive.auto_look_busy",
        "reflex.enabled", "reflex.bubble", "inner.enabled", "inner.reflect", "inner.persona", "call.enabled", "call.auto", "attention.search", "lull.enabled", "addressee.enabled", "backstage.enabled", "vision.source", "env.enabled",
        "perception.enabled",
        "perception.model", "attrs.enabled", "attrs.model", "inbox.enabled", "inbox.ask", "inbox.retrain_min", "places.enabled", "appearance.enabled", "appearance.describe", "catalog.enabled", "icons.enabled", "friend_check.enabled", "panels.enabled", "reply.disclosure_prefix", "owner",
    ]


def test_addressee_field():
    from skydango.console.settings import KNOWN

    f = KNOWN["addressee.enabled"]
    assert (f.label, f.kind, f.group) == ("分清在跟谁说话", "bool", "brain")


def test_console_has_appearance_switches():
    from skydango.console.settings import KNOWN

    on = KNOWN["appearance.enabled"]
    assert (on.label, on.help, on.kind, on.group) == (
        "认装扮", "按外观接回没读到名字的好友、认回来的陌生人；要配合 YOLO 感知层", "bool", "features")
    desc = KNOWN["appearance.describe"]
    assert (desc.label, desc.help, desc.kind, desc.group) == (
        "描述装扮", "让 Haiku 把团子和身边人的装扮写成一句话（花额度）", "bool", "features")


def test_console_has_attrs_switch():
    from skydango.console.settings import KNOWN

    f = KNOWN["attrs.enabled"]
    assert (f.label, f.kind, f.group) == ("人物复核（第二层）", "bool", "features")


def test_console_has_catalog_switch():
    from skydango.console.settings import KNOWN

    f = KNOWN["catalog.enabled"]
    assert (f.label, f.kind, f.group) == ("收集装扮图鉴", "bool", "features")


def test_mask():
    assert mask("") == "没设置" and mask("short") == "已设置" and mask("sk-1234567890abcd") == "已设置（sk-…abcd）"


def test_sources(tmp_path):
    v = store(tmp_path, config='[device]\nserial = "a"\n', console='[llm]\nmodel = "m"\n').view()
    assert (field(v, "device.serial")["source"], field(v, "llm.model")["source"], field(v, "llm.base_url")["source"]) == (
        "config", "console", "default")
    assert field(v, "llm.model")["value"] == "m" and v["error"] is None


def test_secret_is_masked_and_sourced(tmp_path):
    v = store(tmp_path, secrets='[env]\nDEEPSEEK_API_KEY = "sk-1234567890abcd"\n', environ={"SKYDANGO_CLAUDE_TOKEN": "short"}).view()
    assert field(v, "secret.llm").items() >= {"value": "已设置（sk-…abcd）", "source": "secrets"}.items()
    assert (field(v, "secret.claude")["value"], field(v, "secret.claude")["source"]) == ("已设置", "env")
    assert "1234567890" not in json.dumps(v, ensure_ascii=False)


def test_save_writes_console_and_secrets_but_never_config(tmp_path):
    s = store(tmp_path, config='[device]\nserial = "a"  # 手写注释\n')
    before = (tmp_path / "config.toml").read_text(encoding="utf-8")
    r = s.save({"device.serial": "b", "secret.llm": "sk-x", "owner": "卡洛", "console.duration": 30})
    assert r["ok"], r
    assert (tmp_path / "config.toml").read_text(encoding="utf-8") == before
    assert tomllib.loads((tmp_path / "console.toml").read_text(encoding="utf-8")) == {
        "device": {"serial": "b"}, "reply": {"owner_name": "卡洛"}, "brain": {"owner_name": "卡洛"}, "console": {"duration": 30.0}}
    assert tomllib.loads((tmp_path / "secrets.toml").read_text(encoding="utf-8")) == {"env": {"DEEPSEEK_API_KEY": "sk-x"}}


def test_secret_goes_to_configured_env_name(tmp_path):
    s = store(tmp_path, config='[brain]\ntoken_env = "MY_TOKEN"\n')
    assert s.save({"secret.claude": "tok"})["ok"]
    assert s.secret("claude") == "tok"
    assert tomllib.loads((tmp_path / "secrets.toml").read_text(encoding="utf-8")) == {"env": {"MY_TOKEN": "tok"}}


def test_empty_secret_keeps_old_and_revert_clears(tmp_path):
    s = store(tmp_path, secrets='[env]\nDEEPSEEK_API_KEY = "sk-old"\n')
    s.save({"secret.llm": ""})
    assert s.secret("llm") == "sk-old"
    s.save({}, revert=["secret.llm"])
    assert s.secret("llm") == ""


def test_revert_removes_key_from_console_toml(tmp_path):
    s = store(tmp_path, config='[device]\nserial = "a"\n', console='[device]\nserial = "b"\n')
    s.save({}, revert=["device.serial"])
    assert field(s.view(), "device.serial").items() >= {"value": "a", "source": "config"}.items()


@pytest.mark.parametrize("key, value", [("env.enabled", "yes"), ("console.duration", "abc"), ("device.capture", "vnc"),
                                        ("console.duration", -1), ("console.brain", 1)])
def test_bad_values_are_rejected_and_nothing_written(tmp_path, key, value):
    s = store(tmp_path)
    r = s.save({"device.serial": "b", key: value})
    assert not r["ok"] and key in r["errors"] and not (tmp_path / "console.toml").exists()


def test_unknown_key_is_rejected(tmp_path):
    r = store(tmp_path).save({"device.nope": "x"})
    assert not r["ok"] and "device.nope" in r["errors"]


def test_missing_paths_only_warn(tmp_path):
    r = store(tmp_path).save({"device.adb_path": str(tmp_path / "no-adb.exe"), "perception.model": "models/nope.pt"})
    assert r["ok"] and set(r["warnings"]) == {"device.adb_path", "perception.model"}


def test_owner_mismatch_warns(tmp_path):
    v = store(tmp_path, config='[reply]\nowner_name = "A"\n[brain]\nowner_name = "B"\n').view()
    assert "A" in field(v, "owner")["warning"] and "B" in field(v, "owner")["warning"]


def test_broken_console_toml_is_reported_and_not_overwritten(tmp_path):  # Review Focus 2
    s = store(tmp_path, console="[device\n")
    assert "console.toml 读不出来" in s.view()["error"]
    r = s.save({"device.serial": "b"})
    assert not r["ok"] and (tmp_path / "console.toml").read_text(encoding="utf-8") == "[device\n"


def test_unknown_key_in_console_toml_is_reported(tmp_path):
    v = store(tmp_path, console="[device]\nserail = 1\n").view()
    assert "serail" in v["error"]


def test_broken_secrets_structure_is_reported(tmp_path):
    s = store(tmp_path, secrets='env = "x"\n')
    assert "secrets.toml" in s.view()["error"]
    r = s.save({"secret.llm": "sk-x"})
    assert not r["ok"] and "secrets.toml" in r["errors"]["_file"]
    assert (tmp_path / "secrets.toml").read_text(encoding="utf-8") == 'env = "x"\n'


def test_pasted_secret_is_stripped(tmp_path):
    s = store(tmp_path)
    assert s.save({"secret.llm": "  sk-x\n"})["ok"]
    assert s.secret("llm") == "sk-x"
    assert s.save({"secret.llm": "   "})["ok"] and s.secret("llm") == "sk-x"  # 只有空白 = 不改


def test_quota_help_does_not_hardcode_window():
    helps = {f.key: f.help for f in FIELDS}
    for key in ("proactive.quota_busy", "proactive.quota_quiet"):
        assert "10 分钟" not in helps[key] and "quota_window" in helps[key]


def test_console_has_backstage_switch():
    from skydango.console.settings import KNOWN

    f = KNOWN["backstage.enabled"]
    assert (f.label, f.kind, f.group) == ("幕后", "bool", "brain") and "知道团子是 AI" in f.help


def test_console_has_call_switches():
    by = {f.key: f for f in FIELDS}
    assert by["call.enabled"].kind == "bool" and by["call.auto"].kind == "bool"


def test_console_has_vision_source():
    from skydango.console.settings import KNOWN

    f = KNOWN["vision.source"]
    assert (f.label, f.kind, f.group, f.choices) == ("读聊天的方式", "choice", "features", ("a11y", "ocr"))
    assert f.help.startswith("a11y：读游戏的无障碍节点") and "ocr：截图识别" in f.help


def test_console_has_inbox_fields():
    from skydango.console.settings import KNOWN

    for k in ("inbox.enabled", "inbox.ask", "inbox.retrain_min", "perception.model", "attrs.model"):
        assert k in KNOWN, k


def test_example_toml_inbox_matches_defaults():
    from pathlib import Path

    from skydango.config import load_config

    p = Path(__file__).resolve().parent.parent / "config.example.toml"
    cfg, dflt = load_config(p), load_config(None)
    assert cfg.inbox == dflt.inbox and cfg.retrain == dflt.retrain
