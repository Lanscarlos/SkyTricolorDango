"""管理面板「模型」页后端（spec 2026-10-05-model-providers §3）。"""

import hashlib
import tomllib
from types import SimpleNamespace

import pytest

from skydango.config import load_config
from skydango.console.models_view import ModelsView
from skydango.console.probes import test_provider as probe_provider
from skydango.console.settings import SettingsStore
from skydango.models.config import USE_NAMES, ProviderConfig, resolve


@pytest.fixture(autouse=True)
def no_registry(monkeypatch):
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")


def view_of(tmp_path, config="", console="", secrets=""):
    for name, text in (("config.toml", config), ("console.toml", console), ("secrets.toml", secrets)):
        if text:
            (tmp_path / name).write_text(text, encoding="utf-8")
    store = SettingsStore(tmp_path / "config.toml", environ={})
    return ModelsView(store), store


def submit(v, **changes):
    """把 view() 原样提交（可改 uses）：页面第一次保存就是这样。"""
    data = v.view()
    providers = [{k: p[k] for k in p if k not in ("secret", "secret_source", "source", "used_by")} for p in data["providers"]]
    uses = {u["name"]: {"main": u["main"], "backup": u["backup"]} for u in data["uses"]}
    uses.update(changes.pop("uses", {}))
    return v.save({"providers": changes.pop("providers", providers), "uses": uses, "secrets": changes.pop("secrets", {})})


def effective(store):
    return resolve(store.effective())


def test_view_lists_builtin_and_uses(tmp_path):
    v, _ = view_of(tmp_path, secrets='[env]\nDEEPSEEK_API_KEY = "sk-1234567890abcd"\n')
    data = v.view()
    ids = [p["id"] for p in data["providers"]]
    assert ids == ["claude", "deepseek"]
    ds = next(p for p in data["providers"] if p["id"] == "deepseek")
    assert "brain" in ds["used_by"] and ds["secret"] == "已设置（sk-…abcd）" and ds["secret_source"] == "secrets"
    assert "1234567890" not in str(data)
    assert [u["name"] for u in data["uses"]] == list(USE_NAMES)
    assert data["uses"][0]["main"] == "deepseek/deepseek-flash" and data["uses"][0]["backup"] == "claude/sonnet"
    assert (data["uses"][0]["default_main"], data["uses"][0]["default_backup"]) == ("deepseek/deepseek-flash", "claude/sonnet")
    assert data["kinds"] == ["claude-code", "openai"] and data["templates"]


def test_first_save_keeps_legacy_llm_values(tmp_path):  # Review Focus 2
    v, store = view_of(tmp_path, config='[llm]\nbase_url = "https://x"\nmodel = "m1"\n')
    assert submit(v) == (200, {"ok": True, "restart": False})
    s = effective(store)
    assert s.providers["deepseek"].base_url == "https://x" and "m1" in s.providers["deepseek"].models
    assert str(s.uses["reply"].main) == "deepseek/m1"


def test_save_writes_only_changed_uses(tmp_path):
    v, store = view_of(tmp_path)
    assert submit(v, uses={"brain": {"main": "claude/sonnet", "backup": ""}})[0] == 200
    data = tomllib.loads((tmp_path / "console.toml").read_text(encoding="utf-8"))
    assert data["models"] == {"brain": {"main": "claude/sonnet", "backup": ""}}
    assert str(effective(store).uses["brain"].main) == "claude/sonnet" and effective(store).uses["brain"].backup is None
    assert submit(v, uses={"brain": {"main": "deepseek/deepseek-flash", "backup": "claude/sonnet"}})[0] == 200
    data = tomllib.loads((tmp_path / "console.toml").read_text(encoding="utf-8"))
    assert "models" not in data


def test_delete_in_use_rejected(tmp_path):
    v, _ = view_of(tmp_path)
    providers = [p for p in v.view()["providers"] if p["id"] != "deepseek"]
    status, res = submit(v, providers=[{k: p[k] for k in p if k not in ("secret", "secret_source", "source", "used_by")} for p in providers])
    assert status == 400 and "deepseek" in res["text"] and "brain" in res["text"]


def test_delete_config_defined_rejected(tmp_path):
    v, _ = view_of(tmp_path, config='[providers.gpt]\nkind = "openai"\nbase_url = "https://x"\nkey_env = "K"\nmodels = ["g"]\n'
                                     '[providers.deepseek]\nkind = "openai"\nbase_url = "https://api.deepseek.com"\nkey_env = "DEEPSEEK_API_KEY"\n'
                                     'models = ["deepseek-chat"]\n[providers.claude]\nkind = "claude-code"\nmodels = ["sonnet", "haiku"]\n'
                                     'vision = ["sonnet", "haiku"]\n')
    data = v.view()
    providers = [{k: p[k] for k in p if k not in ("secret", "secret_source", "source", "used_by")} for p in data["providers"] if p["id"] != "gpt"]
    status, res = submit(v, providers=providers)
    assert status == 400 and "config.toml" in res["text"] and "gpt" in res["text"]


def test_save_rejects_blind_vision_use(tmp_path):
    v, _ = view_of(tmp_path)
    status, res = submit(v, uses={"eyes": {"main": "deepseek/deepseek-chat", "backup": ""}})
    assert status == 400 and "看不了图" in res["text"]
    assert not (tmp_path / "console.toml").exists()


def test_secret_write_and_clear(tmp_path):
    v, store = view_of(tmp_path)
    assert submit(v, secrets={"deepseek": "sk-1"})[0] == 200
    assert tomllib.loads((tmp_path / "secrets.toml").read_text(encoding="utf-8"))["env"]["DEEPSEEK_API_KEY"] == "sk-1"
    assert store.secret_env("DEEPSEEK_API_KEY") == "sk-1" and "sk-1" not in str(v.view())
    assert submit(v, secrets={"deepseek": ""})[0] == 200
    assert "env" not in tomllib.loads((tmp_path / "secrets.toml").read_text(encoding="utf-8"))


def test_pasted_secret_is_stripped(tmp_path):
    v, store = view_of(tmp_path)
    assert submit(v, secrets={"deepseek": "  sk-x\n"})[0] == 200
    assert store.secret_env("DEEPSEEK_API_KEY") == "sk-x"


def test_config_toml_untouched(tmp_path):
    v, _ = view_of(tmp_path, config='[llm]\nmodel = "m1"\n')
    before = hashlib.sha256((tmp_path / "config.toml").read_bytes()).hexdigest()
    submit(v, uses={"brain": {"main": "claude/sonnet", "backup": ""}}, secrets={"claude": "tok"})
    assert hashlib.sha256((tmp_path / "config.toml").read_bytes()).hexdigest() == before


def test_add_provider_and_use_it(tmp_path):
    v, store = view_of(tmp_path)
    data = v.view()
    providers = [{k: p[k] for k in p if k not in ("secret", "secret_source", "source", "used_by")} for p in data["providers"]]
    providers.append({"id": "gpt", "kind": "openai", "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY",
                      "models": [{"name": "gpt-4o", "vision": True}]})
    assert submit(v, providers=providers, uses={"eyes": {"main": "gpt/gpt-4o", "backup": ""}})[0] == 200
    s = effective(store)
    assert s.providers["gpt"].sees("gpt-4o") and str(s.uses["eyes"].main) == "gpt/gpt-4o"


def test_save_says_restart_when_busy(tmp_path):
    store = SettingsStore(tmp_path / "config.toml", environ={})
    v = ModelsView(store, busy=lambda: True)
    assert submit(v)[1]["restart"] is True


# ---- 测试按钮（probes.test_provider）----
def _client(replies):
    replies = list(replies)

    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies.pop(0)))], usage=None)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_test_provider_openai_and_vision():
    p = ProviderConfig("gpt", "openai", models=("gpt-4o",), vision=("gpt-4o",), base_url="https://x", key_env="K")
    made = []
    ok = probe_provider(p, "sk", client_factory=lambda prov, **kw: made.append(kw) or _client(["ok", "红色"]))
    assert ok["ok"] is True and "gpt-4o" in ok["text"] and made[0]["api_key"] == "sk"
    bad = probe_provider(p, "sk", client_factory=lambda prov, **kw: _client(["ok", "蓝色"]))
    assert bad["ok"] is False and "蓝" in bad["text"]


def test_test_provider_openai_without_key_uses_placeholder():
    p = ProviderConfig("ollama", "openai", models=("qwen",), base_url="http://127.0.0.1:11434/v1", key_env="OLLAMA_API_KEY")
    made = []
    assert probe_provider(p, "", client_factory=lambda prov, **kw: made.append(kw) or _client(["ok"]))["ok"]
    assert made[0]["api_key"] == "ollama"


def test_test_provider_claude(monkeypatch):
    monkeypatch.setattr("skydango.console.probes.resolve_claude", lambda path: ["claude"])
    p = ProviderConfig("claude", "claude-code", models=("sonnet",), vision=("sonnet",))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="2.1 (Claude Code)" if "--version" in cmd else "ok", stderr="")

    assert probe_provider(p, "tok", run=run)["ok"] is True
    assert calls[1][calls[1].index("--model") + 1] == "sonnet"
    assert probe_provider(p, "", run=run) == {"ok": False, "text": "还没填 Claude 令牌"}


def test_restore_default_overrides_legacy_console_fields(tmp_path):  # 终审 I3：console.toml 里旧设置页写的模型字段
    v, store = view_of(tmp_path, console='[brain]\nmodel = "sonnet"\nowner_name = "卡洛"\n[llm]\ntemperature = 0.3\n')
    assert str(effective(store).uses["brain"].main) == "claude/sonnet"
    assert submit(v, uses={"brain": {"main": "deepseek/deepseek-flash", "backup": "claude/sonnet"}})[0] == 200
    s = effective(store)
    assert str(s.uses["brain"].main) == "deepseek/deepseek-flash" and s.uses["reply"].temperature == 0.3
    data = tomllib.loads((tmp_path / "console.toml").read_text(encoding="utf-8"))
    assert data["brain"] == {"owner_name": "卡洛"} and "llm" not in data  # 旧字段迁走，别的照留


def test_untouched_legacy_console_choice_is_kept(tmp_path):
    v, store = view_of(tmp_path, console='[brain]\nmodel = "opus"\n')
    assert submit(v)[0] == 200
    assert str(effective(store).uses["brain"].main) == "claude/opus"
    assert "brain" not in tomllib.loads((tmp_path / "console.toml").read_text(encoding="utf-8"))
