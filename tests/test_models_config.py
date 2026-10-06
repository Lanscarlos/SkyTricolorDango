import pytest

from skydango.config import load_config
from skydango.models.config import ModelRef, resolve


def _cfg(tmp_path, text="", console=""):
    base = tmp_path / "config.toml"; base.write_text(text, encoding="utf-8")
    over = tmp_path / "console.toml"; over.write_text(console, encoding="utf-8")
    return load_config(base, over)

def test_defaults(tmp_path):
    s = resolve(_cfg(tmp_path))
    assert set(s.providers) >= {"claude", "deepseek"}
    assert str(s.uses["brain"].main) == "deepseek/deepseek-flash" and str(s.uses["brain"].backup) == "claude/sonnet"
    assert {n: (str(s.uses[n].main), str(s.uses[n].backup or "")) for n in ("memory", "reflect", "reply", "text_label")} == {
        "memory": ("deepseek/deepseek-flash", "claude/sonnet"),
        "reflect": ("deepseek/deepseek-flash", "claude/sonnet"),
        "reply": ("deepseek/deepseek-flash", ""),
        "text_label": ("deepseek/deepseek-flash", "")}
    assert s.uses["reply"].max_tokens == 200 and s.uses["brain"].max_tokens == 4096
    assert s.problems == [] and s.legacy == []

def test_vision_defaults_on_deepseek_flash(tmp_path):
    s = resolve(_cfg(tmp_path))
    ds = s.providers["deepseek"]
    assert "deepseek-flash" in ds.models and ds.sees("deepseek-flash") and not ds.sees("deepseek-chat")
    assert {n: (str(s.uses[n].main), str(s.uses[n].backup)) for n in ("eyes", "wardrobe", "image_label")} == {
        "eyes": ("deepseek/deepseek-flash", "claude/haiku"),
        "wardrobe": ("deepseek/deepseek-flash", "claude/haiku"),
        "image_label": ("deepseek/deepseek-flash", "claude/sonnet")}
    assert not any(s.uses[n].disabled for n in ("eyes", "wardrobe", "image_label"))

def test_local_layout_brain_on_deepseek(tmp_path):   # Review Focus 1
    s = resolve(_cfg(tmp_path, '[llm]\nprovider = "openai"\nbase_url = "https://api.deepseek.com"\nmodel = "deepseek-chat"\napi_key_env = "DEEPSEEK_API_KEY"\n',
                     '[console]\n[appearance]\nenabled = true\n'))
    assert str(s.uses["brain"].main) == "deepseek/deepseek-flash"   # 新默认；旧 [llm] model 只换算 reply
    assert s.providers["deepseek"].key_env == "DEEPSEEK_API_KEY"
    assert str(s.uses["reply"].main) == "deepseek/deepseek-chat"

def test_builtin_deepseek_models(tmp_path):
    ds = resolve(_cfg(tmp_path)).providers["deepseek"]
    assert ds.models == ("deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner")
    assert ds.vision == ("deepseek-flash",)

def test_legacy_llm_model_added_to_models(tmp_path):
    s = resolve(_cfg(tmp_path, '[llm]\nmodel = "deepseek-v4"\ntemperature = 0.5\nmax_tokens = 300\n'))
    assert "deepseek-v4" in s.providers["deepseek"].models
    assert str(s.uses["reply"].main) == "deepseek/deepseek-v4"
    assert s.uses["reply"].temperature == 0.5 and s.uses["reply"].max_tokens == 300
    assert any("llm" in x for x in s.legacy)

def test_legacy_claude_models(tmp_path):
    s = resolve(_cfg(tmp_path, '[brain]\nmodel = "opus"\neyes_model = "sonnet"\nmemory_model = "haiku"\n[inner]\nreflect_model = "opus"\n'
                               '[appearance]\ndescribe_model = "sonnet"\n[assist]\nmodel = "opus"\n'))
    assert {n: str(s.uses[n].main) for n in ("brain", "eyes", "memory", "reflect", "wardrobe", "image_label")} == {
        "brain": "claude/opus", "eyes": "claude/sonnet", "memory": "claude/haiku", "reflect": "claude/opus",
        "wardrobe": "claude/sonnet", "image_label": "claude/opus"}
    assert s.uses["brain"].source == "legacy"

def test_models_table_beats_legacy(tmp_path):
    s = resolve(_cfg(tmp_path, '[brain]\nmodel = "opus"\n[models.brain]\nmain = "deepseek/deepseek-chat"\n'))
    assert str(s.uses["brain"].main) == "deepseek/deepseek-chat"

def test_llm_echo_and_anthropic(tmp_path):
    assert str(resolve(_cfg(tmp_path, '[llm]\nprovider = "echo"\n')).uses["reply"].main) == "echo/echo"
    s = resolve(_cfg(tmp_path, '[llm]\nprovider = "anthropic"\n'))
    assert str(s.uses["reply"].main) == "deepseek/deepseek-flash" and any("anthropic" in x for x in s.legacy)

def test_explicit_providers_replace_builtin(tmp_path):
    s = resolve(_cfg(tmp_path, '[providers.gpt]\nkind = "openai"\nbase_url = "https://x"\nkey_env = "OPENAI_API_KEY"\nmodels = ["gpt-4o"]\nvision = ["gpt-4o"]\n'
                               '[models.brain]\nmain = "gpt/gpt-4o"\nbackup = ""\n'))
    assert "deepseek" not in s.providers and "claude" not in s.providers and "echo" in s.providers
    assert s.uses["brain"].backup is None
    assert s.uses["memory"].disabled            # 默认的 deepseek 没了
    assert any(p.use == "memory" for p in s.problems)

def test_model_ref_splits_on_first_slash():      # Review Focus 4
    r = ModelRef.parse("ollama/qwen/qwen2.5")
    assert (r.provider, r.model) == ("ollama", "qwen/qwen2.5")
    for bad in ("", "deepseek", "/x", "x/"):
        with pytest.raises(ValueError):
            ModelRef.parse(bad)

def test_vision_use_with_blind_model_is_disabled(tmp_path):   # Review Focus 5
    s = resolve(_cfg(tmp_path, '[models.eyes]\nmain = "deepseek/deepseek-chat"\nbackup = ""\n'))
    assert s.uses["eyes"].disabled and "看图" in s.uses["eyes"].disabled
    assert any(p.use == "eyes" for p in s.problems)

def test_bad_provider_rows(tmp_path):
    s = resolve(_cfg(tmp_path, '[providers.Bad]\nkind = "openai"\n[providers.x]\nkind = "anthropic"\n[providers.y]\nkind = "openai"\nmodels = ["a"]\nvision = ["b"]\n'))
    texts = " ".join(p.text for p in s.problems)
    assert "Bad" in texts and "anthropic" in texts and "vision" in texts

def test_unknown_model_only_warns(tmp_path):
    s = resolve(_cfg(tmp_path, '[models.brain]\nmain = "deepseek/deepseek-v9"\n'))
    assert not s.uses["brain"].disabled
    assert any(p.use == "brain" and "deepseek-v9" in p.text for p in s.problems)


@pytest.mark.parametrize("provider", ["anthropic", "echo"])
def test_legacy_non_openai_llm_not_poured_into_deepseek(tmp_path, provider):  # 终审 I1：旧 anthropic 配置的 Key 别发到 OpenAI
    s = resolve(_cfg(tmp_path, f'[llm]\nprovider = "{provider}"\nbase_url = ""\nmodel = "claude-sonnet-5"\napi_key_env = "ANTHROPIC_API_KEY"\n'))
    ds = s.providers["deepseek"]
    assert (ds.base_url, ds.key_env) == ("https://api.deepseek.com", "DEEPSEEK_API_KEY")
    assert "claude-sonnet-5" not in ds.models


# ---- 单价（spec 2026-10-06-model-usage §3）----
from skydango.models.config import DEEPSEEK_PEAK_HOURS, DEEPSEEK_PRICES, Price, is_deepseek  # noqa: E402


def test_builtin_deepseek_prices(tmp_path):
    s = resolve(_cfg(tmp_path))
    p = s.providers["deepseek"]
    assert p.price("deepseek-flash") == Price(0.02, 1, 4)
    assert p.price("deepseek-v4-pro") == Price(0.15, 4.5, 13.5)
    assert p.price("deepseek-chat").est is True and p.price("deepseek-reasoner").est is True
    assert p.price("deepseek-flash").est is False
    assert p.peak == 2.0 and p.peak_hours == DEEPSEEK_PEAK_HOURS == ("09:00-12:00", "14:00-18:00")
    assert s.providers["claude"].price("sonnet") is None and s.providers["claude"].peak == 1.0
    assert is_deepseek(p) and not is_deepseek(s.providers["claude"])


def test_explicit_prices_parsed(tmp_path):
    s = resolve(_cfg(tmp_path, '[providers.x]\nkind = "openai"\nbase_url = "https://a.example"\nmodels = ["m"]\n'
                               'prices = { m = [1, 2, 3] }\npeak = 1.5\npeak_hours = ["10:00-11:00"]\n'))
    p = s.providers["x"]
    assert p.price("m") == Price(1, 2, 3) and p.peak == 1.5 and p.peak_hours == ("10:00-11:00",)
    assert p.price("other") is None and not is_deepseek(p)
    assert [x for x in s.problems if x.provider == "x"] == []


@pytest.mark.parametrize("extra", ['prices = { m = [1, 2] }', 'prices = { m = ["a", 1, 2] }', 'prices = { m = [-1, 1, 2] }',
                                   'peak_hours = ["25:00-26:00"]', 'peak = 0.5'])
def test_bad_prices_reported(tmp_path, extra):
    s = resolve(_cfg(tmp_path, f'[providers.x]\nkind = "openai"\nmodels = ["m"]\n{extra}\n'))
    assert "x" in s.providers
    assert len([p for p in s.problems if p.provider == "x"]) == 1


def test_explicit_deepseek_without_prices_gets_builtin(tmp_path):   # Review Focus 1
    s = resolve(_cfg(tmp_path, console='[providers.deepseek]\nkind = "openai"\nbase_url = "https://api.deepseek.com"\n'
                                       'key_env = "DEEPSEEK_API_KEY"\nmodels = ["deepseek-flash"]\n'))
    p = s.providers["deepseek"]
    assert p.price("deepseek-flash") == DEEPSEEK_PRICES["deepseek-flash"] and p.peak == 2.0
    assert p.peak_hours == DEEPSEEK_PEAK_HOURS


def test_saved_estimate_stays_estimate(tmp_path):
    s = resolve(_cfg(tmp_path, '[providers.deepseek]\nkind = "openai"\nbase_url = "https://api.deepseek.com"\n'
                               'models = ["deepseek-chat"]\nprices = { "deepseek-chat" = [0.02, 1, 4] }\n'))
    assert s.providers["deepseek"].price("deepseek-chat").est is True
