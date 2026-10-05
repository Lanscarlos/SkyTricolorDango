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
    assert str(s.uses["brain"].main) == "deepseek/deepseek-chat" and str(s.uses["brain"].backup) == "claude/sonnet"
    assert str(s.uses["eyes"].main) == "claude/haiku" and s.uses["eyes"].backup is None
    assert s.uses["reply"].max_tokens == 200 and s.uses["brain"].max_tokens == 4096
    assert s.problems == [] and s.legacy == []

def test_local_layout_brain_on_deepseek(tmp_path):   # Review Focus 1
    s = resolve(_cfg(tmp_path, '[llm]\nprovider = "openai"\nbase_url = "https://api.deepseek.com"\nmodel = "deepseek-chat"\napi_key_env = "DEEPSEEK_API_KEY"\n',
                     '[console]\n[appearance]\nenabled = true\n'))
    assert str(s.uses["brain"].main) == "deepseek/deepseek-chat"
    assert s.providers["deepseek"].key_env == "DEEPSEEK_API_KEY"
    assert str(s.uses["reply"].main) == "deepseek/deepseek-chat"

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
    assert str(s.uses["reply"].main) == "deepseek/deepseek-chat" and any("anthropic" in x for x in s.legacy)

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
    s = resolve(_cfg(tmp_path, '[models.eyes]\nmain = "deepseek/deepseek-chat"\n'))
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
