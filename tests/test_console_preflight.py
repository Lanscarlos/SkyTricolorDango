import subprocess

import pytest

from skydango.config import LlmConfig
from skydango.console.preflight import preflight
from skydango.console.probes import test_claude as probe_claude
from skydango.console.probes import test_llm as probe_llm
from skydango.console.runner import LaunchOptions
from skydango.console.settings import SettingsStore

HAS = lambda name: object()  # noqa: E731  mcp 装了


@pytest.fixture(autouse=True)
def no_registry(monkeypatch):
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")


def store_with(tmp_path, console="", secrets=""):
    if console:
        (tmp_path / "console.toml").write_text(console, encoding="utf-8")
    if secrets:
        (tmp_path / "secrets.toml").write_text(secrets, encoding="utf-8")
    return SettingsStore(tmp_path / "config.toml", environ={})


def test_preflight_brain_needs_token_and_mcp(tmp_path):
    assert preflight(store_with(tmp_path), LaunchOptions(brain=True), busy=False, find_spec=lambda n: None) == [
        {"text": "大脑模式要 Claude 令牌：去设置页填", "setting": "secret.claude"},
        {"text": "大脑要用 mcp：先 pip install --user mcp", "setting": None},
        {"text": "大脑离线时的备用回复要大模型的 API Key：去设置页填", "setting": "secret.llm"}]


def test_preflight_ok_with_token(tmp_path):
    s = store_with(tmp_path, secrets='[env]\nSKYDANGO_CLAUDE_TOKEN = "t"\nDEEPSEEK_API_KEY = "k"\n')
    assert preflight(s, LaunchOptions(), busy=False, find_spec=HAS) == []


def test_preflight_agent_needs_key_unless_echo(tmp_path):
    assert preflight(store_with(tmp_path), LaunchOptions(brain=False), False, HAS) == [{"text": "普通模式要大模型的 API Key：去设置页填", "setting": "secret.llm"}]
    echo = store_with(tmp_path, console='[llm]\nprovider = "echo"\n')
    assert preflight(echo, LaunchOptions(brain=False), False, HAS) == []


def test_preflight_busy_and_broken_config(tmp_path):
    s = store_with(tmp_path, console="[device]\nserail = 1\n", secrets='[env]\nSKYDANGO_CLAUDE_TOKEN = "t"\n')
    problems = preflight(s, LaunchOptions(), busy=True, find_spec=HAS)
    assert problems[0]["text"] == "团子已经在运行" and "serail" in problems[1]["text"] and len(problems) == 2


def test_llm_probe_reports_success_and_failure():
    class Ok:
        def complete(self, system, messages, max_tokens=None):
            return "ok!"

    class Bad:
        def complete(self, *a, **k):
            raise RuntimeError("401")

    seen = {}
    r = probe_llm(LlmConfig(), "sk-x", make=lambda cfg, api_key: seen.update(k=api_key, retries=cfg.max_retries, t=cfg.timeout) or Ok())
    assert r["ok"] and "ok!" in r["text"] and r["text"].startswith("成功") and seen == {"k": "sk-x", "retries": 0, "t": 20.0}
    assert probe_llm(LlmConfig(), "sk-x", make=lambda cfg, api_key: Bad()) == {"ok": False, "text": "失败：401"}
    assert probe_llm(LlmConfig(provider="echo"), "") == {"ok": True, "text": "echo 不调模型"}
    assert probe_llm(LlmConfig(), "") == {"ok": False, "text": "还没填 API Key"}


def test_llm_probe_does_not_touch_the_given_config():
    cfg = LlmConfig()
    probe_llm(cfg, "sk-x", make=lambda c, api_key: type("C", (), {"complete": lambda self, *a, **k: "ok"})())
    assert cfg.max_retries == 2 and cfg.timeout == 30.0


def test_claude_probe(monkeypatch):
    monkeypatch.setattr("skydango.console.probes.resolve_claude", lambda p: ["claude"])
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw.get("env", {}).get("CLAUDE_CODE_OAUTH_TOKEN")))
        return subprocess.CompletedProcess(cmd, 0, stdout="2.1.0" if "--version" in cmd else "ok", stderr="")

    r = probe_claude("claude", ".brain-claude", "tok", run=run)
    assert r["ok"] and "2.1.0" in r["text"] and "ok" in r["text"]
    assert calls[1][0][:4] == ["claude", "-p", "--model", "haiku"] and calls[1][1] == "tok"
    assert probe_claude("claude", ".brain-claude", "", run=run) == {"ok": False, "text": "还没填 Claude 令牌"}


def test_claude_probe_failures(monkeypatch):
    def missing(path):
        raise RuntimeError("找不到 Claude Code（claude）")

    monkeypatch.setattr("skydango.console.probes.resolve_claude", missing)
    assert probe_claude("claude", ".x", "tok") == {"ok": False, "text": "失败：找不到 Claude Code（claude）"}
    monkeypatch.setattr("skydango.console.probes.resolve_claude", lambda p: ["claude"])

    def bad(cmd, **kw):
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="2.1.0", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Invalid API key")

    r = probe_claude("claude", ".x", "tok", run=bad)
    assert not r["ok"] and "Invalid API key" in r["text"]

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 60)

    assert probe_claude("claude", ".x", "tok", run=slow) == {"ok": False, "text": "失败：60 秒没回应"}


TOKEN = 'SKYDANGO_CLAUDE_TOKEN = "t"\n'
KEY = 'DEEPSEEK_API_KEY = "k"\n'


def test_brain_needs_llm_key_for_fallback(tmp_path):
    s = store_with(tmp_path, secrets="[env]\n" + TOKEN)  # 有 Claude 令牌、没有大模型 Key
    assert preflight(s, LaunchOptions(brain=True), busy=False, find_spec=HAS) == [
        {"text": "大脑离线时的备用回复要大模型的 API Key：去设置页填", "setting": "secret.llm"}]


def test_brain_echo_provider_needs_no_llm_key(tmp_path):
    s = store_with(tmp_path, console='[llm]\nprovider = "echo"\n', secrets="[env]\n" + TOKEN)
    assert preflight(s, LaunchOptions(brain=True), busy=False, find_spec=HAS) == []


def test_missing_claude_token_points_to_setting(tmp_path):
    s = store_with(tmp_path, secrets="[env]\n" + KEY)
    assert {"text": "大脑模式要 Claude 令牌：去设置页填", "setting": "secret.claude"} in preflight(s, LaunchOptions(brain=True), busy=False, find_spec=HAS)


def test_all_problems_are_dicts(tmp_path):
    s = store_with(tmp_path, console="[device]\nserail = 1\n")  # 现有测试里的坏配置
    for p in preflight(s, LaunchOptions(), busy=True, find_spec=HAS):
        assert set(p) == {"text", "setting"}
