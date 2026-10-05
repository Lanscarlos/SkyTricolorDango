import subprocess

import pytest

from skydango.console.preflight import preflight
from skydango.console.probes import test_provider as probe_provider
from skydango.console.runner import LaunchOptions
from skydango.console.settings import SettingsStore
from skydango.models.config import ProviderConfig

HAS = lambda name: object()  # noqa: E731  mcp 装了
CLAUDE = ProviderConfig("claude", "claude-code", models=("haiku",), vision=("haiku",))


@pytest.fixture(autouse=True)
def no_registry(monkeypatch):
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    monkeypatch.setattr("skydango.models.claude_code.resolve_claude", lambda path: ["claude"])  # 测试机上没有 claude 命令


def store_with(tmp_path, console="", secrets=""):
    if console:
        (tmp_path / "console.toml").write_text(console, encoding="utf-8")
    if secrets:
        (tmp_path / "secrets.toml").write_text(secrets, encoding="utf-8")
    return SettingsStore(tmp_path / "config.toml", environ={})


TOKEN = 'SKYDANGO_CLAUDE_TOKEN = "t"\n'
KEY = 'DEEPSEEK_API_KEY = "k"\n'


@pytest.mark.parametrize("secrets", ["[env]\n" + KEY, "[env]\n" + TOKEN])
def test_brain_mode_ok_with_main_or_backup(tmp_path, secrets):
    # 大脑主 deepseek / 备 claude 有一家能用就不拦（另一家的用处到时停用或改走备用）
    assert preflight(store_with(tmp_path, secrets=secrets), LaunchOptions(brain=True), busy=False, find_spec=HAS) == []


def test_brain_mode_blocks_only_when_brain_has_no_model(tmp_path):
    [p] = preflight(store_with(tmp_path), LaunchOptions(brain=True), busy=False, find_spec=HAS)
    assert p["setting"] == "models.brain" and p["text"].startswith("大脑没有能用的模型") and "DEEPSEEK_API_KEY" in p["text"]


def test_preflight_brain_needs_mcp(tmp_path):
    s = store_with(tmp_path, secrets="[env]\n" + TOKEN + KEY)
    assert preflight(s, LaunchOptions(brain=True), busy=False, find_spec=lambda n: None) == [
        {"text": "大脑要用 mcp：先 pip install --user mcp", "setting": None}]


def test_plain_mode_needs_reply(tmp_path):
    assert preflight(store_with(tmp_path), LaunchOptions(brain=False), False, HAS) == [
        {"text": "普通模式的回复模型用不了：deepseek 缺 Key（环境变量 DEEPSEEK_API_KEY）", "setting": "models.reply"}]
    echo = store_with(tmp_path, console='[models.reply]\nmain = "echo/echo"\n')
    assert preflight(echo, LaunchOptions(brain=False), False, HAS) == []


def test_preflight_busy_and_broken_config(tmp_path):
    s = store_with(tmp_path, console="[device]\nserail = 1\n", secrets="[env]\n" + TOKEN)
    problems = preflight(s, LaunchOptions(), busy=True, find_spec=HAS)
    assert problems[0]["text"] == "团子已经在运行" and "serail" in problems[1]["text"] and len(problems) == 2


def test_claude_probe_failures(monkeypatch):
    def missing(path):
        raise RuntimeError("找不到 Claude Code（claude）")

    monkeypatch.setattr("skydango.console.probes.resolve_claude", missing)
    assert probe_provider(CLAUDE, "tok") == {"ok": False, "text": "失败：找不到 Claude Code（claude）"}
    monkeypatch.setattr("skydango.console.probes.resolve_claude", lambda p: ["claude"])

    def bad(cmd, **kw):
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="2.1.0", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Invalid API key")

    r = probe_provider(CLAUDE, "tok", run=bad)
    assert not r["ok"] and "Invalid API key" in r["text"]

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 60)

    assert probe_provider(CLAUDE, "tok", run=slow) == {"ok": False, "text": "失败：60 秒没回应"}


def test_claude_probe_passes_token(monkeypatch):
    monkeypatch.setattr("skydango.console.probes.resolve_claude", lambda p: ["claude"])
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw.get("env", {}).get("CLAUDE_CODE_OAUTH_TOKEN")))
        return subprocess.CompletedProcess(cmd, 0, stdout="2.1.0" if "--version" in cmd else "ok", stderr="")

    r = probe_provider(CLAUDE, "tok", run=run)
    assert r["ok"] and "2.1.0" in r["text"] and "ok" in r["text"]
    assert calls[1][0][:4] == ["claude", "-p", "--model", "haiku"] and calls[1][1] == "tok"


def test_all_problems_are_dicts(tmp_path):
    s = store_with(tmp_path, console="[device]\nserail = 1\n")  # 现有测试里的坏配置
    for p in preflight(s, LaunchOptions(), busy=True, find_spec=HAS):
        assert set(p) == {"text", "setting"}
