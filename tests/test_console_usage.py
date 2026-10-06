"""面板的 /api/usage（spec 2026-10-06-model-usage §5.1 §6.1）：停着读账本、自己查余额；在跑转发子进程。"""

import time

import pytest

from skydango.console.settings import SettingsStore
from skydango.console.usage_view import UsageView
from skydango.models.usage_ledger import Ledger

TODAY = time.strftime("%Y-%m-%d")
KEY = "brain|deepseek/deepseek-flash|main"


@pytest.fixture(autouse=True)
def no_user_env(monkeypatch):
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")


def make(tmp_path, monkeypatch, secrets="", fetch=None):
    monkeypatch.setenv("SKYDANGO_USAGE_LEDGER", str(tmp_path / "usage.json"))
    if secrets:
        (tmp_path / "secrets.toml").write_text(secrets, encoding="utf-8")
    calls = []

    def fake(base, key, **kw):
        calls.append(key)
        return {"available": True, "items": [{"currency": "CNY", "total": 9.5, "granted": 0.0, "topped_up": 9.5}]}

    v = UsageView(SettingsStore(tmp_path / "config.toml", environ={}), fetch=fetch or fake)
    return v, calls


def test_console_usage_stopped_reads_ledger(tmp_path, monkeypatch):
    Ledger(tmp_path / "usage.json").add({TODAY: {"sandbox": {KEY: {"calls": 2, "input": 10, "cost": 0.5}}}}, TODAY)
    v, calls = make(tmp_path, monkeypatch, secrets='[env]\nDEEPSEEK_API_KEY = "sk-1"\n')
    got = v.get()
    assert got["ok"] is True and got["run"] is None and got["source"] is None
    assert got["today"]["rows"][0]["calls"] == 2 and got["today"]["cost"] == 0.5
    ds = {p["id"]: p for p in got["providers"]}["deepseek"]
    assert ds["balance"]["items"][0]["total"] == 9.5 and ds["rate"] is None and ds["closed"] is None
    assert calls == ["sk-1"]
    v.get()
    assert calls == ["sk-1"]  # balance_every 秒内不再查
    assert {u["use"]: u for u in got["uses"]}["brain"]["current"] == "deepseek/deepseek-flash"


def test_console_usage_without_key_has_no_balance(tmp_path, monkeypatch):
    v, calls = make(tmp_path, monkeypatch)
    got = v.get()
    assert calls == [] and {p["id"]: p for p in got["providers"]}["deepseek"]["balance"] is None
    assert got["today"]["rows"] == []
