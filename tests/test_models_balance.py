"""DeepSeek 余额（spec 2026-10-06-model-usage §5.1）。"""

import io
import json
import urllib.error

import pytest

from skydango.config import Config
from skydango.models.balance import BalanceError, BalanceWatcher, balance_targets, fetch_balance
from skydango.models.config import ProviderConfig, resolve
from skydango.models.gate import ProviderGates
from skydango.models.usage import UsageMeter

OK = {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "12.34", "granted_balance": "0.00",
                                              "topped_up_balance": "12.34"}]}


class Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def opener_of(body, seen=None):
    def open_(req, timeout=None):
        if seen is not None:
            seen.append(req)
        if isinstance(body, Exception):
            raise body
        return Resp(body if isinstance(body, bytes) else json.dumps(body).encode())

    return open_


def test_fetch_balance_ok():
    seen = []
    got = fetch_balance("https://api.deepseek.com", "k", opener=opener_of(OK, seen))
    assert got == {"available": True, "items": [{"currency": "CNY", "total": 12.34, "granted": 0.0, "topped_up": 12.34}]}
    assert seen[0].full_url == "https://api.deepseek.com/user/balance"
    assert seen[0].get_header("Authorization") == "Bearer k"


def test_fetch_balance_base_url_with_slash():
    seen = []
    fetch_balance("https://api.deepseek.com/", "k", opener=opener_of(OK, seen))
    assert seen[0].full_url == "https://api.deepseek.com/user/balance"


def test_fetch_balance_http_error():
    err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b""))
    with pytest.raises(BalanceError, match="401"):
        fetch_balance("https://api.deepseek.com", "k", opener=opener_of(err))
    with pytest.raises(BalanceError):
        fetch_balance("https://api.deepseek.com", "k", opener=opener_of(TimeoutError("timed out")))


@pytest.mark.parametrize("body", [b"<html>", {"x": 1}, {"balance_infos": [{"total_balance": "abc"}]}])
def test_balance_bad_json(body):   # Review Focus 4
    with pytest.raises(BalanceError, match="看不懂"):
        fetch_balance("https://api.deepseek.com", "k", opener=opener_of(body))


def test_targets_only_deepseek_with_key(monkeypatch):
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")  # 本机 setx 过的 Key 不算
    setup = resolve(Config())
    assert [p.id for p, key in balance_targets(setup, {"DEEPSEEK_API_KEY": "sk"})] == ["deepseek"]
    assert balance_targets(setup, {}) == []
    setup.providers["openai"] = ProviderConfig("openai", "openai", base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY")
    setup.providers["ollama"] = ProviderConfig("ollama", "openai", base_url="http://127.0.0.1:11434/v1", key_env="X")
    assert [p.id for p, _ in balance_targets(setup, {"DEEPSEEK_API_KEY": "sk", "OPENAI_API_KEY": "o", "X": "x"})] == ["deepseek"]


def test_watcher_poll_once_records(monkeypatch):
    setup = resolve(Config())
    meter = UsageMeter(setup, ProviderGates(), source="live")
    w = BalanceWatcher(setup, meter, {"DEEPSEEK_API_KEY": "sk"}, every=300, fetch=lambda base, key, **kw: {"available": True, "items": []})
    w.poll_once()
    bal = {p["id"]: p for p in meter.snapshot()["providers"]}["deepseek"]["balance"]
    assert bal["available"] is True and bal["error"] is None

    def boom(base, key, **kw):
        raise BalanceError("401 Unauthorized")

    w.fetch = boom
    w.poll_once()
    assert {p["id"]: p for p in meter.snapshot()["providers"]}["deepseek"]["balance"]["error"] == "401 Unauthorized"


def test_watcher_start_stop():
    setup = resolve(Config())
    meter = UsageMeter(setup, ProviderGates(), source="live")
    calls = []
    w = BalanceWatcher(setup, meter, {"DEEPSEEK_API_KEY": "sk"}, every=3600, fetch=lambda base, key, **kw: calls.append(1) or {})
    w.start()
    w.stop()
    assert calls == [1]  # 启动时先查一次
