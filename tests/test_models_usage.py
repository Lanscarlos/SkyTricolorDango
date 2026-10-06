"""模型用量（spec 2026-10-06-model-usage §2 §3 §5 §6.1）：记账、算钱、限额、余额、snapshot。"""

from datetime import datetime, timedelta, timezone

import pytest

from skydango.config import Config
from skydango.models.config import DEEPSEEK_PEAK_HOURS, Price, resolve
from skydango.models.gate import ProviderGates
from skydango.models.usage import UsageMeter, call_cost, is_peak

BJ = timezone(timedelta(hours=8))


def bj(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=BJ).timestamp()


IDLE = bj(2026, 10, 6, 20)  # 周二晚上：空闲


def meter(source="live", gates=None):
    return UsageMeter(resolve(Config()), gates or ProviderGates(), source=source, wall=lambda: IDLE)


def rows(snap, part="run"):
    return {(r["use"], r["model"], r["backup"]): r for r in snap[part]["rows"]}


def test_is_peak():
    h = DEEPSEEK_PEAK_HOURS
    assert is_peak(bj(2026, 10, 6, 10), h)          # 周二 10 点
    assert not is_peak(bj(2026, 10, 6, 12), h)      # 12:00 不算（左闭右开）
    assert is_peak(bj(2026, 10, 6, 14), h)
    assert not is_peak(bj(2026, 10, 6, 13), h)
    assert not is_peak(bj(2026, 10, 4, 10), h)      # 周日
    assert not is_peak(bj(2026, 10, 6, 10), ())


def test_call_cost_split_cache():
    flash = Price(0.02, 1, 4)
    usage = {"input_tokens": 1_000_000, "cache_read_input_tokens": 400_000, "output_tokens": 100_000}
    assert call_cost(flash, usage, 1.0) == pytest.approx(0.4 * 0.02 + 0.6 * 1 + 0.1 * 4)
    assert call_cost(flash, usage, 2.0) == pytest.approx(2 * (0.4 * 0.02 + 0.6 * 1 + 0.1 * 4))
    assert call_cost(None, usage, 1.0) is None
    assert call_cost(flash, {"input_tokens": 10, "cache_read_input_tokens": 50}, 1.0) == pytest.approx(50 * 0.02 / 1e6)


def test_record_rows_and_snapshot():
    m = meter()
    u = {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 800}
    m.record("brain", "deepseek", "deepseek-flash", backup=False, usage=u, ok=True)
    m.record("brain", "deepseek", "deepseek-flash", backup=False, usage=u, ok=True)
    m.record("brain", "claude", "sonnet", backup=True, usage=u, ok=True)
    m.record("eyes", "deepseek", "deepseek-flash", backup=False, usage=None, ok=False)
    snap = m.snapshot()
    r = rows(snap)
    assert len(r) == 3
    main = r[("brain", "deepseek/deepseek-flash", False)]
    assert (main["calls"], main["fails"], main["input"], main["output"], main["cache_read"]) == (2, 0, 2000, 200, 1600)
    assert main["label"] == "大脑"
    assert main["cost"] == pytest.approx(2 * (800 * 0.02 + 200 * 1 + 100 * 4) / 1e6)
    eyes = r[("eyes", "deepseek/deepseek-flash", False)]
    assert (eyes["calls"], eyes["fails"], eyes["input"]) == (1, 1, 0)
    assert snap["run"]["rows"][0]["use"] == "brain" and snap["run"]["rows"][0]["backup"] is False  # 钱最多的在前
    assert snap["run"]["cost"] == pytest.approx(main["cost"] + eyes["cost"])
    assert snap["run"]["est"] is False
    assert snap["source"] == "live"


def test_peak_cost_doubles():
    m = meter()
    u = {"input_tokens": 1_000_000, "output_tokens": 0}
    m.record("brain", "deepseek", "deepseek-flash", backup=False, usage=u, ok=True, at=bj(2026, 10, 6, 10))
    assert m.snapshot()["run"]["cost"] == pytest.approx(2.0)


def test_estimated_model_marks_est():
    m = meter()
    m.record("reply", "deepseek", "deepseek-chat", backup=False, usage={"input_tokens": 10}, ok=True)
    snap = m.snapshot()
    assert snap["run"]["rows"][0]["est"] is True and snap["run"]["est"] is True


def test_record_tolerates_missing_usage():   # Review Focus 3
    m = meter()
    for usage in (None, {}, {"input_tokens": None, "output_tokens": "x"}):
        m.record("memory", "deepseek", "deepseek-flash", backup=False, usage=usage, ok=True)
    r = rows(m.snapshot())[("memory", "deepseek/deepseek-flash", False)]
    assert (r["calls"], r["input"], r["output"]) == (3, 0, 0)


def test_claude_has_no_cost():
    m = meter()
    m.record("brain", "claude", "sonnet", backup=False, usage={"input_tokens": 5}, ok=True)
    snap = m.snapshot()
    assert snap["run"]["rows"][0]["cost"] is None and snap["run"]["cost"] is None


def test_unknown_provider_still_counted():
    m = meter()
    m.record("brain", "nope", "x", backup=False, usage={"input_tokens": 5}, ok=True)
    assert m.snapshot()["run"]["rows"][0]["calls"] == 1


def test_snapshot_providers():
    gates = ProviderGates()
    gates.trip("claude", "limit", "429")
    m = meter(gates=gates)
    m.rate_limit("claude", {"status": "allowed_warning", "rateLimitType": "five_hour", "utilization": 0.8, "resetsAt": 123,
                            "unifiedWindows": {"seven_day": {"utilization": 0.3, "resetsAt": 456}, "bad": {"utilization": "x"}},
                            "junk": 1}, at=99)
    m.balance("deepseek", {"available": True, "items": [{"currency": "CNY", "total": 12.34, "granted": 0.0, "topped_up": 12.34}]},
              None, at=98)
    by = {p["id"]: p for p in m.snapshot()["providers"]}
    assert set(by) == {"claude", "deepseek"}
    assert by["claude"]["closed"] == "额度 / 余额用完" and by["deepseek"]["closed"] is None
    assert by["claude"]["rate"] == {"status": "allowed_warning", "type": "five_hour", "utilization": 0.8, "resets_at": 123,
                                    "windows": {"seven_day": {"utilization": 0.3, "resets_at": 456}}, "at": 99}
    assert by["deepseek"]["balance"] == {"available": True, "items": [{"currency": "CNY", "total": 12.34, "granted": 0.0,
                                                                       "topped_up": 12.34}], "at": 98, "error": None}
    m.rate_limit("claude", {"status": "rejected", "utilization": "x"}, at=100)
    assert m.snapshot()["providers"][0]["rate"]["status"] == "rejected"
    assert "utilization" not in m.snapshot()["providers"][0]["rate"]
    m.balance("deepseek", None, "401", at=101)
    assert {p["id"]: p for p in m.snapshot()["providers"]}["deepseek"]["balance"]["error"] == "401"


def test_snapshot_uses():
    gates = ProviderGates()
    gates.trip("deepseek", "limit", "402")
    uses = {u["use"]: u for u in meter(gates=gates).snapshot()["uses"]}
    assert uses["brain"]["label"] == "大脑"
    assert uses["brain"]["current"] == "claude/sonnet" and uses["brain"]["backup"] is True
    assert uses["reply"]["current"] is None and uses["reply"]["backup"] is False
    assert uses["eyes"]["disabled"] is None


def test_delta_take_restore():
    m = meter(source="sandbox")
    m.record("brain", "deepseek", "deepseek-flash", backup=False, usage={"input_tokens": 7}, ok=True)
    d = m.take_delta()
    day = datetime.fromtimestamp(IDLE).strftime("%Y-%m-%d")
    row = d[day]["sandbox"]["brain|deepseek/deepseek-flash|main"]
    assert row["calls"] == 1 and row["input"] == 7 and row["est"] is False
    assert m.take_delta() == {}
    m.restore_delta(d)
    m.record("brain", "deepseek", "deepseek-flash", backup=False, usage={"input_tokens": 3}, ok=True)
    assert m.take_delta()[day]["sandbox"]["brain|deepseek/deepseek-flash|main"]["input"] == 10
    assert rows(m.snapshot())[("brain", "deepseek/deepseek-flash", False)]["input"] == 10  # 这次运行不受 take 影响


def test_recap_row_and_hidden_use():   # spec 2026-10-06-brain-compact §5
    m = meter()
    m.record("recap", "deepseek", "deepseek-flash", backup=True, usage={"input_tokens": 100, "output_tokens": 10}, ok=True)
    snap = m.snapshot()
    r = rows(snap)[("recap", "deepseek/deepseek-flash", True)]
    assert r["label"] == "压缩" and r["calls"] == 1
    assert "recap" not in [u["use"] for u in snap["uses"]]
