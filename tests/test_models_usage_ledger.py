"""用量账本 runs/usage.json（spec 2026-10-06-model-usage §4）：增量合并、锁、按天、坏文件。"""

import json
import os
import time
from datetime import datetime, timedelta

from skydango.config import Config
from skydango.models.config import resolve
from skydango.models.gate import ProviderGates
from skydango.models.usage import UsageMeter
from skydango.models.usage_ledger import Ledger

NOW = time.time()
TODAY = time.strftime("%Y-%m-%d", time.localtime(NOW))
KEY = "brain|deepseek/deepseek-flash|main"


def meter(path, source="live"):
    return UsageMeter(resolve(Config()), ProviderGates(), source=source, wall=lambda: NOW, ledger=Ledger(path))


def brain(m, n=1):
    for _ in range(n):
        m.record("brain", "deepseek", "deepseek-flash", backup=False, usage={"input_tokens": 10}, ok=True)


def test_add_creates_and_merges(tmp_path):
    led = Ledger(tmp_path / "usage.json")
    row = {"calls": 1, "fails": 0, "input": 5, "output": 0, "cache_read": 0, "cache_write": 0, "cost": 0.1, "est": False}
    assert led.add({TODAY: {"live": {KEY: row}}}, TODAY)
    assert led.add({TODAY: {"live": {KEY: row}}}, TODAY)
    data = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert data["version"] == 1
    got = data["days"][TODAY]["live"][KEY]
    assert got["calls"] == 2 and got["input"] == 10 and abs(got["cost"] - 0.2) < 1e-9


def test_two_meters_add_up(tmp_path):   # Review Focus 2
    path = tmp_path / "usage.json"
    a, b = meter(path, "live"), meter(path, "offline")
    brain(a, 2)
    brain(b, 3)
    assert a.flush() and b.flush()
    rows = Ledger(path).day_rows(TODAY)
    assert rows[KEY]["calls"] == 5 and rows[KEY]["input"] == 50


def test_lock_busy_keeps_delta(tmp_path):
    path = tmp_path / "usage.json"
    m = meter(path)
    brain(m)
    lock = tmp_path / "usage.json.lock"
    lock.write_text("x")
    assert m.flush() is False
    assert not path.exists()
    old = time.time() - 31
    os.utime(lock, (old, old))
    assert m.flush() is True
    assert not lock.exists()
    assert Ledger(path).day_rows(TODAY)[KEY]["calls"] == 1


def test_drop_old_days(tmp_path):
    path = tmp_path / "usage.json"
    old = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")
    recent = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
    row = {"calls": 1}
    path.write_text(json.dumps({"version": 1, "days": {old: {"live": {KEY: row}}, recent: {"live": {KEY: row}}}}), encoding="utf-8")
    assert Ledger(path, keep_days=7).add({TODAY: {"live": {KEY: row}}}, TODAY)
    days = json.loads(path.read_text(encoding="utf-8"))["days"]
    assert old not in days and recent in days and TODAY in days


def test_bad_file_renamed(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{坏的", encoding="utf-8")
    assert Ledger(path).add({TODAY: {"live": {KEY: {"calls": 1}}}}, TODAY)
    assert list(tmp_path.glob("usage.json.bad-*"))
    assert Ledger(path).day_rows(TODAY)[KEY]["calls"] == 1


def test_snapshot_today_includes_ledger(tmp_path):
    path = tmp_path / "usage.json"
    sandbox = meter(path, "sandbox")
    brain(sandbox)
    assert sandbox.flush()
    live = meter(path, "live")
    brain(live)

    def today_calls():
        return {(r["use"], r["model"]): r["calls"] for r in live.snapshot()["today"]["rows"]}[("brain", "deepseek/deepseek-flash")]

    assert today_calls() == 2
    assert live.flush()
    assert today_calls() == 2  # 写盘后不重复算
    assert live.snapshot()["run"]["rows"][0]["calls"] == 1


def test_flush_error_not_raised(tmp_path):
    (tmp_path / "usage.json").mkdir()
    m = meter(tmp_path / "usage.json")
    brain(m)
    assert m.flush() is False
    assert m.take_delta()  # 增量还在


def test_close_flushes_and_stops_saver(tmp_path):
    path = tmp_path / "usage.json"
    m = meter(path)
    m.start_saver(3600)
    brain(m)
    m.close()
    assert Ledger(path).day_rows(TODAY)[KEY]["calls"] == 1


def test_save_every_zero_means_off(tmp_path):   # 终审 I1：0 = 关，不是死循环
    m = meter(tmp_path / "usage.json")
    m.start_saver(0)
    assert m._saver is None
    m.close()
