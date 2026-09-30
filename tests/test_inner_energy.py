import time

from skydango.inner.energy import Energy, awake_minutes, energy
from skydango.inner.ledger import Session

T0 = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))


def test_day_fresh():
    assert energy(15.0, 10, False, 0) == Energy("精神", 100, "精神")


def test_late_and_long():
    e = energy(0.8, 130, False, 0)  # 半夜 60，挂了 2 个多小时 −20
    assert (e.level, e.score) == ("有点累", 40) and e.note == "有点累（半夜了，连着挂了 2 个多小时）"


def test_small_hours_sleepy():
    assert energy(3.0, 200, False, 0).level == "困"  # 40 − 30


def test_hour_bands():
    assert [energy(h, 0, False, 0).score for h in (8.0, 9.0, 21.9, 22.0, 23.9, 0.0, 1.9, 2.0, 6.9, 7.0)] == [
        80, 100, 100, 80, 80, 60, 60, 40, 40, 80,
    ]
    assert energy(22.5, 0, False, 0).note == "精神（挺晚了）" and energy(7.5, 0, False, 0).note == "精神（刚起）"
    assert energy(4.0, 0, False, 0).note == "有点累（凌晨了）"


def test_cheered_and_busy():
    assert energy(23.0, 0, True, 0).score == 90  # 80 + 10
    assert energy(15.0, 0, False, 60).score == 85  # (60−30)/10 × 5 = 15
    assert energy(15.0, 0, False, 200).score == 85  # 封顶 −15
    assert energy(15.0, 0, True, 0).score == 100  # 夹到 100
    assert "有人陪着聊" in energy(23.0, 0, True, 0).note and "闹了好一阵" in energy(15.0, 0, False, 60).note
    assert energy(3.0, 900, False, 200).score == 0  # 夹到 0


def test_awake_minutes_continues_without_sleep():
    prev = [Session(start=T0 - 7200, end=T0 - 1800)]  # 上次挂了 90 分钟，半小时前下线
    assert awake_minutes(T0 + 600, T0, prev, 3600) == 100
    assert awake_minutes(T0 + 600, T0, [Session(start=T0 - 20000, end=T0 - 10000)], 3600) == 10
    assert awake_minutes(T0 + 600, T0, [], 3600) == 10
    crash = [Session(start=T0 - 1200, end=None, saved=T0 - 600)]  # 意外断的：用 saved
    assert awake_minutes(T0 + 600, T0, crash, 3600) == 20
