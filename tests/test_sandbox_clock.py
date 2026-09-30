"""沙盒的模拟时钟（brain-sandbox 计划 Task 1）。"""

from __future__ import annotations

import json
import logging
import time

import pytest

from skydango.sandbox.clock import (
    SimClock,
    floor_time,
    load_saved,
    parse_duration,
    resolve_start,
    save,
)

T = time.mktime((2026, 9, 30, 21, 0, 0, 0, 0, -1))  # 真实：21:00


def test_skip_moves_both_clocks():
    c = SimClock(real_wall=lambda: T, real_mono=lambda: 100.0)
    c.skip(3600)
    assert c.wall() == T + 3600 and c.clock() == 3700.0


def test_skip_negative_rejected():
    c = SimClock(real_wall=lambda: T, real_mono=lambda: 0.0)
    with pytest.raises(ValueError):
        c.skip(-1)
    assert c.wall() == T


def test_offset_start():
    c = SimClock(offset=60.0, real_wall=lambda: T, real_mono=lambda: 5.0)
    assert c.wall() == T + 60 and c.clock() == 65.0


def test_set_time_forward_only():
    c = SimClock(real_wall=lambda: T, real_mono=lambda: 0.0)
    assert c.set_time("23:30") == 2.5 * 3600
    assert c.set_time("20:00") == 20.5 * 3600  # 今天已过：明天 20:00
    with pytest.raises(ValueError, match="不能往回拨"):
        c.set_time("2026-09-30 22:00")


def test_set_time_full_forward():
    c = SimClock(real_wall=lambda: T, real_mono=lambda: 0.0)
    assert c.set_time("2026-10-01 09:00") == 12 * 3600
    assert c.wall() == time.mktime((2026, 10, 1, 9, 0, 0, 0, 0, -1))


def test_set_time_bad_format():
    c = SimClock(real_wall=lambda: T, real_mono=lambda: 0.0)
    for bad in ("25:00", "明早", "2026/10/01 09:00", ""):
        with pytest.raises(ValueError):
            c.set_time(bad)


def test_resolve_start_respects_floor():  # Review Focus 2
    floor = T + 3600  # 真记忆里最后一次上线结束在 22:00
    assert resolve_start("20:00", floor, T, 9) == time.mktime((2026, 10, 1, 20, 0, 0, 0, 0, -1))
    assert resolve_start("sleep", floor, T, 9) == time.mktime((2026, 10, 1, 9, 0, 0, 0, 0, -1))
    assert resolve_start("resume", floor, T, 9) == floor
    with pytest.raises(ValueError):
        resolve_start("2026-09-30 21:30", floor, T, 9)


def test_resolve_start_floor_in_past():
    floor = T - 86400  # 昨天 21:00 下的线
    assert resolve_start("resume", floor, T, 9) == T
    assert resolve_start("23:00", floor, T, 9) == T + 2 * 3600
    # 睡一晚：上次下线之后的第一个早上 9 点
    assert resolve_start("sleep", floor, T, 9) == time.mktime((2026, 9, 30, 9, 0, 0, 0, 0, -1))
    assert resolve_start("2026-09-30 10:00", floor, T, 9) == time.mktime((2026, 9, 30, 10, 0, 0, 0, 0, -1))


def test_resolve_start_first_time():
    # 第一次 / 重置后没有下限：接着上次 = 现在；睡一晚 = 现在之后的早上 9 点
    assert resolve_start("resume", 0.0, T, 9) == T
    assert resolve_start("sleep", 0.0, T, 9) == time.mktime((2026, 10, 1, 9, 0, 0, 0, 0, -1))


def test_resolve_start_unknown_choice():
    with pytest.raises(ValueError):
        resolve_start("明天", 0.0, T, 9)


def test_parse_duration():
    assert [parse_duration(x) for x in ("30s", "10m", "2h")] == [30, 600, 7200]
    with pytest.raises(ValueError):
        parse_duration("2d")
    for bad in ("", "m", "-5m", "10"):
        with pytest.raises(ValueError):
            parse_duration(bad)


def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "clock.json"
    assert load_saved(p) is None
    save(p, T + 12.5)
    assert load_saved(p) == T + 12.5


def test_load_bad_file(tmp_path, caplog):
    p = tmp_path / "clock.json"
    p.write_text("{坏的", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert load_saved(p) is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    p.write_text('{"wall": "不是数"}', encoding="utf-8")
    assert load_saved(p) is None


def test_floor_time(tmp_path):
    assert floor_time(tmp_path) == 0.0
    inner = tmp_path / "memory" / "inner"
    inner.mkdir(parents=True)
    rows = [{"start": T - 7200, "end": T - 3600}, {"start": T, "end": T + 3600}]
    (inner / "days.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert floor_time(tmp_path) == T + 3600
    save(tmp_path / "clock.json", T + 100)
    assert floor_time(tmp_path) == T + 3600  # 取大的
    save(tmp_path / "clock.json", T + 7200)
    assert floor_time(tmp_path) == T + 7200


def test_floor_time_ignores_bad_last_line(tmp_path):
    inner = tmp_path / "memory" / "inner"
    inner.mkdir(parents=True)
    (inner / "days.jsonl").write_text(
        json.dumps({"start": T, "end": T + 60}) + "\n" + "坏的一行\n", encoding="utf-8"
    )
    assert floor_time(tmp_path) == T + 60
