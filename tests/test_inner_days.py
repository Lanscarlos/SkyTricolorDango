import time

from skydango.inner.days import days_prompt, recent_days
from skydango.inner.ledger import Card, Session

FRIENDS = ["小明", "阿花"]
NOW = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))  # 周三


def test_first_time():
    assert days_prompt([], {}, FRIENDS, NOW, 7) == (
        "## 日子\n今天 2026年9月30日（周三）。这是你第一次上线。\n这些是给你心里有数的，别跟人报数字。"
    )


def test_nth_time_crash_and_long_gap():
    history = [
        Session(start=NOW - 5 * 86400, end=NOW - 5 * 86400 + 3600, ended="normal"),
        Session(start=NOW - 4 * 3600, end=NOW - 3 * 3600, ended="crash", friends=["小明", "阿花"], summary="一起看了日落"),
    ]
    cards = {
        "阿花": Card(first_met=0, last_seen=NOW - 12 * 86400),
        "小红": Card(first_met=0, last_seen=NOW - 20 * 86400),
        "阿蓝": Card(first_met=0, last_seen=NOW - 30 * 86400),
        "小绿": Card(first_met=0, last_seen=NOW - 40 * 86400),
        "小明": Card(first_met=0, last_seen=NOW - 3600),
        "旧名字": Card(first_met=0, last_seen=NOW - 9 * 86400),
        "小紫": Card(first_met=0),  # 没见过：不算“很久没见”
    }
    text = days_prompt(history, cards, ["小明", "阿花", "小红", "阿蓝", "小绿", "小紫"], NOW, 7)
    assert text.splitlines() == [
        "## 日子",
        "今天 2026年9月30日（周三）。这是你第 3 次上线，今天第 2 次；上次下线是 3 小时前（意外断了）。最近 7 天上线了 2 天。",
        "上次见到了：小明、阿花。上次的经过：一起看了日落",
        "很久没见的好友：阿花（12 天前）、小红（20 天前）、阿蓝（30 天前）",
        "这些是给你心里有数的，别跟人报数字。",
    ]


def test_no_long_gap_line_and_empty_last():
    history = [Session(start=NOW - 7200, end=NOW - 60 * 30, ended="normal")]
    lines = days_prompt(history, {}, FRIENDS, NOW, 7).splitlines()
    assert lines[1].endswith("上次下线是 30 分钟前。最近 7 天上线了 1 天。") and len(lines) == 3


def test_only_summary_or_only_friends():
    only_friends = [Session(start=NOW - 7200, end=NOW - 3600, friends=["小明"])]
    assert days_prompt(only_friends, {}, FRIENDS, NOW, 7).splitlines()[2] == "上次见到了：小明。"
    only_summary = [Session(start=NOW - 7200, end=NOW - 3600, summary="挂机")]
    assert days_prompt(only_summary, {}, FRIENDS, NOW, 7).splitlines()[2] == "上次的经过：挂机"


def test_recent_days():
    history = [
        Session(start=NOW - 86400, end=None, ended="backfill"),
        Session(start=NOW - 3600, end=NOW - 600, ended="crash", friends=["小明"], summary="x" * 60),
    ]
    assert recent_days(history) == [
        "9月30日 19:00  50 分钟  小明  意外断了  " + "x" * 40 + "…",
        "9月29日 20:00  没有结束时间  没见到好友  回填",
    ]
    assert recent_days(history, n=1) == [recent_days(history)[0]]


def test_ledger_days_prompt_uses_history_and_cards():
    from skydango.config import InnerConfig
    from skydango.inner.ledger import Ledger

    led = Ledger(InnerConfig(), lambda: FRIENDS, NOW, cards={"阿花": Card(first_met=0, last_seen=NOW - 10 * 86400)},
                 history=[Session(start=NOW - 7200, end=NOW - 3600, ended="normal")])
    text = led.days_prompt(NOW)
    assert "这是你第 2 次上线" in text and "很久没见的好友：阿花（10 天前）" in text


def test_week_counts_calendar_days():  # 终审 #5：滚动窗口 + 日历日期会数出 8 天
    wed_10 = time.mktime((2026, 9, 30, 10, 0, 0, 0, 0, -1))
    history = [Session(start=wed_10 - 7 * 86400 + 3600, end=wed_10 - 7 * 86400 + 7200, ended="normal")]
    history += [Session(start=wed_10 - d * 86400, end=wed_10 - d * 86400 + 60, ended="normal") for d in range(6, 0, -1)]
    assert "最近 7 天上线了 7 天" in days_prompt(history, {}, FRIENDS, wed_10, 7)
