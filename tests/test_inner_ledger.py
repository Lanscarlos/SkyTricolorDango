import time

import pytest

from skydango.config import InnerConfig
from skydango.inner.ledger import Card, Ledger, card_line, match_friend

FRIENDS = ["小明", "阿花", "懒洋洋大王", "懒洋洋大王子"]
T0 = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))  # 周三 20:00（本地时间）


def ledger(**kw):
    return Ledger(InnerConfig(), lambda: FRIENDS, T0 - 86400 * 30, **kw)


def test_match_friend_prefers_closest():  # Review Focus 2
    assert match_friend("懒洋洋大王子", FRIENDS) == "懒洋洋大王子"
    assert match_friend("懒洋羊大王", FRIENDS) == "懒洋洋大王"  # 错一个字
    assert match_friend("小明", FRIENDS) == "小明"
    assert match_friend("", FRIENDS) is None and match_friend("我", FRIENDS) is None
    assert match_friend("路人甲", FRIENDS) is None


def test_first_meeting_note_and_card():
    led = ledger()
    assert led.present(["小明"], T0) == {"小明": "（第一次在身边见到）"}
    c = led.card("小明")
    assert (c.first_met, c.visits, c.today_visits, c.days) == (T0, 1, 1, ["2026-09-30"])


def test_same_visit_within_gap_counts_once():
    led = ledger()
    led.present(["小明"], T0)
    led.present([], T0 + 60)
    assert led.present(["小明"], T0 + 1799) == {}  # 30 分钟内再出现：同一次
    assert led.card("小明").visits == 1


def test_new_visit_note_after_gap():
    led = ledger()
    led.present(["小明"], T0 - 3 * 3600)
    note = led.present(["小明"], T0)["小明"]
    assert note == "（第 2 次见；上次 3 小时前；今天第 2 次）"


def test_long_gap_and_last_line_before_today():
    led = ledger()
    led.heard("小明", "明天要考试", False, T0 - 12 * 86400)
    led.present(["小明"], T0 - 12 * 86400)
    note = led.present(["小明"], T0)["小明"]
    assert note == "（第 2 次见；上次 12 天前，好久没见了；今天第一次）。他上次最后说的（12 天前）：「明天要考试」"


def test_last_line_today_not_attached():
    led = ledger()
    led.present(["小明"], T0 - 3 * 3600)
    led.heard("小明", "晚点见", False, T0 - 3 * 3600)
    assert "上次最后说的" not in led.present(["小明"], T0)["小明"]


def test_minutes_step_capped_and_clock_back():
    led = ledger()
    led.present(["小明"], T0)
    led.present(["小明"], T0 + 2)  # +2 s
    led.present(["小明"], T0 + 600)  # 卡了 598 s：只算 5
    led.present(["小明"], T0 + 590)  # 时间往回跳：0
    assert led.card("小明").minutes == pytest.approx(7 / 60)
    assert led.card("小明").last_seen == T0 + 600


def test_midnight_resets_today():  # Review Focus 3
    led = ledger()
    night = time.mktime((2026, 9, 30, 23, 59, 58, 0, 0, -1))
    for t in range(int(night - 600), int(night + 4), 2):
        led.present(["小明"], t)
    c = led.card("小明")
    assert c.today == "2026-10-01" and c.today_minutes < 0.1 and c.days == ["2026-09-30", "2026-10-01"]
    assert "今天刚来" in led.status_line(["小明"], night + 4)


def test_status_reads_today_even_if_not_present_since():
    led = ledger()
    for t in range(int(T0), int(T0 + 1205), 5):
        led.present(["小明"], t)
    assert "今天一起 20 分钟" in led.status_line(["小明"], T0 + 1200)
    assert "今天刚来" in led.status_line(["小明"], T0 + 86400)  # 第二天还没更新过：不显示昨天的


def test_heard_counts_friend_lines_only():
    led = ledger()
    led.heard("懒洋羊大王", "x" * 50, True, T0)  # 错字也记到懒洋洋大王
    led.heard("路人", "你好", False, T0)
    led.heard("", "？", False, T0)
    c = led.card("懒洋洋大王")
    assert (c.lines, c.to_me, c.last_line) == (1, 1, {"t": T0, "text": "x" * 40})
    assert led.card("路人") is None and led.session.heard == 3 and led.session.friends == ["懒洋洋大王"]
    assert c.last_seen is None  # 只说话没见过：不算见过


def test_heard_before_seen_then_first_meeting():
    led = ledger()
    led.heard("小明", "我在后面", True, T0)
    assert led.present(["小明"], T0 + 5) == {"小明": "（第一次在身边见到）"}


def test_status_line():
    led = ledger()
    led.present(["小明", "阿花"], T0 - 21 * 86400)
    for t in range(int(T0), int(T0 + 2405), 5):  # 到 T0+2400：480 × 5 s = 40 分钟
        led.present(["小明"], t)
    led.present(["阿花"], T0 + 2400)
    assert led.status_line(["小明", "阿花", "路人"], T0 + 2400) == (
        "小明（今天一起 40 分钟·认识 21 天·一起玩过 2 天）、阿花（今天刚来·认识 21 天·一起玩过 2 天）、路人"
    )


def test_status_line_new_friend():
    led = ledger()
    led.present(["阿花"], T0)
    assert led.status_line(["阿花"], T0 + 10) == "阿花（今天刚来·今天刚认识）"


def test_said_counts():
    led = ledger()
    led.said(T0)
    led.said(T0)
    assert led.session.said == 2


def test_card_is_a_copy():
    led = ledger()
    led.present(["小明"], T0)
    led.card("小明").visits = 99
    assert led.card("小明").visits == 1


def test_card_roundtrip():
    c = Card(first_met=T0, days=["2026-09-30"], last_line={"t": T0, "text": "hi"})
    assert Card.from_dict(c.to_dict()) == c


def test_card_line():
    c = Card(first_met=T0 - 86400, last_seen=T0 - 3600, visits=2, days=["2026-09-29", "2026-09-30"], lines=5, to_me=2)
    assert card_line("小明", c, T0) == "小明（今天刚来·认识 1 天·一起玩过 2 天）；见过 2 次，上次 1 小时前；说过 5 句，跟你说过 2 句"
    assert card_line("阿花", Card(first_met=T0), T0).endswith("；还没在身边见过；说过 0 句，跟你说过 0 句")


F1, F2, F3, F4 = [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]
D1, D2, D3, D4 = T0, T0 + 86400, T0 + 2 * 86400, T0 + 3 * 86400


def test_wear_new_then_same_updates_last_day():
    led = ledger()
    led.wear("小明", F1, "color-v1", False, D1)
    led.wear("小明", F2, "color-v1", False, D2)
    (o,) = led.all_outfits()["小明"]
    assert (o["desc"], o["feat"], o["first"], o["last"]) == ("", F2, "2026-09-30", "2026-10-01")
    assert led.card("小明").first_met == D1  # 没卡就建


def test_wear_changed_appends_and_keeps_three():
    led = ledger()
    for f, d in ((F1, D1), (F2, D2), (F3, D3), (F4, D4)):
        led.wear("小明", f, "k", True, d)
    assert [o["feat"] for o in led.all_outfits()["小明"]] == [F2, F3, F4]
    led.wear("路人甲", F1, "k", True, D1)  # 不认识的名字：什么都不做
    assert "路人甲" not in led.all_outfits()
    led2 = ledger(outfit_keep=1)
    led2.wear("小明", F1, "k", True, D1)
    led2.wear("小明", F2, "k", True, D2)
    assert [o["feat"] for o in led2.all_outfits()["小明"]] == [F2]


def test_describe_outfit_returns_previous_desc_only_for_a_change():
    l = ledger()
    assert l.describe_outfit("小明", "白斗篷", D1) is None  # 没装扮：不建
    l.wear("小明", F1, "color-v1", True, D1)
    assert l.describe_outfit("小明", "白斗篷", D1) is None  # 第一套，没有前一套
    l.wear("小明", F2, "color-v1", True, D2)
    assert l.describe_outfit("小明", "粉斗篷", D2) == "白斗篷"
    assert l.all_outfits()["小明"][-1]["desc"] == "粉斗篷"


def test_describe_outfit_no_event_for_outfit_from_earlier_session_or_empty_prev():
    l = ledger()
    l.wear("小明", F1, "k", True, D1)
    l.wear("小明", F2, "k", True, D1)  # 前一套没描述
    assert l.describe_outfit("小明", "粉斗篷", D1) is None
    card = Card(first_met=D1, outfits=[{"desc": "旧", "feat": F1, "key": "k", "first": "2026-09-30", "last": "2026-09-30"},
                                       {"desc": "", "feat": F2, "key": "k", "first": "2026-09-30", "last": "2026-09-30"}])
    l2 = ledger(cards={"小明": card})
    assert l2.describe_outfit("小明", "新", D1) is None  # 以前上线时加的，不算换装


def test_outfits_are_copies():
    led = ledger()
    led.wear("小明", F1, "k", True, D1)
    led.all_outfits()["小明"][0]["desc"] = "改了"
    led.card("小明").outfits[0]["desc"] = "改了"
    assert led.all_outfits()["小明"][0]["desc"] == ""
