import time

from skydango.chat.memory import SKIP, Turn
from skydango.config import InnerConfig
from skydango.inner import open_ledger
from skydango.inner.backfill import backfill
from skydango.inner.ledger import Session, day_of
from skydango.inner.store import InnerStore

FRIENDS = ["小明", "阿花"]
CFG = InnerConfig()
D1 = time.mktime((2026, 9, 28, 20, 0, 0, 0, 0, -1))
D2 = time.mktime((2026, 9, 29, 20, 0, 0, 0, 0, -1))
NOW = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))


def turn(t, *lines, reply="好"):
    return Turn(t, "新的聊天消息：\n" + "\n".join(lines), reply)


TURNS = [
    turn(D1 + 0, "小明：「在吗」", "路人：「hi」"),
    turn(D1 + 600, "小明：「今天跑图吗」", "「看不出是谁」"),
    turn(D1 + 4 * 3600, "阿花：「晚上好」", reply=SKIP),  # 隔 4 小时：第二次上线
    turn(D2 + 0, "小明：「明天要考试」"),  # 第二天
]


def test_backfill_cards():
    cards, _ = backfill(TURNS, FRIENDS, 7200)
    m = cards["小明"]
    assert (m.first_met, m.lines, m.last_seen) == (D1, 3, D2)
    assert m.last_line == {"t": D2, "text": "明天要考试"}
    assert m.days == [day_of(D1), day_of(D2)] and m.minutes == 0 and m.to_me == 0
    assert m.visits == 2  # 出现在 2 次回填上线里
    assert m.today == "" and m.today_visits == 0
    assert "路人" not in cards and cards["阿花"].visits == 1


def test_backfill_sessions():
    _, sessions = backfill(TURNS, FRIENDS, 7200)
    assert [(s.start, s.end, s.ended, s.friends, s.heard, s.said) for s in sessions] == [
        (D1, D1 + 600, "backfill", ["小明"], 4, 2),
        (D1 + 4 * 3600, D1 + 4 * 3600, "backfill", ["阿花"], 1, 0),
        (D2, D2, "backfill", ["小明"], 1, 1),
    ]


def test_backfill_empty():
    assert backfill([], FRIENDS, 7200) == ({}, [])


def test_open_ledger_backfills_once(tmp_path):
    led = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: TURNS, persist=True, now=NOW)
    assert led.card("小明").lines == 3 and len(led.history) == 3 and led.backfilled == NOW
    assert (tmp_path / "people.json").exists() and len(InnerStore(tmp_path).days()) == 3
    led2 = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: TURNS * 2, persist=True, now=NOW + 10)
    assert led2.card("小明").lines == 3  # 已有 people.json：不再回填
    assert len(led2.history) == 3


def test_open_ledger_days_exist_only_cards(tmp_path):
    InnerStore(tmp_path).append_day(Session(start=D1 - 86400, end=D1 - 80000, ended="normal"))
    led = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: TURNS, persist=True, now=NOW)
    assert len(led.history) == 1 and led.card("小明") is not None


def test_open_ledger_recovers_crash(tmp_path):
    st = InnerStore(tmp_path)
    st.write_people({}, None)
    st.write_current(Session(start=NOW - 900, saved=NOW - 300, friends=["小明"]))
    led = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: [], persist=True, now=NOW)
    assert (led.history[-1].end, led.history[-1].ended) == (NOW - 300, "crash")
    assert not (tmp_path / "current.json").exists()
    again = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: [], persist=True, now=NOW + 5)
    assert [s.ended for s in again.history] == ["crash"]  # 连着重启：只补一次


def test_open_ledger_history_error_still_opens(tmp_path):
    def boom():
        raise OSError("读不了")

    led = open_ledger(CFG, tmp_path, lambda: FRIENDS, boom, persist=True, now=NOW)
    assert led.cards == {} and led.history == []
