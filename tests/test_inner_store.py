import time

from skydango.config import InnerConfig
from skydango.inner.ledger import Card, Ledger, Session
from skydango.inner.store import InnerStore

FRIENDS = ["小明", "阿花"]
T0 = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))
CFG = InnerConfig()


def test_people_roundtrip_and_atomic(tmp_path):
    st = InnerStore(tmp_path)
    st.write_people({"小明": Card(first_met=T0, days=["2026-09-30"])}, backfilled=T0)
    cards, backfilled, ok = st.load_people()
    assert ok and backfilled == T0 and cards["小明"].days == ["2026-09-30"]
    assert not list(tmp_path.glob("*.tmp"))
    assert st.people_exists()


def test_missing_people_is_empty(tmp_path):
    st = InnerStore(tmp_path / "nope")
    assert st.load_people() == ({}, None, True) and not st.people_exists()
    assert not (tmp_path / "nope").exists()  # 读不建目录


def test_bad_people_renamed(tmp_path):
    (tmp_path / "people.json").write_text("{坏", encoding="utf-8")
    cards, backfilled, ok = InnerStore(tmp_path).load_people()
    assert (cards, backfilled, ok) == ({}, None, False)
    assert not (tmp_path / "people.json").exists() and list(tmp_path.glob("people.json.bad-*"))


def test_unknown_version_renamed(tmp_path):
    (tmp_path / "people.json").write_text('{"version": 9, "people": {}}', encoding="utf-8")
    assert InnerStore(tmp_path).load_people()[2] is False
    assert list(tmp_path.glob("people.json.bad-*"))


def test_bad_people_not_renamed_when_read_only(tmp_path):
    (tmp_path / "people.json").write_text("{坏", encoding="utf-8")
    assert InnerStore(tmp_path).load_people(quarantine=False) == ({}, None, False)
    assert (tmp_path / "people.json").exists()


def test_days_skip_bad_lines(tmp_path):
    st = InnerStore(tmp_path)
    st.append_day(Session(start=T0, end=T0 + 60, ended="normal"))
    with (tmp_path / "days.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("不是 json\n")
    st.append_days([Session(start=T0 + 100, end=T0 + 200, ended="normal")])
    assert [s.start for s in st.days()] == [T0, T0 + 100]


def test_take_current_once(tmp_path):  # Review Focus 4
    st = InnerStore(tmp_path)
    st.write_current(Session(start=T0, saved=T0 + 60))
    assert st.peek_current().saved == T0 + 60 and (tmp_path / "current.json").exists()
    assert st.take_current().saved == T0 + 60
    assert st.take_current() is None and st.peek_current() is None


def test_bad_current(tmp_path):
    (tmp_path / "current.json").write_text("坏", encoding="utf-8")
    st = InnerStore(tmp_path)
    assert st.peek_current() is None and (tmp_path / "current.json").exists()
    assert st.take_current() is None
    assert not (tmp_path / "current.json").exists() and list(tmp_path.glob("current.json.bad-*"))


def test_ledger_save_throttled_and_close(tmp_path):
    st = InnerStore(tmp_path)
    led = Ledger(CFG, lambda: FRIENDS, T0, store=st, persist=True)
    led.present(["小明"], T0)
    assert led.save(T0) is True and (tmp_path / "current.json").exists()
    assert st.peek_current().saved == T0
    assert led.save(T0 + 30) is False  # save_every = 60
    led.said(T0 + 40)
    led.close("一起看了日落", T0 + 100)
    led.close("又一次", T0 + 200)  # 只生效一次
    (day,) = st.days()
    assert (day.end, day.ended, day.summary, day.said, day.friends) == (T0 + 100, "normal", "一起看了日落", 1, ["小明"])
    assert not (tmp_path / "current.json").exists() and st.load_people()[0]["小明"].visits == 1


def test_ledger_close_keeps_backfilled(tmp_path):
    st = InnerStore(tmp_path)
    Ledger(CFG, lambda: FRIENDS, T0, store=st, persist=True, backfilled=T0 - 5).close("", T0)
    assert st.load_people()[1] == T0 - 5


def test_ledger_not_persisting_writes_nothing(tmp_path):
    led = Ledger(CFG, lambda: FRIENDS, T0, store=InnerStore(tmp_path), persist=False)
    led.present(["小明"], T0)
    assert led.save(T0) is False
    led.close("x", T0 + 1)
    assert list(tmp_path.iterdir()) == []


def test_old_people_json_without_outfits_loads(tmp_path):
    (tmp_path / "people.json").write_text(
        '{"version": 1, "people": {"小明": {"first_met": 1.0, "days": ["2026-09-30"]}}}', encoding="utf-8")
    cards, _, ok = InnerStore(tmp_path).load_people()
    assert ok and cards["小明"].outfits == []


def test_outfits_written_only_when_live(tmp_path):
    for persist, sub in ((True, "live"), (False, "dry")):
        st = InnerStore(tmp_path / sub)
        led = Ledger(CFG, lambda: FRIENDS, T0, store=st, persist=persist)
        led.wear("小明", [0.1], "k", True, T0)
        led.save(T0)
        led.close("", T0 + 1)
    assert InnerStore(tmp_path / "live").load_people()[0]["小明"].outfits[0]["feat"] == [0.1]
    assert not (tmp_path / "dry").exists() or list((tmp_path / "dry").iterdir()) == []
