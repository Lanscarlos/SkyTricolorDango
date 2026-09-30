import time

from skydango.chat.memory import MemoryStore
from skydango.config import InnerConfig
from skydango.inner import finish_reflection
from skydango.inner.mind import Mind
from skydango.inner.store import InnerStore

T0 = time.mktime((2026, 9, 30, 22, 0, 0, 0, 0, -1))


def test_finish_writes_diary_memos_and_mind(tmp_path):
    st, mem = InnerStore(tmp_path / "inner"), MemoryStore(tmp_path)
    m = Mind()
    s = finish_reflection({"mood": {"level": "开心", "text": "好"}, "diary": "今天不错。", "memos": ["小明考试过了", "约了周六"]},
                          m, st, mem, {}, [], True, T0, InnerConfig())
    assert s == "小明考试过了；约了周六" and st.last_diaries(1) == ["9月30日：今天不错。"]
    assert "2026年9月30日（周三） 的要点：小明考试过了" in mem.inbox() and st.load_mind().mood.level == "开心"


def test_finish_summary_falls_back_to_diary(tmp_path):
    st = InnerStore(tmp_path / "inner")
    s = finish_reflection({"diary": "x" * 150}, Mind(), st, None, {}, [], True, T0, InnerConfig())
    assert s == "x" * 100


def test_finish_dry_run_writes_nothing(tmp_path):  # Review Focus 5
    st, mem = InnerStore(tmp_path / "inner"), MemoryStore(tmp_path)
    m = Mind()
    s = finish_reflection({"mood": {"level": "开心"}, "diary": "今天不错。", "memos": []}, m, st, mem, {}, [], False, T0, InnerConfig())
    assert s == "今天不错。" and not (tmp_path / "inner").exists() and mem.inbox() == ""
    assert m.mood.level == "开心"  # 内存里照样套上


def test_finish_none_and_junk():
    assert finish_reflection(None, Mind(), None, None, {}, [], True, T0, InnerConfig()) == ""
    assert finish_reflection({"diary": 5, "memos": "不是列表"}, Mind(), None, None, {}, [], False, T0, InnerConfig()) == ""


# ---- 第 3 期：下线反思也套性格 ----
def test_finish_applies_persona_live_only(tmp_path):
    from skydango.inner.ledger import Card
    from skydango.inner.persona import Persona

    cards = {"小明": Card(first_met=0, days=["d1", "d2", "d3"])}
    result = {"persona_add": {"catchphrases": ["害，懒得动"], "jokes": [{"who": "小明", "text": "路痴带路"}]}}
    for persist in (False, True):
        st = InnerStore(tmp_path / str(persist) / "inner")
        p = Persona()
        finish_reflection(result, Mind(), st, None, cards, ["小明"], persist, T0, InnerConfig(), persona=p)
        assert [t.text for t in p.catchphrases] == ["害，懒得动"] and [t.who for t in p.jokes] == ["小明"]
        assert st.persona_path.exists() is persist
    p = Persona()
    finish_reflection(result, Mind(), None, None, cards, ["小明"], False, T0, InnerConfig(), persona=p, soft={"小明"})
    assert p.jokes == []  # 收着点的人不记新老梗


def test_finish_without_persona_unchanged(tmp_path):
    st = InnerStore(tmp_path / "inner")
    finish_reflection({"persona_add": {"catchphrases": ["害"]}}, Mind(), st, None, {}, [], True, T0, InnerConfig())
    assert not st.persona_path.exists()


# ---- 内心页：下线反思也记进流水账 ----
def test_finish_logs_final_reflect(tmp_path):
    from skydango.inner.energy import Energy
    from skydango.inner.log import MindLog
    from skydango.inner.persona import Persona

    log = MindLog(None, persist=False)
    finish_reflection({"mood": {"level": "开心", "text": "好"}, "persona_add": {"catchphrases": ["懒得动"]}},
                      Mind(), None, None, {}, [], False, T0, InnerConfig(), persona=Persona(),
                      mind_log=log, energy=Energy("还行", 60, "还行"))
    r = log.recent()[-1]
    assert r["final"] is True and r["t"] == T0 and r["energy"] == {"level": "还行", "score": 60}
    assert r["changes"] == ["心情 平常→开心（好）", "新口头禅：懒得动"]


def test_finish_none_logs_nothing():
    from skydango.inner.log import MindLog

    log = MindLog(None, persist=False)
    finish_reflection(None, Mind(), None, None, {}, [], False, T0, InnerConfig(), mind_log=log)
    assert log.recent() == []
