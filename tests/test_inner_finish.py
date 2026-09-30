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
    assert s == "小明考试过了；约了周六" and st.last_diaries(1) == ["今天不错。"]
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
