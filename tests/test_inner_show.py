import time

from skydango.config import Config
from skydango.inner import show_lines
from skydango.inner.ledger import Card, Session
from skydango.inner.store import InnerStore

NOW = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))


def test_show_lines(tmp_path):
    st = InnerStore(tmp_path)
    st.write_people(
        {"小明": Card(first_met=NOW - 86400, last_seen=NOW - 3600, visits=2, days=["2026-09-29", "2026-09-30"], lines=5, to_me=2)},
        None,
    )
    st.append_day(Session(start=NOW - 7200, end=NOW - 3600, ended="normal", friends=["小明"]))
    lines = show_lines(Config().inner, tmp_path, ["小明"], NOW)
    assert lines[0] == "===== 关系卡 ====="
    assert lines[1] == "小明（今天刚来·认识 1 天·一起玩过 2 天）；见过 2 次，上次 1 小时前；说过 5 句，跟你说过 2 句"
    assert lines[2:] == ["===== 最近 10 次上线 =====", "9月30日 18:00  1 小时  小明  正常下线"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["days.jsonl", "people.json"]  # 只读，没多出文件


def test_show_lines_lists_old_names_last(tmp_path):
    InnerStore(tmp_path).write_people({"旧名字": Card(first_met=NOW), "小明": Card(first_met=NOW)}, None)
    lines = show_lines(Config().inner, tmp_path, ["小明"], NOW)
    assert lines[1].startswith("小明（") and lines[2].startswith("旧名字（不在好友名单里）（")


def test_show_lines_empty(tmp_path):
    assert show_lines(Config().inner, tmp_path / "none", [], NOW) == [
        "===== 关系卡 =====", "（空）", "===== 最近 10 次上线 =====", "（空）",
    ]
    assert not (tmp_path / "none").exists()


def test_show_lines_bad_people_not_renamed(tmp_path):
    (tmp_path / "people.json").write_text("{坏", encoding="utf-8")
    assert show_lines(Config().inner, tmp_path, [], NOW)[1] == "（people.json 读不了）"
    assert (tmp_path / "people.json").exists()
