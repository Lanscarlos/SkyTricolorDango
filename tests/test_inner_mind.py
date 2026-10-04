import tempfile
import time
from pathlib import Path

from skydango.config import InnerConfig
from skydango.inner.energy import Energy
from skydango.inner.ledger import Card
from skydango.inner.mind import Grudge, Mind, Mood, Want, parse_reflection
from skydango.inner.store import InnerStore

CFG = InnerConfig()
T0 = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))
CARDS = {
    "小明": Card(first_met=0, days=["d1", "d2", "d3"]),
    "懒洋洋大王": Card(first_met=0, days=["d1", "d2", "d3"]),
    "阿花": Card(first_met=0, days=["d1"]),
}
FRIENDS = ["小明", "懒洋洋大王", "阿花"]


def test_apply_mood_and_since():
    m = Mind()
    m.apply({"mood": {"level": "低落", "text": "有点闷，小明说好来又没来" + "啊" * 30}}, CARDS, FRIENDS, T0, CFG)
    assert (m.mood.level, len(m.mood.text), m.mood.since) == ("低落", 40, T0)
    m.apply({"mood": {"level": "低落", "text": "还是闷"}}, CARDS, FRIENDS, T0 + 60, CFG)
    assert m.mood.since == T0 and m.mood.text == "还是闷"  # 档位没变：since 不动
    m.apply({"mood": {"level": "狂喜", "text": "x"}}, CARDS, FRIENDS, T0 + 99, CFG)
    assert m.mood.level == "平常" and m.updated == T0 + 99


def test_grudge_rules():  # Review Focus 1
    m = Mind()
    assert m.apply({"grudge": {"who": "路人", "why": "x"}}, CARDS, FRIENDS, T0, CFG) and m.grudge is None
    m.apply({"grudge": {"who": "阿花", "why": "x"}}, CARDS, FRIENDS, T0, CFG)
    assert m.grudge is None  # 不够熟（1 天）
    m.apply({"grudge": {"who": "懒洋羊大王", "why": "放鸽子"}}, CARDS, FRIENDS, T0, CFG)  # 错字对上正名
    assert (m.grudge.who, m.grudge.until) == ("懒洋洋大王", T0 + 7200)
    m.apply({"grudge": {"who": "小明", "why": "又放鸽子"}}, CARDS, FRIENDS, T0, CFG)  # 新的替换旧的
    assert m.grudge.who == "小明"
    m.apply({"grudge": "keep"}, CARDS, FRIENDS, T0 + 1, CFG)
    assert m.grudge is not None
    m.apply({}, CARDS, FRIENDS, T0 + 1, CFG)  # 没写 grudge：不变
    assert m.grudge is not None
    m.apply({"grudge": None}, CARDS, FRIENDS, T0 + 2, CFG)
    assert m.grudge is None


def test_wants_rules():
    m = Mind()
    m.apply({"wants_add": [
        {"kind": "惦记", "text": "阿花考试考得怎么样", "who": "阿花"},
        {"kind": "惦记", "text": "路人怎么样", "who": "路人"},  # 惦记非好友：丢
        {"kind": "想做", "text": "想看日落", "who": "阿花"},  # who 清空
        {"kind": "想去", "text": "霞谷"},  # kind 不认识：丢
        {"kind": "小心思", "text": "想换个发型"},
        {"kind": "小心思", "text": "想换个发型呀"},  # 和已有的几乎一样：丢
        {"kind": "想做", "text": "想听小明弹琴"},  # 第 4 条：挤掉最旧的
    ]}, CARDS, FRIENDS, T0, CFG)
    assert [w.text for w in m.wants] == ["想看日落", "想换个发型", "想听小明弹琴"]
    assert m.wants[0].who == "" and m.wants[0].until == T0 + 7 * 86400
    m.apply({"wants_done": ["想看日落了"]}, CARDS, FRIENDS, T0, CFG)
    assert [w.text for w in m.wants] == ["想换个发型", "想听小明弹琴"]


def test_apply_ignores_junk_shapes():
    m = Mind()
    dropped = m.apply({"mood": "开心", "wants_add": "不是列表", "wants_done": 3, "grudge": 5}, CARDS, FRIENDS, T0, CFG)
    assert m == Mind(updated=T0) and dropped


def test_expire():
    m = Mind(grudge=Grudge("小明", "x", T0, T0 + 10), wants=[Want("想做", "a", "", T0, T0 + 5)])
    m.expire(T0 + 11)
    assert m.grudge is None and m.wants == []


def test_line_and_want_note():
    m = Mind(Mood("低落", "有点闷，小明说好来又没来", T0), Grudge("小明", "说好来又没来", T0, T0 + 2400),
             [Want("惦记", "阿花考试考得怎么样", "阿花", T0, T0 + 99999)])
    e = Energy("困", 20, "困（半夜了）")
    assert m.line(T0, e) == "有点闷，小明说好来又没来 · 困（半夜了） · 跟小明闹别扭（说好来又没来，还有 40 分钟消气） · 惦记：阿花考试考得怎么样"
    assert Mind().line(T0, Energy("精神", 100, "精神")) == "平常 · 精神"
    assert Mind().line(T0, None) == "平常"
    assert m.want_note("阿花") == "。你惦记着：阿花考试考得怎么样" and m.want_note("小明") == ""
    assert m.grudge_on("小明", T0) and not m.grudge_on("小明", T0 + 2400) and not m.grudge_on("阿花", T0)
    kinds = Mind(wants=[Want("想做", "看日落"), Want("小心思", "换发型")])
    assert kinds.line(T0, None) == "平常 · 想：看日落 · 小心思：换发型"


def test_parse_reflection():  # Review Focus 2
    assert parse_reflection('```json\n{"mood": {"level": "开心"}}\n```') == {"mood": {"level": "开心"}}
    assert parse_reflection('好的，结果如下：{"grudge": null} 就这样') == {"grudge": None}
    assert parse_reflection("我今天挺开心的") is None
    assert parse_reflection('{"坏": ') is None
    assert parse_reflection("[1, 2]") is None  # 不是对象
    assert parse_reflection("") is None and parse_reflection(None) is None


def test_parse_reflection_merges_split_objects():
    # 10-03 晚真机：下线反思把心情 / 日记和性格分成两个对象回，第一个 { 到最后一个 } 解析失败，日记丢了
    text = ('```json\n{"mood": {"level": "开心"}, "diary": "今天很热闹"}\n'
            '{"persona_add": {"catchphrases": []}, "persona_used": []}\n```')
    assert parse_reflection(text) == {"mood": {"level": "开心"}, "diary": "今天很热闹",
                                      "persona_add": {"catchphrases": []}, "persona_used": []}
    # 中间夹着字、后面还有一个坏的：好的照收
    assert parse_reflection('{"grudge": null}\n然后是：{"wants_add": []} {"坏": ') == {"grudge": None, "wants_add": []}


def test_mind_roundtrip_and_bad_file(tmp_path):
    st = InnerStore(tmp_path)
    m = Mind(Mood("开心", "x", T0), Grudge("小明", "y", T0, T0 + 5), [Want("想做", "a", "", T0, T0 + 1)], T0)
    st.write_mind(m)
    assert st.load_mind() == m
    (tmp_path / "mind.json").write_text("坏", encoding="utf-8")
    assert st.load_mind(quarantine=False) == Mind() and (tmp_path / "mind.json").exists()
    assert st.load_mind() == Mind() and list(tmp_path.glob("mind.json.bad-*"))
    assert InnerStore(tmp_path / "none").load_mind() == Mind()


def test_diary(tmp_path):
    st = InnerStore(tmp_path)
    st.append_diary("今天和小明看了日落。", T0)
    st.append_diary("晚上又上来挂了会儿。", T0 + 3600)
    st.append_diary("第二天。", T0 + 86400)
    assert st.last_diaries(1) == ["10月1日：第二天。"]
    assert st.last_diaries(2) == ["9月30日：晚上又上来挂了会儿。", "10月1日：第二天。"]
    assert (tmp_path / "diary.md").read_text(encoding="utf-8").count("## ") == 2
    assert InnerStore(tmp_path / "none").last_diaries(1) == [] and st.last_diaries(0) == []


def test_grudge_not_renewed_and_cooldown():  # 终审 I1
    m = Mind()
    m.apply({"grudge": {"who": "小明", "why": "放鸽子"}}, CARDS, FRIENDS, T0, CFG)
    m.apply({"grudge": {"who": "小明", "why": "还是放鸽子"}}, CARDS, FRIENDS, T0 + 1200, CFG)
    assert (m.grudge.since, m.grudge.until, m.grudge.why) == (T0, T0 + 7200, "还是放鸽子")  # 同一个人：不续期
    m.expire(T0 + 7200)
    assert m.grudge is None
    m.apply({"grudge": {"who": "小明", "why": "又来"}}, CARDS, FRIENDS, T0 + 7300, CFG)
    assert m.grudge is None  # 刚消气：冷却一个 grudge_max
    m.apply({"grudge": {"who": "小明", "why": "又来"}}, CARDS, FRIENDS, T0 + 14500, CFG)
    assert m.grudge is not None
    assert Mind.from_dict(m.to_dict()) == m


def test_forgive_sets_cooldown():  # 终审 I2
    m = Mind(grudge=Grudge("小明", "x", T0, T0 + 7200))
    assert m.forgive("小明", T0 + 100) and m.grudge is None and not m.forgive("小明", T0 + 101)
    m.apply({"grudge": {"who": "小明", "why": "y"}}, CARDS, FRIENDS, T0 + 200, CFG)
    assert m.grudge is None


def test_wake_resets_mood_after_rest():  # 终审 I4
    m = Mind(Mood("烦", "被放鸽子", T0), updated=T0)
    assert not m.wake(T0 + 600, 3600) and m.mood.level == "烦"
    assert m.wake(T0 + 3700, 3600) and m.mood == Mood(since=T0 + 3700)
    assert not Mind().wake(T0, 3600)


def test_diary_entries_with_date():  # 终审 I3
    st = InnerStore(Path(tempfile.mkdtemp()))
    st.append_diary("今天和小明看了日落。", T0)
    st.append_diary("晚上又上来\n挂了会儿。", T0 + 600)
    assert st.last_diaries(1) == ["9月30日：晚上又上来 挂了会儿。"]
    assert st.last_diaries(2) == ["9月30日：今天和小明看了日落。", "9月30日：晚上又上来 挂了会儿。"]
