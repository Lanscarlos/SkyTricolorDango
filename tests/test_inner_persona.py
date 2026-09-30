"""内心层第 3 期：性格档案（口头禅、老梗、看法）的规矩、淡出、拼文字和 persona.json 读写。"""

import json
import time

from skydango.config import InnerConfig
from skydango.inner.ledger import Card
from skydango.inner.persona import SENSITIVE, Persona, Trait
from skydango.inner.store import InnerStore

CFG = InnerConfig()
T0 = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))
CARDS = {
    "小明": Card(first_met=0, days=["d1", "d2", "d3"]),
    "懒洋洋大王": Card(first_met=0, days=["d1", "d2", "d3"]),
    "阿花": Card(first_met=0, days=["d1"]),
}
FRIENDS = ["小明", "懒洋洋大王", "阿花"]


def add(**kw):
    return {"persona_add": kw}


def test_config_defaults():
    c = InnerConfig()
    assert (c.persona, c.fade_days, c.catchphrases_max, c.jokes_per_friend, c.jokes_max, c.opinions_max, c.soft_minutes) == (
        True, 14, 5, 3, 20, 10, 30)
    assert "妈" in SENSITIVE and "班" in SENSITIVE and "年龄" in SENSITIVE and "哭" in SENSITIVE and len(SENSITIVE) == 30


def test_add_three_kinds_and_rules():
    p = Persona()
    dropped = p.apply(add(
        catchphrases=["害，懒得动", "妈呀好累"],  # 第二条带“妈”：丢
        jokes=[{"who": "小明", "text": "上次把团子带进冥龙嘴里"},
               {"who": "阿花", "text": "x"},  # 不熟：丢
               {"who": "路人", "text": "y"},  # 不是好友：丢
               {"who": "懒洋羊大王", "text": "路痴带路"},  # 错字对上正名
               {"who": "小明", "text": "小明又胖了"}],  # 敏感词：丢  (Review Focus 1/2)
        opinions=[{"topic": "雨林啊啊啊啊啊啊啊啊啊", "stance": "湿漉漉的，谁爱去谁去" + "呀" * 30}]),
        CARDS, FRIENDS, T0, CFG)
    assert [t.text for t in p.catchphrases] == ["害，懒得动"]
    assert [(t.who, t.text) for t in p.jokes] == [("小明", "上次把团子带进冥龙嘴里"), ("懒洋洋大王", "路痴带路")]
    assert (len(p.opinions[0].topic), len(p.opinions[0].text)) == (10, 30)
    assert len(dropped) == 4


def test_soft_friend_gets_no_new_jokes():
    p = Persona()
    p.apply(add(jokes=[{"who": "小明", "text": "a"}]), CARDS, FRIENDS, T0, CFG, soft={"小明"})
    assert p.jokes == []


def test_same_topic_replaces_and_duplicates_skip():  # Review Focus 3
    p = Persona()
    p.apply(add(opinions=[{"topic": "雨林", "stance": "太湿了"}], catchphrases=["害，懒得动"]), CARDS, FRIENDS, T0, CFG)
    assert [(t.topic, t.text) for t in p.opinions] == [("雨林", "太湿了")]  # 先确实记上了，下面才测得到“换立场”
    p.apply(add(opinions=[{"topic": "雨林", "stance": "其实还行"}], catchphrases=["害，懒得动啊"]), CARDS, FRIENDS, T0 + 9, CFG)
    assert [(t.topic, t.text, t.since) for t in p.opinions] == [("雨林", "其实还行", T0 + 9)]
    assert len(p.catchphrases) == 1


def test_same_topic_replaces_existing_stance():
    p = Persona(opinions=[Trait("湿漉漉的", topic="雨林", since=T0, hits=2)])
    p.apply(add(opinions=[{"topic": "雨林", "stance": "其实还行"}]), CARDS, FRIENDS, T0 + 9, CFG)
    assert [(t.topic, t.text, t.since) for t in p.opinions] == [("雨林", "其实还行", T0 + 9)]


def test_used_bumps_hits():
    p = Persona(catchphrases=[Trait("害，懒得动", since=T0)], opinions=[Trait("丑", topic="雨林", since=T0)])
    p.apply({"persona_used": ["害，懒得动", "雨林"]}, CARDS, FRIENDS, T0 + 5, CFG)
    assert (p.catchphrases[0].hits, p.catchphrases[0].last_used, p.opinions[0].hits) == (1, T0 + 5, 1)


def test_caps_evict_fewest_hits_then_oldest():  # Review Focus 3
    p = Persona(catchphrases=[Trait(f"口头禅{i}", since=T0 + i, hits=1 if i == 0 else 0) for i in range(5)])
    p.apply(add(catchphrases=["新的"]), CARDS, FRIENDS, T0 + 99, CFG)
    assert [t.text for t in p.catchphrases] == ["口头禅0", "口头禅2", "口头禅3", "口头禅4", "新的"]
    jokes = [{"who": "小明", "text": f"梗{i}"} for i in range(4)]
    p.apply(add(jokes=jokes), CARDS, FRIENDS, T0, CFG)
    assert [t.text for t in p.jokes] == ["梗1", "梗2", "梗3"]  # 每人 3 条


def test_jokes_total_cap():
    cfg = InnerConfig(jokes_max=4)
    p = Persona(jokes=[Trait(f"老梗{i}", who="小明", since=T0 + i) for i in range(3)])
    p.apply(add(jokes=[{"who": "懒洋洋大王", "text": "路痴带路"}, {"who": "懒洋洋大王", "text": "手残"}]), CARDS, FRIENDS, T0 + 9, cfg)
    assert [t.text for t in p.jokes] == ["老梗1", "老梗2", "路痴带路", "手残"]


def test_fade():
    p = Persona(catchphrases=[Trait("旧", since=T0), Trait("用过", since=T0, last_used=T0 + 10 * 86400)])
    p.fade(T0 + 15 * 86400, CFG)
    assert [t.text for t in p.catchphrases] == ["用过"]


def test_junk_shapes():
    p = Persona()
    assert p.apply({"persona_add": "不是对象", "persona_used": 5}, CARDS, FRIENDS, T0, CFG) and p == Persona()
    assert p.apply({"persona_add": {"jokes": "x", "opinions": [5]}}, CARDS, FRIENDS, T0, CFG)


def test_section_and_joke_note():
    p = Persona(catchphrases=[Trait("害，懒得动"), Trait("樱花发型天下第一")],
                jokes=[Trait("上次把团子带进冥龙嘴里", who="小明", hits=2), Trait("路痴带路", who="小明"), Trait("第三个", who="小明")],
                opinions=[Trait("湿漉漉的，谁爱去谁去", topic="雨林")])
    assert p.section().splitlines() == [
        "## 你攒下的性格（慢慢和大家玩出来的；人设里写的优先）",
        "口头禅：害，懒得动；樱花发型天下第一",
        "看法：雨林——湿漉漉的，谁爱去谁去",
        "和小明的老梗：上次把团子带进冥龙嘴里；路痴带路；第三个",
        "用得自然，别每句都用；同一个梗一次上线最多用一两回。",
    ]
    assert p.joke_note("小明") == "。你们的老梗：上次把团子带进冥龙嘴里；路痴带路"  # 最多 2 条，hits 多的先
    assert p.joke_note("阿花") == "" and Persona().section() == ""


def test_show_lines():
    p = Persona(catchphrases=[Trait("害，懒得动", hits=2)])
    lines = p.show_lines()
    assert lines[0] == "===== 性格档案 =====" and any("害，懒得动" in s and "用过 2 次" in s for s in lines)
    assert Persona().show_lines() == ["===== 性格档案 =====", "（空）"]


def test_roundtrip_and_bad_file(tmp_path):  # Review Focus 4
    st = InnerStore(tmp_path)
    p = Persona([Trait("a", since=1)], [Trait("b", who="小明")], [Trait("c", topic="雨林", hits=2)])
    st.write_persona(p)
    assert st.load_persona() == p
    assert json.loads((tmp_path / "persona.json").read_text("utf-8"))["opinions"][0]["stance"] == "c"
    (tmp_path / "persona.json").write_text("坏", encoding="utf-8")
    assert st.load_persona(quarantine=False) == Persona() and (tmp_path / "persona.json").exists()
    assert st.load_persona() == Persona() and list(tmp_path.glob("persona.json.bad-*"))


def test_missing_file_is_empty(tmp_path):
    assert InnerStore(tmp_path / "nope").load_persona() == Persona()
    assert not (tmp_path / "nope").exists()


def test_people_not_in_opinions_or_catchphrases():  # 终审 I1
    p = Persona()
    dropped = p.apply(add(opinions=[{"topic": "阿花", "stance": "笨手笨脚的，别跟她走"},
                                    {"topic": "暮土", "stance": "阿花最怕这里"}],
                          catchphrases=["阿花又迷路了哈哈", "害，懒得动"]), CARDS, FRIENDS, T0, CFG)
    assert p.opinions == [] and [t.text for t in p.catchphrases] == ["害，懒得动"] and len(dropped) == 3


def test_filters_age_crying_identity():  # 终审 I2
    p = Persona()
    p.apply(add(jokes=[{"who": "小明", "text": "小学生小明"}, {"who": "小明", "text": "小明今年才十二"},
                       {"who": "小明", "text": "小明上次哭鼻子"}, {"who": "小明", "text": "小明上次好难过"},
                       {"who": "小明", "text": "路痴带路"}],
                catchphrases=["我才不是机器人", "我是真人好吧"]), CARDS, FRIENDS, T0, CFG)
    assert [t.text for t in p.jokes] == ["路痴带路"] and p.catchphrases == []
    for w in ("年龄", "小学生", "初中", "高中", "作业", "老师", "岁", "本名", "QQ", "微信", "哭"):
        assert w in SENSITIVE


def test_prepare_refilters_and_adopts_hand_written():  # 终审 I2 + 手写的 since=0
    p = Persona(catchphrases=[Trait("我才不是机器人", since=T0), Trait("手写的一句")],
                jokes=[Trait("小学生小明", who="小明", since=T0)],
                opinions=[Trait("笨手笨脚的", topic="阿花", since=T0)])
    dropped = p.prepare(T0 + 5, FRIENDS)
    assert [(t.text, t.since) for t in p.catchphrases] == [("手写的一句", T0 + 5)]
    assert p.jokes == [] and p.opinions == [] and len(dropped) == 3
    p.fade(T0 + 5, CFG)
    assert len(p.catchphrases) == 1  # 手写的不会一启动就淡出


def test_new_entry_not_evicted_when_full():  # 终审 I4
    p = Persona(catchphrases=[Trait(f"说法{i}号", since=T0 + i, hits=3) for i in range(5)])
    p.apply(add(catchphrases=["全新的一句"]), CARDS, FRIENDS, T0 + 99, CFG)
    assert "全新的一句" in [t.text for t in p.catchphrases] and len(p.catchphrases) == 5
    assert "说法0号" not in [t.text for t in p.catchphrases]  # 挤掉旧的里最没用、最旧的


def test_changed_stance_counts_as_used_now():  # 终审：换了立场不该一两天就淡出
    p = Persona(opinions=[Trait("丑丑的", topic="雨林", since=T0, last_used=T0 + 86400)])
    p.opinions[0].text = "太湿了"
    p.apply(add(opinions=[{"topic": "雨林", "stance": "其实还行"}]), CARDS, FRIENDS, T0 + 13 * 86400, CFG)
    p.fade(T0 + 15 * 86400, CFG)
    assert [(t.text, t.last_used) for t in p.opinions] == [("其实还行", T0 + 13 * 86400)]
