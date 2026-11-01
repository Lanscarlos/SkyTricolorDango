from datetime import datetime

from skydango.brain.events import Event
from skydango.brain.recollect import Recollector, bigrams
from skydango.config import ReplyConfig

NOTES = (
    "## 番茄炒蛋盖饭\n- 下周去青岛出差三天 〔2026-10-21｜出差〕\n- 养仓鼠，叫芝麻球 〔2026-10-21〕\n"
    "- 在追长风渡 〔2026-10-30〕\n- 买了新耳机 〔2026-10-29〕\n## 小明\n- 腿摔了 〔2026-10-30〕\n## 自己\n- 不会吹萨克斯 〔2026-10-20〕\n"
)
NOW = datetime(2026, 10, 31, 22, 0).timestamp()


def chat(who, text):
    return Event("chat", f"{who}：{text}", 0.0, speech=(who, text))


def make(notes=NOTES, cfg=None, match=None, inbox="", wall=None, load=None):
    return Recollector(load or (lambda: (notes, inbox)), match or (lambda w: None), cfg or ReplyConfig(), wall or (lambda: NOW))


def test_bigrams():
    assert bigrams("我下周出差，今天") >= {"下周", "周出", "出差"}
    assert "今天" not in bigrams("今天") and bigrams("好") == set()


def test_topic_and_speaker():
    r = make(NOTES)
    out = r.line([chat("番茄炒蛋盖饭", "我下周出差要去青岛")], 0)
    assert out.startswith("想起来的：\n- 番茄炒蛋盖饭：下周去青岛出差三天（10 天前）")
    assert "小明" not in out


def test_threshold_needs_person_or_words():
    assert make(NOTES).line([Event("notice", "天黑了", 0.0)], 0) is None


def test_once_per_session_until_epoch_changes():
    r = make(NOTES)
    ev = [chat("番茄炒蛋盖饭", "青岛出差")]
    assert "青岛" in r.line(ev, 0)
    assert (r.line(ev, 0) or "").find("青岛") < 0
    assert "青岛" in r.line(ev, 1)


def test_arrive_brings_latest_three():
    out = make(NOTES).line([Event("arrive", "番茄炒蛋盖饭 来到身边", 0.0, who="番茄炒蛋盖饭")], 0)
    assert [x.split("：", 1)[1].split("（")[0] for x in out.splitlines()[1:4]] == ["在追长风渡", "买了新耳机", "下周去青岛出差三天"]


def test_limits():
    out = make(NOTES, ReplyConfig(recall_per_turn=2)).line([Event("arrive", "", 0.0, who="番茄炒蛋盖饭")], 0)
    assert len(out.splitlines()) == 3
    out = make(NOTES, ReplyConfig(recall_chars=30)).line([Event("arrive", "", 0.0, who="番茄炒蛋盖饭")], 0)
    assert len(out.splitlines()) == 2  # 第一条总是带，第二条就超字数了


def test_self_section_when_talked_to():
    assert "萨克斯" in make(NOTES).line([chat("小明", "你会吹萨克斯吗")], 0)


def test_speaker_alias_and_typo_match_section():
    r = make(NOTES, match=lambda w: "番茄炒蛋盖饭" if w in ("番茄", "番茄炒蛋盖坂") else None)
    assert "仓鼠" in r.line([chat("番茄炒蛋盖坂", "我家仓鼠又跑了")], 0)


def test_mid_session_inbox_not_in_pool():
    calls = []

    def load():
        calls.append(1)
        if len(calls) == 1:
            return NOTES, ""
        return NOTES, "- 番茄炒蛋盖饭：去了霞谷 〔2026-10-31〕\n"

    r = make(load=load)
    out = r.line([chat("番茄炒蛋盖饭", "我去霞谷了")], 0)
    assert "霞谷" not in out


def test_startup_inbox_in_pool():
    r = make(inbox="- 番茄炒蛋盖饭：煮了酸辣粉 〔2026-10-31〕\n")
    assert "酸辣粉" in r.line([chat("番茄炒蛋盖饭", "酸辣粉好吃")], 0)


def test_recollect_uses_given_clock():
    ev = [chat("番茄炒蛋盖饭", "我下周出差要去青岛")]
    earlier = datetime(2026, 10, 26, 12, 0).timestamp()
    assert "下周去青岛出差三天（5 天前）" in make(NOTES, wall=lambda: earlier).line(ev, 0)
    later = datetime(2026, 11, 10, 12, 0).timestamp()  # 20 天前跨月：按 show 的规矩写「上个月 10-21」
    assert "下周去青岛出差三天（上个月 10-21）" in make(NOTES, wall=lambda: later).line(ev, 0)


def test_notes_rewritten_rebuilds_pool():
    state = {"n": NOTES}
    r = make(load=lambda: (state["n"], ""))
    state["n"] = NOTES + "- 去了霞谷 〔2026-10-31〕\n"  # 位置在最后一节(自己)，换成番茄节
    state["n"] = NOTES.replace("## 小明", "- 去了霞谷 〔2026-10-31〕\n## 小明")
    assert "霞谷" in r.line([chat("番茄炒蛋盖饭", "霞谷好看")], 0)


def test_pinned_entries_not_in_pool_unless_overflow():
    notes = "## 番茄\n- 爱喝奶茶 〔2026-10-30｜长期〕\n- 去霞谷 〔2026-10-30〕\n"
    out = make(notes).line([chat("番茄", "奶茶霞谷")], 0)
    assert "霞谷" in out and "奶茶" not in out
    out = make(notes, ReplyConfig(pinned_max=10)).line([chat("番茄", "奶茶霞谷")], 0)
    assert "奶茶" in out


def test_error_returns_none():
    def boom():
        raise RuntimeError("坏了")

    r = make(NOTES)
    r.load = boom
    assert r.line([chat("番茄炒蛋盖饭", "青岛")], 0) is None


# ---- 11-01：聊到了就提醒性格里的看法 / 老梗 ----
def test_persona_cues_by_topic_and_friend():
    from skydango.brain.recollect import persona_cues
    from skydango.inner.persona import Persona, Trait

    p = Persona(opinions=[Trait("无翼不敢进，怕坠机", topic="霞谷"), Trait("从来没准过", topic="识别模块")],
                jokes=[Trait("管我叫沙子团子，我回一句你们嘴都打歪了", who="懒洋洋大王"), Trait("路痴带路", who="小明")])
    skip: set = set()
    assert persona_cues(p, ["走，去霞谷跑图"], {"懒洋洋大王"}, skip) == ["- 你对「霞谷」的看法：无翼不敢进，怕坠机"]
    assert persona_cues(p, ["沙子团子在干嘛"], {"懒洋洋大王"}, skip) == [
        "- 和懒洋洋大王的老梗：管我叫沙子团子，我回一句你们嘴都打歪了"]
    assert persona_cues(p, ["霞谷霞谷", "沙子团子"], {"懒洋洋大王"}, skip) == []  # 一次上线同一条只带一次
    assert persona_cues(p, ["路痴带路"], {"懒洋洋大王"}, set()) == []  # 老梗只在那个好友在说话时带
    many = persona_cues(Persona(opinions=[Trait("a", topic="雨林"), Trait("b", topic="云野"), Trait("c", topic="禁阁")]),
                        ["雨林云野禁阁"], set(), set())
    assert len(many) == 2
