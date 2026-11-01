import time

from skydango.config import InnerConfig
from skydango.inner.feelings import (
    STALE,
    apply_feelings,
    distressed,
    feeling_bit,
    feeling_note,
    trend_shown,
)
from skydango.inner.ledger import Card

FRIENDS = ["小明", "阿花", "懒洋洋大王"]
NOW = time.mktime((2026, 11, 1, 20, 0, 0, 0, 0, -1))
TODAY = "2026-11-01"
cfg = InnerConfig()


def card(days=5, feeling=None, last_seen=NOW):
    return Card(first_met=NOW - 86400 * 30, last_seen=last_seen, days=[f"2026-10-{i + 1:02d}" for i in range(days)],
                feeling=dict(feeling or {}))


def feel(text="嘴贫但靠谱", close="熟了", trend="平稳", who="小明"):
    return {"who": who, "text": text, "close": close, "trend": trend}


def apply(cards, proposed, met=("小明", "阿花"), soft=()):
    return apply_feelings(cards, proposed, met=set(met), soft=set(soft), today=TODAY, friends=FRIENDS, cfg=cfg)


def test_old_card_without_feeling_loads():
    assert Card.from_dict({"first_met": 0.0}).feeling == {}


def test_apply_new_feeling():
    cards = {"小明": card(days=5)}
    changes, dropped = apply(cards, [feel(trend="在变近")])
    assert cards["小明"].feeling == {"text": "嘴贫但靠谱", "close": "熟了", "trend": "在变近", "since": TODAY, "past": []}
    assert changes and dropped == []


def test_not_met_or_unknown_dropped():
    cards = {"小明": card(), "阿花": card()}
    changes, dropped = apply(cards, [feel(who="阿花"), feel(who="路人甲")], met={"小明"})
    assert changes == [] and cards["阿花"].feeling == {}
    assert any("这次没见到" in d for d in dropped) and len(dropped) == 2


def test_ocr_typo_matches_friend():
    cards = {"懒洋洋大王": card()}
    apply(cards, [feel(who="懒洋羊大王")], met={"懒洋洋大王"})
    assert cards["懒洋洋大王"].feeling["text"] == "嘴贫但靠谱"


def test_duplicate_who_keeps_first():
    cards = {"小明": card()}
    apply(cards, [feel(text="第一条印象"), feel(text="第二条印象")])
    assert cards["小明"].feeling["text"] == "第一条印象"


def test_close_moves_one_step():
    cards = {"小明": card(days=10, feeling={"text": "刚来的", "close": "刚认识", "trend": "平稳", "since": "2026-10-01", "past": []})}
    apply(cards, [feel(close="很铁")])
    assert cards["小明"].feeling["close"] == "熟了"


def test_close_needs_days():
    cards = {"小明": card(days=1), "阿花": card(days=5, feeling={"text": "爱弹琴", "close": "熟了", "trend": "平稳", "since": "2026-10-01", "past": []})}
    _, dropped = apply(cards, [feel(close="熟了"), feel(who="阿花", text="爱弹琴还爱唱歌", close="很铁")])
    assert cards["小明"].feeling["close"] == "刚认识"
    assert cards["阿花"].feeling["close"] == "熟了"
    assert len(dropped) == 2


def test_cap_still_one_step_down():  # 终审：旧「很铁」但天数不够时别一次掉两档
    old = {"text": "嘴贫但靠谱", "close": "很铁", "trend": "平稳", "since": "2026-10-20", "past": []}
    cards = {"小明": card(days=2, feeling=old)}
    apply(cards, [feel(close="很铁")])
    assert cards["小明"].feeling["close"] == "熟了"


def test_favorite_thing_is_not_clingy():  # 终审：「最爱吃辣」不是依恋
    cards = {"小明": card()}
    apply(cards, [feel(text="最爱吃辣，嘴还硬")])
    assert cards["小明"].feeling["text"] == "最爱吃辣，嘴还硬"
    cards = {"小明": card()}
    apply(cards, [feel(text="我最爱他了")])
    assert cards["小明"].feeling == {}


def test_bad_level_dropped():
    cards = {"小明": card(), "阿花": card()}
    changes, dropped = apply(cards, [feel(close="好朋友"), feel(who="阿花", trend="变远")])
    assert changes == [] and cards["小明"].feeling == {} and cards["阿花"].feeling == {} and len(dropped) == 2


def test_too_long_dropped_not_cut():
    cards = {"小明": card()}
    apply(cards, [feel(text="很" * 41)])
    assert cards["小明"].feeling == {}


def test_sensitive_clingy_distress_dropped():
    for bad in ("有点胖但是人好", "离不开他了", "他今天很难过"):
        cards = {"小明": card()}
        apply(cards, [feel(text=bad)])
        assert cards["小明"].feeling == {}, bad


def test_soft_no_cooling_no_grumble():
    old = {"text": "嘴贫但靠谱", "close": "熟了", "trend": "平稳", "since": "2026-10-20", "past": []}
    cards = {"小明": card(feeling=old)}
    apply(cards, [feel(text="又放鸽子，有点烦", trend="有点生疏")], soft={"小明"})
    f = cards["小明"].feeling
    assert f["text"] == "嘴贫但靠谱" and f["trend"] == "平稳"


def test_same_text_new_close_pushes_past():
    old = {"text": "嘴贫但靠谱", "close": "刚认识", "trend": "平稳", "since": "2026-10-20", "past": []}
    cards = {"小明": card(days=5, feeling=old)}
    apply(cards, [feel(text="嘴贫但靠谱。", close="熟了")])
    f = cards["小明"].feeling
    assert f["close"] == "熟了" and f["past"] == [{"text": "嘴贫但靠谱", "close": "刚认识", "date": "2026-10-20"}]


def test_trend_only_no_past():
    old = {"text": "嘴贫但靠谱", "close": "熟了", "trend": "平稳", "since": "2026-10-20", "past": []}
    cards = {"小明": card(feeling=old)}
    apply(cards, [feel(trend="在变近")])
    assert cards["小明"].feeling["trend"] == "在变近" and cards["小明"].feeling["past"] == []


def test_past_max_three():
    cards = {"小明": card()}
    for i in range(5):
        apply(cards, [feel(text=f"第{i}版的印象完全不同{'甲乙丙丁戊'[i] * 4}")])
    f = cards["小明"].feeling
    assert len(f["past"]) == 3 and f["past"][0]["text"].startswith("第1版")


def test_non_list_or_bad_items_ignored():
    cards = {"小明": card()}
    for bad in ("x", {"who": "小明"}, [{"who": None}], [{"text": "a"}], ["小明"], None):
        changes, _ = apply(cards, bad)
        assert changes == []
    assert cards["小明"].feeling == {}


def test_distressed():
    assert distressed([(0, "小明", "我今天好难过"), (1, "我", "难过"), (2, "阿花", "哈哈")], FRIENDS) == {"小明"}


def test_notes_and_bits():
    c = card(feeling={"text": "嘴贫但靠谱", "close": "熟了", "trend": "平稳", "since": TODAY, "past": []})
    assert feeling_note(c, NOW, cfg, soft=False) == "。你对他：嘴贫但靠谱（熟了，最近平稳）"
    assert feeling_note(c, NOW, cfg, soft=True) == "。你对他：（熟了）"
    assert feeling_bit(c, NOW, cfg) == "熟了"
    assert feeling_note(card(), NOW, cfg, soft=False) == "" and feeling_bit(card(), NOW, cfg) == ""


def test_cooling_shown_in_bit():
    c = card(feeling={"text": "最近不怎么来", "close": "熟了", "trend": "有点生疏", "since": TODAY, "past": []})
    assert feeling_bit(c, NOW, cfg) == "熟了·有点生疏"


def test_feelings_lines():
    from skydango.inner.feelings import feelings_lines

    cards = {"小明": card(days=5, feeling={"text": "嘴贫但靠谱", "close": "熟了", "trend": "平稳", "since": "2026-10-20", "past": []}),
             "阿花": card(days=1)}
    assert feelings_lines(cards, ["小明", "阿花", "路人"], NOW, cfg) == [
        "小明：嘴贫但靠谱（熟了，平稳，2026-10-20 写的；一起玩过 5 天）",
        "阿花：还没有印象（一起玩过 1 天）",
    ]


def test_describe_with_feel():
    from skydango.inner.ledger import describe

    c = card(days=9)
    assert describe("小明", c, NOW, feel="熟了").endswith("·一起玩过 9 天·熟了）")
    assert describe("小明", c, NOW) == describe("小明", c, NOW, feel="")
    assert "熟了" not in describe("小明", c, NOW)


def test_introspect_text():
    from skydango.inner.feelings import introspect_text

    past = [{"text": "刚认识的时候话少", "close": "刚认识", "date": "2026-10-02"}]
    cards = {"小明": card(feeling={"text": "嘴贫但靠谱", "close": "熟了", "trend": "平稳", "since": "2026-10-20", "past": past}),
             "阿花": card()}
    text = introspect_text(cards, FRIENDS, NOW, cfg)
    for s in ("小明", "嘴贫但靠谱", "熟了", "平稳", "2026-10-20", "刚认识的时候话少", "2026-10-02"):
        assert s in text
    assert "阿花" not in text
    assert introspect_text({"阿花": card()}, FRIENDS, NOW, cfg) == "还没对谁有印象"


def test_stale_shown():
    c = card(feeling={"text": "嘴贫但靠谱", "close": "熟了", "trend": "平稳", "since": TODAY, "past": []}, last_seen=NOW - 86400 * 15)
    assert trend_shown(c, NOW, cfg) == STALE
    assert feeling_bit(c, NOW, cfg) == "熟了·好久没见了"
