from skydango.brain.addressee import LABEL_NAMES, Addressee, Verdict, parse_aliases
from skydango.config import AddresseeConfig

FR = {"小明": ["小明", "明哥"], "阿花": ["阿花"], "小红": ["小红"]}
TWO = ["小明", "阿花"]


def make(**cfg):
    return Addressee(AddresseeConfig(**cfg), ["团子", "三彩"], 30.0)


def j(a, now, who, text, nearby=TWO, friends=FR):
    return a.judge(now, who, text, friends=friends, nearby=nearby)


def lt(v):
    return (v.label, v.target)


def test_verdict_tag():
    assert Verdict("me", "叫了你").tag() == "跟你说：叫了你"
    assert Verdict("unsure", "").tag() == LABEL_NAMES["unsure"] == "拿不准"


def test_rule1_self_name():
    v = j(make(), 0, "小明", "团子不认地图")  # 第三人称也算
    assert v.label == "me" and "叫了你" in v.reason


def test_rule2_other_friend_alias():
    a = make()
    v = j(a, 0, "阿花", "明哥你看")
    assert lt(v) == ("other", "小明") and "小明" in v.reason
    assert j(make(), 0, "小明", "明哥你看").label != "other"  # 自己的叫法不算


def test_rule2_self_beats_other():
    assert j(make(), 0, "阿花", "团子和明哥").label == "me"


def test_rule3_group_words():
    assert j(make(), 0, "小明", "你们去哪").label == "all"


def test_rule3_greet_needs_two_nearby():
    assert j(make(), 0, "小明", "晚上好呀", nearby=TWO).label == "all"
    v = j(make(), 0, "小明", "晚上好呀", nearby=["小明"])
    assert v.label == "me" and "身边只有他" in v.reason


def test_rule4_followup():
    a = make()
    a.said(0)
    v = j(a, 5, "小明", "哈哈")
    assert v.label == "me" and "接你的话" in v.reason
    assert j(a, 6, "小明", "嘿嘿").label == "me"
    assert j(a, 7, "小明", "嘻嘻").label == "unsure"


def test_rule4_window_expires():
    a = make()
    a.said(0)
    assert j(a, 31, "小明", "哈哈").label == "unsure"


def test_rule4_broken_by_aside():
    a = make()
    a.said(0)
    assert lt(j(a, 2, "阿花", "明哥你看")) == ("other", "小明")
    v = j(a, 3, "小明", "好了")
    assert lt(v) == ("other", "阿花") and "在回阿花" in v.reason


def test_rule5a_still_talking():
    a = make()
    assert lt(j(a, 0, "小明", "阿花你看")) == ("other", "阿花")
    v = j(a, 10, "小明", "好了")
    assert lt(v) == ("other", "阿花") and "还在跟阿花说" in v.reason


def test_rule5_cut_by_dango():
    a = make()
    j(a, 0, "小明", "阿花你看")
    a.said(5)
    assert j(a, 10, "小明", "好了").label == "me"


def test_rule5c_alternating():
    a = make()
    j(a, 0, "小明", "嘻嘻")
    j(a, 3, "阿花", "哎呀")
    v = j(a, 6, "小明", "你呀")
    assert lt(v) == ("other", "阿花") and "一来一回" in v.reason


def test_rule5_window():
    a = make()
    j(a, 0, "小明", "阿花你看")
    assert j(a, 21, "小明", "好了").label != "other"


def test_rule6_only_friend_nearby():
    assert j(make(), 0, "小明", "哈哈", nearby=["小明"]).label == "me"


def test_rule6_not_if_other_spoke():
    a = make()
    j(a, 0, "阿花", "哈哈", nearby=["小明"])
    assert j(a, 5, "小明", "哈哈", nearby=["小明"]).label == "unsure"


def test_rule0_not_friend():
    for who in ("路人甲", ""):
        v = j(make(), 0, who, "团子在吗")
        assert v.label == "unsure" and "不是好友" in v.reason


def test_ocr_speaker_maps_to_friend():
    a = make()
    j(a, 0, "小明", "阿花你看")
    assert lt(j(a, 5, "小明>", "好了")) == ("other", "阿花")


def test_fuzzy_alias_three_chars():
    fr = {"小明": ["小明", "懒洋洋"], "阿花": ["阿花", "明哥"]}
    assert lt(j(make(), 0, "阿花", "懒羊洋你看", friends=fr)) == ("other", "小明")
    assert j(make(), 0, "阿花", "明天去哪", friends={"小明": ["小明", "明哥"], "阿花": ["阿花"]}).label != "other"


def test_aliases_formats():
    want = {"小明": ["小明", "明哥", "小明明"]}
    for line in ("- 叫法：明哥、小明明", "- 叫法: 明哥,小明明", "-  叫法：明哥 小明明"):
        md = f"# 好友\n\n## 小明\n- 关系：同学\n{line}\n\n## 阿花\n- 叫法：花花\n"
        assert parse_aliases(md, ["小明"]) == want
    assert parse_aliases("## 小明\n- 关系：同学\n", ["小明"]) == {"小明": ["小明"]}
    assert parse_aliases("## 阿花\n- 叫法：花花\n", ["小明"]) == {"小明": ["小明"]}


def test_thread_note():
    a = make()
    for i in range(3):
        j(a, i * 10, "小明", "阿花你看")
        j(a, i * 10 + 5, "阿花", "明哥你看")
    assert a.thread_note(30) == "小明 和 阿花 在聊（1 分钟内 6 句）"
    assert a.thread_note(200) == ""  # 超过 60 秒
    b = make()
    for i in range(4):
        j(b, i * 5, "小明", "阿花你看")
    assert b.thread_note(20) == ""  # 只有单向


def test_state_kept_two_minutes():
    a = make()
    for t in range(0, 301, 10):
        j(a, t, "小明", "哈哈")
    assert len(a._lines) <= 13
