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
    assert j(make(), 0, "小明", "我先下了 晚安", nearby=TWO).label == "all"  # 句尾也算（10-11 沙盒）
    assert j(make(), 0, "小明", "困了，拜拜~", nearby=TWO).label == "all"  # 句尾的标点不算
    assert j(make(), 0, "小明", "拜拜啦我走了", nearby=TWO).label == "all"  # 句首照旧
    for text in ("bug又来了", "他刚走了", "雨要来了"):  # 句尾的得是单独一个分句，不然是在说事
        assert j(make(), 0, "小明", text, nearby=TWO).label != "all", text
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
    j(a, 0, "小明", "团子你看")
    j(a, 2, "阿花", "明哥你看")
    a.said(5)  # 团子在回小明：这之后阿花说的不再算她还在回小明
    assert j(a, 10, "小明", "好了").label == "me"


# 规则 4 的连锁（10-11 沙盒）：团子插一句 / 回了别人，不该把旁边两个人互相说的都算成在接团子的话
def test_rule4_interjection_opens_no_window():
    a = make()
    assert lt(j(a, 0, "小明", "阿花你看")) == ("other", "阿花")
    assert lt(j(a, 4, "阿花", "看啥")) == ("other", "小明")
    a.said(8)  # 整批都是跟别人说的：插话
    v = j(a, 12, "小明", "你那个斗篷")
    assert lt(v) == ("other", "阿花") and "还在跟阿花说" in v.reason
    assert lt(j(a, 15, "阿花", "季节送的")) == ("other", "小明")


def test_rule4_interjection_then_unknown_is_unsure():
    a = make()
    j(a, 0, "小明", "阿花你看")
    a.said(5)
    v = j(a, 25, "阿花", "哪里像了")  # 对话串过期了，又没点名：可能在接团子插的那句，拿不准
    assert v.label == "unsure" and "插的那句" in v.reason


def test_rule4_only_for_who_dango_answered():
    a = make()
    assert j(a, 0, "阿花", "团子你说像不像").label == "me"
    j(a, 3, "小明", "明明不像")  # 两个人都在，没点名：拿不准
    a.said(6)  # 团子回这一批：两个人都开接话窗口
    assert j(a, 9, "阿花", "就是像").label == "me"
    b = make()
    assert j(b, 0, "阿花", "团子你看").label == "me"
    assert lt(j(b, 2, "小明", "阿花你看")) == ("other", "阿花")
    b.said(5)  # 团子回的是阿花：小明那句是跟阿花说的，不给他开窗口
    assert j(b, 8, "小明", "哈哈").label != "me"


def test_rule4_sandbox_chain_1011():
    """10-11 沙盒：辣椒和老登斗嘴，团子插一句之后两人互相回的被判成「在接你的话」。"""
    fr = {"番茄炒蛋盖饭": ["番茄炒蛋盖饭", "老登"], "辣椒": ["辣椒"]}
    s = dict(nearby=["番茄炒蛋盖饭", "辣椒"], friends=fr)
    a = make()
    assert lt(j(a, 0, "辣椒", "老登你那个新斗篷哪来的", **s)) == ("other", "番茄炒蛋盖饭")
    assert lt(j(a, 4, "番茄炒蛋盖饭", "季节送的 好看吧", **s)) == ("other", "辣椒")
    assert lt(j(a, 8, "辣椒", "还行吧 有点像窗帘", **s)) == ("other", "番茄炒蛋盖饭")
    assert lt(j(a, 12, "番茄炒蛋盖饭", "你才窗帘", **s)) == ("other", "辣椒")
    a.said(16)  # 「窗帘也挺好啊」：插话
    assert lt(j(a, 19, "辣椒", "就是像", **s)) == ("other", "番茄炒蛋盖饭")
    assert lt(j(a, 22, "番茄炒蛋盖饭", "哪里像了", **s)) == ("other", "辣椒")


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
    fr = {"小明": ["小明", "胖胖虎"], "阿花": ["阿花", "明哥"]}
    assert lt(j(make(), 0, "阿花", "胖胖乎你看", friends=fr)) == ("other", "小明")
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


# 说难过 / 不舒服（inner.mind.DISTRESS）是说给在场所有人的：没点名别人就不走对话串的 other（10-06 沙盒）
SANDBOX = {"懒洋洋大王": ["懒洋洋大王", "卡洛"], "番茄炒蛋盖饭": ["番茄炒蛋盖饭", "老登"]}
BOTH = ["懒洋洋大王", "番茄炒蛋盖饭"]


def test_distress_breaks_thread_sandbox_1006():
    a = make()
    s = dict(nearby=BOTH, friends=SANDBOX)
    assert lt(j(a, 0, "懒洋洋大王", "老登你周末打不打瓦", **s)) == ("other", "番茄炒蛋盖饭")
    assert lt(j(a, 8, "番茄炒蛋盖饭", "打啊卡洛，晚上八点上线", **s)) == ("other", "懒洋洋大王")
    v = j(a, 20, "番茄炒蛋盖饭", "唉今天被老板骂了，真的有点难过", **s)
    assert v.label == "all" and "难过" in v.reason and v.target == ""
    # 判成 all 之后，这句不再算她在跟懒洋洋大王的对话串里
    assert "番茄炒蛋盖饭" in a.thread_note(20)


def test_distress_reply_and_alternating_become_all():
    a = make()
    assert lt(j(a, 0, "阿花", "明哥你看")) == ("other", "小明")
    assert j(a, 3, "小明", "我心情不好").label == "all"  # 本来是「在回阿花」
    b = make()
    j(b, 0, "小明", "哈哈")
    j(b, 2, "阿花", "嘿嘿")
    assert j(b, 4, "小明", "好难受").label == "all"  # 本来是「和阿花一来一回」


def test_distress_unsure_becomes_all():
    assert j(make(), 0, "小明", "今天好难过").label == "all"


def test_distress_naming_other_stays_other():
    assert lt(j(make(), 0, "小明", "阿花我好难过")) == ("other", "阿花")


def test_distress_keeps_me():
    v = j(make(), 0, "小明", "好难过", nearby=["小明"])
    assert v.label == "me" and "身边只有他" in v.reason
    assert j(make(), 0, "小明", "团子我好难过").label == "me"
    a = make()
    a.said(0)
    assert j(a, 5, "小明", "其实我有点难过").label == "me"


def test_rule4_second_line_same_turn_keeps_targets():
    """团子一轮说两句：第二句前面没有新聊天，沿用第一句的对象，不当成主动开口给所有人开窗口。"""
    a = make()
    j(a, 0, "小明", "阿花你看")
    a.said(4)
    a.said(6)  # 插话说了两句
    assert lt(j(a, 9, "小明", "好看吧")) == ("other", "阿花")
    b = make()
    assert j(b, 0, "阿花", "团子你看").label == "me"
    j(b, 2, "小明", "阿花你看")
    b.said(4)
    b.said(6)  # 回阿花说了两句：还是只给阿花
    assert j(b, 9, "小明", "哈哈").label != "me"
    assert j(b, 10, "阿花", "嘿嘿").label == "me"
    c = make()
    c.said(0)
    c.said(100)  # 隔了很久又开口：主动开口，谁接都算
    assert j(c, 105, "小明", "嗯").label == "me"


def test_rule4_unnamed_banter_breaks_after_one_line():
    """评审：两人不点名斗嘴（都判拿不准），团子回一句之后两人接着互相说，不该一直算在接团子的话。"""
    a = make()
    assert j(a, 0, "小明", "这个斗篷好看吗").label == "unsure"
    assert j(a, 3, "阿花", "还行").label == "unsure"
    a.said(6)  # 团子回这一批（两个人都拿不准）
    assert j(a, 9, "小明", "哪里还行了").label == "me"  # 第一句可能是接团子
    assert j(a, 12, "阿花", "就是还行").label != "me"  # 别人已经接过话了：交给对话串
    assert lt(j(a, 15, "小明", "你眼光不行")) == ("other", "阿花")
    b = make()  # 直接叫过团子的人不受影响
    assert j(b, 0, "小明", "团子你说呢").label == "me"
    j(b, 2, "阿花", "嗯嗯")
    b.said(5)
    j(b, 7, "阿花", "哈哈")
    assert j(b, 9, "小明", "你也这么觉得").label == "me"


def test_rule4_late_interjection():
    """评审：插话来得晚（大脑攒了一会儿），那批聊天早于 60 秒也照样算插话。"""
    a = make()
    j(a, 0, "小明", "阿花你看")
    j(a, 3, "阿花", "明哥你看")
    a.said(70)
    assert j(a, 75, "阿花", "哪里像了").label != "me"
