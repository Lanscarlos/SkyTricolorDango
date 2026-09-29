from skydango.brain.occasion import Spoken, assess, is_friend_fn, reply_state
from skydango.config import ProactiveConfig

NOW = 10_000.0
CFG = ProactiveConfig()
FRIEND = is_friend_fn(["懒洋洋大王", "阿花"])


def said(ago, text="好无聊", proactive=True):
    return Spoken(NOW - ago, text, proactive)


def line(ago, who, text="嗯"):
    return (NOW - ago, who, text)


def occ(friends=("阿花",), chat=(), spoken=(), strangers=0, cfg=CFG):
    return assess(cfg, NOW, friends, strangers, list(chat), list(spoken), FRIEND)


def test_levels():
    assert occ(friends=()).level == "alone"
    assert occ(chat=[line(100, "阿花")] * 3).level == "quiet"
    assert occ(chat=[line(100, "阿花")] * 4).level == "busy"
    assert occ(chat=[line(181, "阿花")] + [line(100, "阿花")] * 3).level == "quiet"  # 181 秒前的不算
    assert occ(chat=[line(100, "我")] * 5).level == "quiet"  # 自己说的不算


def test_fuzzy_friend_and_unknown_speaker():
    assert FRIEND("懒洋洋大玉") and not FRIEND("") and not FRIEND("路人甲乙丙")
    o = occ(chat=[line(90, ""), line(80, "")], spoken=[said(100)])
    assert o.others == 2  # 看不出是谁的也算别人说话
    assert o.last_reply == "none"  # 但不算有好友接
    assert occ(chat=[line(90, "懒洋洋大玉")], spoken=[said(100)]).last_reply == "replied"


def test_reply_state():
    s = said(100)
    assert reply_state(CFG, s, [line(70, "阿花", "哈哈")], NOW, FRIEND) == ("replied", "阿花")
    assert reply_state(CFG, s, [line(70, "阿花", "团子你好")], NOW, FRIEND) == ("called", "阿花")
    assert reply_state(CFG, s, [line(80, "懒洋洋大王"), line(70, "阿花", "三彩在吗")], NOW, FRIEND) == ("called", "阿花")
    assert reply_state(CFG, said(60), [], NOW, FRIEND) == ("waiting", "")
    assert reply_state(CFG, s, [], NOW, FRIEND) == ("none", "")
    assert reply_state(CFG, said(200), [line(100, "阿花")], NOW, FRIEND) == ("none", "")  # 窗口外才说
    assert reply_state(CFG, s, [line(120, "阿花")], NOW, FRIEND) == ("none", "")  # 那句之前说的不算


def test_alone_blocks():
    o = occ(friends=())
    assert o.left == 0 and o.blocked == "身边没有好友，不主动开口"


def test_quota_by_level():
    o = occ(spoken=[said(500), said(70)])
    assert o.level == "quiet" and o.left == 0
    assert o.blocked == "最近 10 分钟已经主动说了 2 句，100 秒后才能再主动开口"
    busy = occ(chat=[line(170, "阿花"), line(150, "阿花"), line(130, "阿花"), line(110, "阿花")], spoken=[said(500), said(70)])
    assert busy.level == "busy" and busy.left == 2 and busy.blocked == ""


def test_min_gap():
    o = occ(spoken=[said(20)])
    assert o.left == 1 and o.blocked == "刚主动说过，40 秒后才能再主动开口"


def test_cold_and_recovery():
    three = [said(500), said(350), said(200)]
    cold = occ(spoken=three)
    assert cold.blocked.startswith("连着 3 句主动的话都没人接") and cold.left == 0
    thawed = occ(spoken=three, chat=[line(50, "阿花")])  # 最后一句之后好友说话了（窗口外）
    assert not thawed.blocked.startswith("连着")
    assert not occ(spoken=[said(350), said(200)]).blocked.startswith("连着")
    mixed = occ(spoken=[said(500), said(350, proactive=False), said(200)], cfg=ProactiveConfig(quota_quiet=10))
    assert mixed.blocked == ""  # 接话的那句不参与冷场


def test_line():
    chat = [line(210, "阿花", "哈哈")] + [line(t, "懒洋洋大王") for t in (170, 150, 130, 110, 90, 70, 50)]
    o = occ(friends=("懒洋洋大王", "阿花"), chat=chat, spoken=[said(500, "嗯"), said(240, "这图好黑")])
    assert o.line(NOW) == (
        "热闹（身边 懒洋洋大王、阿花；最近 3 分钟别人说了 7 句） / 上次主动开口 4 分钟前「这图好黑」，阿花接了话"
        " / 最近 10 分钟主动说了 2 句，1 句有人接 / 这会儿还能主动说 2 句"
    )
    alone = occ(friends=(), strangers=2).line(NOW)
    assert alone.startswith("没熟人（身边只有 2 个陌生人） / 还没主动开过口")
    assert alone.endswith("这会儿不能主动开口：身边没有好友，不主动开口")
    assert occ(friends=()).line(NOW).startswith("没熟人（身边没人）")
    assert "上次主动开口 20 秒前「好无聊」，还在等人接" in occ(spoken=[said(20)]).line(NOW)
    assert "，没人接" in occ(spoken=[said(200)]).line(NOW)
    assert "，阿花叫了你" in occ(spoken=[said(200)], chat=[line(150, "阿花", "团子？")]).line(NOW)
