import re

from a11ysnap import bubble, placeholder, row, self_row, snap, tag
from skydango.chat.a11yreader import A11yChatReader
from skydango.chat.tracker import SelfFilter
from skydango.config import ChatConfig, OcrConfig


def panel(*texts, pkg="com.netease.sky.vivo"):
    """texts: (内容, 说话人)；自己的行写 ("self", 内容)。"""
    nodes = [placeholder()]
    y = 100
    for t in texts:
        if t[0] == "self":
            nodes.append(self_row(t[1], y))
        else:
            nodes.append(row(t[0], t[1], y))
        y += 60
    return snap(*nodes, pkg=pkg)


def make():
    holder = [None]
    sf = SelfFilter(30, 0.8)
    r = A11yChatReader(lambda: holder[0], lambda: ["小明", "小红"], ChatConfig(), OcrConfig(), sf)
    return r, holder, sf


A, B, C, D = ("你好呀", "小明"), ("今天天气好", "小红"), ("我也觉得", "小明"), ("出去玩吧", "小红")


def test_first_snapshot_is_baseline():
    r, h, _ = make()
    h[0] = panel(A, B, C)
    assert r.read(None, 1.0) == []


def test_new_rows_reported():
    r, h, _ = make()
    h[0] = panel(A, B, C)
    r.read(None, 1.0)
    h[0] = panel(A, B, C, ("在吗", "小明"))
    msgs = r.read(None, 2.0)
    assert len(msgs) == 1
    assert (msgs[0].speaker, msgs[0].text, msgs[0].source) == ("小明", "在吗", "panel")


def test_same_line_twice_reported_twice():
    r, h, _ = make()
    h1 = ("哈哈哈", "小明")
    h[0] = panel(A, h1)
    r.read(None, 1.0)
    h[0] = panel(A, h1, h1)
    assert [m.text for m in r.read(None, 2.0)] == ["哈哈哈"]
    h[0] = panel(A, h1, h1, h1)
    assert [m.text for m in r.read(None, 3.0)] == ["哈哈哈"]


def test_top_rows_scrolled_away():
    r, h, _ = make()
    h[0] = panel(A, B, C)
    r.read(None, 1.0)
    h[0] = panel(B, C, D)
    assert [m.text for m in r.read(None, 2.0)] == ["出去玩吧"]


def test_rebaseline_when_nothing_matches():
    r, h, _ = make()
    h[0] = panel(A, B)
    r.read(None, 1.0)
    X, Y, Z, W = ("甲甲甲", "小明"), ("乙乙乙", "小红"), ("丙丙丙", "小明"), ("丁丁丁", "小红")
    h[0] = panel(X, Y, Z)
    assert r.read(None, 2.0) == []
    h[0] = panel(X, Y, Z, W)
    assert [m.text for m in r.read(None, 3.0)] == ["丁丁丁"]


def test_rows_added_while_panel_closed():
    r, h, _ = make()
    h[0] = panel(A)
    r.read(None, 1.0)
    h[0] = snap()
    assert r.read(None, 2.0) == []
    h[0] = panel(A, B, C)
    assert [m.text for m in r.read(None, 3.0)] == ["今天天气好", "我也觉得"]


def test_self_masked_and_filtered_rows_not_reported():
    r, h, sf = make()
    h[0] = panel(A)
    r.read(None, 1.0)
    sf.remember("我刚说的", 1.5)
    h[0] = panel(A, ("self", "我自己说的话"), ("...", "小明"), ("我刚说的", "小明"))
    assert r.read(None, 2.0) == []


def test_panel_closed_since():
    r, h, _ = make()
    assert r.panel_closed_since is None
    h[0] = panel(A)
    r.read(None, 1.0)
    assert r.panel_closed_since is None and r.panel_visible()
    h[0] = snap()
    r.read(None, 2.0)
    assert r.panel_closed_since == 2.0 and not r.panel_visible()
    r.read(None, 3.0)
    assert r.panel_closed_since == 2.0
    h[0] = panel(A)
    r.read(None, 4.0)
    assert r.panel_closed_since is None


def test_not_in_game_reads_nothing():
    r, h, _ = make()
    h[0] = panel(A, B)
    r.read(None, 1.0)
    h[0] = panel(A, B, C, pkg="com.other")
    assert r.read(None, 2.0) == []
    assert r.panel_closed_since is not None
    h[0] = panel(A, B, C)
    assert [m.text for m in r.read(None, 3.0)] == ["我也觉得"]


def test_log_rows_and_detect():
    r, h, _ = make()
    h[0] = panel(("你好", "小明"))
    r.read(None, 1.0)
    assert [d.text for d in r.detect()] == ["小明：你好"]
    assert r.log_rows()[0].speaker == "小明"
    assert r.reads_bubbles and not r.settling


def test_empty_panel_keeps_baseline():
    r, h, _ = make()
    h[0] = panel(A)
    r.read(None, 1.0)
    h[0] = panel()  # 面板开着、一行都没有（只有占位）：不当新基准
    assert r.read(None, 2.0) == []
    h[0] = panel(A, B)
    assert [m.text for m in r.read(None, 3.0)] == ["今天天气好"]


# ---- 头顶气泡（面板关着） ----

TX, TY = 1053, 279  # 小明的名字标签


def head(*texts, name="小明", cx=TX, with_tag=True):
    """面板关着时一个人头顶：名字标签 + 下面一摞气泡。"""
    nodes = [tag(name, cx, TY)] if with_tag else []
    y = TY + 50
    for t in texts:
        nodes.append(bubble(t, cx, y))
        y += 45
    return nodes


def closed(*nodes):
    return snap(*nodes)


def opened(rows, *nodes):
    """面板开着（rows 同 panel()），画面里另外还有 nodes。"""
    return snap(*panel(*rows).nodes, *nodes)


def test_bubble_new_message():
    r, h, _ = make()
    h[0] = closed(*head())
    assert r.read(None, 1.0) == []
    h[0] = closed(*head("怎么样"))
    msgs = r.read(None, 2.0)
    assert len(msgs) == 1
    m = msgs[0]
    assert (m.text, m.speaker, m.source, m.seen_at) ==("怎么样", "小明", "bubble", 2.0)
    assert m.box.y > TY  # 气泡的框，不是名字标签


def test_typing_animation_not_new():
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head("怎么样"))
    assert [m.text for m in r.read(None, 2.0)] == ["怎么样"]
    assert r.typing() == []
    h[0] = closed(*head("怎么样.. "))
    assert r.read(None, 3.0) == []
    assert r.typing() == ["小明"]
    h[0] = closed(*head("怎么样..."))
    assert r.read(None, 4.0) == []
    assert r.typing() == ["小明"]
    h[0] = closed(*head("怎么样"))
    assert r.read(None, 5.0) == []
    assert r.typing() == []


def test_dots_only_bubble_is_typing():
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head(" .."))
    assert r.read(None, 2.0) == []
    assert r.typing() == ["小明"]


def test_same_bubble_twice():
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head("哈哈"))
    assert [m.text for m in r.read(None, 2.0)] == ["哈哈"]
    h[0] = closed(*head("哈哈", "哈哈"))
    assert [m.text for m in r.read(None, 3.0)] == ["哈哈"]
    h[0] = closed(*head("哈哈", "哈哈"))
    assert r.read(None, 4.0) == []


def test_first_snapshot_bubbles_not_reported():
    r, h, _ = make()
    h[0] = closed(*head("怎么样", "在吗"))
    assert r.read(None, 1.0) == []
    assert r.read(None, 2.0) == []


def test_unattached_text_bubble_wants_peek():
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    assert r.want_peek() is None
    h[0] = closed(*head(), *head("在吗", cx=1500, with_tag=False))
    assert r.read(None, 2.0) == []
    assert r.want_peek() == "bubble_text"
    assert r.want_peek() is None
    assert r.read(None, 3.0) == []  # 同一句下一份不再记
    assert r.want_peek() is None


def test_unattached_dots_bubble_ignored():
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head(), *head("...", cx=1500, with_tag=False))
    assert r.read(None, 2.0) == []
    assert r.want_peek() is None
    assert r.typing() == []


def test_own_bubble_dropped():
    r, h, sf = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    sf.remember("测试", 1.5)
    h[0] = closed(*head(), *head("测试", cx=960, with_tag=False))
    assert r.read(None, 2.0) == []
    assert r.want_peek() is None


def test_bubble_filters():
    r, h, sf = make()
    r.ignore = [re.compile("^广告")]
    h[0] = closed(*head())
    r.read(None, 1.0)
    sf.remember("我说的话", 1.5)
    h[0] = closed(*head("广告位招租", "我说的话"))
    assert r.read(None, 2.0) == []


def test_panel_open_ignores_bubbles():
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    h[0] = opened([A], *head("怎么样"))
    assert r.read(None, 2.0) == []
    h[0] = closed(*head("怎么样"))
    assert r.read(None, 3.0) == []


def test_bubble_then_panel_row_deduped():
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    h[0] = closed(*head())
    r.read(None, 2.0)
    h[0] = closed(*head("怎么样"))
    assert [m.source for m in r.read(None, 3.0)] == ["bubble"]
    q = ("怎么样", "小明")
    h[0] = opened([A, q])
    assert r.read(None, 4.0) == []  # 气泡报过了，一条抵一条
    h[0] = opened([A, q, q])
    msgs = r.read(None, 5.0)
    assert [(m.text, m.source) for m in msgs] == [("怎么样", "panel")]


def test_reported_expires():
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    h[0] = closed(*head())
    r.read(None, 2.0)
    h[0] = closed(*head("怎么样"))
    assert len(r.read(None, 3.0)) == 1
    h[0] = opened([A, ("怎么样", "小明")])
    assert [m.source for m in r.read(None, 3.0 + 181)] == ["panel"]  # 过了 180 秒不再抵


def test_bubble_not_rereported_across_panel_toggle():
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    q = ("怎么样", "小明")
    total = []
    h[0] = opened([A, q], *head("怎么样.."))
    total += r.read(None, 2.0)
    h[0] = closed(*head("怎么样..."))
    total += r.read(None, 3.0)
    h[0] = opened([A, q], *head("怎么样 .."))
    total += r.read(None, 4.0)
    h[0] = closed(*head("怎么样"))
    total += r.read(None, 5.0)
    assert [(m.text, m.source) for m in total] == [("怎么样", "panel")]


def test_tag_flicker_does_not_rereport():
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head("怎么样"))
    assert len(r.read(None, 2.0)) == 1
    h[0] = closed(*head("怎么样", with_tag=False))  # 名字标签闪了一下
    assert r.read(None, 2.5) == []
    h[0] = closed(*head("怎么样"))
    assert r.read(None, 3.0) == []


def test_tags_in_view():
    r, h, _ = make()
    assert r.tags_in_view() == []
    h[0] = closed(*head("怎么样"))
    r.read(None, 1.0)
    assert r.tags_in_view() == ["小明"]


def test_bubble_traced(tmp_path):
    r, h, _ = make()
    r.trace_path = tmp_path / "rows.log"
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head("怎么样"))
    r.read(None, 2.0)
    assert "气泡 小明：怎么样" in r.trace_path.read_text(encoding="utf-8")
