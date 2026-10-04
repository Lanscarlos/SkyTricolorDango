from a11ysnap import placeholder, row, self_row, snap
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
