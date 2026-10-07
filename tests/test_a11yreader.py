import re

from a11ysnap import bubble, node, placeholder, row, self_row, snap, tag
from skydango.chat.a11yreader import A11yChatReader, FallbackReader
from skydango.chat.tracker import SelfFilter
from skydango.config import ChatConfig, OcrConfig
from skydango.device.a11y import A11yError


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
    assert r.want_peek() is None  # 这句已经认出是小明的，不用看一眼
    h[0] = closed(*head("怎么样"))
    assert r.read(None, 3.0) == []


def test_panel_then_bubble_offscreen_not_rereported():
    """面板开着时小明在画面外：面板行报了，面板关上后他头上那条气泡不再报。"""
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    q = ("看我", "小明")
    h[0] = opened([A, q])  # 画面里看不到小明
    assert [(m.text, m.source) for m in r.read(None, 2.0)] == [("看我", "panel")]
    h[0] = closed(*head("看我"))
    assert r.read(None, 3.0) == []


def test_peek_then_tag_returns_not_rereported():
    """标签淡了 → 看一眼 → 面板行报出；面板关上后他走近、标签回来，气泡不再报。"""
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    h[0] = closed(*head())
    r.read(None, 2.0)
    h[0] = closed(*head("在吗", with_tag=False))
    assert r.read(None, 3.0) == []
    assert r.want_peek() == "bubble_text"
    h[0] = opened([A, ("在吗", "小明")], *head("在吗", with_tag=False))
    assert [(m.text, m.source) for m in r.read(None, 4.0)] == [("在吗", "panel")]
    h[0] = closed(*head("在吗"))
    assert r.read(None, 5.0) == []


def test_panel_entry_consumed_then_real_repeat_reported():
    """面板开着时气泡也看到了（抵掉面板那条）：之后他真的又说一遍，气泡照报。"""
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    q = ("怎么样", "小明")
    h[0] = opened([A, q], *head("怎么样"))
    assert [m.source for m in r.read(None, 2.0)] == ["panel"]
    h[0] = closed(*head("怎么样"))
    assert r.read(None, 3.0) == []
    h[0] = closed(*head("怎么样", "怎么样"))
    assert [(m.text, m.source) for m in r.read(None, 4.0)] == [("怎么样", "bubble")]


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


def test_edge_tag_then_back_not_rereported():
    """小明走出画面、名字标签贴在屏幕边上：不算在画面里，回来时还挂着的那句不重报（C1）。"""
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head("怎么样"))
    assert len(r.read(None, 2.0)) == 1
    off = (tag("小明", 78, TY), node("怎么样", (1000, 330, 1100, 368), visible=False))  # 气泡在画面外看不见
    for t in (3.0, 6.0, 10.0):
        h[0] = closed(*off)
        assert r.read(None, t) == []
        assert r.tags_in_view() == []
    h[0] = closed(*head("怎么样"))
    assert r.read(None, 15.0) == []


def test_first_seen_with_dots_reported_clean_and_deduped():
    """第一次看到时这句后面已挂着打字的点：报的是去掉点的原文，面板行对得上、不再报（I1）。"""
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    h[0] = closed(*head())
    r.read(None, 2.0)
    h[0] = closed(*head("怎么样.."))
    assert [(m.text, m.source) for m in r.read(None, 3.0)] == [("怎么样", "bubble")]
    h[0] = opened([A, ("怎么样", "小明")])
    assert r.read(None, 4.0) == []


def test_wrapped_bubble_matches_panel_row():
    """折行的气泡（换行 / 多空格）和面板行文字对得上：两边只报一次（I1）。"""
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    h[0] = closed(*head())
    r.read(None, 2.0)
    h[0] = closed(*head("一起去" + chr(10) + "雨林  吗"))
    assert len(r.read(None, 3.0)) == 1
    h[0] = opened([A, ("一起去雨林 吗", "小明")])
    assert r.read(None, 4.0) == []
    # 反过来：面板行先报，面板关上后他头上折行的同一句不再报
    h[0] = opened([A, ("一起去雨林 吗", "小明"), ("出发吧 走", "小明")])
    assert [m.text for m in r.read(None, 5.0)] == ["出发吧 走"]
    h[0] = closed(*head("一起去" + chr(10) + "雨林  吗", "出发吧" + chr(10) + "走"))
    assert r.read(None, 6.0) == []


def transition(*texts):
    """面板刚打开的过渡快照：行都“看不见”，自己的行是很小的框（不满足自己行的规则）。"""
    nodes = [placeholder()]
    y = 100
    for t in texts:
        if t[0] == "self":
            nodes.append(node(t[1], (13, 301, 52, 311), visible=False))
        else:
            nodes.append(row(t[0], t[1], y, visible=False))
        y += 60
    return snap(*nodes)


def test_transition_snapshot_does_not_lose_rows():
    """上次最后一行是团子自己说的 → 过渡快照 → 正常快照多一行好友的话：报出来（I2）。"""
    r, h, _ = make()
    S, N = ("self", "我在呢"), ("走吧", "小红")
    h[0] = panel(A, S)
    r.read(None, 1.0)
    assert r.settling is False
    h[0] = snap()
    r.read(None, 2.0)
    h[0] = transition(A, S, N)
    assert r.read(None, 3.0) == []
    assert r.settling is True  # 过渡态：面板管理器先别关
    h[0] = panel(A, S, N)
    assert [(m.speaker, m.text) for m in r.read(None, 3.3)] == [("小红", "走吧")]
    assert r.settling is False


def test_self_rows_not_used_for_alignment(tmp_path):
    """对齐只用别人的行：自己的行一会儿有一会儿没有也不打乱；rows.log 照样记自己的行（I2）。"""
    r, h, _ = make()
    r.trace_path = tmp_path / "rows.log"
    S = ("self", "我在呢")
    h[0] = panel(A, S)
    r.read(None, 1.0)
    h[0] = panel(A, ("走吧", "小红"))  # 自己那行这一份没认出来
    assert [m.text for m in r.read(None, 2.0)] == ["走吧"]
    h[0] = panel(A, S, ("走吧", "小红"))
    assert r.read(None, 3.0) == []
    assert "我在呢" in r.trace_path.read_text(encoding="utf-8")


def test_unknown_tag_reappearing_does_not_peek():
    """名单外的名字标签消失又出现不叫看一眼；他头上的原文气泡照样看一眼（I3）。"""
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    for t in (2.0, 4.0):
        h[0] = closed(*head(), tag("路人甲", 1500, TY))
        assert r.read(None, t) == []
        assert r.want_peek() is None
        assert r.tags_in_view() == ["小明"]
        h[0] = closed(*head())
        r.read(None, t + 1)
    h[0] = closed(*head(), *head("你好", name="路人甲", cx=1500))
    assert r.read(None, 6.0) == []
    assert r.want_peek() == "bubble_text"
    r.read(None, 7.0)
    assert r.want_peek() is None


def test_rebaseline_consumes_bubble_reported():
    """面板对不上、当新基准时，基准里气泡报过的那句划掉：之后他在画面外又说一遍照报（M1）。"""
    r, h, _ = make()
    h[0] = opened([A])
    r.read(None, 1.0)
    h[0] = closed(*head())
    r.read(None, 2.0)
    h[0] = closed(*head("怎么样"))
    assert len(r.read(None, 3.0)) == 1
    X, Y, q = ("甲甲甲", "小红"), ("乙乙乙", "小红"), ("怎么样", "小明")
    h[0] = opened([X, Y, q])  # 对不上：当新基准
    assert r.read(None, 4.0) == []
    h[0] = opened([X, Y, q, q])
    assert [(m.text, m.source) for m in r.read(None, 5.0)] == [("怎么样", "panel")]


def test_friends_cached():
    """好友名单最多每 2 秒重取一次（M3）。"""
    calls, clock = [], [0.0]

    def friends():
        calls.append(clock[0])
        return ["小明", "小红"]

    holder = [closed(*head())]
    r = A11yChatReader(
        lambda: holder[0], friends, ChatConfig(), OcrConfig(), SelfFilter(30, 0.8), clock=lambda: clock[0]
    )
    r.read(None, 1.0)
    r.panel_visible()
    r.log_rows()
    clock[0] = 1.9
    r.read(None, 2.0)
    assert len(calls) == 1
    clock[0] = 2.1
    r.read(None, 3.0)
    assert len(calls) == 2


# ---- FallbackReader（spec §5）----
GOOD = object()  # 假快照
FRAME = object()  # 假截图


class FakeClient:
    def __init__(self):
        self.alive = True
        self.error = ""
        self.snap = GOOD
        self.starts = 0
        self.stops = 0
        self.fail: Exception | None = None
        self.others = False  # 设备上有别的读取器的客户端（别的进程起的 / 上次留下的）

    def others_running(self):
        return self.others

    def start(self):
        self.starts += 1
        if self.fail is not None:
            raise self.fail
        self.alive = True

    def stop(self):
        self.stops += 1

    def latest(self, max_age=None):
        return self.snap


class FakeOcrReader:
    settling = False

    def __init__(self):
        self.msgs = ["ocr 的消息"]
        self.visible = True
        self.panel_closed_since = None
        self.trace_path = None
        self.reads = 0

    def read(self, frame, now):
        self.reads += 1
        return list(self.msgs)

    def panel_visible(self, frame):
        return self.visible

    def log_rows(self, frame):
        return ["ocr 行"]

    def detect(self, frame):
        return ["ocr 框"]


class FakeA11y:
    settling = False
    reads_bubbles = True

    def __init__(self):
        self.msgs = ["a11y 的消息"]
        self.panel_closed_since = None
        self.trace_path = None

    def read(self, frame, now):
        return list(self.msgs)

    def panel_visible(self, frame=None):
        return self.panel_closed_since is None

    def log_rows(self, frame=None):
        return ["a11y 行"]

    def detect(self, frame=None):
        return ["a11y 框"]

    def tags_in_view(self):
        return ["小明"]

    def typing(self):
        return ["小红"]

    def want_peek(self):
        return "bubble_text"


def fallback(fail: Exception | None = None, others: bool = False, sleep=None):
    client, a11y, ocr = FakeClient(), FakeA11y(), FakeOcrReader()
    client.fail, client.others = fail, others
    fb = FallbackReader(client, a11y, ocr, sleep=sleep or (lambda s: None))
    fb.start()
    return fb, client, a11y, ocr


def test_another_reader_connected_at_start_uses_ocr():  # 后起的那个退回 OCR，不踢别人
    waits = []
    fb, client, a11y, ocr = fallback(others=True, sleep=waits.append)
    assert waits == [2.0]  # 等一下再看（上次被强杀留下的会自己退）
    assert fb.using_ocr and "另一个读取器正连着" in fb.reason
    assert client.starts == 0 and client.stops == 0  # 没起、也没清别人的
    assert fb.read(FRAME, 1.0) == ["ocr 的消息"]
    fb.stop()
    assert client.stops == 0  # 收尾也不清（会把别人的客户端杀掉）


def test_stale_leftover_that_exits_is_fine():
    client_box = []

    def sleep(s):
        client_box[0].others = False  # 上次留下的那个在等的时候退了

    client, a11y, ocr = FakeClient(), FakeA11y(), FakeOcrReader()
    client.others = True
    client_box.append(client)
    fb = FallbackReader(client, a11y, ocr, sleep=sleep)
    fb.start()
    assert not fb.using_ocr and client.starts == 1


def test_another_reader_took_over_while_ours_was_dead():
    fb, client, a11y, ocr = fallback()
    fb.read(FRAME, 1.0)
    client.alive, client.snap = False, None  # 被踢掉了
    client.others = True  # 别人连着
    fb.read(FRAME, 2.0)
    assert not fb.using_ocr  # 还没到重启的时候
    fb.read(FRAME, 4.0)  # 到点：先看有没有别人
    assert fb.using_ocr and "另一个读取器正连着" in fb.reason
    assert client.starts == 1 and client.stops == 0  # 不重启、不清别人的
    fb.read(FRAME, 30.0)
    assert client.starts == 1


def test_healthy_reads_a11y():
    fb, client, a11y, ocr = fallback()
    assert fb.read(FRAME, 1.0) == ["a11y 的消息"]
    assert not fb.using_ocr and fb.describe() == "无障碍"
    assert fb.reads_bubbles is True and fb.want_peek() == "bubble_text"
    assert fb.tags_in_view() == ["小明"] and fb.typing() == ["小红"]
    assert fb.log_rows(FRAME) == ["a11y 行"] and fb.detect(FRAME) == ["a11y 框"]
    assert fb.panel_visible(FRAME) is True
    assert ocr.reads == 0


def test_start_failure_falls_back_to_ocr(caplog):
    fb, client, a11y, ocr = fallback(A11yError("IllegalStateException: already registered!"))
    assert fb.using_ocr
    assert "already registered" in fb.describe() and fb.describe().startswith("OCR（无障碍读不到：")
    assert "读聊天退回 OCR" in caplog.text
    assert fb.read(FRAME, 1.0) == ["ocr 的消息"]
    assert fb.log_rows(FRAME) == ["ocr 行"] and fb.detect(FRAME) == ["ocr 框"]
    ocr.panel_closed_since = 3.0
    assert fb.panel_closed_since == 3.0


def test_restart_then_recover():
    fb, client, a11y, ocr = fallback()
    assert fb.read(FRAME, 9.0) == ["a11y 的消息"]
    client.alive, client.snap = False, None
    assert fb.read(FRAME, 10.0) == []
    assert fb.read(FRAME, 11.0) == [] and client.starts == 1  # 还没到 2 秒
    fb.read(FRAME, 12.0)
    assert client.starts == 2  # 按 2 秒重启
    client.snap = GOOD
    assert fb.read(FRAME, 13.0) == ["a11y 的消息"]  # 收到快照：次数清零
    client.alive, client.snap = False, None
    fb.read(FRAME, 20.0)
    fb.read(FRAME, 21.9)
    assert client.starts == 2
    fb.read(FRAME, 22.0)
    assert client.starts == 3  # 还是按 2 秒（不是 5 秒）
    assert not fb.using_ocr


def test_three_failed_restarts_switch_to_ocr(caplog):
    fb, client, a11y, ocr = fallback()
    fb.read(FRAME, 0.0)
    client.alive, client.snap = False, None
    client.error = "error: already registered!"
    t = 10.0
    fb.read(FRAME, t)
    for delay in (2.0, 5.0, 10.0):
        t += delay
        fb.read(FRAME, t)  # 到点重启
        assert not fb.using_ocr
        client.alive = False  # 没快照又死了
        t += 0.1
        fb.read(FRAME, t)
    assert client.starts == 4  # 起一次 + 重启三次
    assert fb.using_ocr and "already registered" in fb.reason
    assert client.stops == 1
    for later in (t + 20, t + 60):
        assert fb.read(FRAME, later) == ["ocr 的消息"]
    assert client.starts == 4  # 不再起
    assert "读聊天退回 OCR" in caplog.text


def test_restarts_used_up_with_another_reader_connected_does_not_stop():
    """重启用完要退回 OCR 时有别人连着：不清（按名字清会把别人的客户端杀掉）。"""
    fb, client, a11y, ocr = fallback()
    fb.read(FRAME, 0.0)
    client.alive, client.snap = False, None
    t = 10.0
    fb.read(FRAME, t)
    for delay in (2.0, 5.0, 10.0):
        t += delay
        fb.read(FRAME, t)  # 到点重启
        client.alive = False  # 没快照又死了
        client.others = delay == 10.0  # 最后一次是别人连上来把我们踢了
        t += 0.1
        fb.read(FRAME, t)
    assert fb.using_ocr and client.starts == 4
    assert client.stops == 0


def test_restart_raising_counts_as_another_death():
    fb, client, a11y, ocr = fallback()
    fb.read(FRAME, 0.0)
    client.alive, client.snap = False, None
    client.fail = A11yError("adb 没了")
    t = 10.0
    fb.read(FRAME, t)
    for delay in (2.0, 5.0, 10.0):
        t += delay
        fb.read(FRAME, t)
        t += 0.1
        fb.read(FRAME, t)
    assert fb.using_ocr and client.starts == 4


def test_stale_switches_to_ocr():
    fb, client, a11y, ocr = fallback()
    client.snap = None
    for now in range(0, 31):
        assert fb.read(FRAME, float(now)) == []
        assert not fb.using_ocr
    fb.read(FRAME, 31.0)
    assert fb.using_ocr and "30 秒" in fb.reason
    assert client.stops == 1


def test_unhealthy_uses_ocr_panel_visibility():
    fb, client, a11y, ocr = fallback()
    fb.read(FRAME, 1.0)
    assert fb.panel_closed_since is None
    client.alive, client.snap = False, None
    ocr.visible = False
    assert fb.read(FRAME, 2.0) == []
    assert fb.panel_closed_since == 2.0
    assert fb.panel_visible(FRAME) is False
    fb.read(None, 3.0)  # 没有画面：保持原值
    assert fb.panel_closed_since == 2.0
    ocr.visible = True
    fb.read(FRAME, 3.5)
    assert fb.panel_closed_since is None
    assert fb.panel_visible(FRAME) is True
    assert ocr.reads == 0  # 不健康时只借 OCR 看面板，不读消息


def test_closed_since_does_not_jump_when_turning_unhealthy():
    fb, client, a11y, ocr = fallback()
    a11y.panel_closed_since = 5.0  # 无障碍说面板 5 秒起关着
    fb.read(FRAME, 6.0)
    assert fb.panel_closed_since == 5.0
    client.alive, client.snap = False, None
    ocr.visible = False  # OCR 也看着关着
    fb.read(FRAME, 7.0)
    assert fb.panel_closed_since == 5.0  # 不跳成 7
    fb.read(None, 8.0)
    assert fb.panel_closed_since == 5.0


def test_unhealthy_has_no_stale_bubbles():
    fb, client, a11y, ocr = fallback()
    fb.read(FRAME, 1.0)
    client.alive, client.snap = False, None
    fb.read(FRAME, 2.0)
    assert fb.tags_in_view() == [] and fb.typing() == [] and fb.want_peek() is None


def test_trace_path_propagates(tmp_path):
    fb, client, a11y, ocr = fallback()
    fb.trace_path = tmp_path / "rows.log"
    assert a11y.trace_path == ocr.trace_path == tmp_path / "rows.log"
    assert fb.trace_path == tmp_path / "rows.log"


def test_stop_calls_client_stop():
    fb, client, a11y, ocr = fallback()
    fb.stop()
    assert client.stops == 1


def test_no_restart_after_stop():  # 收尾时身体还会借 reader 看面板：别把设备上的客户端又起起来
    fb, client, a11y, ocr = fallback()
    fb.read(FRAME, 1.0)
    fb.stop()
    client.alive, client.snap = False, None
    for now in (2.0, 5.0, 20.0, 60.0):
        assert fb.read(FRAME, now) == []
    assert client.starts == 1 and not fb.using_ocr
    ocr.visible = False
    assert fb.panel_visible(FRAME) is False


def test_stop_swallows_errors():
    fb, client, a11y, ocr = fallback()

    def boom():
        raise RuntimeError("adb 断了")

    client.stop = boom
    fb.stop()  # 不抛


def test_stop_after_falling_back_does_not_stop_again():  # 退回 OCR 时停过了；再 stop 会把后来连上的别人的客户端杀掉
    fb, client, a11y, ocr = fallback()
    client.snap = None
    fb.read(FRAME, 0.0)
    fb.read(FRAME, 31.0)
    assert fb.using_ocr and client.stops == 1
    fb.stop()
    assert client.stops == 1


def test_ocr_mode_has_no_bubbles():
    fb, client, a11y, ocr = fallback(A11yError("x"))
    assert fb.reads_bubbles is False
    assert fb.want_peek() is None and fb.tags_in_view() == [] and fb.typing() == []
    ocr.settling = True
    assert fb.settling is True


def test_bubbles_lagging_behind_tag_while_panning_not_rereported():
    """转镜头时节点位置不是一起更新的：标签和第一句先动、下面几句晚一帧，那一帧挂不上名字。
    不能把计数掉下去再回来当成新冒出来的话（10-04 真机录像 132.09 秒）。"""
    r, h, _ = make()
    h[0] = closed(tag("小明", 1672, 194))
    r.read(None, 1.0)
    h[0] = closed(tag("小明", 1672, 194), bubble("团子", 1672, 256), bubble("你怎么不理我", 1671, 318))
    assert [m.text for m in r.read(None, 2.0)] == ["团子", "你怎么不理我"]
    h[0] = closed(tag("小明", 1330, 116), bubble("团子", 1330, 178), bubble("你怎么不理我", 1256, 239))
    assert r.read(None, 2.2) == []
    assert r.want_peek() is None  # 这句刚才还挂在小明名下：不用为它看一眼面板
    h[0] = closed(tag("小明", 996, 138), bubble("团子", 996, 200), bubble("你怎么不理我", 996, 262))
    assert r.read(None, 2.4) == []


def test_same_sentence_again_after_old_bubble_faded_still_reported():
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head("哈哈"))
    assert [m.text for m in r.read(None, 2.0)] == ["哈哈"]
    h[0] = closed(*head())  # 气泡淡掉了
    assert r.read(None, 25.0) == []
    h[0] = closed(*head("哈哈"))  # 过了一阵又说了一遍
    assert [m.text for m in r.read(None, 40.0)] == ["哈哈"]


def test_hold_does_not_expire_while_tag_is_out_of_view():
    """他走到屏幕最上面、名字标签整个看不见了好几秒；回来第一份里有一句被切掉、偏了没挂上，
    下一份才挂上：不能当成新冒出来的话（10-04 真机录像 141 秒）。"""
    r, h, _ = make()
    h[0] = closed(tag("小明", 607, 300))
    r.read(None, 1.0)
    h[0] = closed(tag("小明", 607, 300), bubble("你怎么不理我", 607, 362), bubble("呜呜呜", 607, 424))
    assert [m.text for m in r.read(None, 2.0)] == ["你怎么不理我", "呜呜呜"]
    h[0] = closed(bubble("你怎么不理我", 607, 0), bubble("呜呜呜", 607, 50))  # 标签出了屏幕上沿
    assert r.read(None, 5.0) == []
    assert r.read(None, 9.0) == []
    h[0] = closed(tag("小明", 612, 0), bubble("你怎么不理我", 673, 0), bubble("呜呜呜", 612, 61))
    assert r.read(None, 9.2) == []
    h[0] = closed(tag("小明", 653, 0), bubble("你怎么不理我", 653, 0), bubble("呜呜呜", 653, 49))
    assert r.read(None, 9.4) == []


def test_out_of_order_snapshot_does_not_rereport_old_rows():
    """10-04 真机：一份乱序快照被当成基准后，下一份正常快照把后面几行全当成新行又报了一遍。"""
    from a11ysnap import row as prow

    r, h, _ = make()
    rows = [("团子", 553), ("昨天说什么了", 692), ("哎呀", 832), ("在干啥呢", 886)]
    h[0] = snap(placeholder(), *[prow(t, "小明", y) for t, y in rows])
    r.read(None, 1.0)
    shuffled = [("昨天说什么了", 692), ("哎呀", 832), ("在干啥呢", 886), ("怎么不说话", 940), ("团子", 553)]
    h[0] = snap(placeholder(), *[prow(t, "小明", y) for t, y in shuffled])
    assert [m.text for m in r.read(None, 2.0)] == ["怎么不说话"]
    h[0] = snap(placeholder(), *[prow(t, "小明", y) for t, y in sorted(shuffled, key=lambda x: x[1])])
    assert r.read(None, 3.0) == []


def test_peek_for_unattached_bubble_reports_it_even_on_first_baseline():
    """10-04 真机：团子刚上线，好友的标签淡了、说了句「团子」挂不上名字 → 看一眼面板；
    这是上线后第一次开面板，只建基准的话这句就被当成历史吞了。"""
    r, h, _ = make()
    h[0] = closed()
    r.read(None, 1.0)
    h[0] = closed(bubble("团子", 900, 400))  # 没有名字标签
    assert r.read(None, 2.0) == []
    assert r.want_peek() == "bubble_text"
    h[0] = panel(A, B, ("团子", "小明"))  # 第一次开面板
    assert [(m.speaker, m.text, m.source) for m in r.read(None, 3.0)] == [("小明", "团子", "panel")]
    h[0] = panel(A, B, ("团子", "小明"))
    assert r.read(None, 4.0) == []


def test_baseline_without_pending_peek_still_reports_nothing():
    r, h, _ = make()
    h[0] = panel(A, B, ("团子", "小明"))
    assert r.read(None, 1.0) == []


def test_rows_filling_in_right_after_panel_appears_are_not_new():
    """重新登录后第一次开面板，列表从上往下陆续填进来（10-04 晚真机：只填了上面几行的那份当了基准，下面 8 句旧话被当成新的）。
    面板刚出现 0.8 秒内只在底部往下长的，算还在加载。"""
    r, h, _ = make()
    h[0] = closed()
    r.read(None, 1.0)
    h[0] = panel(A, B)
    assert r.read(None, 10.0) == []
    h[0] = panel(A, B, C, D)
    assert r.read(None, 10.3) == []
    h[0] = panel(A, B, C, D, ("真的新消息", "小明"))
    assert [m.text for m in r.read(None, 12.0)] == ["真的新消息"]


def test_top_part_only_snapshot_keeps_baseline():
    r, h, _ = make()
    h[0] = panel(A, B, C, D)
    r.read(None, 1.0)
    h[0] = panel(A, B)  # 只看到上半截（没加载全）
    assert r.read(None, 2.0) == []
    h[0] = panel(A, B, C, D, ("在吗", "小明"))
    assert [m.text for m in r.read(None, 3.0)] == ["在吗"]


def test_bubble_that_is_just_a_friend_name_is_not_a_message():
    """游戏偶尔多出一个名字节点，挂在他的标签下面像一句话（10-04 晚真机：「懒洋洋大王：懒洋洋大王」，他没打过自己名字）。"""
    r, h, _ = make()
    h[0] = closed(*head())
    r.read(None, 1.0)
    h[0] = closed(*head("小明", "在吗"))
    assert [m.text for m in r.read(None, 2.0)] == ["在吗"]


def _stack(gap: int):
    """10-07 晚真机：小明和团子挨着站，团子的气泡夹在小明两句中间；gap = 团子那条到小明下一句的距离。"""
    return closed(
        node("小明", (628, 188, 784, 230)),
        node("不行啊你", (649, 250, 762, 288)),
        node("我小短腿走得慢，等不了怪我咯", (370, 312, 763, 350)),
        node("你还是拜倒在你造物主脚下吧", (523, 350 + gap, 888, 388 + gap)),
    )


def test_interleaved_stack_moving_does_not_rereport():
    """夹着别人气泡的一摞随动画上下错动：小明那句只报一次。"""
    r, h, sf = make()
    sf.remember("我小短腿走得慢，等不了怪我咯", 0.0)
    h[0] = closed(node("小明", (628, 188, 784, 230)), node("不行啊你", (649, 250, 762, 288)),
                  node("我小短腿走得慢，等不了怪我咯", (370, 312, 763, 350)))
    r.read(None, 1.0)
    h[0] = _stack(-2)  # 挨着：旧规则也挂得上
    assert [m.text for m in r.read(None, 2.0)] == ["你还是拜倒在你造物主脚下吧"]
    t = 2.0
    for gap in (19, -2, 19, -2, 19):  # 来回错动（19 = 隔 81 px，旧规则挂不上）、跨过 HOLD
        t += 4.0
        h[0] = _stack(gap)
        assert r.read(None, t) == []


M = ("......", "陌生人")  # 被屏蔽的陌生人行


def test_masked_rows_do_not_misalign_reopened_panel():
    """关面板期间进来一串被屏蔽的陌生人行，重开时游戏只留了几行：别靠一串一样的被屏蔽行对齐，
    把早就报过的几句当成新的（10-07 晚 22:19:06 真机）。"""
    r, h, _ = make()
    w, go, ok = ("白眼狼", "小明"), ("你走", "小明"), ("那你倒是走啊", "小明")
    no, bai, pull = ("不行啊你", "小明"), ("拜倒吧", "小明"), ("电源拔了", "小明")
    h[0] = panel(w, go, ok, M, M, M, no, bai, pull)
    r.read(None, 1.0)
    h[0] = panel(w, go, ok, M, M, M, no, bai, pull, M, M, M, M, M, M)
    assert r.read(None, 2.0) == []
    h[0] = closed()
    r.read(None, 3.0)
    h[0] = panel(M, M, w, go, ok, M, M, M, no, bai, pull, M, M)
    assert r.read(None, 90.0) == []


def test_line_back_after_vanishing_long_is_reported():
    """面板开着时某句消失够久（被挤出历史）后又出现：是又说了一遍，照报。"""
    r, h, _ = make()
    haha = ("哈哈", "小明")
    h[0] = panel(A, haha)
    r.read(None, 1.0)
    h[0] = panel(B, C)
    r.read(None, 2.0)
    h[0] = panel(B, C)
    r.read(None, 9.0)
    h[0] = panel(B, C, haha)
    assert [m.text for m in r.read(None, 10.0)] == ["哈哈"]
