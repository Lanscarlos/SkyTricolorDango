from a11ysnap import bubble, input_bar, node, placeholder, row, self_row, snap, tag
from skydango.vision.a11yui import Tag, bubble_key, classify, is_dots

FRIENDS = {"小明", "小红"}


def cl(s):
    return classify(s, 1920, 1080, FRIENDS)


def test_panel_rows_self_rows_and_placeholder():
    v = cl(
        snap(
            row("你好", "小明", 100, visible=False),
            row(".....", "陌生人", 150),
            self_row("嗯", 200),
            placeholder(),
        )
    )
    assert v.in_game and v.panel_open
    assert [r.key() for r in v.rows] == [(False, "小明", "你好"), (False, "陌生人", ""), (True, "", "嗯")]
    assert v.rows[1].masked and not v.rows[0].masked
    assert not v.rows[0].visible and v.rows[1].visible
    assert v.bubbles == () and v.tags == ()


def test_panel_closed_tags_and_bubbles():
    v = cl(
        snap(
            tag("小明", 1053, 279),  # 实测：标签底 321，下面一摞气泡相隔 20 多 px
            bubble("怎么样", 1053, 341, lines=2),
            bubble("团子", 1053, 437),
            bubble(".....", 1387, 279),
        )
    )
    assert v.panel_open is False and v.rows == ()
    assert v.tags == (Tag("小明", (975, 279, 1131, 321)),)
    assert [b.speaker for b in v.bubbles] == ["小明", "小明", None]
    assert v.bubbles[2].typing_only and not v.bubbles[0].typing_only


def test_bubble_saying_friend_name_is_not_a_tag():
    v = cl(snap(tag("小明", 1053, 279), bubble("小红", 1053, 341)))
    assert [t.name for t in v.tags] == ["小明"]
    assert len(v.bubbles) == 1 and v.bubbles[0].text == "小红" and v.bubbles[0].speaker == "小明"


def test_two_matching_same_column_topmost_is_tag():
    v = cl(snap(tag("小明", 1053, 279), tag("小红", 1053, 340)))  # 竖着挨着：一摞里只有最上面的是标签
    assert [t.name for t in v.tags] == ["小明"]
    assert [b.text for b in v.bubbles] == ["小红"]


def test_far_below_in_same_column_is_another_column():
    """同一条竖线上隔得远（不挨着）的是另一个人：他自己的标签，近处那人的气泡不挂到远处的标签上。"""
    v = cl(snap(tag("小红", 1053, 100), tag("小明", 1053, 500), bubble("嗨", 1053, 560)))
    assert [t.name for t in v.tags] == ["小红", "小明"]
    assert [(b.text, b.speaker) for b in v.bubbles] == [("嗨", "小明")]
    v = cl(snap(tag("小红", 1053, 100), bubble("在吗", 1053, 600)))  # 近处那人的标签淡了
    assert v.bubbles[0].speaker is None


def test_stack_chains_from_tag():
    """一摞气泡一句挨一句往下：每句都挂在最上面的标签上。"""
    v = cl(snap(tag("小明", 1053, 279), bubble("一", 1053, 341, lines=2), bubble("二", 1053, 437), bubble("三", 1053, 498)))
    assert [b.speaker for b in v.bubbles] == ["小明", "小明", "小明"]


def test_edge_tag_is_flagged():
    """好友在画面外：光遇把名字标签贴在屏幕边上，标 edge，下面的气泡不挂他。"""
    v = cl(snap(tag("小明", 78, 279), bubble("嗨", 78, 329), tag("小红", 1842, 279), tag("小明", 1053, 600)))
    assert [(t.name, t.edge) for t in v.tags] == [("小明", True), ("小红", True), ("小明", False)]
    assert v.bubbles[0].speaker is None


def test_unknown_name_tag():
    """名单外的名字标签照样是标签（known=False），不当气泡；他列里的气泡挂不上已知好友。"""
    v = cl(snap(tag("路人甲", 1053, 279), bubble("你好", 1053, 329)))
    assert [(t.name, t.known) for t in v.tags] == [("路人甲", False)]
    assert [(b.text, b.speaker) for b in v.bubbles] == [("你好", None)]


def test_friend_tag_behind_panel_is_not_self_row():
    """面板开着时好友名字标签落在自己行的 x 范围里：不是自己的行。"""
    v = cl(snap(row("你好", "小明", 100), placeholder(), tag("小明", 560, 600)))
    assert [r.key() for r in v.rows] == [(False, "小明", "你好")]


def test_placeholder_ignored_when_input_open():
    """输入栏开着时左下角的占位不算面板开着，只认看得见的别人的面板行。"""
    assert cl(snap(input_bar(), placeholder())).panel_open is False
    assert cl(snap(input_bar(), placeholder(), row("你好", "小明", 100))).panel_open is True
    assert cl(snap(placeholder())).panel_open is True


def test_bubble_attaches_to_tag_in_its_column():
    v = cl(snap(tag("小明", 1000, 100), tag("小红", 1200, 100), bubble("嗨", 1200, 200)))
    assert v.bubbles[0].speaker == "小红"


def test_invisible_bubble_ignored():
    v = cl(snap(tag("小明", 1053, 279), node("远处", (900, 400, 1000, 438), visible=False)))
    assert v.bubbles == ()


def test_input_bar():
    v = cl(snap(input_bar("测试")))
    assert v.input_text == "测试"
    assert not v.panel_open and v.bubbles == () and v.tags == ()
    assert cl(snap(tag("小明", 1053, 279))).input_text is None


def test_not_in_game():
    for s in (snap(row("你好", "小明", 100), pkg="android"), None):
        v = cl(s)
        assert v.in_game is False and v.panel_open is False
        assert v.rows == () and v.bubbles == () and v.tags == () and v.input_text is None


def test_bubble_key_and_dots():
    assert bubble_key("怎么样.. ") == "怎么样"
    assert bubble_key("怎么样") == "怎么样"
    assert bubble_key("一起去" + chr(10) + "雨林 吗 ..") == "一起去雨林吗"  # 折行 / 空白都不算
    assert is_dots(" ..") and is_dots("…。")
    assert not is_dots("好吧...") and not is_dots("")


def test_friend_tag_clipped_at_top_edge():
    """好友站得靠画面顶上，名字标签被屏幕上沿切掉一截（框顶 0、高 38 / 28，10-04 真机录像）：照样是他的标签。"""
    from a11ysnap import node

    v = classify(snap(node("小明", (401, 0, 557, 38)), bubble("团子", 479, 57)), 1920, 1080, FRIENDS)
    assert [t.name for t in v.tags] == ["小明"]
    assert [b.speaker for b in v.bubbles] == ["小明"]
    v = classify(snap(node("小明", (529, 0, 685, 28)), bubble("呜呜呜", 607, 48)), 1920, 1080, FRIENDS)
    assert [b.speaker for b in v.bubbles] == ["小明"]
    # 被切掉的不是好友名字（气泡被切了一截）：不当标签
    v = classify(snap(node("在吗", (440, 0, 518, 30))), 1920, 1080, FRIENDS)
    assert v.tags == ()


def test_panel_rows_sorted_by_position_not_tree_order():
    """面板刚打开时游戏在复用 TextView，树里的顺序暂时不是从上到下（10-04 真机：y=553 那行排在了最后）。
    行按框的底边排：滚出去的行框被裁成顶 0、底是负数，越往上越负，照样排得对。"""
    from a11ysnap import node

    nodes = [
        placeholder(),
        node("很早以前 - 小明", (21, 0, 188, -954), visible=False),
        row("在干啥呢", "小明", 886),
        row("怎么不说话", "小明", 940),
        row("团子", "小明", 553),
        self_row("我在呀", 623),
    ]
    v = classify(snap(*nodes), 1920, 1080, FRIENDS)
    assert [r.text for r in v.rows] == ["很早以前", "团子", "我在呀", "在干啥呢", "怎么不说话"]
