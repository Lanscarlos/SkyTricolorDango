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
            tag("小明", 1053, 279),
            bubble("怎么样", 1053, 437),
            bubble("团子", 1053, 498),
            bubble(".....", 1387, 279),
        )
    )
    assert v.panel_open is False and v.rows == ()
    assert v.tags == (Tag("小明", (975, 279, 1131, 321)),)
    assert [b.speaker for b in v.bubbles] == ["小明", "小明", None]
    assert v.bubbles[2].typing_only and not v.bubbles[0].typing_only


def test_bubble_saying_friend_name_is_not_a_tag():
    v = cl(snap(tag("小明", 1053, 279), bubble("小红", 1053, 437)))
    assert [t.name for t in v.tags] == ["小明"]
    assert len(v.bubbles) == 1 and v.bubbles[0].text == "小红" and v.bubbles[0].speaker == "小明"


def test_two_matching_same_column_topmost_is_tag():
    v = cl(snap(tag("小明", 1053, 279), tag("小红", 1053, 400)))
    assert [t.name for t in v.tags] == ["小明"]
    assert [b.text for b in v.bubbles] == ["小红"]


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
    assert is_dots(" ..") and is_dots("…。")
    assert not is_dots("好吧...") and not is_dots("")
