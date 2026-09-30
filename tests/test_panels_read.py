import pytest
from conftest import scene
from panels_helpers import CARDS, ListOcr, line, write_card

from skydango.config import PanelsConfig
from skydango.vision.bubbles import Rect
from skydango.vision.panels import (
    UNKNOWN,
    Panel,
    PanelReading,
    PanelWatcher,
    classify,
    describe_reading,
    load_cards,
    split_reading,
)

INVITE = Panel("shared_invite", "共享空间邀请", Rect(192, 58, 896, 604), False, 90)
INVITE_LINES = [
    line("共享空间", 300, 10),
    line("这位旅人正在共享空间内，你想一起加入吗？", 100, 60),
    line("取消", 50, 400),
    line("加入", 500, 400),
]


def watcher(ocr, cards=CARDS):
    return PanelWatcher(PanelsConfig(), load_cards(cards), ocr, {"chat_input": lambda f: False}, background=False)


@pytest.mark.parametrize(
    "text, kind",
    [("退出并删除", "never"), ("取消邀请", "other"), ("返回遇境", "other"), ("关闭好友申请", "other"), ("取消。", "retreat"),
     ("关闭", "retreat"), ("×", "retreat"), ("确定", "other"), ("购买", "never")],
)
def test_classify(text, kind):
    assert classify(text, PanelsConfig()) == kind


def test_classify_card_lists():
    cfg = PanelsConfig()
    assert classify("确定", cfg, allow=["确定"]) == "allow"
    assert classify("加入", cfg, never=["加入"]) == "never"
    assert classify("取消", cfg, allow=["取消"]) == "retreat"


def test_read_splits_title_text_buttons(clock):
    w = watcher(ListOcr(INVITE_LINES))
    r = w.read(scene(), INVITE, clock())
    assert r.panel is INVITE and r.t == clock()
    assert r.title == "共享空间" and r.text == "这位旅人正在共享空间内，你想一起加入吗？"
    assert [(b.text, b.kind) for b in r.buttons] == [("取消", "retreat"), ("加入", "never")]
    assert r.buttons[0].box == Rect(192 + 50, 58 + 400, 60, 30)
    assert w.readings["shared_invite"] is r


def test_short_title_without_button_word_is_not_a_button(clock):
    w = watcher(ListOcr([line("提示", 300, 10), line("网络连接断开，请重试", 100, 60), line("好的", 300, 300)]))
    r = w.read(scene(), Panel("unknown", "不认识的面板", Rect(192, 58, 896, 604), False, 100), clock())
    assert r.title == "提示" and [b.text for b in r.buttons] == ["好的"]


def test_buttons_roi_takes_any_short_text(tmp_path, clock):
    write_card(tmp_path, "demo", '''
label = "演示"
region = [0.0, 0.0, 1.0, 1.0]
[[features]]
kind = "dark"
roi = [0.0, 0.0, 1.0, 1.0]
[buttons]
roi = [0.0, 0.5, 1.0, 1.0]
''')
    w = watcher(ListOcr([line("标题", 10, 10), line("正文写在上半部分", 10, 100), line("好耶", 10, 500)]), cards=tmp_path)
    r = w.read(scene(), Panel("demo", "演示", Rect(0, 0, 1280, 720), False, 50), clock())
    assert [(b.text, b.kind) for b in r.buttons] == [("好耶", "other")]
    assert r.title == "标题" and r.text == "正文写在上半部分"


def test_read_cache(clock):
    ocr = ListOcr(INVITE_LINES)
    w = watcher(ocr)
    frame = scene()
    first = w.read(frame, INVITE, clock())
    assert w.read(frame, INVITE, clock()) is first and ocr.calls == 1
    frame[100:500, 300:900] = 255
    w.read(frame, INVITE, clock())
    assert ocr.calls == 2


def test_describe_reading():
    r = watcher(ListOcr(INVITE_LINES)).read(scene(), INVITE, 0.0)
    assert describe_reading(r) == "「共享空间」这位旅人正在共享空间内，你想一起加入吗？，按钮：取消、加入"
    bare = PanelReading(INVITE, "", "字" * 50, (), 0.0)
    out = describe_reading(bare)
    assert out.startswith("字" * 40) and "字" * 41 not in out and out.endswith("按钮：没认出按钮")


DISCONNECT_LINES = [  # 2026-09-30 真机掉线弹框当时的 OCR（整张图坐标，1920×1080）
    line("连接错误", 495, 447, 120, 36),
    line("网络连接失败。 (错误码：140)", 495, 518, 320, 30),
    line("请确认您的网络连接状态", 495, 546, 245, 30),
    line("取消", 1245, 609, 50, 30),
    line("重试", 1357, 609, 50, 30),
]


def test_retry_is_a_button():
    panel = Panel(UNKNOWN, "不认识的面板", Rect(460, 407, 1000, 265), False, 100)
    r = split_reading(DISCONNECT_LINES, panel, None, PanelsConfig(), 1920, 1080, 0.0)
    assert r.title == "连接错误"
    assert "重试" not in r.text
    assert [(b.text, b.kind) for b in r.buttons] == [("取消", "retreat"), ("重试", "other")]


def test_retry_inside_long_sentence_is_not_a_button():
    panel = Panel(UNKNOWN, "不认识的面板", Rect(0, 0, 1920, 1080), False, 100)
    lines = [line("出错了", 300, 20), line("网络连接断开，请重试", 200, 90)]
    r = split_reading(lines, panel, None, PanelsConfig(), 1920, 1080, 0.0)
    assert r.buttons == ()
