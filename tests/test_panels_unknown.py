import time

import numpy as np
from conftest import scene
from panels_helpers import CARDS, ListOcr, line

from skydango.config import PanelsConfig
from skydango.vision.panels import UNKNOWN, PanelWatcher, load_cards, looks_like_panel

DIALOG = [line("出错了", 300, 20), line("网络连接断开，请重试", 200, 90), line("确定", 330, 300)]
INVITE = [line("这位旅人正在共享空间内，你想一起加入吗？", 100, 60), line("取消", 100, 300), line("加入", 500, 300)]


def watcher(ocr, background=False):
    return PanelWatcher(PanelsConfig(), load_cards(CARDS), ocr, {"chat_input": lambda f: False}, background=background)


def changed():
    """和 scene() 差很多的一帧（像弹出了大框）。"""
    frame = scene()
    frame[50:670, 100:1180] = (235, 235, 235)
    return frame


def names(changes):
    return [(c.kind, c.panel.name) for c in changes]


def test_changed_frame_is_a_big_change():
    from skydango.brain.images import difference, thumb

    assert difference(thumb(scene()), thumb(changed())) >= PanelsConfig().change


def test_looks_like_panel():
    cfg = PanelsConfig()
    assert looks_like_panel(DIALOG, cfg)
    assert not looks_like_panel([line("懒洋洋大王", 10, 10), line("2级", 10, 60)], cfg)
    assert not looks_like_panel([line("确定", 10, 10), line("好", 10, 60)], cfg)  # 正文太短


def test_scene_change_triggers_scan_and_opens_unknown(clock):
    w = watcher(ListOcr([], DIALOG))
    w.observe(scene(), clock())
    assert w.pop_changes() == []
    clock.advance(3)
    w.observe(changed(), clock())
    [c] = w.pop_changes()
    assert c.kind == "open" and c.panel.name == UNKNOWN and c.panel.label == "不认识的面板"
    assert c.reading.title == "出错了" and [b.text for b in c.reading.buttons] == ["确定"]
    assert [p.name for p in w.state.others()] == [UNKNOWN] and w.readings[UNKNOWN] is c.reading
    area_x = round(0.15 * 1280)
    assert c.panel.box.x == area_x + 200 - round(0.02 * 1280)  # 文字外接框往外扩 unknown_pad


def test_name_tags_only_is_not_panel(clock):
    w = watcher(ListOcr([line("懒洋洋大王", 10, 10), line("2级", 10, 60)]))
    w.observe(scene(), clock())
    clock.advance(3)
    w.observe(changed(), clock())
    assert w.pop_changes() == [] and w.state.others() == ()


def test_periodic_scan(clock):
    ocr = ListOcr([])
    w = watcher(ocr)
    w.observe(scene(), clock())
    clock.advance(4)
    w.observe(scene(), clock())
    assert ocr.calls == 1
    clock.advance(1)
    w.observe(scene(), clock())
    assert ocr.calls == 2


def test_cooldown(clock):
    ocr = ListOcr([])
    w = watcher(ocr)
    w.observe(scene(), clock())
    clock.advance(1)
    w.observe(changed(), clock())
    assert ocr.calls == 1


def test_black_frame_does_not_scan(clock):
    ocr = ListOcr(DIALOG)
    w = watcher(ocr)
    w.observe(np.zeros((720, 1280, 3), np.uint8), clock())
    clock.advance(10)
    w.observe(np.zeros((720, 1280, 3), np.uint8), clock())
    assert ocr.calls == 0 and w.pop_changes() == []


def test_text_card_identified(clock):
    w = watcher(ListOcr(INVITE))
    w.observe(scene(), clock())
    [c] = w.pop_changes()
    assert c.panel.name == "shared_invite" and c.panel.label == "共享空间邀请" and c.panel.layer == 90
    assert [(b.text, b.kind) for b in c.reading.buttons] == [("取消", "retreat"), ("加入", "never")]


def test_unknown_closes_when_rescan_finds_nothing(clock):
    w = watcher(ListOcr(DIALOG, []))
    w.observe(scene(), clock())
    assert names(w.pop_changes()) == [("open", UNKNOWN)]
    clock.advance(5)
    w.observe(scene(), clock())
    assert names(w.pop_changes()) == [("close", UNKNOWN)] and w.state.others() == ()


def test_unknown_ttl_closes(clock):
    w = watcher(ListOcr(DIALOG, RuntimeError("OCR 挂了")))
    w.observe(scene(), clock())
    w.pop_changes()
    for _ in range(6):
        clock.advance(5)
        w.observe(scene(), clock())
    assert names(w.pop_changes()) == []
    clock.advance(2)
    w.observe(scene(), clock())
    assert names(w.pop_changes()) == [("close", UNKNOWN)]


def test_ocr_error_keeps_state(clock):
    w = watcher(ListOcr(DIALOG, RuntimeError("OCR 挂了")))
    w.observe(scene(), clock())
    w.pop_changes()
    clock.advance(5)
    w.observe(scene(), clock())
    assert w.pop_changes() == [] and [p.name for p in w.state.others()] == [UNKNOWN]


def test_mark_closed_unknown(clock):
    w = watcher(ListOcr(DIALOG))
    w.observe(scene(), clock())
    w.pop_changes()
    w.mark_closed(UNKNOWN)
    assert names(w.pop_changes()) == [("close", UNKNOWN)] and w.state.others() == ()


def test_present_unknown(clock):
    ocr = ListOcr(DIALOG)
    w = watcher(ocr)
    w.observe(scene(), clock())
    panel = w.state.top()
    assert w.present(scene(), panel)
    ocr.results = [[]]
    assert not w.present(scene(), panel)


def test_find_close_without_template(clock):
    assert watcher(ListOcr([])).find_close(scene()) is None


def test_background_scan_merges_next_observe(clock):
    w = watcher(ListOcr(DIALOG), background=True)
    try:
        w.observe(scene(), clock())
        for _ in range(200):
            time.sleep(0.01)
            w.observe(scene(), clock())
            got = w.pop_changes()
            if got:
                break
        assert names(got) == [("open", UNKNOWN)]
    finally:
        w.close()
