import threading
import time

import numpy as np
from conftest import scene
from panels_helpers import CARDS, ListOcr, line

from skydango.config import PanelsConfig
from skydango.vision.panels import UNKNOWN, PanelWatcher, load_cards, looks_like_panel

DIALOG = [line("出错了", 300, 20), line("网络连接断开，请重试", 200, 90), line("确定", 330, 300)]
INVITE = [line("这位旅人正在共享空间内，你想一起加入吗？", 100, 60), line("取消", 100, 300), line("加入", 500, 300)]
DISCONNECT = [  # 2026-09-30 真机掉线弹框：「取消」是灰的、「重试」是蓝的
    line("连接错误", 300, 20),
    line("网络连接失败。 (错误码：140)", 200, 90),
    line("请确认您的网络连接状态", 200, 130),
    line("取消", 400, 300),
    line("重试", 500, 300),
]


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


def test_looks_like_panel_with_only_retry():  # 掉线弹框：OCR 漏读灰色的「取消」
    lines = [line("连接错误", 300, 20), line("网络连接失败。 (错误码：140)", 200, 90), line("重试", 500, 300)]
    assert looks_like_panel(lines, PanelsConfig())


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


def test_disconnect_card_identified(clock):
    w = watcher(ListOcr(DISCONNECT))
    w.observe(scene(), clock())
    [c] = w.pop_changes()
    assert c.panel.name == "disconnect" and c.panel.layer == 100
    assert [(b.text, b.kind) for b in c.reading.buttons] == [("取消", "never"), ("重试", "never")]


def test_disconnect_card_without_grey_cancel(clock):  # OCR 漏读灰色的「取消」
    w = watcher(ListOcr([ln for ln in DISCONNECT if ln.text != "取消"]))
    w.observe(scene(), clock())
    [c] = w.pop_changes()
    assert c.panel.name == "disconnect" and [(b.text, b.kind) for b in c.reading.buttons] == [("重试", "never")]


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


# ---- 评审修复 ----
def test_chat_panel_lines_ignored(clock):
    # 聊天记录面板开着：团子自己的「好的」和别人消息都在面板里（x < 429），不能凑成"不认识的面板"
    lines = [line("我们一起去雨林跑图吧", 0, 350, w=220), line("好的", 150, 400)]
    w = PanelWatcher(PanelsConfig(), load_cards(CARDS), ListOcr(lines), {"chat_input": lambda f: True}, background=False)
    w.observe(scene(), clock())
    assert [p.name for p in w.state.panels] == ["chat_log"]


def test_button_word_inside_sentence_is_not_a_button():
    cfg = PanelsConfig()
    assert not looks_like_panel([line("好的，走吧", 10, 10), line("懒洋洋大王今天好开心", 10, 60)], cfg)
    assert not looks_like_panel([line("知道了吗", 10, 10), line("一段足够长的正文文字", 10, 60)], cfg)
    assert looks_like_panel([line("确定！", 10, 10), line("一段足够长的正文文字", 10, 60)], cfg)


def test_unknown_panel_does_not_block_by_default(clock):
    w = watcher(ListOcr(DIALOG))
    w.observe(scene(), clock())
    assert w.state.top().name == UNKNOWN and w.blocking("say") == [] and w.blocking("camera") == []
    cfg = PanelsConfig(unknown_blocks=True)
    strict = PanelWatcher(cfg, load_cards(CARDS), ListOcr(DIALOG), {"chat_input": lambda f: False}, background=False)
    strict.observe(scene(), clock())
    assert [p.name for p in strict.blocking("say")] == [UNKNOWN]


class GateOcr:
    """第一次马上返回 DIALOG；之后等 gate 放行才返回 DIALOG（模拟关面板时还在跑的后台扫描）。"""

    def __init__(self):
        self.gate = threading.Event()
        self.calls = 0
        self.finished = threading.Event()

    def recognize(self, img):
        self.calls += 1
        if self.calls > 1:
            self.gate.wait(2)
            self.finished.set()
        return list(DIALOG)


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)


def test_stale_scan_does_not_revive_closed_panel(clock):
    ocr = GateOcr()
    w = watcher(ocr, background=True)
    try:
        w.observe(scene(), clock())
        wait_for(lambda: w._future is not None and w._future.done())
        w.observe(scene(), clock())
        assert names(w.pop_changes()) == [("open", UNKNOWN)]
        clock.advance(5)
        w.observe(scene(), clock())  # 提交第二次扫描（卡在 gate 上）
        w.mark_closed(UNKNOWN)
        assert names(w.pop_changes()) == [("close", UNKNOWN)]
        ocr.gate.set()
        ocr.finished.wait(2)
        wait_for(lambda: w._future is None or w._future.done())
        w.observe(scene(), clock())
        assert w.pop_changes() == [] and w.state.others() == ()
    finally:
        w.close()


class SlowOcr:
    def __init__(self):
        self.active = 0
        self.most = 0
        self.lock = threading.Lock()

    def recognize(self, img):
        with self.lock:
            self.active += 1
            self.most = max(self.most, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return list(DIALOG)


def test_ocr_not_used_from_two_threads_at_once(clock):
    from skydango.vision.bubbles import Rect
    from skydango.vision.panels import Panel

    ocr = SlowOcr()
    w = watcher(ocr, background=True)
    try:
        w.observe(scene(), clock())  # 后台开始扫描
        time.sleep(0.01)
        w.read(scene(), Panel(UNKNOWN, "不认识的面板", Rect(200, 100, 800, 500), False, 100), clock())
        wait_for(lambda: w._future is None or w._future.done())
        assert ocr.most == 1
    finally:
        w.close()


# 11-02 真机：头顶名字「懒洋洋大王」+ OCR 从图形里读出的「24」「x」凑成了不认识的面板，大脑 panel_close 真去点了那个「x」
TAG_JUNK = [line("懒洋洋大王", 400, 200), line("24", 420, 240), line("x", 600, 300)]


def tag_box():
    from skydango.vision.panels import Rect

    return Rect(round(0.15 * 1280) + 395, round(0.08 * 720) + 195, 160, 40)  # 名字在整张图上的位置（OCR 区域偏移 + 行坐标）


def test_world_text_lines_are_ignored(clock):
    w = watcher(ListOcr(TAG_JUNK))
    w.world_text = lambda: [tag_box()]
    w.observe(scene(), clock())
    clock.advance(3)
    w.observe(changed(), clock())
    assert w.pop_changes() == [] and w.state.others() == ()


def test_without_world_text_same_lines_open(clock):  # 对照：不给头顶文字的框就照旧认成面板
    w = watcher(ListOcr(TAG_JUNK))
    w.observe(scene(), clock())
    clock.advance(3)
    w.observe(changed(), clock())
    assert [c.panel.name for c in w.pop_changes()] == [UNKNOWN]


def test_present_ignores_world_text(clock):
    w = watcher(ListOcr(TAG_JUNK))
    w.observe(scene(), clock())
    clock.advance(3)
    w.observe(changed(), clock())
    [c] = w.pop_changes()
    from skydango.vision.panels import Rect

    assert w.present(changed(), c.panel)
    w.world_text = lambda: [Rect(c.panel.box.x + 395, c.panel.box.y + 195, 160, 40)]  # present 只 OCR 面板框：行坐标按它偏移
    assert not w.present(changed(), c.panel)  # 去掉名字后不像面板：关面板时当它已经不在了


def test_world_text_error_is_ignored(clock):
    w = watcher(ListOcr(DIALOG))

    def broken():
        raise RuntimeError("坏了")

    w.world_text = broken
    w.observe(scene(), clock())
    clock.advance(3)
    w.observe(changed(), clock())
    assert [c.panel.name for c in w.pop_changes()] == [UNKNOWN]
