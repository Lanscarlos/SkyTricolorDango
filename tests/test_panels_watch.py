import time

import numpy as np
import pytest
from conftest import scene
from panels_helpers import CARDS, ListOcr, draw_cross, write_card

from skydango.config import PanelsConfig
from skydango.vision.panels import CHAT, Feature, PanelWatcher, check_feature, load_cards, load_template

DEMO = '''
label = "演示面板"
verified = true
layer = {layer}
region = [0.7, 0.0, 0.95, 1.0]
confirm_frames = {confirm}
[[features]]
kind = "template"
image = "icon.png"
roi = [0.7, 0.3, 0.9, 0.5]
[close]
ways = ["key:18"]
auto = true
'''
CHAT_CARD = '''
label = "聊天记录面板"
layer = 10
region = [0.0, 0.0, 0.335, 0.855]
confirm_frames = 1
allows = ["say", "emote", "camera", "check_friend", "social"]
[[features]]
kind = "builtin"
name = "chat_input"
'''


def demo_card(root, name="demo", layer=20, confirm=2):
    write_card(root, name, DEMO.format(layer=layer, confirm=confirm), images=("icon.png",))


def watcher(root, builtins=None, **kw):
    return PanelWatcher(PanelsConfig(), load_cards(root), ListOcr(), builtins or {}, background=False, **kw)


def icon_frame(size=(1280, 720)):
    return draw_cross(scene(size=size), at=(0.8, 0.4), size=40 * size[1] / 1080)


def names(changes):
    return [(c.kind, c.panel.name) for c in changes]


def test_template_matches_on_1280x720_and_1920x1080(tmp_path):
    demo_card(tmp_path)
    feature = Feature("template", (0.7, 0.3, 0.9, 0.5), "icon.png", load_template(tmp_path / "demo" / "icon.png"))
    for size in ((1280, 720), (1920, 1080)):
        hit, score = check_feature(icon_frame(size), feature, {})
        assert hit and score >= 0.8, size
        assert not check_feature(scene(size=size), feature, {})[0]


def test_missing_template_never_hits():
    assert check_feature(icon_frame(), Feature("template", (0.7, 0.3, 0.9, 0.5), "icon.png", None), {}) == (False, 0.0)


def test_dark_feature():
    feature = Feature("dark", (0.1, 0.1, 0.5, 0.5), max_value=70, max_std=40)
    frame = scene()
    assert not check_feature(frame, feature, {})[0]
    frame[50:400, 100:700] = 30
    hit, value = check_feature(frame, feature, {})
    assert hit and value == 30


def test_builtin_feature():
    assert check_feature(scene(), Feature("builtin", name="chat_input"), {"chat_input": lambda f: True}) == (True, 1.0)
    assert check_feature(scene(), Feature("builtin", name="chat_input"), {"chat_input": lambda f: False}) == (False, 0.0)


def test_unknown_builtin_rejected(tmp_path):
    write_card(tmp_path, "chat_log", CHAT_CARD.replace("chat_input", "nope"))
    with pytest.raises(ValueError, match="nope"):
        watcher(tmp_path)


def test_opens_after_confirm_frames(tmp_path, clock):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    w.observe(icon_frame(), clock())
    assert w.pop_changes() == [] and w.state.panels == ()
    w.observe(icon_frame(), clock())
    [c] = w.pop_changes()
    assert c.kind == "open" and c.panel.name == "demo" and c.reading is None
    assert c.panel.label == "演示面板" and c.panel.verified and c.panel.box.x == 896
    assert w.pop_changes() == []


def test_flicker_does_not_toggle(tmp_path, clock):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    for frame in (icon_frame(), scene(), icon_frame(), scene(), icon_frame()):
        w.observe(frame, clock())
    assert w.pop_changes() == []


def test_closes_after_misses(tmp_path, clock):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    for frame in (icon_frame(), icon_frame(), scene()):
        w.observe(frame, clock())
    assert names(w.pop_changes()) == [("open", "demo")]
    w.observe(scene(), clock())
    assert names(w.pop_changes()) == [("close", "demo")] and w.state.panels == ()


def test_expect_suppresses_changes_and_blocking(tmp_path, clock):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    with w.expect("demo"):
        w.observe(icon_frame(), clock())
        w.observe(icon_frame(), clock())
        assert w.pop_changes() == [] and w.blocking("camera") == []
        assert [p.name for p in w.state.panels] == ["demo"]
        w.observe(scene(), clock())
        w.observe(scene(), clock())
    assert w.pop_changes() == []


def test_expect_exit_still_open_emits_open(tmp_path, clock):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    with w.expect("demo"):
        w.observe(icon_frame(), clock())
        w.observe(icon_frame(), clock())
    assert names(w.pop_changes()) == [("open", "demo")]
    assert [p.name for p in w.blocking("camera")] == ["demo"]


def test_expect_nested_and_none(tmp_path):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    with w.expect("demo"), w.expect("demo"), w.expect(None):
        pass
    assert w.expected == set()


def test_expect_restored_after_exception(tmp_path):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    with pytest.raises(RuntimeError):
        with w.expect("demo"):
            raise RuntimeError("boom")
    assert w.expected == set()


def test_blocking_respects_allows_and_layer(tmp_path, clock):
    demo_card(tmp_path)
    demo_card(tmp_path, name="demo2", layer=40)
    write_card(tmp_path, "chat_log", CHAT_CARD)
    w = watcher(tmp_path, builtins={"chat_input": lambda f: True})
    w.observe(icon_frame(), clock())
    w.observe(icon_frame(), clock())
    assert [p.name for p in w.state.panels] == ["demo2", "demo", CHAT]
    assert [p.name for p in w.blocking("camera")] == ["demo2", "demo"]
    assert w.state.top().name == "demo2" and [p.name for p in w.state.others()] == ["demo2", "demo"]
    assert (CHAT not in [p.name for p in w.blocking("say")])


def test_mark_closed(tmp_path, clock):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    w.observe(icon_frame(), clock())
    w.observe(icon_frame(), clock())
    w.pop_changes()
    w.mark_closed("demo")
    assert w.state.panels == () and names(w.pop_changes()) == [("close", "demo")]
    w.observe(scene(), clock())  # 已经关了：不命中不再报一次
    w.observe(scene(), clock())
    assert w.pop_changes() == []


def test_explain_reports_each_feature(tmp_path):
    demo_card(tmp_path)
    w = watcher(tmp_path)
    [check] = w.explain(icon_frame())
    assert check.card.name == "demo" and check.open
    [(feature, hit, score)] = check.hits
    assert feature.image == "icon.png" and hit and score >= 0.8
    assert not w.explain(scene())[0].open


def test_observe_is_fast():
    w = PanelWatcher(PanelsConfig(), load_cards(CARDS), ListOcr(), {"chat_input": lambda f: False}, background=False)
    frame = scene(size=(1920, 1080))
    w.observe(frame, 0.0)
    started = time.perf_counter()
    for i in range(20):
        w.observe(frame, 0.1 * i)
    assert (time.perf_counter() - started) / 20 < 0.02


def test_black_frame_skips_cards(tmp_path, clock):
    write_card(tmp_path, "darkish", '''
label = "暗面板"
verified = true
region = [0.5, 0.0, 1.0, 1.0]
confirm_frames = 1
[[features]]
kind = "dark"
roi = [0.6, 0.1, 0.9, 0.9]
''')
    w = watcher(tmp_path)
    w.observe(np.zeros((720, 1280, 3), np.uint8), clock())
    assert w.pop_changes() == [] and w.state.panels == ()


def test_real_cards_do_not_open_on_dark_scene(clock):
    w = PanelWatcher(PanelsConfig(), load_cards(CARDS), ListOcr(), {"chat_input": lambda f: False}, background=False)
    dark = np.full((720, 1280, 3), 35, np.uint8)  # 暗色地图：整体偏暗但不是黑屏
    for _ in range(3):
        w.observe(dark, clock())
    assert w.state.panels == ()
