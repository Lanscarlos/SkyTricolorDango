"""聊天记录（沙盒和真机共用，spec 2026-10-01-console-live-page §3.2）：长轮询、事件变成一行。"""
import threading
import time
from types import SimpleNamespace

from skydango.brain.transcript import Transcript, event_line


def tr():
    return Transcript(SimpleNamespace(wall=lambda: 1_790_000_000.0))


def test_wait_since_returns_new_lines_at_once():
    t = tr()
    t.add("heard", "在吗", "小明")
    v, lines = t.wait_since(0, 5.0)
    assert v == 1 and [r["text"] for r in lines] == ["在吗"]


def test_wait_since_times_out_empty():
    t = tr()
    t.add("heard", "在吗", "小明")
    started = time.monotonic()
    v, lines = t.wait_since(1, 0.2)
    assert v == 1 and lines == [] and time.monotonic() - started >= 0.15


def test_wait_since_wakes_on_add():
    t = tr()
    threading.Timer(0.1, lambda: t.add("said", "在呢", "团子")).start()
    v, lines = t.wait_since(0, 3.0)
    assert v == 1 and lines[0]["kind"] == "said"


def test_wait_since_restarts_when_client_is_ahead():  # 浏览器记的比这边大：这边重启过，从头给
    t = tr()
    t.add("heard", "一", "小明")
    v, lines = t.wait_since(99, 0.0)
    assert v == 1 and len(lines) == 1


def test_event_line():
    assert event_line("arrive", "小明 来到身边（第 3 次见）", "小明") == "── 小明 来到身边 ──"
    assert event_line("leave", "小明 走开了（20 秒没看到名字）", "小明") == "── 小明 走开了 ──"
    assert event_line("return", "小明 回来了", "小明") == "── 小明 回来了 ──"
    assert event_line("lull", "小明 回来了（走开了 3 分钟，你刚才在想：…）", "小明") == "── 小明 回来了 ──"
    assert event_line("lull", "冷场  小明 1 分钟没说话了。…") is None  # 冷场节点不记（心里想的另有一行）
    assert event_line("holding", "（推测）牵上了 小明 的手") == "── （推测）牵上了 小明 的手 ──"
    assert event_line("released", "（推测）和 小明 松手了") == "── （推测）和 小明 松手了 ──"
    assert event_line("scene_change", "画面整屏黑了（可能在切场景）") == "── 画面黑了（可能在切场景） ──"
    assert event_line("scene_change", "画面恢复了") == "── 画面恢复了 ──"
    assert event_line("scene_change", "画面变化很大（换了地方或者镜头动了）") is None
    assert event_line("stranger_back", "刚才那个陌生人A又回来了") == "── 刚才那个陌生人A又回来了 ──"
    assert event_line("chat", "聊天  小明：「在吗」", "") is None
