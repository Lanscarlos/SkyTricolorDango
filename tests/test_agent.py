from conftest import FakeDevice, FakeOcr, scene
from skydango.agent import Agent, RateLimiter
from skydango.chat.llm import EchoClient
from skydango.chat.reader import ChatReader
from skydango.chat.responder import Responder
from skydango.chat.sender import ChatSender, to_pixels
from skydango.chat.tracker import SelfFilter
from skydango.config import Config

import pytest


def build(cfg: Config, frames, ocr_texts, clock):
    device = FakeDevice(frames)
    self_filter = SelfFilter(cfg.chat.self_window, cfg.chat.similarity, cfg.reply.disclosure_prefix)
    reader = ChatReader(FakeOcr(ocr_texts), cfg.vision, cfg.ocr, cfg.chat, self_filter)
    responder = Responder(EchoClient(), cfg.reply)
    sender = ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None)
    agent = Agent(cfg, device, reader, responder, sender, self_filter, clock=clock, sleep=lambda s: None)
    return agent, device


def live_config() -> Config:
    cfg = Config()
    cfg.reply.dry_run = False
    cfg.sender.open_chat = [0.05, 0.1]
    return cfg


def test_full_loop_reads_debounces_and_sends(clock):
    cfg = live_config()
    frame = scene([(400, 200, 300, 50)])
    agent, device = build(cfg, [frame], ["一起去霞谷吗"] * 10, clock)

    assert agent.step() is None  # 刚读到，先等 debounce
    clock.advance(2)
    sent = agent.step()
    assert sent == "【AI】收到：一起去霞谷吗"
    assert device.calls == [("tap", 64, 72), ("text", sent), ("editor", 4)]

    # 气泡还在屏幕上：不重复回复
    clock.advance(10)
    assert agent.step() is None


def test_ignores_own_bubble_after_sending(clock):
    cfg = live_config()
    frame = scene([(400, 200, 300, 50)])
    texts = ["在吗"] * 2 + ["【AI】收到：在吗"] * 5
    agent, device = build(cfg, [frame], texts, clock)
    agent.step()
    clock.advance(2)
    assert agent.step() is not None
    clock.advance(10)
    assert agent.step() is None  # 读到的是自己刚发的
    assert agent.pending == []


def test_dry_run_does_not_touch_device(clock):
    cfg = Config()  # 默认 dry_run
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["你好"] * 5, clock)
    agent.step()
    clock.advance(2)
    assert agent.step() == "【AI】收到：你好"
    assert device.calls == []


def test_rate_limit_holds_pending(clock):
    cfg = live_config()
    cfg.reply.min_interval = 30
    cfg.chat.dedupe_ttl = 60  # 测试里跳过了中间的截屏，气泡的“最后看见时间”不会被刷新
    frame = scene([(400, 200, 300, 50)])
    agent, device = build(cfg, [frame], ["第一句", "第一句", "第二句", "第二句", "第二句"], clock)
    agent.step()
    clock.advance(2)
    assert agent.step() is not None
    clock.advance(2)
    agent.step()  # 读到第二句
    clock.advance(2)
    assert agent.step() is None and agent.pending  # 被限速，消息保留
    clock.advance(30)
    assert agent.step() == "【AI】收到：第二句"


def test_rate_limiter_per_minute():
    rl = RateLimiter(min_interval=0, max_per_minute=2)
    rl.record(0)
    rl.record(1)
    assert not rl.allow(2)
    assert rl.allow(61.5)


def test_to_pixels_validates():
    assert to_pixels([0.5, 0.5], 1280, 720) == (640, 360)
    with pytest.raises(ValueError):
        to_pixels([640, 360], 1280, 720)


def test_send_by_tap_requires_button():
    cfg = Config()
    cfg.sender.submit = "tap"
    device = FakeDevice([scene()])
    with pytest.raises(ValueError):
        ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None).send("hi")
    cfg.sender.send_button = [0.9, 0.9]
    ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None).send("hi")
    assert device.calls[-1] == ("tap", 1152, 648)
