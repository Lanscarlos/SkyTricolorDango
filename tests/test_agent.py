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


def test_open_chat_with_hardware_key_only_when_closed():
    cfg = Config()
    cfg.sender.open_chat_key = 28
    device = FakeDevice([scene()])
    sender = ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None)
    sender.send("hi")
    assert device.calls[:2] == [("hw_key", 28), ("text", "hi")]

    device.calls.clear()
    device.shown = True  # 输入框已经开着（光遇发送后不会自动关闭）
    sender.send("again")
    assert ("hw_key", 28) not in device.calls
    assert device.calls[0] == ("text", "again")


def test_screenshot_failure_does_not_block_pending_reply(clock):
    """实测 adb 截图会偶尔连续失败几秒；已经读到的消息照样要回复。"""
    cfg = live_config()
    cfg.sender.close_with_back = True
    frame = scene([(400, 200, 300, 50)])
    agent, device = build(cfg, [frame], ["你在干啥"] * 10, clock)
    assert agent.step() is None  # 读到，等 debounce

    def broken():
        raise RuntimeError("adb 失败 (3221225794)")

    device.screenshot = broken
    clock.advance(2)
    assert agent.step() == "【AI】收到：你在干啥"
    assert device.calls[-1] == ("key", 4)  # 发完按 BACK 关掉输入框，头顶不再一直显示“正在输入”


def test_open_chat_waits_only_until_input_box_is_shown():
    """实测按 Enter 后输入框约 0.07 s 就开了，不必每次固定等 open_delay。"""
    cfg = Config()
    cfg.sender.open_chat_key = 28
    device = FakeDevice([scene()])
    slept = []

    def sleep(s):
        slept.append(s)
        if len(slept) == 2:  # 第二次查看时输入框开了
            device.shown = True

    ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=sleep).send("hi")
    assert device.calls[:2] == [("hw_key", 28), ("text", "hi")]
    assert sum(slept[:3]) < 0.3  # 等了两小步 + 一点缓冲，而不是 open_delay


def test_open_chat_gives_up_waiting_after_open_delay():
    cfg = Config()
    cfg.sender.open_chat_key = 28
    device = FakeDevice([scene()])  # 输入框一直没显示
    slept = []
    ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=slept.append).send("hi")
    assert abs(sum(slept) - (cfg.sender.open_delay + cfg.sender.type_delay + cfg.sender.after_delay)) < 0.11


class RecordingResponder:
    """记下调模型那一刻设备上已经发生了什么。"""

    def __init__(self, device, reply):
        self.device = device
        self.reply_text = reply
        self.calls_at_reply = None

    def reply(self, batch):
        self.calls_at_reply = list(self.device.calls)
        return self.reply_text


def _typing_agent(clock, reply, shown=False):
    cfg = live_config()
    cfg.sender.open_chat = []
    cfg.sender.open_chat_key = 28
    cfg.sender.close_with_back = True
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["在吗"] * 10, clock)
    device.shown = shown
    agent.responder = RecordingResponder(device, reply)
    return agent, device


def test_opens_input_box_while_waiting_for_model(clock):
    """等模型回复时先打开输入框：头顶显示“正在输入”，发送时也省掉打开这一步。"""
    agent, device = _typing_agent(clock, "在呢")
    agent.step()
    clock.advance(2)
    assert agent.step() == "【AI】在呢"
    assert agent.responder.calls_at_reply == [("hw_key", 28)]  # 调模型之前输入框已经打开了
    assert device.calls == [("hw_key", 28), ("text", "【AI】在呢"), ("editor", 4), ("key", 4)]  # 只按了一次 Enter


def test_closes_input_box_when_model_skips(clock):
    agent, device = _typing_agent(clock, None)
    agent.step()
    clock.advance(2)
    assert agent.step() is None
    assert device.calls == [("hw_key", 28), ("key", 4)]  # 像打了几个字又删了


def test_does_not_touch_input_box_someone_else_opened(clock):
    agent, device = _typing_agent(clock, None, shown=True)  # 输入框本来就开着（比如用户在手动打字）
    agent.step()
    clock.advance(2)
    agent.step()
    assert device.calls == []


def test_no_type_ahead_in_dry_run(clock):
    agent, device = _typing_agent(clock, "在呢")
    agent.cfg.reply.dry_run = True
    agent.step()
    clock.advance(2)
    agent.step()
    assert device.calls == []


def test_run_stops_after_duration(clock):
    cfg = Config()
    agent, _ = build(cfg, [scene()], [""], clock)
    agent.sleep = lambda s: clock.advance(max(s, 0.1))
    agent.run(duration=5.0)  # 不传 duration 会一直跑；传了就自己退出，不用在外面套 timeout
    assert clock() >= 5.0
