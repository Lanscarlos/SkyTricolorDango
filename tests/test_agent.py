from conftest import FakeDevice, FakeOcr, scene
from skydango.agent import Agent, RateLimiter
from skydango.chat.llm import EchoClient
from skydango.chat.memory import MemoryStore
from skydango.chat.reader import ChatReader, Message
from skydango.chat.responder import Reply, Responder
from skydango.chat.sender import ChatSender, to_pixels
from skydango.chat.tracker import SelfFilter
from skydango.config import Config
from skydango.game.wheel import WheelError
from skydango.vision.bubbles import Rect

import pytest


def build(cfg: Config, frames, ocr_texts, clock, llm=None, emotes=None, store=None):
    device = FakeDevice(frames)
    self_filter = SelfFilter(cfg.chat.self_window, cfg.chat.similarity, cfg.reply.disclosure_prefix)
    reader = ChatReader(FakeOcr(ocr_texts), cfg.vision, cfg.ocr, cfg.chat, self_filter)
    available = emotes.available if emotes else None
    responder = Responder(llm or EchoClient(), cfg.reply, available_emotes=available)
    sender = ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None)
    agent = Agent(
        cfg, device, reader, responder, sender, self_filter, clock=clock, sleep=lambda s: None, emotes=emotes, store=store
    )
    if emotes:
        emotes.device = device
    return agent, device


class ScriptedReader:
    """跳过真实 OCR：直接按顺序吐出预设的消息批次（每轮一批，用光了就不再有新消息）。"""

    def __init__(self, batches: list[list[Message]]) -> None:
        self.batches = list(batches)
        self.panel_closed_since = None

    def read(self, frame, now):
        return self.batches.pop(0) if self.batches else []


def msg(text: str, speaker: str = "") -> Message:
    return Message(text, Rect(0, 0, 1, 1), 0.0, speaker)


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
        return Reply(self.reply_text) if self.reply_text else None


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


class FixedLlm:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, messages):
        return self.reply


class FakeEmotes:
    def __init__(self, names=("鞠躬", "害羞"), fail=False):
        self.names = list(names)
        self.fail = fail
        self.pretended = []
        self.device = None

    def available(self):
        return list(self.names)

    def perform(self, name):
        if self.fail:
            raise WheelError("动作列表里没找到")
        self.device.calls.append(("emote", name))
        return 1

    def pretend(self, name):
        self.pretended.append(name)


def run_one_turn(agent, clock):
    agent.step()
    clock.advance(2)
    return agent.step()


def test_emote_before_text(clock):
    cfg = live_config()
    cfg.sender.type_ahead = False  # 提前开输入框的情况见 test_emote_with_type_ahead_closes_our_input_box_first
    emotes = FakeEmotes()
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["你真可爱"] * 5, clock, FixedLlm("[害羞]哪有啦"), emotes)
    assert run_one_turn(agent, clock) == "【AI】哪有啦"
    assert device.calls == [("emote", "害羞"), ("tap", 64, 72), ("text", "【AI】哪有啦"), ("editor", 4)]
    assert agent.emoted == ["害羞"]


def test_emote_only_turn_does_not_use_send_quota(clock):
    cfg = live_config()
    cfg.sender.type_ahead = False  # 提前开输入框的情况见 test_emote_with_type_ahead_closes_our_input_box_first
    emotes = FakeEmotes()
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["晚安"] * 5, clock, FixedLlm("[鞠躬]"), emotes)
    assert run_one_turn(agent, clock) is None
    assert device.calls == [("emote", "鞠躬")]
    assert agent.sent == [] and agent.limiter.allow(clock())


def test_dry_run_pretends_emote(clock):
    cfg = Config()  # 默认 dry_run
    emotes = FakeEmotes()
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["你真可爱"] * 5, clock, FixedLlm("[害羞]哪有啦"), emotes)
    assert run_one_turn(agent, clock) == "【AI】哪有啦"
    assert device.calls == [] and emotes.pretended == ["害羞"]


def test_emote_failure_still_sends_text(clock):
    cfg = live_config()
    emotes = FakeEmotes(fail=True)
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["你真可爱"] * 5, clock, FixedLlm("[害羞]哪有啦"), emotes)
    assert run_one_turn(agent, clock) == "【AI】哪有啦"
    assert ("text", "【AI】哪有啦") in device.calls


def test_emote_with_type_ahead_closes_our_input_box_first(clock):
    """等模型时提前开了输入框；要做动作就先关掉（数字键会变成打字），发文字时再打开。"""
    cfg = live_config()
    cfg.sender.open_chat = []
    cfg.sender.open_chat_key = 28
    emotes = FakeEmotes()
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["你真可爱"] * 5, clock, FixedLlm("[害羞]哪有啦"), emotes)
    assert run_one_turn(agent, clock) == "【AI】哪有啦"
    assert device.calls == [
        ("hw_key", 28), ("key", 4), ("emote", "害羞"), ("hw_key", 28), ("text", "【AI】哪有啦"), ("editor", 4)
    ]


# ---- 主人命令（# 开头） ----


def owner_agent(clock, batches, tmp_path, camera=None, run=None, frames=None):
    cfg = live_config()
    cfg.reply.owner_name = "懒洋洋大王"
    cfg.chat.debounce = 999  # 正常聊天要等很久才凑够 debounce 才回复，命令不受影响
    device = FakeDevice(frames or [scene()])
    self_filter = SelfFilter(cfg.chat.self_window, cfg.chat.similarity, cfg.reply.disclosure_prefix)
    reader = ScriptedReader(batches)
    responder = Responder(EchoClient(), cfg.reply)
    sender = ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None)
    store = MemoryStore(tmp_path)
    agent = Agent(cfg, device, reader, responder, sender, self_filter, clock=clock, sleep=lambda s: None, store=store,
                  camera=camera, run=run)
    return agent, device, store


def test_owner_remember_command_sends_confirmation_and_skips_pending(clock, tmp_path):
    agent, device, store = owner_agent(clock, [[msg("#remember 卡洛周三要加班", "懒洋洋大王")]], tmp_path)
    agent.step()
    assert agent.sent == ["【AI】记下了"]
    assert agent.pending == []  # 没进 pending，不占 debounce
    assert "卡洛周三要加班" in store.inbox()


def test_owner_friend_command_writes_friends_md(clock, tmp_path):
    agent, device, store = owner_agent(clock, [[msg("#friend 新朋友 第一次见", "懒洋洋大王")]], tmp_path)
    agent.step()
    assert agent.sent == ["【AI】记好啦"]
    assert "新朋友" in store.friends() and "第一次见" in store.friends()


def test_owner_command_does_not_use_send_rate_limit(clock, tmp_path):
    agent, device, store = owner_agent(clock, [[msg("#remember 一", "懒洋洋大王")], [msg("#remember 二", "懒洋洋大王")]], tmp_path)
    agent.limiter.max_per_minute = 0  # 普通聊天此时会被限速卡住，发不出任何一句
    agent.step()
    agent.step()  # 同一秒内连续两条命令：命令不占用限速，照样都发出去
    assert agent.sent == ["【AI】记下了", "【AI】记下了"]


def test_owner_pause_stops_other_peoples_messages(clock, tmp_path):
    agent, device, store = owner_agent(
        clock,
        [[msg("#pause", "懒洋洋大王")], [msg("在吗", "路人")], [msg("#resume", "懒洋洋大王")], [msg("在吗", "路人")]],
        tmp_path,
    )
    agent.step()
    assert agent.paused is True
    agent.step()
    assert agent.pending == []  # 暂停期间别人的消息不进 pending
    agent.step()
    assert agent.paused is False
    agent.step()
    assert len(agent.pending) == 1  # 恢复后照常收消息


def test_owner_status_command_reports_state(clock, tmp_path):
    agent, device, store = owner_agent(clock, [[msg("#status", "懒洋洋大王")]], tmp_path)
    agent.step()
    assert agent.sent == ["【AI】live｜运行中｜待处理0｜限速8/8"]


def test_unknown_owner_command(clock, tmp_path):
    agent, device, store = owner_agent(clock, [[msg("#跳舞", "懒洋洋大王")]], tmp_path)
    agent.step()
    assert agent.sent == ["【AI】没这个命令"]


def test_hash_prefix_from_non_owner_is_normal_chat(clock, tmp_path):
    agent, device, store = owner_agent(clock, [[msg("#remember 我也想试试", "路人")]], tmp_path)
    agent.step()
    assert agent.sent == []  # 不是主人发的，不当命令，走普通聊天流程
    assert len(agent.pending) == 1
    assert store.inbox() == ""


def test_owner_name_empty_disables_commands(clock, tmp_path):
    agent, device, store = owner_agent(clock, [[msg("#remember 测试", "懒洋洋大王")]], tmp_path)
    agent.cfg.reply.owner_name = ""
    agent.step()
    assert agent.sent == []  # 功能关闭时，即使 speaker 匹配也当普通聊天
    assert len(agent.pending) == 1
    assert store.inbox() == ""


class HoldEnv:
    requests = {}

    def __init__(self):
        self.holds = []

    def observe(self, frame, now, panel_visible):
        pass

    def nearby(self, now):
        return []

    def describe(self, now):
        return ""

    def held(self, reason):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            self.holds.append(("hold", reason))
            try:
                yield
            finally:
                self.holds.append(("release", reason))

        return ctx()


def test_agent_emote_wrapped_in_held(clock):
    cfg = live_config()
    emotes = FakeEmotes()
    agent, _ = build(cfg, [scene([(400, 200, 300, 50)])], ["你真可爱"] * 5, clock, FixedLlm("[害羞]哪有啦"), emotes)
    agent.env = HoldEnv()
    run_one_turn(agent, clock)
    assert agent.env.holds == [("hold", "wheel"), ("release", "wheel")]


# ---- #spin ----
class FakeCamera:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def spin(self, capture, turns=1, seconds_per_turn=2.0, fps=15.0):
        from skydango.brain.camera import SpinResult

        self.calls.append((turns, seconds_per_turn, fps))
        if self.error:
            raise self.error
        f = capture()
        return SpinResult(f, [(i / 15, f) for i in range(28)], f, 1.9, True, False)


def test_owner_spin_turns_camera_and_confirms(clock, tmp_path):
    cam = FakeCamera()
    agent, device, _ = owner_agent(clock, [[msg("#spin", "懒洋洋大王")]], tmp_path, camera=cam)
    agent.step()
    assert cam.calls == [(1, 2.0, 15.0)]
    assert agent.sent == ["【AI】转完了，1.9 秒 28 张"]
    assert agent.pending == [] and agent.limiter.remaining(clock()) == agent.limiter.max_per_minute


def test_second_spin_within_min_interval_is_refused(clock, tmp_path):
    cam = FakeCamera()
    agent, _, _ = owner_agent(clock, [[msg("#spin", "懒洋洋大王")], [msg("#spin 2", "懒洋洋大王")]], tmp_path, camera=cam)
    agent.step()
    agent.step()
    assert len(cam.calls) == 1 and agent.sent[-1] == "【AI】刚转过，等 10 秒"


def test_spin_in_dry_run_still_turns_but_does_not_send(clock, tmp_path):
    cam = FakeCamera()
    agent, device, _ = owner_agent(clock, [[msg("#spin", "懒洋洋大王")]], tmp_path, camera=cam)
    agent.cfg.reply.dry_run = True
    agent.step()
    assert len(cam.calls) == 1 and agent.sent == ["【AI】转完了，1.9 秒 28 张"]
    assert not any(c[0] == "text" for c in device.calls)


def test_spin_refused_on_black_screen(clock, tmp_path):
    import numpy as np

    cam = FakeCamera()
    agent, _, _ = owner_agent(clock, [[msg("#spin", "懒洋洋大王")]], tmp_path, camera=cam,
                              frames=[np.zeros((720, 1280, 3), np.uint8)])
    agent.step()
    assert cam.calls == [] and agent.sent == ["【AI】画面黑着，转不了"]


def test_spin_error_is_reported(clock, tmp_path):
    cam = FakeCamera(RuntimeError("截图失败\n详细"))
    agent, _, _ = owner_agent(clock, [[msg("#spin", "懒洋洋大王")]], tmp_path, camera=cam)
    agent.step()
    assert agent.sent == ["【AI】没转成：截图失败"]


def test_spin_without_camera_says_so(clock, tmp_path):
    agent, _, _ = owner_agent(clock, [[msg("#spin", "懒洋洋大王")]], tmp_path)
    agent.step()
    assert agent.sent == ["【AI】这次没开视角控制"]


def test_spin_wrapped_in_camera_hold(clock, tmp_path):
    agent, _, _ = owner_agent(clock, [[msg("#spin", "懒洋洋大王")]], tmp_path, camera=FakeCamera())
    agent.env = HoldEnv()
    agent.step()
    assert agent.env.holds == [("hold", "camera"), ("release", "camera")]


def test_spin_saves_to_run_dir(clock, tmp_path):
    from skydango.runlog import RunDir

    cfg = Config()
    cfg.run.dir = str(tmp_path / "runs")
    run = RunDir.create(cfg, "live", now=0)
    agent, _, _ = owner_agent(clock, [[msg("#spin", "懒洋洋大王")]], tmp_path, camera=FakeCamera(), run=run)
    agent.step()
    [folder] = list((run.path / "spin").iterdir())
    assert (folder / "summary.json").exists()
