import numpy as np

from skydango.brain.events import EventQueue
from skydango.brain.eyes import AROUND_REQUEST, Eyes, eyes_command
from skydango.config import BrainConfig


def frame():
    return np.full((1080, 1920, 3), (60, 90, 40), np.uint8)


class Describer:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, content):
        self.calls.append(content)
        if self.fail:
            raise RuntimeError("Haiku 没回")
        return f" 描述{len(self.calls)} "


def eyes(clock, blackout=False, labels=None, fail=False):
    d = Describer(fail)
    e = Eyes(BrainConfig(), d, frame, lambda: dict(labels or {}), lambda: blackout, clock=clock)
    return e, d


def test_looks_first_then_only_on_triggers_or_timeout(clock):
    e, _ = eyes(clock)
    assert e.tick(clock()) is True  # 从没看过：马上看
    assert e.tick(clock() + 30) is False
    e.notice("chat")
    assert e.tick(clock() + 30) is False  # 聊天不触发
    e.notice("arrive")
    assert e.tick(clock() + 10) is False  # 离上次不到 20 秒
    assert e.tick(clock() + 25) is True
    assert e.tick(clock() + 25 + 181) is True  # 超过 3 分钟兜底


def test_blackout_skips(clock):
    e, d = eyes(clock, blackout=True)
    assert e.tick(clock()) is False and d.calls == []


def test_describe_frame_sends_image_names_and_caches(clock):
    e, d = eyes(clock, labels={"懒洋洋大王": (1320, 300, 160, 44, clock())})
    text = e.describe_frame(frame(), clock())
    content = d.calls[0]
    assert text == "描述1" and content[0]["type"] == "image"
    assert "懒洋洋大王：名字在 (933, 200)" in content[1]["text"] and "地点和环境" in content[1]["text"]
    assert e.summary(clock() + 35) == "场景（35 秒前看的）：\n描述1"


def test_stale_names_are_not_passed(clock):
    e, d = eyes(clock, labels={"懒洋洋大王": (1320, 300, 160, 44, clock() - 60)})
    e.describe_frame(frame(), clock())
    assert "图里没认出好友的名字" in d.calls[0][1]["text"]


def test_describe_around_marks_directions(clock):
    e, d = eyes(clock)
    e.describe_around([frame()] * 4, clock())
    content = d.calls[0]
    texts = [b["text"] for b in content if b["type"] == "text"]
    assert texts[:4] == ["【前】", "【右】", "【后】", "【左】"] and texts[-1] == AROUND_REQUEST
    assert sum(b["type"] == "image" for b in content) == 4


def test_failure_is_swallowed_and_waits(clock):
    e, _ = eyes(clock, fail=True)
    assert e.tick(clock()) is True
    assert e.latest is None and e.summary(clock()) == "场景：还没有场景描述"
    assert e.tick(clock() + 5) is False  # 别每秒都重试


def test_event_queue_pokes_eyes(clock):
    q = EventQueue(clock=clock)
    e, _ = eyes(clock)
    e.last_look = clock()
    q.subscribe(e.notice)
    q.put("scene_change", "画面变化很大")
    assert e.tick(clock() + 21) is True


def test_eyes_command_is_locked_down():
    cmd = eyes_command(["claude"], BrainConfig())
    assert cmd[cmd.index("--model") + 1] == "haiku" and cmd[cmd.index("--tools") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--system-prompt", "--verbose"):
        assert flag in cmd


def test_eyes_use_scene_note_when_given(clock):
    d = Describer()
    seen = []
    e = Eyes(BrainConfig(), d, frame, lambda: {}, lambda: False, clock=clock, note=lambda now, s: seen.append(s) or "NOTE")
    e.describe_frame(frame(), clock())
    assert d.calls[0][1]["text"].startswith("NOTE") and seen == [1280 / 1920]


def test_eyes_prompts_mention_listed_people_and_skip_self():
    from skydango.brain.eyes import EYES_SYSTEM, LOOK_REQUEST

    assert "没列出的人都叫“陌生人”" in EYES_SYSTEM and "团子、陌生人位置" not in EYES_SYSTEM
    assert "给出的陌生人、团子位置照用" in EYES_SYSTEM
    assert "团子自己不用描述" in LOOK_REQUEST


def test_eyes_do_not_auto_look_on_approach():
    from skydango.brain.eyes import AUTO_LOOK_KINDS

    assert "approach" not in AUTO_LOOK_KINDS
