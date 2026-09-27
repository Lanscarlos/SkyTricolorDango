import threading

from skydango.brain.body import ToolError
from skydango.brain.tools import ACTIONS, TOOL_NAMES, ToolBox
from skydango.config import Config


class FakeBody:
    def __init__(self):
        self.calls = []
        self.timeouts = []
        self.cfg = Config()

    def clock(self):
        return 100.0

    def call(self, fn, timeout=None):
        self.timeouts.append(timeout)
        return fn()

    def look(self):
        return [{"type": "image"}]

    def fresh_frame(self):
        return "帧"

    def capture_around(self):
        return ["帧"] * 4

    def look_at(self, x, y, w, h):
        self.calls.append(("look_at", x, y, w, h))
        return "ok"

    def status(self):
        return "状态"

    def chat_log(self, n):
        self.calls.append(("chat_log", n))
        return "log"

    def say(self, text):
        if text == "太快":
            raise ToolError("说得太快了")
        return f"说了{text}"

    def emote(self, name, force):
        self.calls.append(("emote", name, force))
        return "ok"

    def set_policy(self, who, kind, accept):
        self.calls.append(("policy", who, kind, accept))
        return "ok"

    def camera_move(self, action, steps):
        self.calls.append(("camera", action, steps))
        return "ok"

    def camera_reset(self):
        raise ToolError("没有视角控制")


class FakeEyes:
    def __init__(self):
        self.latest = None
        self.last_look = float("-inf")

    def describe_frame(self, frame, now):
        return f"描述{frame}"

    def describe_around(self, frames, now):
        return f"四周{len(frames)}张"

    def summary(self, now):
        return "旧描述"


def test_tool_names_and_actions():
    assert TOOL_NAMES == [
        "look", "look_at", "look_around", "status", "chat_log", "say", "emote", "set_request_policy", "camera", "camera_reset"
    ]
    assert ACTIONS == {"say", "emote", "set_request_policy", "camera", "camera_reset"}


def test_look_uses_eyes_unless_image_requested():
    eyes = FakeEyes()
    tb = ToolBox(FakeBody(), eyes)
    assert tb.run("look", {}) == ("描述帧", False)
    assert tb.run("look", {"image": True}) == ([{"type": "image"}], False)
    eyes.latest, eyes.last_look = ("旧", 99.0), 99.0
    assert tb.run("look", {}) == ("旧描述", False)  # 5 秒内刚看过：给缓存，不再花一次


def test_look_around_gets_longer_timeout():
    body = FakeBody()
    assert ToolBox(body, FakeEyes()).run("look_around", {}) == ("四周4张", False)
    assert body.timeouts[-1] == 30
    assert ToolBox(body).run("look_around", {})[1] is True  # 没开眼睛


def test_per_turn_limits():
    tb = ToolBox(FakeBody(), FakeEyes(), max_steps=3, max_says=1)
    assert tb.run("say", {"text": "a"}) == ("说了a", False)
    assert tb.run("say", {"text": "b"})[1] is True
    assert tb.run("status", {})[1] is False
    content, err = tb.run("status", {})
    assert err and content.startswith("这一轮做的事够多了")
    tb.begin_turn()
    assert tb.run("say", {"text": "c"}) == ("说了c", False)


def test_failed_say_does_not_use_quota():
    tb = ToolBox(FakeBody(), max_says=1)
    assert tb.run("say", {"text": "太快"}) == ("说得太快了", True)
    assert tb.run("say", {"text": "好"}) == ("说了好", False)


def test_acted_used_and_status_not_counted():
    tb = ToolBox(FakeBody())
    assert tb.status() == "状态" and tb.used == []
    tb.run("status", {})
    assert tb.acted is False
    tb.run("camera", {"action": "left"})
    assert tb.acted is True and tb.used == ["status", "camera"]


def test_concurrent_says_respect_limit():
    tb = ToolBox(FakeBody(), max_steps=50, max_says=2)
    results = []
    threads = [threading.Thread(target=lambda: results.append(tb.run("say", {"text": "x"}))) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(not err for _, err in results) == 2


def test_run_dispatches_with_defaults():
    body = FakeBody()
    tb = ToolBox(body)
    tb.run("chat_log", {})
    tb.run("emote", {"name": "鞠躬"})
    tb.run("camera", {"action": "left"})
    tb.run("set_request_policy", {"who": "*", "kind": "hug", "accept": False})
    tb.run("look_at", {"x": 1, "y": 2, "w": 30, "h": 40})
    assert body.calls == [
        ("chat_log", 20), ("emote", "鞠躬", False), ("camera", "left", 1), ("policy", "*", "hug", False), ("look_at", 1, 2, 30, 40)
    ]


def test_bad_arguments_and_errors_become_error_results():
    tb = ToolBox(FakeBody())
    content, err = tb.run("say", {})
    assert err and "缺少参数 text" in content
    content, err = tb.run("look_at", {"x": "1", "y": 0, "w": 10, "h": 10})
    assert err and "整数" in content
    assert tb.run("look_at", {"x": True, "y": 0, "w": 10, "h": 10})[1] is True  # bool 不算整数
    assert tb.run("camera_reset", {}) == ("没有视角控制", True)
    content, err = tb.run("fly", {})
    assert err and "没有这个工具" in content


def test_crash_in_body_does_not_escape():
    body = FakeBody()
    body.status = lambda: 1 / 0
    content, err = ToolBox(body).run("status", {})
    assert err and content.startswith("出错了")
