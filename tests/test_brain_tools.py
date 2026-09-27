from skydango.brain.body import ToolError
from skydango.brain.tools import ACTIONS, TOOLS, ToolBox


class FakeBody:
    def __init__(self):
        self.calls = []

    def call(self, fn):
        return fn()

    def look(self):
        return [{"type": "image"}]

    def look_at(self, x, y, w, h):
        self.calls.append(("look_at", x, y, w, h))
        return "ok"

    def status(self):
        return "状态"

    def chat_log(self, n):
        self.calls.append(("chat_log", n))
        return "log"

    def say(self, text):
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


def test_tool_definitions_cover_dispatch():
    names = [t["name"] for t in TOOLS]
    assert names == ["look", "look_at", "status", "chat_log", "say", "emote", "set_request_policy", "camera", "camera_reset"]
    assert ACTIONS == {"say", "emote", "set_request_policy", "camera", "camera_reset"}
    assert all(t["input_schema"]["type"] == "object" and t["description"] for t in TOOLS)


def test_run_dispatches_with_defaults():
    body = FakeBody()
    tb = ToolBox(body)
    assert tb.run("say", {"text": "在呢"}) == ("说了在呢", False)
    assert tb.run("look", {}) == ([{"type": "image"}], False)
    tb.run("chat_log", {})
    tb.run("emote", {"name": "鞠躬"})
    tb.run("camera", {"action": "left"})
    tb.run("set_request_policy", {"who": "*", "kind": "hug", "accept": False})
    assert body.calls == [("chat_log", 20), ("emote", "鞠躬", False), ("camera", "left", 1), ("policy", "*", "hug", False)]


def test_bad_arguments_and_errors_become_error_results():
    tb = ToolBox(FakeBody())
    content, err = tb.run("say", {})
    assert err and "缺少参数 text" in content
    content, err = tb.run("look_at", {"x": "1", "y": 0, "w": 10, "h": 10})
    assert err and "x" in content and "整数" in content
    content, err = tb.run("look_at", {"x": True, "y": 0, "w": 10, "h": 10})  # bool 不算整数
    assert err
    assert tb.run("camera_reset", {}) == ("没有视角控制", True)
    content, err = tb.run("fly", {})
    assert err and "没有这个工具" in content


def test_crash_in_body_does_not_escape():
    body = FakeBody()
    body.status = lambda: 1 / 0
    content, err = ToolBox(body).run("status", {})
    assert err and content.startswith("出错了")
