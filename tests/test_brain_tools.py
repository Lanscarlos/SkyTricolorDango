import threading
import time

import pytest

from skydango.brain.body import ToolError
from skydango.brain.tools import ACTIONS, SAY_ENOUGH, TOOL_NAMES, ToolBox
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

    def look_person(self, name):
        self.calls.append(("look_person", name))
        return "ok"

    def status(self):
        return "状态"

    def stop_task(self):
        self.calls.append(("stop_task",))
        return "停下了"

    def track(self, name, seconds):
        self.calls.append(("track", name, seconds))
        return f"开始盯着{name}了"

    def set_attention(self, mode, focus=None):
        self.calls.append(("set_attention", mode, focus))
        return "注意力：" + mode

    def chat_log(self, n):
        self.calls.append(("chat_log", n))
        return "log"

    def say(self, text):
        if text == "太快":
            raise ToolError("说得太快了")
        return f"说了{text}"

    def emote(self, name):
        self.calls.append(("emote", name))
        return "ok"

    def set_policy(self, who, kind, accept):
        self.calls.append(("policy", who, kind, accept))
        return "ok"

    def camera_move(self, action, steps):
        self.calls.append(("camera", action, steps))
        return "ok"

    def camera_reset(self):
        raise ToolError("没有视角控制")

    def move(self, direction, steps, force):
        self.calls.append(("move", direction, steps, force))
        return "ok"


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
        "look", "look_at", "look_person", "look_around", "status", "chat_log", "recall", "say", "emote", "set_request_policy", "camera",
        "camera_reset", "attention", "move", "check_friend", "track", "find", "stop_task", "panel_read", "panel_press", "panel_close",
    ]
    assert ACTIONS == {
        "call", "say", "emote", "set_request_policy", "camera", "camera_reset", "move", "check_friend", "track", "find", "stop_task",
        "panel_press", "panel_close",
    }


def test_tools_track_passes_args():
    body = FakeBody()
    tb = ToolBox(body)
    out, err = tb.run("track", {"name": "小明", "seconds": 10})
    assert not err and out == "开始盯着小明了" and tb.acted
    tb.run("track", {"name": "阿白"})
    assert body.calls == [("track", "小明", 10), ("track", "阿白", 30)]
    assert "track" in ACTIONS
    out, err = tb.run("track", {})
    assert err and "name" in out


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


def test_look_person_gets_room_to_peek():  # 被挡住时边转边看：按键 + 等画面停稳最多 [peek] max_seconds 秒，再截图裁图
    from skydango.brain.tools import PEEK_TIMEOUT
    from skydango.config import Config

    body = FakeBody()
    ToolBox(body).run("look_person", {"name": "小明"})
    assert body.timeouts[-1] == PEEK_TIMEOUT and PEEK_TIMEOUT >= Config().peek.max_seconds + 10


def test_look_person_description_says_camera_is_not_reset():
    from skydango.brain.tools import DESCRIPTIONS

    assert "挡住" in DESCRIPTIONS["look_person"] and "camera_reset" in DESCRIPTIONS["look_person"]


def test_camera_reset_gets_longer_timeout():
    from skydango.brain.tools import RESET_TIMEOUT

    body = FakeBody()
    ToolBox(body).run("camera_reset", {})
    assert body.timeouts[-1] == RESET_TIMEOUT and RESET_TIMEOUT >= 45  # 闭环细调最多 60 下，每下要等画面停稳


def test_per_turn_limits():
    tb = ToolBox(FakeBody(), FakeEyes(), max_steps=3, max_says=1)
    assert tb.run("say", {"text": "a"}) == ("说了a", False)
    assert tb.run("say", {"text": "b"})[1] is True
    assert tb.run("status", {})[1] is False
    content, err = tb.run("status", {})
    assert err and content.startswith("这一轮做的事够多了")
    tb.begin_turn()
    assert tb.run("say", {"text": "c"}) == ("说了c", False)


def test_first_say_of_turn_reminds_one_is_enough():
    """10-03 晚一轮两句，第二句常是把第一句换个说法再说一遍（「不要啦我才不叫」→「要让他睡饱呀」）：
    第一句说完就在结果里提醒，第二句不再提醒；名额只剩一句时不提醒。"""
    tb = ToolBox(FakeBody(), max_says=2)
    first, err = tb.run("say", {"text": "a"})
    assert not err and first.startswith("说了a") and SAY_ENOUGH in first
    assert tb.run("say", {"text": "b"}) == ("说了b", False)
    tb = ToolBox(FakeBody(), max_says=1)
    assert tb.run("say", {"text": "a"}) == ("说了a", False)


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
    tb.run("look_person", {"name": "小明"})
    tb.begin_turn()  # 一轮最多 6 次工具
    tb.run("stop_task", {})
    tb.run("move", {"direction": "forward"})
    tb.run("move", {"direction": "left", "steps": 2, "force": True})
    assert body.calls == [
        ("chat_log", 20), ("emote", "鞠躬"), ("camera", "left", 1), ("policy", "*", "hug", False), ("look_at", 1, 2, 30, 40),
        ("look_person", "小明"), ("stop_task",), ("move", "forward", 1, False), ("move", "left", 2, True),
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


class SweepEnv:
    def sweep(self, frames, spin):
        return None


def test_look_around_uses_sweep_when_perception_is_on():
    body = FakeBody()
    body.env = SweepEnv()
    body.sweep_around = lambda: "正前方：懒洋洋大王"
    eyes = FakeEyes()
    eyes.describe_around = lambda frames, now: pytest.fail("不该叫眼睛")
    assert ToolBox(body, eyes).run("look_around", {}) == ("正前方：懒洋洋大王", False)
    assert body.timeouts[-1] == 30
    assert ToolBox(body).run("look_around", {}) == ("正前方：懒洋洋大王", False)  # 没开眼睛也能扫


def test_look_around_description_switches():
    from skydango.brain.tools import DESCRIPTIONS, descriptions

    assert "几秒就好" in descriptions(True)["look_around"] and "十几秒" in descriptions(False)["look_around"]
    assert DESCRIPTIONS == descriptions(False) and list(descriptions(True)) == TOOL_NAMES


# ---- recall：翻以前的聊天（只读 memory/，不走身体线程） ----
def test_recall_searches_memory(tmp_path):
    from skydango.chat.memory import MemoryStore

    store = MemoryStore(tmp_path)
    store.history.append("新的聊天消息：\n懒洋洋大王：「明天跑暴风眼吗」", "行啊", time.time() - 86400)
    body = FakeBody()
    tb = ToolBox(body, memory=store)
    out, err = tb.run("recall", {"query": "暴风眼", "days": 3})
    assert not err and "懒洋洋大王：「明天跑暴风眼吗」 → 我：行啊" in out
    assert body.timeouts == [] and not tb.acted  # 不占身体线程、不算做了事
    out, err = tb.run("recall", {"who": "懒洋洋"})
    assert not err and "暴风眼" in out


def test_recall_errors_go_back_to_the_brain(tmp_path):
    from skydango.chat.memory import MemoryStore

    tb = ToolBox(FakeBody(), memory=MemoryStore(tmp_path))
    out, err = tb.run("recall", {})
    assert err and "关键词" in out
    assert tb.run("recall", {"query": "x", "days": "三天"})[1] is True
    out, err = ToolBox(FakeBody()).run("recall", {"query": "x"})
    assert err and "记忆" in out


# ---- 沙盒计划 Task 2：text_only（沙盒里没有画面）----
def test_text_only_look_always_gives_eyes_text():
    from skydango.brain.tools import SANDBOX_NO_PERSON

    body = FakeBody()
    tb = ToolBox(body, FakeEyes(), text_only=True)
    out, err = tb.run("look", {"image": True})
    assert not err and out == "描述帧"  # 不附图、只要眼睛的文字
    out, err = tb.run("look_person", {"name": "小明"})
    assert not err and out == SANDBOX_NO_PERSON == "沙盒里看不到人，只能靠聊天和场景"
    assert ("look_person", "小明") not in body.calls


def test_not_text_only_is_unchanged():
    tb = ToolBox(FakeBody(), FakeEyes())
    assert tb.run("look", {"image": True}) == ([{"type": "image"}], False)


def test_sandbox_says_missing_parts():  # 沙盒计划 Task 5：镜头、好友树、面板、互动请求 → "沙盒里没有这个"
    body = FakeBody()
    tb = ToolBox(body, FakeEyes(), max_steps=50, sandbox=True)
    for name, args in (("camera", {"action": "left"}), ("camera_reset", {}), ("track", {"name": "小明"}), ("look_around", {}),
                       ("check_friend", {"x": 1, "y": 2}), ("panel_read", {}), ("panel_close", {}),
                       ("set_request_policy", {"who": "*", "kind": "hand", "accept": True})):
        out, err = tb.run(name, args)
        assert err and out.startswith("沙盒里没有这个"), (name, out)
    assert body.calls == []
    assert tb.run("emote", {"name": "鞠躬"}) == ("ok", False)  # 别的照旧
    assert ToolBox(FakeBody()).run("camera", {"action": "left"}) == ("ok", False)


def test_attention_tool_is_listed_and_not_an_action():
    from skydango.brain.tools import DESCRIPTIONS

    assert "attention" in DESCRIPTIONS and "attention" not in ACTIONS  # 不在游戏里做事：不让心跳退档归零
    assert "随意" in DESCRIPTIONS["attention"] and "别动" in DESCRIPTIONS["attention"]


def test_attention_tool_calls_body():
    body = FakeBody()
    tb = ToolBox(body)
    out, err = tb.run("attention", {"mode": "专心", "focus": "小明"})
    assert not err and body.calls[-1] == ("set_attention", "专心", "小明")
    tb.run("attention", {"mode": "随意", "focus": ""})
    assert body.calls[-1] == ("set_attention", "随意", None)
    tb.run("attention", {"mode": "好奇"})
    assert body.calls[-1] == ("set_attention", "好奇", None)


def test_tools_find_passes_args():
    class B(FakeBody):
        def find(self, name, seconds=30, live=False):
            self.calls.append(("find", name, seconds))
            return f"开始找{name}了"

    body = B()
    tb = ToolBox(body)
    out, err = tb.run("find", {"name": "小明", "seconds": 20})
    assert not err and out == "开始找小明了" and tb.acted
    tb.run("find", {"name": "阿白"})
    assert body.calls == [("find", "小明", 20), ("find", "阿白", 30)]
    out, err = ToolBox(B(), FakeEyes(), max_steps=50, sandbox=True).run("find", {"name": "小明"})
    assert err and "沙盒里没有这个" in out


# ---- 代看：大脑看不了图时看图工具交给眼睛（spec 2026-10-05-model-providers §2.4）----
def _image_body():
    import numpy as np

    from skydango.brain.images import image_block

    body = FakeBody()
    blocks = [image_block(np.zeros((8, 8, 3), np.uint8)), {"type": "text", "text": "小明在 (300,200)"}]
    body.look_person = lambda name: blocks
    return body, blocks


def test_proxy_replaces_image():
    body, _ = _image_body()
    tb = ToolBox(body, proxy=lambda b, q: f"{len(b)}|{q}", sees=lambda: False)
    assert tb.run("look_person", {"name": "小明", "question": "衣服"}) == ("（眼睛代看）2|衣服", False)


def test_sees_keeps_image():
    body, blocks = _image_body()
    tb = ToolBox(body, proxy=lambda b, q: "不该调", sees=lambda: True)
    assert tb.run("look_person", {"name": "小明"}) == (blocks, False)


def test_proxy_when_eyes_unavailable():  # Review Focus 5
    from skydango.models.errors import ModelUnavailable

    body, _ = _image_body()

    def proxy(b, q):
        raise ModelUnavailable("现在看不了图")

    out, is_error = ToolBox(body, proxy=proxy, sees=lambda: False).run("look_person", {"name": "小明"})
    assert is_error is True and "看不了图" in out


def test_look_image_when_blind_reuses_recent_eyes():  # 刚看过：不再起代看，直接给眼睛的描述
    from types import SimpleNamespace

    body = FakeBody()
    eyes = SimpleNamespace(latest=("在雨林", 99.0), last_look=99.0, summary=lambda now: "场景：在雨林")
    tb = ToolBox(body, eyes=eyes, proxy=lambda b, q: "不该调", sees=lambda: False)
    assert tb.run("look", {"image": True}) == ("场景：在雨林", False)


class JabBody(FakeBody):
    def say(self, text, jab=False):
        self.calls.append(("say", text, jab))
        return f"说了{text}"


def test_toolbox_say_passes_jab():
    """jab 开着时把大脑标的 jab 交给身体；关着时 say 只收到 text（spec 2026-10-07-chat-pacing §2）。"""
    b = JabBody()
    tb = ToolBox(b, jab=True)
    tb.run("say", {"text": "嗯", "jab": True})
    tb.begin_turn()
    tb.run("say", {"text": "好"})
    assert b.calls == [("say", "嗯", True), ("say", "好", False)]
    plain = FakeBody()  # say 只接 text
    out, err = ToolBox(plain).run("say", {"text": "嗯", "jab": True})
    assert not err and out.startswith("说了嗯")
