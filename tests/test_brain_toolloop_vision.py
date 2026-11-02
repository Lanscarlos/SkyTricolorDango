import json
from types import SimpleNamespace

import numpy as np
import pytest

from skydango.brain.images import image_block
from skydango.brain.toolloop import (
    BLIND_NOTE, IMAGE_HOLDER, SIGHTED_NOTE, TOOL_IMAGE_HOLDER, TOOL_IMAGE_NOTE, VISION_OFF_NOTE, ToolLoopBrain,
)
from skydango.brain.vision import mark_frame
from skydango.models.errors import ModelError


def att(src="auto"):
    return mark_frame(np.zeros((1080, 1920, 3), np.uint8), [{"x": 100, "y": 300, "w": 200, "h": 300, "kind": "friend", "label": "小明"}],
                      (960, 540), 80, src)


def shot():
    return image_block(np.zeros((90, 160, 3), np.uint8), 80)


def call(name, args="{}", cid="c1"):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=args))


def reply(content=None, calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))])


class Recorder:
    """记下每次 create 收到的 messages（深拷贝：toolloop 会原地改）；script 里的 Exception 直接抛。"""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[list[dict]] = []

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw["messages"])))
        item = self.script.pop(0) if self.script else reply("好")
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def client(self):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))


def rejected(text="image_url is not supported"):
    exc = Exception(text)
    exc.status_code = 400
    return exc


def toolbox(returns):
    """returns：工具名 → (内容, 是否出错)。"""
    calls = []

    def run(name, args):
        calls.append(name)
        return returns.get(name, ("好", False))

    return SimpleNamespace(calls=calls, run=run)


def tools(*names):
    return [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}} for n in names]


def make(rec, tb=None, **kw):
    return ToolLoopBrain(rec.client, "sys", tb or toolbox({}), tools("look", "look_person", "say"), model="m", temperature=0.8,
                         max_tokens=4096, history=8, **kw)


def look_result(note="位置说明"):
    return ([{"type": "text", "text": note}, shot()], False)


def test_wake_image_in_request_and_placeholder_in_history():
    rec = Recorder([reply("好"), reply("好")])
    brain = make(rec)
    a = att()
    r = brain.send("[t] 事件：\n- 小明：在吗\n状态：身边 小明", image=a)
    content = rec.requests[0][-1]["content"]
    assert [p["type"] for p in content] == ["text", "text", "image_url"] and content[1]["text"] == a.legend
    assert r["images"] == [{"src": "auto", "w": 960, "h": 540}]
    brain.send("再来一轮")
    hist = rec.requests[1][1]["content"]
    assert isinstance(hist, str) and a.legend in hist and IMAGE_HOLDER in hist and "image_url" not in json.dumps(rec.requests[1])
    assert hist.index(IMAGE_HOLDER) < hist.index("状态：")  # 插在状态前面，截断时不会丢


def test_tool_image_goes_to_next_user_message():
    rec = Recorder([reply(None, [call("look", '{"image": true}')]), reply("好"), reply("好")])
    brain = make(rec, toolbox({"look": look_result()}))
    r = brain.send("醒来")
    second = rec.requests[1]
    assert second[-2] == {"role": "tool", "tool_call_id": "c1", "content": TOOL_IMAGE_NOTE + "\n位置说明"}
    user = second[-1]
    assert user["role"] == "user" and user["content"][0]["text"] == "（look 的截图）" and user["content"][1]["type"] == "image_url"
    assert r["images"] == [{"src": "look", "w": 160, "h": 90}]
    brain.send("下一轮")
    assert {"role": "user", "content": TOOL_IMAGE_HOLDER} in rec.requests[2]
    assert "image_url" not in json.dumps(rec.requests[2])


def test_two_image_tools_one_user_message():  # Review Focus 1
    rec = Recorder([reply(None, [call("look", cid="c1"), call("look_person", '{"name": "小明"}', cid="c2")]), reply("好")])
    brain = make(rec, toolbox({"look": look_result("A"), "look_person": look_result("B")}))
    brain.send("醒来")
    second = rec.requests[1]
    assert [m["role"] for m in second[-4:]] == ["assistant", "tool", "tool", "user"]
    texts = [p["text"] for p in second[-1]["content"] if p["type"] == "text"]
    assert texts == ["（look 的截图）", "（look_person 的截图）"]  # 按调用顺序
    assert sum(p["type"] == "image_url" for p in second[-1]["content"]) == 2


def test_tool_error_no_image_message():  # Review Focus 1
    rec = Recorder([reply(None, [call("look")]), reply("好")])
    brain = make(rec, toolbox({"look": ("没找到", True)}))
    r = brain.send("醒来")
    assert rec.requests[1][-1] == {"role": "tool", "tool_call_id": "c1", "content": "没找到"}
    assert "images" not in r


def test_image_not_added_for_unexecuted_calls():  # Review Focus 2
    def create(**kw):
        return reply(None, [call("look")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tb = toolbox({"look": look_result()})
    brain = ToolLoopBrain(client, "sys", tb, tools("look"), model="m", temperature=0.8, max_tokens=4096, max_steps=2, history=8)
    brain.send("醒来")
    turn = brain._past[-1][1]
    # 最后一次没执行的调用不留在历史里；每张图都有对应的 tool 消息
    assert len(tb.calls) == 4 and sum(m["role"] == "user" for m in turn) == 4
    assert all(m["role"] != "assistant" or not m.get("tool_calls") or all(
        any(t["tool_call_id"] == c["id"] for t in turn if t["role"] == "tool") for c in m["tool_calls"]) for m in turn)


def test_rejected_image_turns_vision_off():  # Review Focus 4
    rec = Recorder([rejected(), reply("好"), reply("好"), reply("好")])
    brain = make(rec)
    r = brain.send("第一轮", image=att())
    assert len(rec.requests) == 2 and "image_url" in json.dumps(rec.requests[0]) and "image_url" not in json.dumps(rec.requests[1])
    assert "图发不过去" in rec.requests[1][-1]["content"]
    assert brain.vision_off and "images" not in r and r["num_turns"] == 1  # 重发不算轮数
    brain.send("第二轮", image=att())  # 拒收之后给的图也忽略
    assert rec.requests[2][-1]["content"] == VISION_OFF_NOTE + "\n第二轮"
    brain.send("第三轮")
    assert rec.requests[3][-1]["content"] == "第三轮"  # 只提醒一次


def test_other_400_still_model_error():
    rec = Recorder([rejected("bad request: max_tokens too large")])
    brain = make(rec)
    with pytest.raises(ModelError):
        brain.send("在吗", image=att())
    assert brain.vision_off == ""


def test_no_image_request_unchanged():
    rec = Recorder([reply("好")])
    make(rec).send("在吗")
    assert rec.requests[0] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "在吗"}]


def test_make_session_picks_sighted_note(tmp_path):
    from skydango.brain.sessions import make_session
    from skydango.config import BrainConfig, Config
    from skydango.models.config import ModelRef, resolve
    from skydango.models.gate import ProviderGates
    from skydango.models.registry import Registry
    reg = Registry(resolve(Config()), ProviderGates(), tmp_path, environ={"DEEPSEEK_API_KEY": "k"})
    ref = ModelRef("deepseek", "deepseek-flash")

    def build(cfg, **kw):
        tb = toolbox({})  # make_session 要给 openai_tools 用：有 body.env、calling、backstage
        tb.body, tb.calling, tb.backstage = SimpleNamespace(env=object()), False, False
        return make_session(reg, ref, prompt="prompt", toolbox=tb, mcp_url="http://x", workdir=tmp_path,
                            cfg=cfg, addressee=False, on_message=None, **kw)

    assert build(BrainConfig()).system.endswith(SIGHTED_NOTE)
    off = BrainConfig()
    off.vision.enabled = False
    assert build(off).system.endswith(BLIND_NOTE)
    assert build(BrainConfig(), text_only=True).system.endswith(BLIND_NOTE)


def test_shots_saved_with_legend_and_named_in_images(tmp_path):
    rec = Recorder([reply(None, [call("look", '{"image": true}')]), reply("核对：①是小明")])
    brain = make(rec, toolbox({"look": look_result("图例 ①你自己")}), shots_dir=tmp_path / "images", wall=lambda: 1793628000.0)
    a = att()
    r = brain.send("[t] 事件：\n- 小明：在吗\n状态：身边 小明", image=a)
    files = [s["file"] for s in r["images"]]
    assert [s["src"] for s in r["images"]] == ["auto", "look"] and len(set(files)) == 2
    for s in r["images"]:
        jpg = tmp_path / "images" / s["file"]
        assert jpg.suffix == ".jpg" and jpg.read_bytes()[:2] == b"\xff\xd8"
    auto_txt = (tmp_path / "images" / files[0]).with_suffix(".txt").read_text(encoding="utf-8")
    assert a.legend in auto_txt and "小明：在吗" in auto_txt  # 图例 + 唤醒消息
    look_txt = (tmp_path / "images" / files[1]).with_suffix(".txt").read_text(encoding="utf-8")
    assert "look" in look_txt and '"image": true' in look_txt and "图例 ①你自己" in look_txt


def test_shots_not_saved_without_dir():
    rec = Recorder([reply("好")])
    r = make(rec).send("醒来", image=att())
    assert r["images"] == [{"src": "auto", "w": 960, "h": 540}]


def test_shots_stop_at_max(tmp_path):
    rec = Recorder([reply("好")] * 3)
    brain = make(rec, shots_dir=tmp_path, shots_max=2)
    got = [brain.send("醒来", image=att())["images"][0] for _ in range(3)]
    assert ["file" in g for g in got] == [True, True, False]
    assert len(list(tmp_path.glob("*.jpg"))) == 2
