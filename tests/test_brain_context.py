from skydango.brain.context import Context, wake_message
from skydango.brain.events import Event
from skydango.config import BrainConfig

IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "x"}}


def ctx(**kw):
    loads = []

    def memory():
        loads.append(1)
        return f"记忆{len(loads)}"

    return Context(BrainConfig(**kw), "规则", memory), loads


def test_system_has_two_cached_blocks():
    c, _ = ctx()
    blocks = c.system()
    assert [b["text"] for b in blocks] == ["规则", "记忆1"]
    assert all(b["cache_control"] == {"type": "ephemeral"} for b in blocks)


def test_prune_keeps_latest_images_including_tool_results():
    c, _ = ctx(keep_images=2, prune_at=4)
    c.add_user([{"type": "text", "text": "a"}, dict(IMG)])
    c.add_assistant([{"type": "tool_use", "id": "t1", "name": "look", "input": {}}])
    c.add_tool_results([{"type": "tool_result", "tool_use_id": "t1", "content": [dict(IMG), {"type": "text", "text": "名字"}]}])
    c.add_assistant([{"type": "text", "text": "看到了"}])
    c.add_user([{"type": "text", "text": "b"}, dict(IMG)])
    assert c.prune() == 0  # 3 张，还没到 prune_at
    c.add_user([dict(IMG)])
    assert c.prune() == 2
    assert c.messages[0]["content"][1] == {"type": "text", "text": "[截图已移除]"}
    assert c.messages[2]["content"][0]["content"][0] == {"type": "text", "text": "[截图已移除]"}
    assert len(c.images()) == 2


def test_reset_starts_from_summary_and_rereads_memory():
    c, _ = ctx()
    c.add_user("x")
    c.reset("在雨林，懒懒在旁边")
    assert len(c.messages) == 1 and "在雨林" in c.messages[0]["content"][0]["text"]
    assert c.system()[1]["text"] == "记忆2"


def test_drop_last_assistant_only_drops_assistant():
    c, _ = ctx()
    c.add_user("x")
    c.add_assistant([{"type": "text", "text": "y"}])
    c.drop_last_assistant()
    c.drop_last_assistant()
    assert [m["role"] for m in c.messages] == ["user"]


def test_wake_message_lists_events_status_and_images():
    msg = wake_message("18:32:05", [Event("chat", "聊天  懒懒：在吗", 0.0)], "面板开", [IMG])
    assert msg[0]["text"] == "[18:32:05] 事件：\n- 聊天  懒懒：在吗\n状态：面板开"
    assert msg[1] is IMG
    assert wake_message("18:33:00", [], "面板开", [])[0]["text"].startswith("[18:33:00] 没有新事件")
