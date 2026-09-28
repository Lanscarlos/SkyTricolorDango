import json
import threading
import time

from skydango.brain.trace import MAX_CHARS, BrainTrace


def asst(*blocks):
    return {"type": "assistant", "message": {"content": list(blocks)}}


def user(*blocks):
    return {"type": "user", "message": {"content": list(blocks)}}


def say(tid="t1", text="你好"):
    return {"type": "tool_use", "id": tid, "name": "mcp__sky__say", "input": {"text": text}}


def turns(t, after=0):
    return t.since(after, 0.0)["turns"]


def test_steps_in_order_and_tool_prefix_stripped():
    t = BrainTrace(wall=lambda: 100.0)
    t.begin("events", "[..] 事件：\n- 卡洛：你好")
    t.feed(asst({"type": "thinking", "thinking": "想想"}, {"type": "text", "text": " 嗯 "}, say()))
    t.feed(user({"type": "tool_result", "tool_use_id": "t1", "content": "已发送", "is_error": False}))
    [turn] = turns(t)
    assert turn["reason"] == "events" and turn["prompt"].endswith("卡洛：你好") and turn["end"] is None
    assert turn["start"] == 100.0
    assert [s["kind"] for s in turn["steps"]] == ["thinking", "text", "tool", "result"]
    assert turn["steps"][1]["text"] == "嗯"
    assert turn["steps"][2] == {"kind": "tool", "id": "t1", "name": "say", "input": {"text": "你好"}}
    assert turn["steps"][3] == {"kind": "result", "tool_use_id": "t1", "text": "已发送", "error": False}
    assert turn["tools"] == ["say"]


def test_empty_thinking_and_blank_text_are_skipped():
    t = BrainTrace()
    t.begin("heartbeat", "没有新事件")
    t.feed(asst({"type": "thinking", "thinking": ""}, {"type": "text", "text": "  "}))
    assert turns(t)[0]["steps"] == []


def test_tool_result_blocks_with_image():
    t = BrainTrace()
    t.begin("events", "x")
    content = [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}, {"type": "text", "text": "画面里有卡洛"}]
    t.feed(user({"type": "tool_result", "tool_use_id": "t1", "content": content}))
    out = t.since(0, 0.0)
    assert out["turns"][0]["steps"][0]["text"] == "[图片]\n画面里有卡洛"
    assert "AAAA" not in json.dumps(out, ensure_ascii=False)


def test_error_result_marked():
    t = BrainTrace()
    t.begin("events", "x")
    t.feed(user({"type": "tool_result", "tool_use_id": "t1", "content": "这一轮已经说得够多了", "is_error": True}))
    assert turns(t)[0]["steps"][0]["error"] is True


def test_long_text_truncated():
    t = BrainTrace()
    t.begin("events", "x")
    t.feed(asst({"type": "text", "text": "字" * (MAX_CHARS + 5)}))
    text = turns(t)[0]["steps"][0]["text"]
    assert len(text) <= MAX_CHARS + 30
    assert text.endswith(f"（截断，原长 {MAX_CHARS + 5} 字）")


def test_finish_records_result_tokens_and_seconds():
    t = BrainTrace(wall=lambda: 100.0)
    t.begin("events", "x")
    usage = {
        "input_tokens": 1460, "output_tokens": 88, "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 6369, "output_tokens_details": {"thinking_tokens": 25},
    }
    t.finish({"subtype": "success", "num_turns": 3, "total_cost_usd": 0.02, "usage": usage}, 3.2)
    [turn] = turns(t)
    assert turn["end"] == 100.0 and turn["seconds"] == 3.2 and turn["error"] is None
    assert turn["result"]["tokens"] == {"input": 1460, "output": 88, "cache_read": 10, "cache_write": 6369, "thinking": 25}
    assert turn["result"]["cost"] == 0.02 and turn["result"]["subtype"] == "success" and turn["result"]["num_turns"] == 3
    t.begin("heartbeat", "y")
    t.finish({}, 1.0)
    assert set(turns(t)[1]["result"]["tokens"].values()) == {None}


def test_fail_keeps_steps():
    t = BrainTrace()
    t.begin("events", "x")
    t.feed(asst(say()))
    t.fail("超时（120s）", 120.0)
    [turn] = turns(t)
    assert turn["error"] == "超时（120s）" and turn["end"] is not None and turn["seconds"] == 120.0
    assert len(turn["steps"]) == 1


def test_begin_interrupts_unfinished_turn():
    t = BrainTrace()
    t.begin("events", "x")
    t.begin("heartbeat", "y")
    first, second = turns(t)
    assert first["error"] == "被打断（下一轮开始了）" and first["end"] is not None
    assert second["end"] is None and second["error"] is None


def test_messages_between_turns_go_outside():
    t = BrainTrace()
    t.begin("events", "x")
    t.finish({}, 1.0)
    t.feed(user({"type": "tool_result", "tool_use_id": "t9", "content": "晚到的"}))
    got = {turn["id"]: turn for turn in turns(t)}
    assert got[1]["steps"] == []
    assert got[0]["reason"] == "outside" and got[0]["steps"][0]["text"] == "晚到的"


def test_keeps_last_50_turns_and_reports_oldest():
    t = BrainTrace()
    for i in range(55):
        t.begin("heartbeat", str(i))
        t.finish({}, 0.1)
    out = t.since(0, 0.0)
    assert [turn["id"] for turn in out["turns"]] == list(range(6, 56))
    assert out["oldest"] == 6


def test_since_is_incremental():
    t = BrainTrace()
    for _ in range(2):
        t.begin("heartbeat", "x")
        t.finish({}, 0.1)
    v = t.since(0, 0.0)["version"]
    t.begin("events", "y")
    assert [turn["id"] for turn in turns(t, v)] == [3]
    now = t.since(0, 0.0)["version"]
    out = t.since(now, 0.0)
    assert out["turns"] == [] and "turns" in out["state"]


def test_after_larger_than_version_returns_everything():
    t = BrainTrace()
    t.begin("events", "x")
    assert len(turns(t, 10**9)) == 1


def test_since_waits_for_change():
    t = BrainTrace()
    v = t.since(0, 0.0)["version"]
    threading.Timer(0.1, lambda: t.begin("events", "x")).start()
    start = time.monotonic()
    out = t.since(v, 2.0)
    assert time.monotonic() - start < 1.0
    assert out["turns"]


def test_state_comes_from_callable_and_survives_errors():
    t = BrainTrace()
    t.begin("events", "x")
    t.state = lambda: {"failures": 2}
    assert t.since(0, 0.0)["state"] == {"failures": 2, "turns": 1}

    def broken():
        raise RuntimeError("坏了")

    t.state = broken
    assert t.since(0, 0.0)["state"] == {"turns": 1}


def test_feed_survives_garbage():
    t = BrainTrace()
    t.begin("events", "x")
    for m in (None, {"type": "assistant"}, {"type": "assistant", "message": {"content": "字符串"}}, {"type": "user", "message": None}):
        t.feed(m)
    assert turns(t)[0]["steps"] == []


def test_chain_calls_fn_then_feeds_even_if_fn_raises():
    t = BrainTrace()
    t.begin("events", "x")
    seen = []

    def broken(m):
        seen.append(m)
        raise RuntimeError("日志坏了")

    handler = t.chain(broken)
    m = asst({"type": "text", "text": "嗯"})
    handler(m)
    assert seen == [m]
    assert turns(t)[0]["steps"][0]["text"] == "嗯"


def test_feed_from_another_thread_while_polling():
    t = BrainTrace()
    t.begin("events", "x")
    errors = []

    def writer():
        for i in range(200):
            t.feed(asst({"type": "text", "text": str(i)}))

    th = threading.Thread(target=writer)
    th.start()
    while th.is_alive():
        try:
            t.since(0, 0.0)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    th.join()
    assert errors == []
    assert len(turns(t)[0]["steps"]) == 200


def test_boot_marks_each_instance():
    a, b = BrainTrace(), BrainTrace()
    assert a.since(0, 0.0)["boot"] == a.since(0, 0.0)["boot"]
    assert a.since(0, 0.0)["boot"] != b.since(0, 0.0)["boot"]  # 程序重启过：网页据此清空重来
