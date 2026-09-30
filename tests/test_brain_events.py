import threading

from skydango.brain.events import BACKGROUND, EventQueue


def test_drain_returns_events_in_order_and_empties(clock):
    q = EventQueue(clock=clock)
    q.put("chat", "聊天  懒洋洋大王：在吗")
    clock.advance(1)
    q.put("arrive", "番茄炒蛋盖饭 来到身边")
    events = q.drain()
    assert [e.kind for e in events] == ["chat", "arrive"] and events[0].t == 100.0
    assert q.drain() == [] and len(q) == 0


def test_repeated_event_is_merged_with_count(clock):
    q = EventQueue(clock=clock)
    for _ in range(3):
        q.put("error", "截图失败")
    (e,) = q.drain()
    assert e.count == 3 and e.line() == "截图失败（×3）"


def test_overflow_drops_oldest_and_says_so(clock):
    q = EventQueue(limit=3, clock=clock)
    for i in range(5):
        q.put("chat", f"消息{i}")
    events = q.drain()
    assert events[0].kind == "dropped" and "2" in events[0].text
    assert [e.text for e in events[1:]] == ["消息2", "消息3", "消息4"]


def test_wait_wakes_up_when_event_arrives():
    q = EventQueue()
    threading.Timer(0.05, lambda: q.put("chat", "hi")).start()
    assert q.wait(2.0) is True


def test_has_kind_and_last_put(clock):
    q = EventQueue(clock=clock)
    assert not q.has("chat") and q.last_put == float("-inf")
    q.put("chat", "x")
    assert q.has("chat") and not q.has("arrive") and q.last_put == 100.0


def test_subscribers_hear_every_event(clock):
    q = EventQueue(clock=clock)
    heard = []
    q.subscribe(heard.append)
    q.subscribe(lambda kind: 1 / 0)  # 订阅者出错不影响放事件
    q.put("arrive", "懒懒 来到身边")
    q.put("chat", "x")
    assert heard == ["arrive", "chat"] and len(q) == 2


# ---- 背景事件：自己不叫醒大脑，攒着等下一次醒来（2026-09-29 真机 5 分钟醒 42 次，41 次是背景事件） ----


def test_background_kinds():
    assert {"stranger", "leave", "return", "request", "accepted", "holding", "released", "scene_change", "reflex"} <= BACKGROUND
    assert not {"chat", "owner_command", "arrive", "approach", "gesture", "notice", "task_done", "error"} & BACKGROUND


def test_urgent_and_oldest_background(clock):
    q = EventQueue(clock=clock)
    assert not q.urgent() and q.oldest_background() is None
    q.put("stranger", "身边来了陌生人（1 个，头顶没有名字）")
    clock.advance(5)
    q.put("leave", "懒懒 走开了（5 秒没看到名字）", who="懒懒")
    assert not q.urgent() and q.oldest_background() == 100.0
    q.put("chat", "聊天  懒懒：「在吗」")
    assert q.urgent()
    q.drain()
    assert not q.urgent() and q.oldest_background() is None


def test_leave_then_return_cancel_out(clock):
    q = EventQueue(clock=clock)
    q.put("leave", "懒懒 走开了（5 秒没看到名字）", who="懒懒")
    q.put("stranger", "陌生人都走开了")
    q.put("return", "懒懒 回来了", who="懒懒")
    assert [e.kind for e in q.drain()] == ["stranger"]


def test_return_then_leave_cancel_out(clock):
    q = EventQueue(clock=clock)
    q.put("return", "懒懒 回来了", who="懒懒")
    q.put("leave", "懒懒 走开了（5 秒没看到名字）", who="懒懒")
    assert len(q) == 0 and q.drain() == []


def test_cancel_only_matches_the_same_person(clock):
    q = EventQueue(clock=clock)
    q.put("leave", "懒懒 走开了（5 秒没看到名字）", who="懒懒")
    q.put("return", "番茄 回来了", who="番茄")
    assert [e.who for e in q.drain()] == ["懒懒", "番茄"]


def test_only_latest_stranger_event_is_kept(clock):
    q = EventQueue(clock=clock)
    q.put("stranger", "身边来了陌生人（1 个，头顶没有名字）")
    q.put("chat", "聊天  懒懒：「在吗」")
    q.put("stranger", "陌生人都走开了")
    q.put("stranger", "身边来了陌生人（2 个，头顶没有名字）")
    assert [e.text for e in q.drain()] == ["聊天  懒懒：「在吗」", "身边来了陌生人（2 个，头顶没有名字）"]


def test_same_background_text_merges_even_when_not_adjacent(clock):
    q = EventQueue(clock=clock)
    q.put("request", "懒懒 发起了牵手")
    q.put("accepted", "身体按规则接受了 懒懒 的牵手")
    q.put("request", "懒懒 发起了牵手")
    events = q.drain()
    assert [e.line() for e in events] == ["懒懒 发起了牵手（×2）", "身体按规则接受了 懒懒 的牵手"]


def test_urgent_text_still_only_merges_when_adjacent(clock):
    q = EventQueue(clock=clock)
    q.put("chat", "聊天  懒懒：「哈哈」")
    q.put("chat", "聊天  懒懒：「在吗」")
    q.put("chat", "聊天  懒懒：「哈哈」")
    assert len(q.drain()) == 3  # 聊天是按顺序的原话，不能挪位置合并


def test_repeated_background_event_keeps_its_first_time(clock):
    q = EventQueue(clock=clock)
    for _ in range(30):  # dry-run 下没接的请求每秒报一次：不能把兜底叫醒一直往后推
        q.put("request", "懒懒 发起了牵手")
        clock.advance(1)
    assert q.oldest_background() == 100.0
    (e,) = q.drain()
    assert e.count == 30


def test_flipping_stranger_count_keeps_first_time_and_no_count(clock):
    q = EventQueue(clock=clock)
    for i in range(30):  # 陌生人数量 0 / 1 来回跳：兜底叫醒不能一直往后推，也不显示"×N"
        q.put("stranger", "身边来了陌生人（1 个，头顶没有名字）" if i % 2 == 0 else "陌生人都走开了")
        clock.advance(1)
    assert q.oldest_background() == 100.0
    (e,) = q.drain()
    assert e.line() == "陌生人都走开了"
