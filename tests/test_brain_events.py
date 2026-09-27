import threading

from skydango.brain.events import EventQueue


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
