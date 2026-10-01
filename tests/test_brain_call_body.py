"""按 Q 喊一声：身体那一边（spec 2026-10-01-q-call §2、§3.2）。"""

from types import SimpleNamespace

import numpy as np
from test_brain_body import FakeEnv, body

from skydango.brain.calling import CallResult, event_text, halo_text, seen_text, tool_text, wait_result
from skydango.device.base import LINUX_KEY_Q
from skydango.vision.bubbles import Rect
from skydango.vision.halo import head_region
from skydango.vision.people import CallSeen, Seen

W, H = 1920, 1080
MID = Rect(940, 500, 60, 150)


def gray(v=80):
    return np.full((H, W, 3), v, np.uint8)


def lit(box=MID):
    f = gray()
    r = head_region(box, W, H)
    f[r.y:r.y2, r.x:r.x2] = 140
    return f


class CallEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.last_tracks = []
        self.self_box = None
        self.results = {}
        self.unnamed_n = 0

    def called(self, at, *, by_self=True):
        self.calls.append(at)

    def call_result(self, at):
        return self.results.get(at)

    def unnamed(self, now):
        return self.unnamed_n


class TickingDevice:
    """截图时让时钟走 step 秒（连拍要看时间），截图次数记下来；press：按键本身花的时间（adb 往返）。"""

    def __init__(self, device, clock, frames, step=0.1, press=0.0):
        self.device, self.clock, self.frames, self.shots = device, clock, list(frames), 0
        self.step, self.press = step, press

    def __getattr__(self, name):
        return getattr(self.device, name)

    def hw_key(self, code):
        self.device.hw_key(code)
        self.clock.advance(self.press)

    def screenshot(self):
        self.clock.advance(self.step)
        self.shots += 1
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]


def track(tid, cls, box, last):
    return SimpleNamespace(id=tid, cls=cls, box=box, last=last, data={})


def call_body(clock, live=True, frames=None, tick=True, env=None, step=0.1, press=0.0, **kw):
    env = env or CallEnv()
    b, device, reader, events = body(clock, live=live, env=env, panel_mode=kw.pop("panel_mode", "always"), **kw)
    device.calls.clear()  # 启动时开面板那一下不算
    if tick:
        b.device = TickingDevice(device, clock, frames or [gray()], step, press)
    return b, device, env, events


def test_call_presses_q_inside_borrow_and_opens_window(clock):
    b, device, env, _ = call_body(clock)
    r = b.call_out("brain", live=True)
    assert not r.refused and not r.dry and r.halo == "off"
    keys = [c[1] for c in device.calls if c[0] == "hw_key"]
    assert keys == [46, LINUX_KEY_Q, 46]  # 先关聊天面板（开着按 Q 没反应），按 Q，再开回来
    assert env.calls == [r.at] and b.last_call is r


def test_call_refused_when_ime_open(clock):
    b, device, env, _ = call_body(clock)
    device.shown = True
    r = b.call_out("brain", live=True)
    assert "输入框开着" in r.refused
    assert ("hw_key", LINUX_KEY_Q) not in device.calls and env.calls == []


def test_call_min_gap_and_owner_window(clock):
    b, device, env, _ = call_body(clock)
    b.call_out("brain", live=True)
    clock.advance(10)
    r = b.call_out("brain", live=True)
    assert "刚喊过" in r.refused
    b._owner_window_until = clock() + 60  # 卡洛 # 命令窗口里大脑不受限
    r = b.call_out("brain", live=True)
    assert not r.refused and r.reason == "brain-owner"
    clock.advance(5)
    assert "刚喊过" in b.call_out("manual", live=True).refused  # 手动的不放宽


def test_call_dry_run_does_not_press(clock):
    b, device, env, _ = call_body(clock, live=False)
    r = b.call_out("brain")
    assert r.dry and not r.refused
    assert ("hw_key", LINUX_KEY_Q) not in device.calls and env.calls == []
    clock.advance(5)
    assert "刚喊过" in b.call_out("brain").refused  # dry-run 也计间隔


def test_call_refused_while_blackout_or_disabled(clock):
    b, device, env, _ = call_body(clock)
    b.blackout = True
    assert "画面黑着" in b.call_out("brain", live=True).refused
    b.blackout = False
    b.cfg.call.enabled = False
    assert b.call_out("brain", live=True).refused == "没开"


def test_call_burst_stops_when_clock_frozen(clock):
    b, device, env, _ = call_body(clock, tick=False)
    shots = []
    real = device.screenshot
    device.screenshot = lambda: (shots.append(1), real())[1]
    b.call_out("brain", live=True)
    assert len(shots) <= int(b.cfg.call.burst * 100) + 20  # 连拍有张数上限（另外几张是聊天面板借还时看开没开）


def test_call_halo_writes_self_box_only_without_yolo_self(clock):
    # 聊天面板关着（auto 模式平时关着）：感知层的框和画面对得上
    b, device, env, _ = call_body(clock, frames=[gray(), lit(), lit(), lit(), gray()], panel_mode="auto")
    b.cfg.call.halo = True
    env.last_tracks = [track(1, "player", MID, clock())]
    r = b.call_out("brain", live=True)
    assert r.halo == "self" and r.self_box == MID and env.self_box == MID

    b2, device2, env2, _ = call_body(clock, frames=[gray(), lit(), lit(), lit(), gray()], panel_mode="auto")
    b2.cfg.call.halo = True
    env2.last_tracks = [track(1, "player", MID, clock()), track(2, "self", Rect(300, 480, 80, 200), clock())]
    b2.call_out("brain", live=True)
    assert env2.self_box is None  # YOLO 已经稳稳认出一个团子：不覆盖


def test_call_halo_self_box_overlapping_player_is_one_person(clock):
    # 团子身上同时有 YOLO 的 self 框和几乎重合的 player 框：是同一个人，不是"别人也在喊"
    b, device, env, _ = call_body(clock, frames=[gray(), lit(), lit(), lit(), gray()], panel_mode="auto")
    b.cfg.call.halo = True
    env.last_tracks = [track(1, "player", MID, clock()), track(2, "self", Rect(942, 502, 60, 150), clock())]
    assert b.call_out("brain", live=True).halo == "self"


def test_call_halo_skipped_when_chat_panel_was_open(clock):
    # 面板开着：借面板关掉时画面横移、感知层这段时间没新帧，框对不上 → 放弃，不报"没看到"
    b, device, env, _ = call_body(clock, frames=[gray(), lit(), lit(), lit(), gray()])
    b.cfg.call.halo = True
    env.last_tracks = [track(1, "player", MID, clock())]
    assert b.call_out("brain", live=True).halo == "skipped" and env.self_box is None


def test_call_halo_press_time_taken_before_the_key(clock):
    # 按键要一次 adb 往返（0.15 s）：光圈最亮那一下离"按完"很近，按完才记时间会把它算到窗口之前
    b, device, env, _ = call_body(clock, frames=[gray(), lit(), gray()], panel_mode="auto", step=0.05, press=0.15)
    b.cfg.call.halo = True
    env.last_tracks = [track(1, "player", MID, clock())]
    assert b.call_out("brain", live=True).halo == "self"


def test_call_refused_without_perception(clock):
    # 整图 OCR（EnvWatcher）也有 called()，但收不到呼喊窗口：手动直接 POST 也别真的按 Q
    class OcrEnv(FakeEnv):
        def called(self, at, *, by_self=True):
            raise AssertionError("不该开窗口")

    b, device, env, _ = call_body(clock, env=OcrEnv())
    assert "没开感知层" in b.call_out("manual", live=True).refused
    assert ("hw_key", LINUX_KEY_Q) not in device.calls


def test_call_halo_skipped_when_camera_just_moved(clock):
    b, device, env, _ = call_body(clock, frames=[gray(), lit(), lit(), gray()], panel_mode="auto")
    b.cfg.call.halo = True
    env.last_tracks = [track(1, "player", MID, clock())]
    b._camera_moved_at = clock()
    assert b.call_out("brain", live=True).halo == "skipped" and env.self_box is None


def test_status_line_after_call(clock):
    b, device, env, _ = call_body(clock)
    assert "上次喊" not in b.status()
    b.last_call = CallResult(clock() - 130, "brain", halo="self",
                             seen=CallSeen(clock() - 130, {"小明": Seen("右边", "远")}, ended=True))
    line = [p for p in b.status().split(" / ") if p.startswith("上次喊")][0]
    assert line == "上次喊：2 分钟前（认出小明；你自己在中间）"


def test_texts():
    seen = CallSeen(1.0, {"小明": Seen("右边", "远"), "懒洋洋大王": Seen("左边", "近"), "小红": Seen("左边", None, False),
                          "番茄": Seen("前面", None)}, unnamed=1, ended=True)
    assert seen_text(seen) == "认出 小明（右边·远）、懒洋洋大王（左边·近）、番茄（前面）；小红在画面外（左边）；还有 1 个没挂名字的人"
    assert seen_text(CallSeen(1.0, {}, ended=True)) == "没看到谁的名字"
    assert halo_text("self") == "光圈：认出了你自己（中间）"
    assert halo_text("others") == "光圈：别人也在喊，没认出你自己"
    assert halo_text("off") == halo_text("skipped") == ""
    r = CallResult(1.0, "brain", halo="self", seen=seen)
    assert tool_text(r) == f"喊了一声：{seen_text(seen)}。光圈：认出了你自己（中间）。"
    assert tool_text(CallResult(1.0, "brain")) == "喊了一声：没等到结果（感知层暂停了？）。"
    assert event_text(CallSeen(1.0, {}, unnamed=2, ended=True)) == "你下意识喊了一声：没看到谁的名字；还有 2 个没挂名字的人。"


def test_wait_result_polls_and_gives_up(clock):
    b, device, env, _ = call_body(clock, tick=False)
    b.call = lambda fn, timeout=None: fn()
    r = CallResult(5.0, "brain")
    polls = []

    def sleep(s):
        polls.append(s)
        clock.advance(s)
        if len(polls) == 3:
            env.results[5.0] = CallSeen(5.0, {}, ended=True)

    assert wait_result(b, r, sleep=sleep).seen is not None and len(polls) == 3
    r2 = wait_result(b, CallResult(9.0, "brain"), sleep=lambda s: clock.advance(s))
    assert r2.seen is None  # 一直没有：window + 4 秒后放弃

    frozen = wait_result(b, CallResult(9.0, "brain"), sleep=lambda s: None)  # 时钟不走也不会死等
    assert frozen.seen is None


# ---- 自动兜底（spec §2.3） ----
def left(b, clock, name="小明", ago=5.0, unnamed=1):
    b._left_at[name] = clock() - ago
    b._nearby = set()
    b.env.unnamed_n = unnamed


def pressed(device):
    return sum(c == ("hw_key", LINUX_KEY_Q) for c in device.calls)


def test_auto_call_after_friend_leaves_with_unnamed_people(clock):
    b, device, env, events = call_body(clock)
    left(b, clock)
    b._watch_call(clock())
    assert pressed(device) == 1 and b.last_call.reason == "auto"
    b._watch_call(clock())
    assert not [e for e in events.drain() if e.kind == "call"]  # 窗口还没结束
    env.results[b.last_call.at] = CallSeen(b.last_call.at, {"小明": Seen("右边", "远")}, unnamed=0, ended=True)
    b._watch_call(clock())
    calls = [e for e in events.drain() if e.kind == "call"]
    assert [e.text for e in calls] == ["你下意识喊了一声：认出 小明（右边·远）。"]
    assert b.last_call.seen is not None


def test_auto_call_once_per_leave_and_quota(clock):
    b, device, env, _ = call_body(clock)
    left(b, clock)
    b._watch_call(clock())
    env.results[b.last_call.at] = CallSeen(b.last_call.at, {}, ended=True)
    clock.advance(25)
    left(b, clock, ago=10.0)  # 又走开了一次（新的走开时间）：可以再喊
    b._left_at["小明"] = b._auto_called["小明"]  # ……但还是同一次走开：不喊
    b._watch_call(clock())
    b._watch_call(clock())
    assert pressed(device) == 1
    for i in range(3):  # 每次都是新的走开
        clock.advance(25)
        left(b, clock, name=f"好友{i}")
        b._watch_call(clock())
        if b.last_call is not None:
            env.results[b.last_call.at] = CallSeen(b.last_call.at, {}, ended=True)
        b._watch_call(clock())
    assert pressed(device) == 3  # 10 分钟最多 3 次


def test_auto_call_skipped_when_busy(clock):
    for busy in ("skill", "bubble", "request", "brain", "ime", "candle"):
        b, device, env, _ = call_body(clock)
        left(b, clock)
        if busy == "skill":
            b.skills.active = object()
        elif busy == "bubble":
            b._bubble_at = clock()
        elif busy == "request":
            from skydango.game.social import Request

            env.requests = {"小红": Request("小红", "hand", (0, 0), clock())}
        elif busy == "brain":
            b.brain_busy = lambda: True
        elif busy == "ime":
            device.shown = True
        elif busy == "candle":
            b._raised = (1, (0, 0), clock())
        b._watch_call(clock())
        assert pressed(device) == 0, busy


def test_auto_call_not_when_friend_back_or_nobody_unnamed(clock):
    b, device, env, _ = call_body(clock)
    left(b, clock, unnamed=0)
    b._watch_call(clock())
    left(b, clock)
    b._nearby = {"小明"}  # 已经回来了
    b._watch_call(clock())
    left(b, clock, ago=40.0)  # 走开太久（auto_after_leave 30 秒）
    b._watch_call(clock())
    b.cfg.call.auto = False
    left(b, clock)
    b._watch_call(clock())
    assert pressed(device) == 0


def test_auto_call_dry_run_logs_only(clock, caplog):
    b, device, env, _ = call_body(clock, live=False)
    left(b, clock)
    with caplog.at_level("INFO"):
        b._watch_call(clock())
    assert pressed(device) == 0 and "会喊一声" in caplog.text
    b._watch_call(clock())
    assert caplog.text.count("会喊一声") == 1  # 同一次走开只判一次


def test_auto_call_gives_up_waiting(clock):
    b, device, env, events = call_body(clock)
    left(b, clock)
    b._watch_call(clock())
    clock.advance(b.cfg.call.window + 11)
    b._watch_call(clock())
    assert b._pending_auto is None and not [e for e in events.drain() if e.kind == "call"]


def test_auto_call_not_blocked_by_stale_request(clock):
    # 好友带着请求圈走开了：他的请求留在 env.requests 里不会被清，过了 social.max_age 就别再拦自动喊
    from skydango.game.social import Request

    b, device, env, _ = call_body(clock)
    left(b, clock)
    env.requests = {"小明": Request("小明", "hand", (0, 0), clock() - b.cfg.social.max_age - 1)}
    b._watch_call(clock())
    assert pressed(device) == 1
    b2, device2, env2, _ = call_body(clock)
    left(b2, clock)
    env2.requests = {"小明": Request("小明", "hand", (0, 0), clock() - 1)}  # 新鲜的请求照样拦
    b2._watch_call(clock())
    assert pressed(device2) == 0
