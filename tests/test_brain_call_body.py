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
    """截图时让时钟走 0.1 秒（连拍要看时间），截图次数记下来。"""

    def __init__(self, device, clock, frames):
        self.device, self.clock, self.frames, self.shots = device, clock, list(frames), 0

    def __getattr__(self, name):
        return getattr(self.device, name)

    def screenshot(self):
        self.clock.advance(0.1)
        self.shots += 1
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]


def track(tid, cls, box, last):
    return SimpleNamespace(id=tid, cls=cls, box=box, last=last, data={})


def call_body(clock, live=True, frames=None, tick=True, **kw):
    env = CallEnv()
    b, device, reader, events = body(clock, live=live, env=env, panel_mode=kw.pop("panel_mode", "always"), **kw)
    device.calls.clear()  # 启动时开面板那一下不算
    if tick:
        b.device = TickingDevice(device, clock, frames or [gray()])
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
    b, device, env, _ = call_body(clock, frames=[gray(), lit(), lit(), lit(), gray()])
    b.cfg.call.halo = True
    env.last_tracks = [track(1, "player", MID, clock() + 0.1)]
    r = b.call_out("brain", live=True)
    assert r.halo == "self" and r.self_box == MID and env.self_box == MID

    b2, device2, env2, _ = call_body(clock, frames=[gray(), lit(), lit(), lit(), gray()])
    b2.cfg.call.halo = True
    yolo_self = track(2, "self", Rect(900, 480, 80, 200), clock() + 0.1)
    env2.last_tracks = [track(1, "player", MID, clock() + 0.1), yolo_self]
    r2 = b2.call_out("brain", live=True)
    assert env2.self_box is None  # YOLO 已经稳稳认出团子：不覆盖
    assert r2.halo in ("self", "others", "none")


def test_call_halo_skipped_when_camera_just_moved(clock):
    b, device, env, _ = call_body(clock, frames=[gray(), lit(), lit(), gray()])
    b.cfg.call.halo = True
    env.last_tracks = [track(1, "player", MID, clock() + 0.1)]
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
