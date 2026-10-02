"""团子自己别被当成陌生人（docs/progress/2026-10-03-plan.md ① 第 1 项）。

10-02 晚真机：YOLO 在团子身上常常只出 player（0.7~0.85）、self 只有 0.3 左右或干脆没有；以前只在同一帧有 self 框时才去掉
团子身上的 player 轨迹，一帧没 self 就过 stranger_after 判成陌生人（图鉴 50 张"陌生人"里约 26 张是团子）。
团子在屏幕上的位置随聊天面板开关差约 400 px（关着 x≈550、开着 x≈950），所以按面板开关分别记。
"""

from test_perception import FakeDetector, FakeOcr, frame, player, tag, watcher

from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.perception import DANGO_MEMORY, same_body

ME = Rect(550, 470, 165, 395)  # 团子框（10-02 晚面板关着时的位置）
ON_ME = Rect(549, 468, 287, 399)  # 团子身上的 player 框：带着斗篷比 self 框宽一圈（IoU ≈ 0.57）
ME_OPEN = Rect(950, 470, 170, 390)  # 面板开着时团子在这


def me(box=ME, score=0.9):
    return Detection("self", box, score)


def on_me(box=ON_ME, score=0.8):
    return Detection("player", box, score)


def run(w, det, frames, start=0.0, step=0.25, panel=False):
    """依次喂 frames（每帧一组检测），返回最后的时间。"""
    t = start
    for dets in frames:
        det.frames = [dets]
        w.process(frame(), t, panel)
        t += step
    return t - step


def test_same_body():
    assert same_body(ON_ME, ME)  # 宽一圈的斗篷框
    assert same_body(ME, ME)
    assert not same_body(Rect(580, 520, 90, 220), ME)  # 团子身后站着个矮一截的人：框在团子框里，但高度差太多
    assert not same_body(Rect(900, 470, 165, 395), ME)
    # play-1001-1 103.25 s / 105 s：贴在团子身前的陌生人（黑影 / 橙斗篷），框比团子宽 2.5 倍、把团子框整个包住（IoU 0.26 / 0.33）
    assert not same_body(Rect(1160, 188, 590, 760), Rect(1393, 394, 211, 564))
    assert not same_body(Rect(1138, 388, 498, 692), Rect(1411, 399, 199, 568))


def test_player_on_dango_without_self_box_is_never_a_stranger():
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[me(), on_me()]])
    t = run(w, det, [[on_me()]] * 24, start=t + 0.25)  # 6 秒都没出 self：以前 1 秒后就判陌生人
    assert w.strangers(t) == 0
    assert w.unnamed(t) == 0  # 自动喊一声靠它：团子不算没挂名字的人
    assert all(p.kind != "stranger" for p in w.people(t))


def test_dango_position_is_remembered_per_panel_state():
    det = FakeDetector()
    w = watcher(det, track_buffer=0.5)
    t = run(w, det, [[me()]])
    t = run(w, det, [[me(ME_OPEN)]], start=t + 1, panel=True)
    t = run(w, det, [[on_me()]] * 8, start=t + 1)  # 面板关着：团子在左边那个位置
    assert w.strangers(t) == 0
    t = run(w, det, [[on_me(Rect(948, 468, 280, 395))]] * 8, start=t + 1, panel=True)  # 面板开着：在右边那个位置
    assert w.strangers(t) == 0
    t = run(w, det, [[on_me(Rect(948, 468, 280, 395))]] * 8, start=t + 1)  # 面板关着时右边那个位置上的不是团子
    assert w.strangers(t) == 1


def test_memory_expires():
    det = FakeDetector()
    w = watcher(det, track_buffer=0.5)
    t = run(w, det, [[me()]])
    t = run(w, det, [[on_me()]] * 8, start=t + DANGO_MEMORY + 1)  # 半分钟以上没见过团子框：不靠旧位置瞎猜
    assert w.strangers(t) == 1


def test_mark_sticks_while_no_self_box_even_if_zooming():
    """镜头拉近时团子框慢慢变大、离记住的位置越来越远：轨迹一直接着就一直是团子。"""
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[me(), on_me()]])
    frames = []
    for i in range(1, 25):
        grow = 12 * i
        frames.append([on_me(Rect(549 - grow // 2, 468 - grow, 287 + grow, 399 + grow))])
    t = run(w, det, frames, start=t + 0.25)
    assert w.strangers(t) == 0


def test_mark_dropped_when_self_box_is_elsewhere():
    """self 框明明在别处、这条轨迹也不在记住的位置上：它不是团子（好友从团子身上走过、轨迹跟着他走了）。"""
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[me(), on_me()]])
    walk = [[on_me(Rect(549 + 40 * i, 468, 287, 399))] for i in range(1, 12)]  # 一路往右走开
    t = run(w, det, walk, start=t + 0.25)
    t = run(w, det, [[me(), on_me(Rect(549 + 40 * 12, 468, 287, 399))]] * 6, start=t + 0.25)
    assert w.strangers(t) == 1


def test_mark_dropped_when_track_walks_off_without_self_box():
    """一直没有 self 框时，轨迹横着离开记住的团子位置超过一个框宽：它不是团子（是从团子身上走过的人）。"""
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[me(), on_me()]])
    t = run(w, det, [[on_me()]] * 4, start=t + 0.25)
    walk = [[on_me(Rect(549 + 30 * i, 468, 287, 399))] for i in range(1, 16)]
    t = run(w, det, walk, start=t + 0.25)
    assert w.strangers(t) == 1


def test_mark_dropped_when_box_suddenly_swells():
    """团子身上那条轨迹被突然挡到团子身前的陌生人接走（框一下子胀成两倍多宽）：标记不跟过去（play-1001-1 103 s）。"""
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[me(Rect(1411, 399, 199, 568)), on_me(Rect(1400, 395, 230, 570))]])
    t = run(w, det, [[on_me(Rect(1400, 395, 230, 570))]] * 2, start=t + 0.25)
    t = run(w, det, [[on_me(Rect(1290, 330, 420, 700))], [on_me(Rect(1160, 188, 590, 760))]] + [[on_me(Rect(1160, 188, 590, 760))]] * 6,
            start=t + 0.25)
    assert w.strangers(t) == 1


def test_stranger_wrapped_around_dango_is_not_dango():
    """陌生人贴在团子身前、框把团子框整个包住：不是团子（点亮陌生人要靠他）。"""
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[me(Rect(1411, 399, 199, 568))]])
    t = run(w, det, [[on_me(Rect(1138, 388, 498, 692))]] * 8, start=t + 0.25)
    assert w.strangers(t) == 1


def test_person_behind_dango_is_not_dango():
    det = FakeDetector()
    w = watcher(det)
    behind = player(590, y=520, w=90, h=220)  # 团子身后远一点的人：框在团子框里，矮一截
    t = run(w, det, [[me(), behind]] * 8)
    assert w.strangers(t) == 1


def test_friend_tag_over_dango_is_not_hung_on_dangos_box():
    """好友站在团子正后方：名字标签在团子头顶，YOLO 这一帧没出 self、只在团子身上出 player 框。
    标签不能挂到团子身上（look_person 会裁团子自己的背影给大脑看），好友照样算在身边。"""
    det = FakeDetector()
    w = watcher(det, ocr=FakeOcr({110: "懒洋洋大王"}))
    t = run(w, det, [[me(), on_me()]])
    t = run(w, det, [[on_me(), tag(640, 110, y=400)]] * 4, start=t + 0.25)
    assert w.nearby(t) == ["懒洋洋大王"]
    assert [p for p in w.people(t) if p.name] == []
