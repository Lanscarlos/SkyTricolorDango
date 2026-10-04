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


# ---- self 框的位置连续性（10-03 晚真机复盘：docs/progress/2026-10-03-plan.md「认人」）----
# 21:28~21:46 团子和番茄并排坐着、聊天面板开着：团子在右边（self 0.4~0.9 / player 0.88~0.95），YOLO 偶尔把高分 self 框
# （0.70~0.75）打在番茄身上（hard/213242、213406）。以前团子当场被摘掉标记、记住的位置被番茄覆盖，之后团子的新轨迹
# 全被判成陌生人（图鉴"陌生人" 105 张里约 70 张是团子）
SIT_ME = Rect(1430, 340, 190, 442)
SIT_ON_ME = Rect(1436, 328, 187, 445)
SIT_FRIEND = Rect(1090, 457, 259, 415)
JUMP = Rect(1052, 447, 302, 416)  # 打在番茄身上的 self 框


def friend(box=SIT_FRIEND, score=0.7):
    return Detection("player", box, score)


def dango_strangers(w, t):
    return [p for p in w.people(t) if p.kind == "stranger" and same_body(p.box, SIT_ON_ME)]


def test_self_box_jumping_to_neighbour_does_not_unmark_dango():
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[me(SIT_ME), on_me(SIT_ON_ME, 0.9), friend()]] * 4, panel=True)
    cycle = [[me(JUMP, 0.75), on_me(SIT_ON_ME, 0.9)]] + [[on_me(SIT_ON_ME, 0.9), friend()]] * 3
    t = run(w, det, cycle * 8, start=t + 0.25, panel=True)  # 8 秒，每秒跳一次
    assert dango_strangers(w, t) == []


def test_weak_self_on_neighbour_does_not_steal_dango():
    """番茄身上只有低分 self（0.3）、和她的 player 框重合：不能把她提升成团子、把团子挤掉。"""
    det = FakeDetector()
    w = watcher(det)
    t = run(w, det, [[me(SIT_ME), on_me(SIT_ON_ME, 0.9), friend()]] * 4, panel=True)
    weak = [[on_me(SIT_ON_ME, 0.9), friend(), Detection("self", SIT_FRIEND, 0.3)]]
    t = run(w, det, (weak + [[on_me(SIT_ON_ME, 0.9), friend()]]) * 12, start=t + 0.25, panel=True)
    assert dango_strangers(w, t) == []


def test_self_box_that_stays_somewhere_new_is_believed():
    """self 框在新位置连着待了 SELF_JUMP_HOLD 秒以上：团子真的换了地方（比如走开了），照新位置认。"""
    from skydango.vision.perception import SELF_JUMP_HOLD

    det = FakeDetector()
    w = watcher(det, track_buffer=0.5)
    t = run(w, det, [[me(), on_me()]] * 2)
    new_me, new_on = Rect(1300, 470, 165, 395), Rect(1299, 468, 287, 399)
    n = int(SELF_JUMP_HOLD / 0.25) + 3
    t = run(w, det, [[me(new_me)]] * n, start=t + 1)
    t = run(w, det, [[on_me(new_on)]] * 8, start=t + 0.25)
    assert w.strangers(t) == 0


def test_one_self_prefers_box_near_dango():
    from skydango.vision.perception import one_self

    dets = [Detection("self", JUMP, 0.75), Detection("self", SIT_ME, 0.45)]
    out = one_self(dets, near=[SIT_ON_ME])
    assert [d.cls for d in out] == ["player", "self"]
    assert [d.cls for d in one_self(dets)] == ["self", "player"]  # 没有参考位置：照旧留分数最高的


def test_dango_sliding_with_panel_toggle_is_still_dango():
    """聊天面板开关时画面横移的 2 秒动画里，团子的 self 框一帧挪 100~150 px（q-call-20260930-c 16.0~16.4 s、-b 38.4~38.6 s）：
    不能当成"跳到别人身上"。身体按 C 时先通知（camera_moved panel），面板标志比动画晚 0.1~0.3 秒才翻。
    整幅画面一起横移：平移估计（这里直接给）让团子身上那条 player 轨迹跟着滑过去。"""
    small = Rect(1009, 536, 105, 230)  # 镜头离得远，团子框小（q-call-c）
    on_small = Rect(1006, 532, 156, 236)  # 团子身上同时还有个宽一圈的 player 框（-c 的 #5）：轨迹活了一秒多
    det = FakeDetector()
    w = watcher(det, track_pan=True)
    pans = []
    w._pan_step = lambda frame, panel: pans.pop(0) if pans else (0.0, 0.0)
    t = run(w, det, [[me(small), on_me(on_small, 0.45)]] * 8, panel=True)
    w.camera_moved(t + 0.1, "panel")
    pans[:] = [(-145.0, 0.0)] * 3
    slide = [[me(Rect(small.x - 145 * i, small.y, small.w, small.h)),
              on_me(Rect(on_small.x - 145 * i, on_small.y, on_small.w, on_small.h), 0.45)] for i in range(1, 4)]
    t = run(w, det, slide, start=t + 0.25, step=0.36, panel=True)  # 标志还没翻
    end, on_end = Rect(small.x - 435, small.y, small.w, small.h), Rect(on_small.x - 435, on_small.y, on_small.w, on_small.h)
    t = run(w, det, [[me(end), on_me(on_end, 0.45)]] * 3 + [[on_me(on_end, 0.9)]] * 8, start=t + 0.36)
    assert w.strangers(t) == 0


# ---- 评审（10-04）后补 ----
def test_turning_camera_does_not_unmark_dango_as_pivot():
    """评审 1：团子是镜头支点，转镜头时背景平移、团子在屏幕上不动。先拉近（框离开记住的位置），再转、一直不出 self：
    团子轨迹的连续性不能只拿"按背景平移挪过的上一帧框"比（团子自己就对不上了）。"""
    det = FakeDetector()
    w = watcher(det, track_pan=True)
    pans = []
    w._pan_step = lambda frame, panel: pans.pop(0) if pans else (0.0, 0.0)
    small, on_small = Rect(1009, 536, 105, 230), Rect(1006, 532, 156, 236)
    t = run(w, det, [[me(small), on_me(on_small)]] * 4)
    grow = [[on_me(Rect(on_small.x - 6 * i, on_small.y - 14 * i, on_small.w + 12 * i, on_small.h + 14 * i))] for i in range(1, 11)]
    t = run(w, det, grow, start=t + 0.25)  # 拉近：框慢慢变大
    big = grow[-1][0].box
    pans[:] = [(-150.0, 0.0)] * 6
    t = run(w, det, [[on_me(big)]] * 12, start=t + 0.25)  # 转镜头：团子不动
    assert w.strangers(t) == 0


def test_demoted_self_on_neighbour_with_own_player_box_is_not_a_ghost():
    """评审 2：邻居身上 YOLO 常常同时出 player 和 self（按类别各自 NMS）：self 改成 player 后和她自己的 player 框重复，
    不能多出一条没挂名字的"幽灵陌生人"。"""
    det = FakeDetector()
    w = watcher(det, ocr=FakeOcr({110: "懒洋洋大王"}))
    head = tag(1165, 110, y=400)
    t = run(w, det, [[me(SIT_ME), on_me(SIT_ON_ME, 0.9), friend(), head]] * 4, panel=True)
    cycle = [[me(JUMP, 0.75), on_me(SIT_ON_ME, 0.9), friend(), head]] + [[on_me(SIT_ON_ME, 0.9), friend(), head]] * 3
    t = run(w, det, cycle * 8, start=t + 0.25, panel=True)
    assert w.nearby(t) == ["懒洋洋大王"] and w.strangers(t) == 0


def test_relocation_with_flickering_self_is_believed():
    """评审 3：团子身上的 self 常常断帧：在新位置隔一帧出一次，满 SELF_JUMP_HOLD 也要信（不能每次断帧就重新计时）。"""
    from skydango.vision.perception import SELF_JUMP_HOLD

    det = FakeDetector()
    w = watcher(det, track_buffer=0.5)
    t = run(w, det, [[me(), on_me()]] * 2)
    new_me, new_on = Rect(1300, 470, 165, 395), Rect(1299, 468, 287, 399)
    n = int(SELF_JUMP_HOLD / 0.25) + 3
    t = run(w, det, [[me(new_me), on_me(new_on)], [on_me(new_on)]] * n, start=t + 1)
    assert w._dango_mem[False][0] == new_me
    assert w.strangers(t) == 0


def test_one_self_near_preference_only_among_confident_boxes():
    """评审 4：贴着团子的垃圾低分 self（点亮陌生人时检测器降到 0.2）不能把别处的高分 self 挤成 player。"""
    from skydango.vision.perception import one_self

    dets = [Detection("self", JUMP, 0.8), Detection("self", SIT_ME, 0.22)]
    assert [d.cls for d in one_self(dets, near=[SIT_ON_ME], conf=0.35)] == ["self", "player"]
