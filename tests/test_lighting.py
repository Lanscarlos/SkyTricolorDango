"""点亮陌生人的线索和判定（spec 2026-10-03-light-flame-vanish §2~§3）：按真跑的节拍（0.3 秒扫一次）造观测。"""

import pytest

from skydango.config import SocialConfig
from skydango.vision.bubbles import Rect
from skydango.vision.candle import Disk
from skydango.vision.lighting import FlameWatch, flame_area, person_under

CFG = SocialConfig()
ME = Rect(900, 500, 100, 250)  # 团子框：H = 250，cx = 950
AREA = flame_area(ME, CFG, 1920, 1080)  # x 600~1300、y 375~750
HOME = (1050, 560)  # 黑影胸口的火焰：团子右边 0.4 倍框高
DARK = Rect(1000, 520, 100, 220)  # 火焰下面的黑影


def nobody(pos, r):
    return None


def person(box=DARK, blk=0.9, unlit=True):
    return lambda pos, r: (box, blk, unlit) if person_under([box], pos, r) is not None else None


def disk(pos=HOME, score=0.9, r=25.0):
    return Disk(pos[0], pos[1], r, score)


def raised(watch=None, at=HOME, who=nobody):
    """火焰在 at 连着看到 2 秒、出请求，0 秒时举起蜡烛。"""
    watch = watch or FlameWatch(CFG)
    for k in range(7):
        watch.scan(-2.0 + 0.3 * k, ME, AREA, [disk(at)], who)
    clue = watch.ready(-0.2)
    assert clue is not None
    watch.start(clue.id, 0.0)
    return watch, clue.id


def scans(watch, start, end, flames=lambda t: [], who=nobody, who_at=None):
    """从 start 到 end 每 0.3 秒扫一次，返回每次扫完的判定 [(t, 结果)]。who_at(t)：随时间变的"下面那个人"。"""
    out, t = [], start
    while t <= end + 1e-9:
        watch.scan(t, ME, AREA, flames(t), who_at(t) if who_at else who)
        out.append((round(t, 2), watch.verdict(watch.lighting.clue, 0.0, t, stale=0.8)))
        t += 0.3
    return out


def first_decision(results):
    return next(((t, r) for t, r in results if r is not False), None)


# ---- 出请求 ----
def test_ready_after_light_after_with_a_sure_frame():
    watch = FlameWatch(CFG)
    for k in range(5):  # 1.2 秒：不够 light_after（1.5）
        watch.scan(0.3 * k, ME, AREA, [disk()], nobody)
    assert watch.ready(1.2) is None
    watch.scan(1.5, ME, AREA, [disk()], nobody)
    assert watch.ready(1.5).pos == HOME


def test_never_sure_never_ready():
    watch = FlameWatch(CFG)
    for k in range(20):
        watch.scan(0.3 * k, ME, AREA, [disk(score=0.8)], nobody)
    assert watch.ready(5.7) is None


def test_two_flames_two_clues():
    """身边两个黑影：两条线索，不会因为分数高低来回换而断开。"""
    watch = FlameWatch(CFG)
    other = (850, 560)
    for k in range(8):
        a, b = (0.9, 0.75) if k % 2 else (0.75, 0.9)
        watch.scan(0.3 * k, ME, AREA, [disk(HOME, a), disk(other, b)], nobody)
    assert len(watch.clues) == 2 and all(c.first == 0.0 for c in watch.clues)


# ---- 灯笼：不动、从没到过 disk_sure ----
LANTERN = (1150, 520)


def test_lantern_becomes_static_and_never_ready():
    watch = FlameWatch(CFG)
    for k in range(20):
        watch.scan(0.3 * k, ME, AREA, [disk(LANTERN, 0.72)], nobody)
    (c,) = watch.clues
    assert c.static and watch.ready(5.7) is None and watch.main(5.7) is None


def test_lantern_does_not_keep_lighting_alive():
    """举蜡烛时旁边（< light_jump）有个灯笼：他的火焰原地没了，灯笼的候选配给灯笼自己的线索，照样判点亮（不会一直"火焰还在"等到超时）。"""
    watch = FlameWatch(CFG)
    for k in range(20):  # 灯笼先在那儿 6 秒
        watch.scan(-8.0 + 0.3 * k, ME, AREA, [disk(LANTERN, 0.72)], nobody)
    for k in range(7):
        watch.scan(-2.0 + 0.3 * k, ME, AREA, [disk(LANTERN, 0.72), disk(HOME)], nobody)
    clue = watch.ready(-0.2)
    assert clue.pos == HOME
    watch.start(clue.id, 0.0)
    res = scans(watch, 0.1, 3.0, flames=lambda t: [disk(LANTERN, 0.72)] + ([disk(HOME)] if t < 1.0 else []))
    assert first_decision(res) == (1.6, True)  # 最后看到 0.7，1.6 离它 0.9 秒


def test_lantern_flicker_still_static_for_a_while():
    """灯笼分数在门槛上下晃（这一次没找到）：断开 STATIC_KEEP 内还记着它，下次出现还是那条不动的线索。"""
    watch = FlameWatch(CFG)
    for k in range(20):
        watch.scan(0.3 * k, ME, AREA, [disk(LANTERN, 0.72)], nobody)
    watch.scan(8.0, ME, AREA, [], nobody)
    watch.scan(9.5, ME, AREA, [disk(LANTERN, 0.7)], nobody)
    (c,) = watch.clues
    assert c.static and c.first == 0.0


def test_standing_dark_stranger_with_sure_frame_is_not_static():
    watch = FlameWatch(CFG)
    for k in range(20):
        watch.scan(0.3 * k, ME, AREA, [disk(score=0.95 if k == 3 else 0.75)], nobody)
    (c,) = watch.clues
    assert not c.static and watch.ready(5.7) is c


# ---- 判点亮 ----
def test_flame_vanishes_in_place_is_lit():
    watch, _ = raised()
    res = scans(watch, 0.1, 3.0, flames=lambda t: [disk()] if t < 0.5 else [])
    assert first_decision(res) == (1.3, True)  # 最后看到 0.4，0.7 / 1.0 没了（< 0.8 秒），1.3 判出来


def test_nothing_before_lit_min():
    watch, _ = raised()
    res = scans(watch, 0.1, 3.0)  # 举起就没了
    assert first_decision(res)[0] >= CFG.lit_min


def test_flame_still_there_waits():
    watch, _ = raised(who=person())
    res = scans(watch, 0.1, 8.0, flames=lambda t: [disk(score=0.75)], who=person())
    assert first_decision(res) is None


def test_one_missed_scan_is_not_vanished():
    watch, _ = raised()
    res = scans(watch, 0.1, 6.0, flames=lambda t: [] if abs(t - 2.5) < 0.05 else [disk()])
    assert first_decision(res) is None


def test_still_black_under_vanished_flame_waits():
    """火焰没了但下面的人量得到、还是黑的（被挡住 / 镜头晃）：接着等，他亮了再判。"""
    watch, _ = raised(who=person())
    res = scans(watch, 0.1, 4.0, flames=lambda t: [disk()] if t < 0.5 else [],
                who_at=lambda t: person(blk=0.9 if t < 2.4 else 0.05))
    assert first_decision(res) == (2.5, True)  # 2.4 之前一直黑着：火焰早没了也不判；他一变亮就判


def test_still_black_then_brightens_is_lit():
    watch, _ = raised(who=person())
    res = scans(watch, 0.1, 4.0, flames=lambda t: [disk()] if t < 0.5 else [],
                who_at=lambda t: person(blk=0.9 if t < 2.4 else 0.05, unlit=False))
    assert first_decision(res) == (2.5, True)  # 变亮的第一次扫描：火焰早就没够 lit_vanish 了


def test_unmeasurable_person_does_not_block():
    """他躲在团子身后，black() 量不准（None）：不挡，火焰原地没了就判点亮（10-02 21:54 / 21:58）。"""
    watch, _ = raised(who=person(blk=None))
    res = scans(watch, 0.1, 3.0, flames=lambda t: [disk()] if t < 0.5 else [], who=person(blk=None))
    assert first_decision(res) == (1.3, True)


def test_brightening_same_person_speeds_up():
    """火焰没了一次扫描、同一个人连着两次看到变亮：不用等 lit_vanish。"""
    watch, _ = raised(who=person(blk=0.9))
    res = scans(watch, 0.1, 3.0, flames=lambda t: [disk()] if t < 0.5 else [], who=person(blk=0.05, unlit=False))
    assert first_decision(res) == (1.0, True)  # 0.7 第一次变亮、1.0 第二次（lit_min 1.0 正好到）


def test_near_threshold_or_yolo_unlit_counts_as_seen_dark():
    """10-02 21:42：举起时 black() 0.34（差一点到 0.35）；或者 YOLO 认成 player_unlit：都算见过他黑。"""
    for blk, unlit in ((0.34, False), (0.1, True)):
        watch, _ = raised(who=person(blk=blk, unlit=unlit))
        assert watch.lighting.dark_box == DARK


def test_walks_to_edge_then_vanishes_is_gone():
    """火焰跟着人往右挪到范围边上（右边 1300）才没的：走开了。"""
    watch, _ = raised()
    pos = lambda t: (round(1050 + 120 * t), 560)  # noqa: E731
    res = scans(watch, 0.1, 4.0, flames=lambda t: [disk(pos(t))] if pos(t)[0] < 1240 else [])
    assert first_decision(res)[1] is None


def test_walks_while_dark_then_lit_in_the_middle_is_lit():
    """10-02 21:44：他黑着往右挪了 160 px（0.6 倍框高）再原地亮了：不算走开。"""
    watch, _ = raised()
    pos = lambda t: (round(1050 + 100 * t), 560)  # noqa: E731
    res = scans(watch, 0.1, 4.0, flames=lambda t: [disk(pos(t))] if t < 1.6 else [])
    assert first_decision(res)[1] is True


def test_flame_shrinks_then_vanishes_is_gone():
    """往远处走：火焰变小再没的（位置还在中间）。"""
    watch, _ = raised()
    res = scans(watch, 0.1, 4.0, flames=lambda t: [disk(r=25 - 10 * t)] if t < 1.2 else [])
    assert first_decision(res)[1] is None


def test_clue_gone_before_raise_is_gone():
    """举之前线索刚断（start 找不到线索）：不知道火焰在哪，没了算走开，不会判点亮。"""
    watch = FlameWatch(CFG)
    watch.start(99, 0.0)
    res = scans(watch, 0.1, 3.0)
    assert first_decision(res)[1] is None


def test_flame_comes_back_near_keeps_waiting():
    watch, _ = raised()
    res = scans(watch, 0.1, 6.0, flames=lambda t: [disk((1060, 565))] if t < 0.5 or t > 0.9 else [])
    assert first_decision(res) is None


def test_other_strangers_flame_elsewhere_is_ignored():
    """他原地亮了，另一个黑影的火焰在团子另一边（> light_jump）：照样判点亮。"""
    watch, _ = raised()
    res = scans(watch, 0.1, 3.0, flames=lambda t: [disk()] if t < 0.5 else [disk((750, 560))])
    assert first_decision(res) == (1.3, True)


def test_stale_scan_is_false():
    watch, cid = raised()
    scans(watch, 0.1, 1.9, flames=lambda t: [disk()] if t < 0.5 else [])
    assert watch.verdict(cid, 0.0, 1.9, stale=0.8) is True
    assert watch.verdict(cid, 0.0, 3.0, stale=0.8) is False  # 1.1 秒没扫描了


def test_wrong_clue_is_false_and_stop_drops_clue():
    watch, cid = raised()
    assert watch.verdict(cid + 1, 0.0, 1.0, stale=0.8) is False
    watch.stop()
    assert watch.lighting is None and watch.get(cid) is None


def test_shift_moves_times():
    watch, cid = raised()
    watch.scan(0.4, ME, AREA, [disk()], nobody)
    watch.shift(3.0)
    L = watch.lighting
    assert L.raised == 3.0 and L.flame_last == pytest.approx(3.4) and L.scan_at == pytest.approx(3.4)
    assert all(c.last >= 3.0 for c in watch.clues)


# ---- 评审后补的 ----
def test_screen_pan_is_not_vanishing():
    """举蜡烛时画面整体平移 200 px（转镜头、聊天面板开关）：给了平移量就跟得上，不当成"原地没了"。"""
    watch, cid = raised()
    shift = (200, 0)
    out = []
    for k, t in enumerate((0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2)):
        moved = k >= 2  # 0.7 起画面往右挪了 200
        watch.scan(t, ME, AREA, [disk((HOME[0] + 200, HOME[1]) if moved else HOME)], nobody, shift=shift if k == 2 else (0, 0))
        out.append(watch.verdict(cid, 0.0, t, stale=0.8))
    assert out == [False] * 8 and watch.lighting.misses == 0


def test_brightening_while_flame_still_there_does_not_count():
    """火焰还在时看到"变亮"（夜里 black() 一抖）不攒：再丢一帧也不能立刻判点亮。"""
    watch, cid = raised(who=person(blk=0.9))
    res = scans(watch, 0.1, 1.9, flames=lambda t: [disk()] if t < 1.5 else [], who=person(blk=0.05, unlit=False))
    assert dict(res)[1.6] is False  # 1.6 第一次没看到火焰：变亮只算这一次


def test_pan_near_pivot_person_moves_less_than_background():
    """镜头绕团子转：背景平移 300 px，贴着团子的他只挪了 150（一半）：照样配得上。"""
    watch, cid = raised()
    watch.scan(0.1, ME, AREA, [disk()], nobody)
    watch.scan(0.4, ME, AREA, [disk((HOME[0] + 150, HOME[1]))], nobody, shift=(300, 0))
    assert watch.lighting.misses == 0 and watch.lighting.pos == (HOME[0] + 150, HOME[1])


def test_pan_keeps_lantern_static():
    """灯笼是背景：画面平移后它跟着挪，home 一起挪，还是不动的假火焰。"""
    watch = FlameWatch(CFG)
    for k in range(20):
        dx = 200 if k >= 10 else 0
        watch.scan(0.3 * k, ME, AREA, [disk((LANTERN[0] - 400 + dx, LANTERN[1]), 0.72)], nobody, shift=(200, 0) if k == 10 else (0, 0))
    (c,) = watch.clues
    assert c.static and c.moved < 1
