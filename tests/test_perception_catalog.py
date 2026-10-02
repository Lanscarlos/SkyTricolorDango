"""感知层接图鉴收集（spec 2026-10-02-catalog-collect §4）。"""

from test_perception import FakeDetector, FakeOcr, frame, player, tag, watcher

from skydango.vision.bubbles import Rect
from skydango.vision.catalog import Who
from skydango.vision.detect import Detection
from skydango.vision.track import Track


class FakeCatalog:
    def __init__(self):
        self.updates, self.drops, self.closed = [], [], 0

    def update(self, frame, players, selfs, now, panel, place, judge):
        self.updates.append((now, [judge(t) for t in players], [judge(t) for t in selfs], panel, place))

    def dropped(self, tracks):
        self.drops += [t.id for t in tracks]

    def close(self):
        self.closed += 1


def test_judge_friend_stranger_unlit_self():
    w = watcher(FakeDetector())
    friend = Track(1, "player", Rect(0, 0, 10, 10), 0.9, 0, 0, data={"tagged": True, "name": "懒洋洋大王"})
    lit = Track(2, "player", Rect(0, 0, 10, 10), 0.9, 0, 0, data={"maybe": "懒洋洋大王", "maybe_by": "relink"})
    unlit = Track(3, "player_unlit", Rect(0, 0, 10, 10), 0.9, 0, 0)
    me = Track(4, "self", Rect(0, 0, 10, 10), 0.9, 0, 0)
    unknown = Track(5, "player", Rect(0, 0, 10, 10), 0.9, 0, 0, data={"tagged": True})  # 有标签但不在好友名单
    assert w._catalog_who(friend) == Who("懒洋洋大王", "friend", True)
    assert w._catalog_who(lit) == Who("陌生人-t2", "stranger", False, "懒洋洋大王")
    assert w._catalog_who(unlit) is None
    assert w._catalog_who(me) == Who("团子", "self", True)
    assert w._catalog_who(unknown) == Who("陌生人-t5", "stranger", False)


def test_process_calls_update_with_frame_people():
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110), Detection("self", Rect(1500, 400, 90, 220), 0.9)]]
    w = watcher(det, ocr=FakeOcr({110: "懒洋洋大王"}))
    w.catalog = FakeCatalog()
    w.process(frame(), 0.0, True)
    (now, players, selfs, panel, place), = w.catalog.updates
    assert now == 0.0 and players == [Who("懒洋洋大王", "friend", True)] and selfs == [Who("团子", "self", True)]
    assert panel is not None  # 面板开着：给面板区域
    w.process(frame(), 0.1, False)
    assert w.catalog.updates[-1][3] is None  # 面板关着：None


def test_paused_skips_update_and_dropped_is_forwarded():
    det = FakeDetector()
    det.frames = [[player(1000)], []]
    w = watcher(det, track_buffer=0.5)
    w.catalog = FakeCatalog()
    w.process(frame(), 0.0, False)
    w.hold("test")
    w.process(frame(), 0.1, False)
    assert len(w.catalog.updates) == 1  # 暂停时不收
    w.release("test")
    w.process(frame(), 2.0, False)  # 人不见了、过了 track_buffer：轨迹删掉
    assert w.catalog.drops


def test_stop_closes_catalog():
    w = watcher(FakeDetector())
    w.catalog = FakeCatalog()
    w.stop()
    assert w.catalog.closed == 1


def test_no_catalog_is_fine():
    det = FakeDetector()
    det.frames = [[player(1000)]]
    w = watcher(det)
    assert w.catalog is None
    w.process(frame(), 0.0, False)
    w.stop()


def test_stranger_where_dango_just_was_is_not_collected():
    """YOLO 这一帧没给团子出 self 框、只出了 player 框：最近 1 秒团子待过的地方冒出来的"陌生人"就是团子，不收。"""
    me = Detection("self", Rect(900, 400, 120, 260), 0.9)
    on_me = Detection("player", Rect(905, 405, 118, 255), 0.9)
    far = player(300)
    det = FakeDetector()
    det.frames = [[me], [on_me, far], [on_me, far]]
    w = watcher(det)
    w.catalog = FakeCatalog()
    w.process(frame(), 0.0, False)
    w.process(frame(), 0.5, False)  # 团子框 0.5 秒前还在：这个 player 是团子
    _, players, _, _, _ = w.catalog.updates[-1]
    assert players.count(None) == 1 and any(p is not None and p.kind == "stranger" for p in players)
    w.process(frame(), 1.6, False)  # 超过 1 秒没见过团子框：不再这样认（宁可漏，不靠旧位置瞎猜）
    _, players, _, _, _ = w.catalog.updates[-1]
    assert None not in players


def test_player_overlapping_self_box_in_same_frame_is_not_collected():
    """同一帧里 self 框和团子身上的 player 框重叠不到 0.5（_is_self 没去掉），但 ≥ 0.3（IoU ≈ 0.39）：也是团子。"""
    me = Detection("self", Rect(900, 400, 120, 260), 0.9)
    bigger = Detection("player", Rect(940, 440, 120, 260), 0.9)
    det = FakeDetector()
    det.frames = [[me, bigger]]
    w = watcher(det)
    w.catalog = FakeCatalog()
    w.process(frame(), 0.0, False)
    _, players, selfs, _, _ = w.catalog.updates[-1]
    assert selfs == [Who("团子", "self", True)]
    assert players and all(p is None for p in players)
