"""图鉴收集（spec 2026-10-02-catalog-collect）：门槛、挑图、写盘。"""

import cv2
import numpy as np
import pytest

from skydango.config import CatalogConfig
from skydango.vision import catalog as cat
from skydango.vision.bubbles import Rect
from skydango.vision.catalog import CatalogCollector, Who, sharpness, stranger_key
from skydango.vision.track import Track

H, W = 1080, 1920


def frame(v=100):
    """纯色画面：测挑图时把 sharpness 换成"裁图平均亮度"，用亮度控制清晰度。"""
    return np.full((H, W, 3), v, np.uint8)


def track(tid, x=800, y=300, w=200, h=400, cls="player"):
    return Track(tid, cls, Rect(x, y, w, h), 0.9, 0.0, 0.0)


def friend(t):
    return Who("懒洋洋大王", "friend", True)


def stranger(t):
    return Who(stranger_key(t.id), "stranger", False)


def collector(tmp_path, trace=True, **kw):
    cfg = CatalogConfig(**{"sharp_min": 10.0, **kw})
    return CatalogCollector(cfg, tmp_path / "catalog", "20261002-210000-dry-brain", "2026-10-02",
                            wall=lambda: 1700000000.0, trace=trace)


@pytest.fixture
def brightness(monkeypatch):
    monkeypatch.setattr(cat, "sharpness", lambda crop: float(crop.mean()))


def test_sharpness_noise_vs_blur():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (400, 200, 3), dtype=np.uint8)
    assert sharpness(noise) > 1000
    assert sharpness(cv2.GaussianBlur(noise, (0, 0), 8)) < 50
    assert sharpness(np.full((400, 200, 3), 90, np.uint8)) == 0.0


def test_padded_clamps_to_frame():
    assert cat.padded(Rect(100, 100, 100, 200), 0.1, W, H) == Rect(90, 80, 120, 240)
    assert cat.padded(Rect(0, 0, 100, 200), 0.1, W, H) == Rect(0, 0, 110, 220)


def test_gates(tmp_path, brightness):
    c = collector(tmp_path)
    f = frame(100)
    c.update(f, [track(1, h=200)], [], 0.0, None, "", stranger)  # 太小：200 / 1080 < 0.25
    c.update(f, [track(2, x=2)], [], 0.0, None, "", stranger)  # 贴左边
    c.update(f, [track(3, y=H - 400)], [], 0.0, None, "", stranger)  # 贴下边
    c.update(f, [track(4), track(5, x=850)], [], 0.0, None, "", stranger)  # 两个人叠在一起
    c.update(f, [track(6, x=100)], [], 0.0, Rect(0, 0, 640, 920), "", stranger)  # 压在聊天面板上
    c.update(frame(5), [track(7)], [], 0.0, None, "", stranger)  # 模糊（亮度 5 < sharp_min 10）
    c.update(f, [track(8)], [], 0.0, None, "", lambda t: None)  # judge 说不收
    assert c.total == 0
    fails = {r["who"]: r["fail"] for r in c.candidates}
    assert fails == {"陌生人-t1": "small", "陌生人-t2": "edge", "陌生人-t3": "edge", "陌生人-t4": "blocked",
                     "陌生人-t5": "blocked", "陌生人-t6": "blocked", "陌生人-t7": "blurry"}
    c.update(f, [track(9)], [], 0.0, None, "", stranger)
    assert [s.row["who"] for s in c.buffer("陌生人-t9")] == ["陌生人-t9"]


def test_every_limits_how_often_a_track_is_looked_at(tmp_path, brightness):
    c = collector(tmp_path)
    for now in (0.0, 0.2, 0.4, 0.5, 0.7, 1.0):
        c.update(frame(), [track(1)], [], now, None, "", stranger)
    assert [r["t"] for r in c.candidates] == [0.0, 0.5, 1.0]


def test_keeps_at_most_per_who_spread_by_gap(tmp_path, brightness):
    c = collector(tmp_path, per_who=3, gap=3.0)
    for i, now in enumerate([0.0, 1.0, 3.0, 6.0, 9.0, 12.0]):
        c.update(frame(100 + i), [track(1)], [], now, None, "", friend)
    shots = c.buffer("懒洋洋大王")
    assert len(shots) == 3
    ts = sorted(s.t for s in shots)
    assert all(b - a >= 3.0 for a, b in zip(ts, ts[1:]))
    assert ts == [6.0, 9.0, 12.0]  # 越往后越亮 = 越清楚：留下最好的三张


def test_close_in_time_keeps_the_better_one(tmp_path, brightness):
    c = collector(tmp_path, gap=3.0)
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(150), [track(1)], [], 1.0, None, "", friend)  # 1 秒后、更清楚：换掉
    c.update(frame(120), [track(1)], [], 2.0, None, "", friend)  # 不如 1.0 那张：丢掉
    (s,) = c.buffer("懒洋洋大王")
    assert s.t == 1.0 and s.row["sharp"] == 150.0


def test_one_new_shot_can_replace_two_close_ones(tmp_path, brightness):
    c = collector(tmp_path, gap=3.0)
    # 时间不按顺序来，所以每次换一条轨迹（同一条轨迹 every 秒内只看一次）；judge 都说是他
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(100), [track(2)], [], 3.0, None, "", friend)
    c.update(frame(200), [track(3)], [], 1.5, None, "", friend)  # 离两张都不到 3 秒、比两张都好
    assert [s.t for s in c.buffer("懒洋洋大王")] == [1.5]


def test_score_grows_with_height_until_half_frame(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(100), [track(1, x=300, y=100, h=300)], [], 0.0, None, "", stranger)
    c.update(frame(100), [track(2, x=1300, y=100, h=700)], [], 0.0, None, "", stranger)
    (a,), (b,) = c.buffer("陌生人-t1"), c.buffer("陌生人-t2")
    assert b.score == pytest.approx(100.0)  # 超过半个画面高不再加分
    assert a.score == pytest.approx(100.0 * (300 / H) / 0.5, rel=0.02)


def test_friend_keeps_one_buffer_across_tracks(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(), [track(2)], [], 5.0, None, "", friend)  # 轨迹断了换了编号，还是他
    assert len(c.buffer("懒洋洋大王")) == 2


def test_self_and_row_fields(tmp_path, brightness):
    c = collector(tmp_path)
    me = track(3, cls="self")
    c.update(frame(), [], [me], 2.0, None, "雨林", lambda t: Who("团子", "self", True))
    (s,) = c.buffer("团子")
    assert s.row == {
        "kind": "outfit", "who": "团子", "who_kind": "self", "sure": True, "maybe": None,
        "run": "20261002-210000-dry-brain", "t": 2.0, "wall": 1700000000.0, "box": [800, 300, 200, 400],
        "height": round(400 / H, 4), "sharp": 100.0, "score": round(100.0 * (400 / H) / 0.5, 1), "place": "雨林",
    }
    assert s.crop.shape == (480, 240, 3)  # 四周各放宽 10%


def test_maybe_is_recorded_on_stranger(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(), [track(4)], [], 0.0, None, "", lambda t: Who(stranger_key(t.id), "stranger", False, "懒洋洋大王"))
    assert c.buffer("陌生人-t4")[0].row["maybe"] == "懒洋洋大王"


def test_max_per_run_stops_growth_but_allows_replacement(tmp_path, brightness):
    c = collector(tmp_path, max_per_run=2, gap=3.0)
    c.update(frame(100), [track(1)], [], 0.0, None, "", stranger)
    c.update(frame(100), [track(1)], [], 5.0, None, "", stranger)
    c.update(frame(100), [track(2, x=1300)], [], 5.0, None, "", stranger)  # 满了：新身份开不了
    c.update(frame(100), [track(1)], [], 10.0, None, "", stranger)  # 满了：同一个人也不再加
    assert c.total == 2 and c.buffer("陌生人-t2") == []
    c.update(frame(200), [track(1)], [], 15.0, None, "", stranger)  # 但比最差的好：替换
    assert c.total == 2 and max(s.t for s in c.buffer("陌生人-t1")) == 15.0
