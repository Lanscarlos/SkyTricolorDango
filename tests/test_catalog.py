"""图鉴收集（spec 2026-10-02-catalog-collect）：门槛、挑图、写盘。"""

import cv2
import json
import numpy as np
import pytest
import threading

from skydango.config import CatalogConfig
from skydango.imageio import imread
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


def test_public_readers_take_the_lock(tmp_path, brightness):
    """证明 total / saved / buffer 读的时候也加锁（别的线程读统计时不撞 update）。"""
    c = collector(tmp_path, trace=False)
    # 先放一张
    c.update(frame(100), [track(1)], [], 0.0, None, "", stranger)
    assert c.total == 1

    # 开一个线程，试图在主线程持锁期间读 total
    result = []
    def reader_thread():
        result.append(c.total)

    # 主线程持锁，阻止读者获取锁
    with c._lock:
        t = threading.Thread(target=reader_thread)
        t.start()
        t.join(timeout=0.2)
        # 在主线程还持锁的情况下，读者线程应该被阻塞（还活着）
        assert t.is_alive(), "读者线程应该被锁阻塞"

    # 释放锁后，读者线程应该能获取锁并完成
    t.join(timeout=1.0)
    assert not t.is_alive(), "读者线程应该能获取锁并完成"
    assert result == [1], "读者线程应该读到 1"


def rows(c):
    return [json.loads(line) for line in (c.folder / "index.jsonl").read_text(encoding="utf-8").splitlines()]


def test_close_writes_files_and_index(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(150), [track(1)], [], 5.0, None, "", friend)
    c.close()
    folder = tmp_path / "catalog" / "inbox" / "2026-10-02" / "20261002-210000-dry-brain"
    assert c.folder == folder
    assert sorted(p.name for p in (folder / "懒洋洋大王").iterdir()) == ["1.jpg", "2.jpg"]
    assert imread(folder / "懒洋洋大王" / "1.jpg").mean() == pytest.approx(150, abs=2)  # 名次按得分
    rs = rows(c)
    assert [r["file"] for r in rs] == [
        "inbox/2026-10-02/20261002-210000-dry-brain/懒洋洋大王/1.jpg",
        "inbox/2026-10-02/20261002-210000-dry-brain/懒洋洋大王/2.jpg",
    ]
    assert rs[0]["kind"] == "outfit" and rs[0]["who"] == "懒洋洋大王" and rs[0]["t"] == 5.0
    assert c.saved == 2


def test_nothing_collected_leaves_no_folder(tmp_path):
    c = collector(tmp_path)
    c.update(frame(), [], [], 0.0, None, "", friend)
    c.close()
    assert not (tmp_path / "catalog").exists()


def test_dropped_stranger_is_written_and_cleared(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(), [track(7)], [], 0.0, None, "", stranger)
    c.dropped([track(7), track(99)])  # 99 没收过：不管
    assert (c.folder / "陌生人-t7" / "1.jpg").exists()
    assert c.buffer("陌生人-t7") == [] and c.total == 1  # 清掉了，但还算在这次运行的张数里
    c.update(frame(), [track(8, x=1300)], [], 1.0, None, "", stranger)
    c.close()
    assert [r["who"] for r in rows(c)] == ["陌生人-t7", "陌生人-t8"]  # 清掉的那份还在索引里


def test_periodic_flush_and_shrink_removes_extra_file(tmp_path, brightness):
    c = collector(tmp_path, flush_every=10.0, gap=3.0)
    # 时间不按顺序来，每次换一条轨迹（同一条轨迹 every 秒内只看一次）
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(100), [track(2)], [], 3.0, None, "", friend)
    c.update(frame(100), [track(3)], [], 10.0, None, "", friend)  # 到 10 秒：写一次盘
    assert len(list((c.folder / "懒洋洋大王").iterdir())) == 3
    c.update(frame(250), [track(4)], [], 11.5, None, "", friend)  # 离 10.0 不到 3 秒、更好：换掉 10.0 那张
    c.update(frame(255), [track(5)], [], 1.5, None, "", friend)  # 离 0.0 和 3.0 都不到 3 秒、都更好：两张换一张
    c.close()
    assert sorted(p.name for p in (c.folder / "懒洋洋大王").iterdir()) == ["1.jpg", "2.jpg"]
    assert len(rows(c)) == 2


def test_write_error_is_logged_not_raised(tmp_path, brightness, monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("磁盘满了")

    monkeypatch.setattr(cat, "imwrite", boom)
    c = collector(tmp_path)
    c.update(frame(), [track(1)], [], 0.0, None, "", friend)
    with caplog.at_level("WARNING"):
        c.close()
    assert "图鉴收集写盘出错" in caplog.text


def test_closed_collector_ignores_updates(tmp_path, brightness):
    c = collector(tmp_path)
    c.close()
    c.update(frame(), [track(1)], [], 0.0, None, "", friend)
    c.dropped([track(1)])
    assert c.total == 0


def test_contact_sheet_one_row_per_identity(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(120), [track(1)], [], 5.0, None, "", friend)
    c.update(frame(140), [track(2, x=1300)], [], 0.0, None, "", stranger)
    c.close()
    sheet, legend = cat.contact_sheet(c.folder, cell=100)
    assert legend == ["懒洋洋大王（2 张）", "陌生人-t2（1 张）"]
    assert sheet.shape == (2 * 100, 2 * 100, 3)  # 2 行；最多的一行 2 张
    assert cat.contact_sheet(tmp_path / "nothing") is None
