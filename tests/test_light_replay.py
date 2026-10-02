"""点亮陌生人：拿 10-02 晚 6 次举蜡烛的存图回放判定（spec 2026-10-03-light-flame-vanish §5）。

观测由 tests/light_replay_extract.py 从 runs/ 的存图抽出来（每帧团子框、火焰候选、YOLO 人物框 + black()），
存图大约 0.7 秒一张，比真跑时（0.3 秒扫一次）稀，判定照样要对。期望看存图判的：lit / lit / timeout / lit / lit / lit。
"""

import json
from pathlib import Path

import pytest

from skydango.config import SocialConfig
from skydango.vision.bubbles import Rect
from skydango.vision.candle import Disk
from skydango.vision.lighting import FlameWatch, flame_area, person_under

DATA = json.loads((Path(__file__).parent / "data" / "light_replay.json").read_text(encoding="utf-8"))
CFG = SocialConfig()


def observe(frame):
    me = Rect(*frame["me"])
    area = flame_area(me, CFG, *frame["size"])
    flames = [Disk(x, y, r, s) for x, y, r, s in frame["flames"] if s >= CFG.disk_min_score]
    people = [(Rect(x, y, w, h), blk, unlit) for x, y, w, h, unlit, _, blk in frame["people"]]

    def person_at(pos, r):
        i = person_under([p[0] for p in people], pos, r)
        return None if i is None else people[i]

    return me, area, flames, person_at


def replay(case):
    """举蜡烛前的帧先扫（出请求那条线索）；举起前一帧都没看到火焰（21:44 的请求帧正好没找到）就拿举起后第一帧先扫一次当举起前。
    返回 (结果, 判出的时间)：lit / gone / timeout（存图到头了还在等：身体会在 light_timeout 秒超时）。"""
    frames = DATA[case]["frames"]
    watch = FlameWatch(CFG)
    before = [f for f in frames if f["t"] < 0]
    if not any(observe(f)[2] for f in before):
        before.append(dict(next(f for f in frames if f["t"] >= 0 and observe(f)[2]), t=-0.05))
    for f in before:
        watch.scan(f["t"], *observe(f))
    clue = watch.main(before[-1]["t"])  # 不用 ready()：举起前只存了一两张图，凑不够 light_after；真跑时出请求的就是这条
    assert clue is not None, case
    watch.start(clue.id, 0.0)
    for f in frames:
        if f["t"] < 0:
            continue
        watch.scan(f["t"], *observe(f))
        result = watch.verdict(clue.id, 0.0, f["t"], stale=1.0)
        if result is True:
            return "lit", f["t"]
        if result is None:
            return "gone", f["t"]
    return "timeout", frames[-1]["t"]


@pytest.mark.parametrize("case", sorted(DATA))
def test_replay_10_02_night(case):
    result, at = replay(case)
    assert result == DATA[case]["expect"], (case, result, at)
    if result == "lit":
        assert at <= 3.0, (case, at)  # 以前最快的一次 3.8 秒（21:44），别的都判成走开了


def test_replay_lit_soon_after_flame_vanishes():
    """21:54 / 21:58：火焰举起后 0.6~0.8 秒就原地没了（他躲在团子身后，人物框量不到），存图下一张就该判出来。"""
    assert replay("215454") == ("lit", 1.4)
    assert replay("215823") == ("lit", 1.6)
