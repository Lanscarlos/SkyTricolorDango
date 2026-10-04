"""点亮陌生人：拿 10-02 晚 6 次、10-03 晚 8 次举蜡烛的存图回放判定（spec 2026-10-03-light-flame-vanish §5；10-03 的见 docs/progress/2026-10-03-plan.md）。

观测由 tests/light_replay_extract.py 从 runs/ 的存图抽出来（每帧团子框、火焰候选、YOLO 人物框 + black()），
存图大约 0.7 秒一张，比真跑时（0.3 秒扫一次）稀，判定照样要对。期望是看存图判的，写在 light_replay_extract.py 的 CASES 里。
"""

import json
from pathlib import Path

import pytest

from skydango.config import SocialConfig
from skydango.vision.bubbles import Rect
from skydango.vision.candle import Disk
from skydango.vision.lighting import LIT_WEAK, FlameWatch, flame_area, person_under

DATA = json.loads((Path(__file__).parent / "data" / "light_replay.json").read_text(encoding="utf-8"))
CFG = SocialConfig()


def observe(frame):
    me = Rect(*frame["me"])
    area = flame_area(me, CFG, *frame["size"])
    flames = [Disk(x, y, r, s) for x, y, r, s in frame["flames"] if s >= CFG.disk_min_score]
    # 平时不算、只给举着蜡烛时他那团接续用的：名字标签下面的 + 分数在 LIT_WEAK ~ disk_min_score 之间的（同感知层）
    extra = ([Disk(x, y, r, s) for x, y, r, s in frame.get("masked", []) if s >= LIT_WEAK]
             + [Disk(x, y, r, s) for x, y, r, s in frame["flames"] if LIT_WEAK <= s < CFG.disk_min_score])
    people = [(Rect(x, y, w, h), blk, unlit) for x, y, w, h, unlit, _, blk in frame["people"]]

    def person_at(pos, r):
        i = person_under([p[0] for p in people], pos, r)
        return None if i is None else people[i]

    return me, area, flames, person_at, extra


def scan(watch, f):
    me, area, flames, person_at, extra = observe(f)
    watch.scan(f["t"], me, area, flames, person_at, extra=extra)


def replay(case):
    """举蜡烛前的帧先扫（出请求那条线索）。举起前的存图里一团火焰都没有时：summary 记着那条线索最后在举起前 flame_last 秒看到、
    在 flame_pos（10-03 晚：举起前约 1 秒就没了 / 落在名字标签下面），就在那一刻造一次"看到"；10-02 的没有 summary，
    拿举起后第一帧先扫一次当举起前（21:44 的请求帧正好没找到）。
    返回 (结果, 判出的时间)：lit / gone / timeout（存图到头了还在等：身体会在 light_timeout 秒超时）。"""
    data = DATA[case]
    frames = data["frames"]
    watch = FlameWatch(CFG)
    before = [f for f in frames if f["t"] < 0]
    if not any(observe(f)[2] for f in before):
        last, pos = data.get("flame_last"), data.get("flame_pos")
        if last is not None and last < 0 and pos:
            seed = dict(before[0] if before else frames[0], t=last)
            seed["flames"], seed["masked"] = [[pos[0], pos[1], 25.0, 0.9]], []
            before = sorted(before + [seed], key=lambda f: f["t"])
        else:
            before.append(dict(next(f for f in frames if f["t"] >= 0 and observe(f)[2]), t=-0.05))
    for f in before:
        scan(watch, f)
    clue = watch.main(max(f["t"] for f in before if observe(f)[2]))  # 不用 ready()：举起前只存了一两张图，凑不够 light_after
    assert clue is not None, case
    watch.start(clue.id, 0.0)
    for f in frames:
        if f["t"] < 0:
            continue
        scan(watch, f)
        result = watch.verdict(clue.id, 0.0, f["t"], stale=1.0)
        if result is True:
            return "lit", f["t"]
        if result is None:
            return "gone", f["t"]
    return "timeout", frames[-1]["t"]


LIT_BY = {"212739": 4.5}  # 10-03 21:27：约 +4 秒才亮（深色衣服），以前等到 8.1 秒


@pytest.mark.parametrize("case", sorted(DATA))
def test_replay_real_nights(case):
    result, at = replay(case)
    assert result == DATA[case]["expect"], (case, result, at)
    if result == "lit":
        assert at <= LIT_BY.get(case, 3.0), (case, at)  # 10-02：以前最快的一次 3.8 秒（21:44），别的都判成走开了


def test_replay_lit_soon_after_flame_vanishes():
    """21:54 / 21:58：火焰举起后 0.6~0.8 秒就原地没了（他躲在团子身后，人物框量不到），存图下一张就该判出来。"""
    assert replay("215454") == ("lit", 1.4)
    assert replay("215823") == ("lit", 1.6)
