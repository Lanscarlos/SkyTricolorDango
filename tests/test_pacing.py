"""攒话的纯计算 Pacer（spec 2026-10-07-chat-pacing §1）。"""

import random

from skydango.brain.pacing import Pacer, is_question
from skydango.config import PacingConfig


def pacer(quiet=None, seed=1, **kw):
    cfg = PacingConfig(**kw)
    if quiet is not None:
        cfg.quiet_min = cfg.quiet_max = quiet
    return Pacer(cfg, random.Random(seed))


def release_time(p, start, typing=()):
    """从 start 起每 0.1 秒问一次，返回第一次 ready 的时刻。"""
    t = start
    while not p.ready(t, typing):
        t = round(t + 0.1, 3)
        assert t < start + 100
    return t


def test_quiet_random_in_range():
    p = pacer()
    waits = []
    t = 0.0
    for _ in range(8):
        p.heard(t, [("小明", "哈哈")], False)
        r = release_time(p, t)
        waits.append(r - t)
        p.released(r)
        t = r + 50
    assert all(2.0 <= w <= 6.1 for w in waits)
    assert len({round(w, 1) for w in waits}) > 1


def test_not_ready_before_quiet():
    p = pacer()
    p.heard(0.0, [("小明", "哈哈")], False)
    assert not p.ready(1.9, [])


def test_typing_waits_then_restarts():
    p = pacer(quiet=3.0)
    p.heard(0.0, [("小明", "哈哈")], False)
    assert not p.ready(2.0, ["小明"])
    assert not p.ready(5.0, ["小明"])  # 还在打字
    assert not p.ready(6.0, [])  # 从最后看到打字（5 秒）起算
    assert p.ready(8.0, [])


def test_other_typist_does_not_block():
    p = pacer(quiet=3.0)
    p.heard(0.0, [("小明", "哈哈")], False)
    assert p.ready(3.0, ["阿花"])


def test_max_wait_releases_while_typing():
    p = pacer(quiet=3.0)
    p.heard(0.0, [("小明", "哈哈")], False)
    assert not p.ready(14.9, ["小明"])
    assert p.ready(15.0, ["小明"])


def test_urgent_releases_now():
    p = pacer()
    p.heard(0.0, [("小明", "我好难过")], True)
    assert p.ready(0.0, ["小明"])


def test_question_uses_quiet_min():
    p = pacer(seed=3)
    p.heard(0.0, [("小明", "你在干嘛？")], False)
    assert not p.ready(1.9, [])
    assert p.ready(2.0, [])
    q = pacer(seed=3, quiet_min=2.0, quiet_max=6.0)
    q.heard(0.0, [("小明", "哈哈")], False)
    q.heard(1.0, [("小明", "你吃了吗")], False)  # 后来一句问句：整批改取 quiet_min
    assert q.ready(3.0, [])


def test_is_question():
    assert is_question("去哪") and is_question("几点了") and is_question("真的?")
    assert not is_question("哈哈哈")


def test_ocr_no_typing():
    p = pacer(quiet=3.0)
    p.heard(0.0, [("小明", "哈哈")], False)
    assert not p.ready(2.9, [])
    assert p.ready(3.0, [])


def test_released_clears_and_batches_independent():
    p = pacer(quiet=3.0)
    p.heard(0.0, [("小明", "哈哈")], False)
    assert p.pending() and p.batch() == [("小明", "哈哈")]
    p.released(3.0)
    assert not p.pending() and p.batch() == []
    assert not p.ready(100.0, [])
    p.heard(100.0, [("阿花", "嗨")], False)  # 新的一批：max_wait 从这一句算
    assert not p.ready(114.0, ["阿花"])
    assert p.ready(115.0, ["阿花"])


def test_urgent_without_lines_flushes_pending():
    p = pacer(quiet=3.0)
    p.heard(0.0, [], True)
    assert not p.pending() and not p.ready(0.0, [])
    p.heard(1.0, [("小明", "哈哈")], False)
    p.heard(1.5, [], True)
    assert p.ready(1.5, ["小明"])


def test_describe():
    p = pacer(quiet=3.0)
    p.heard(0.0, [("小明", "哈哈")], False)
    p.heard(0.5, [("小明", "笑死")], False)
    assert p.describe(0.6, ["小明"]) == "在攒话：小明说了 2 句，等他说完（还在打字）"
    assert p.describe(1.0, []) == "在攒话：小明说了 2 句，再等 3 秒"  # 2.5 秒向上取整
    p.heard(1.0, [("阿花", "嗯")], False)
    assert p.describe(1.0, ["阿花"]) == "在攒话：小明、阿花说了 3 句，等他们说完（还在打字）"
