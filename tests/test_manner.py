"""回法那一行（spec 2026-10-07-chat-pacing §2）。"""

import random

from skydango.brain.manner import jab_note, manner
from skydango.config import PacingConfig

T, F = True, False
EMOTE = "这几句可以只回个动作，不用说话"


class Rng(random.Random):
    """random() 固定返回 r，记调用次数。"""

    def __init__(self, r):
        super().__init__(0)
        self.r = r
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.r


ASK = [("小明", "今天去哪个图？")]


def m(batch=ASK, recent=(), mood="平常", sleepy=False, r=0.99, **kw):
    return manner(list(batch), list(recent), mood, sleepy, Rng(r), PacingConfig(**kw))


def test_too_many_jabs():
    assert m(recent=[T, F, T, F, F, F]).startswith("回法：最近 6 句里贱了 2 句，这轮好好说")


def test_last_jab_counts():
    assert "最近 3 句里贱了 1 句，这轮好好说" in m(recent=[F, F, T])


def test_tired_or_low():
    assert "没力气贫，懒一点回" in m(mood="低落")
    assert "没力气贫，懒一点回" in m(mood="烦")
    assert "没力气贫，懒一点回" in m(mood="平常", sleepy=True)
    line = m(recent=[T, T], mood="低落")
    assert "这轮好好说" in line and "没力气贫" not in line  # 三选一，命中即停


def test_happy_may_jab():
    assert "想贫可以贫一句" in m(mood="开心", recent=[T, F, F])
    line = m(mood="开心", recent=[T, T, F])
    assert "这轮好好说" in line and "想贫" not in line


def test_nothing_to_say():
    assert m(recent=[F], mood="平常") is None
    assert m(mood=None) is None


def test_emote_hint_short_no_question():
    short = [("小明", "哈哈"), ("小明", "笑 死")]
    assert EMOTE in m(batch=short, r=0.2)
    assert EMOTE not in (m(batch=short, r=0.4) or "")


def test_emote_hint_not_for_question_or_long_or_upset():
    for batch in ([("小明", "哈哈"), ("小明", "去哪")], [("小明", "我跟你说个好玩的事")], [("小明", "难过")]):
        rng = Rng(0.0)
        line = manner(batch, [], "平常", False, rng, PacingConfig())
        assert EMOTE not in (line or "")
        assert rng.calls == 0


def test_sleepy_emote_chance_capped():
    short = [("小明", "哈哈")]
    assert EMOTE in m(batch=short, sleepy=True, r=0.44)  # 0.3 × 1.5 = 0.45
    assert EMOTE in m(batch=short, sleepy=True, r=0.59, emote_chance=0.5)
    assert EMOTE not in m(batch=short, sleepy=True, r=0.61, emote_chance=0.5)  # 最多 0.6


def test_many_lines():
    one = [("小明", "今天去哪个图？"), ("小明", "雨林还是霞谷？"), ("小明", "快说？")]
    assert "他连着说了 3 句，挑最想接的回一句就行，不用每句都回" in m(batch=one)
    two = [("小明", "今天去哪个图？"), ("阿花", "雨林吧？")]
    assert "他们连着说了 2 句，挑最想接的回一句就行，不用每句都回" in m(batch=two)
    assert m(batch=ASK) is None


def test_order_and_join():
    line = m(batch=[("小明", "哈哈"), ("小明", "笑死")], recent=[T, T, T, F, F, F], r=0.0)
    assert line == ("回法：最近 6 句里贱了 3 句，这轮好好说；这几句可以只回个动作，不用说话；"
                    "他连着说了 2 句，挑最想接的回一句就行，不用每句都回")


def test_jab_note():
    assert jab_note([T, F]) == "最近说的：2 句里贱了 1 句"
    assert jab_note([]) == ""
