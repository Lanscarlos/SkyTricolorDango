import random
from pathlib import Path

from skydango.brain.reflex import Reflexes, addressed
from skydango.config import Config, ReflexConfig, load_config

ROOT = Path(__file__).resolve().parent.parent


def FRIENDS(s):
    return s in ("小明", "阿花")


KW = dict(is_friend=FRIENDS, self_names=["团子", "三彩"], followup_window=30.0)


def test_addressed_by_name_followup_or_only_friend():
    assert addressed("小明", "团子在吗", nearby=["小明", "阿花"], since_said=None, **KW)
    assert addressed("小明", "哈哈", nearby=["小明", "阿花"], since_said=12.0, **KW)
    assert addressed("小明", "哈哈", nearby=["小明"], since_said=None, **KW)


def test_not_addressed():
    assert not addressed("小明", "哈哈", nearby=["小明", "阿花"], since_said=31.0, **KW)  # 过了窗口
    assert not addressed("小明", "哈哈", nearby=["小明", "阿花"], since_said=None, **KW)  # 两个好友在、没叫名字
    assert not addressed("路人甲", "团子", nearby=[], since_said=None, **KW)  # 陌生人
    assert not addressed("我", "团子", nearby=["小明"], since_said=None, **KW)  # 自己
    assert not addressed("", "团子", nearby=["小明"], since_said=None, **KW)  # 看不出是谁
    assert not addressed("小明", "#过来", nearby=["小明"], since_said=None, owner="小明", **KW)  # 主人命令


class Rng(random.Random):
    """random() 固定返回 r；choice 照常（按 getrandbits）。"""

    def __init__(self, r):
        super().__init__(0)
        self.r = r

    def random(self):
        return self.r


def make(r=0.0, **kw):
    cfg = ReflexConfig(addressed=["点头"], idle=["伸懒腰", "坐下"], return_map={"wave": "挥手"}, **kw)
    return Reflexes(cfg, Rng(r), now=0.0)


def test_chances_and_wheel():
    assert make(0.29).pick_addressed(0.0, ["点头"]) == "点头"
    assert make(0.31).pick_addressed(0.0, ["点头"]) is None  # addressed_chance 0.3
    assert make(0.0).pick_addressed(0.0, ["鞠躬"]) is None  # 不在轮盘上
    assert make(0.69).pick_return(0.0, "wave", ["挥手"]) == "挥手"
    assert make(0.71).pick_return(0.0, "wave", ["挥手"]) is None  # return_chance 0.7
    assert make(0.0).pick_return(0.0, "bow", ["挥手"]) is None  # return_map 里没有


def test_quota():
    rx = make(0.0, quota=2)
    rx.done(0.0, "a")
    rx.done(1.0, "b")
    assert rx.left(2.0) == 0 and rx.pick_addressed(2.0, ["点头"]) is None
    assert rx.left(600.5) == 1  # quota_window 600：0 秒那个过期了，1 秒那个还在
    assert [t for t, _ in rx.recent] == [0.0, 1.0]


def test_idle_waits_random_interval_and_avoids_repeat():
    rx = make(0.0)  # idle_min 180：Rng 0.0 → 截止 = stir 时刻 + 180
    assert rx.pick_idle(179.0, ["伸懒腰", "坐下"]) is None
    first = rx.pick_idle(180.0, ["伸懒腰", "坐下"])
    assert first in ("伸懒腰", "坐下")
    rx.done(180.0, first)
    assert rx.pick_idle(359.0, ["伸懒腰", "坐下"]) is None
    assert rx.pick_idle(360.0, ["伸懒腰", "坐下"]) not in (None, first)
    rx.stir(400.0)
    assert rx.pick_idle(579.0, ["伸懒腰", "坐下"]) is None
    assert make(1.0).pick_idle(419.0, ["伸懒腰"]) is None  # idle_max 420
    assert make(0.0).pick_idle(180.0, ["鞠躬"]) is None  # 清单里的都不在轮盘上


def test_usable_keeps_only_wheel_emotes():
    assert make().usable(["点头", "鞠躬"], ["鞠躬", "挥手"]) == ["鞠躬"]


def test_example_config_has_reflex():
    cfg = load_config(ROOT / "config.example.toml")
    assert cfg.reflex.enabled is True and cfg.reflex.bubble is True and cfg.reflex.idle == []
    assert Config().reflex.min_gap == 4.0 and Config().reflex.bubble_max == 45.0
