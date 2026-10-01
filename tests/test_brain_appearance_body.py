"""身体用上装扮（spec 2026-10-01-appearance §6）：陌生人回来了、好友换装、status 的装扮、look_person 认编号 / 像谁。"""

import pytest
from test_brain_body import FakeEnv, body

from skydango.brain.body import ToolError
from skydango.brain.events import BACKGROUND
from skydango.config import Config
from skydango.inner.ledger import Ledger
from skydango.vision.bubbles import Rect
from skydango.vision.people import Person
from skydango.vision.perception import OutfitNote

FRIENDS = ["懒洋洋大王", "阿花"]
WALL0 = 1_790_000_000.0
F1, F2 = [0.1, 0.2], [0.3, 0.4]


class FakeBook:
    def __init__(self):
        self.strangers = {}

    def look(self, kind, key):
        return self.strangers.get(key, "") if kind == "stranger" else ""


class FakeLookEnv(FakeEnv):
    """感知层开着认装扮的样子（PerceptionWatcher 的 pop_stranger_backs / pop_outfits / my_look / looks）。"""

    def __init__(self):
        super().__init__()
        self.backs = []
        self.outfits = []
        self.me = ""
        self.friend_looks = {}
        self.appearance = FakeBook()

    def pop_stranger_backs(self):
        out, self.backs = self.backs, []
        return out

    def pop_outfits(self):
        out, self.outfits = self.outfits, []
        return out

    def my_look(self):
        return self.me

    def looks(self, names):
        return {n: self.friend_looks[n] for n in names if self.friend_looks.get(n)}


def make(clock, ledger=True, env=None):
    env = env or FakeLookEnv()
    led = Ledger(Config().inner, lambda: FRIENDS, WALL0) if ledger else None
    b, dev, reader, events = body(clock, env=env, ledger=led)
    b.friend_names = lambda: FRIENDS
    return b, env, events, led


def test_stranger_back_is_background_event(clock):
    b, env, events, _ = make(clock)
    env.appearance.strangers["陌生人A"] = "白斗篷"
    env.backs = ["陌生人A", "陌生人B"]
    b.step()
    got = [e for e in events.drain() if e.kind == "stranger_back"]
    assert [e.text for e in got] == ["刚才那个陌生人A（白斗篷）又回来了", "刚才那个陌生人B又回来了"]
    assert "stranger_back" in BACKGROUND and not events.urgent()


def test_outfit_notes_go_to_ledger_and_change_event(clock):
    b, env, events, led = make(clock)
    env.outfits = [OutfitNote("懒洋洋大王", F1, "k", "new"), OutfitNote("懒洋洋大王", F1, "k", "described", "白斗篷"),
                   OutfitNote("懒洋洋大王", F2, "k", "changed"), OutfitNote("懒洋洋大王", F2, "k", "described", "粉斗篷")]
    b.step()
    outfits = led.all_outfits()["懒洋洋大王"]
    assert [(o["feat"], o["desc"]) for o in outfits] == [(F1, "白斗篷"), (F2, "粉斗篷")]
    got = [e for e in events.drain() if e.kind == "outfit"]
    assert [(e.text, e.who) for e in got] == [("懒洋洋大王换了装扮：上次是「白斗篷」，现在「粉斗篷」", "懒洋洋大王")]
    assert "outfit" in BACKGROUND
    env.outfits = [OutfitNote("懒洋洋大王", F2, "k", "same"), OutfitNote("懒洋洋大王", F2, "k", "described", "粉色斗篷")]
    clock.advance(1)
    b.step()
    assert [e for e in events.drain() if e.kind == "outfit"] == []  # 同一套再描述一次：不重复
    assert len(led.all_outfits()["懒洋洋大王"]) == 2


def test_new_state_with_old_model_outfit_is_not_a_change(clock):
    # 关系卡里有一套，但特征是旧模型算的（key 不同）→ 感知层判 new：不是换装，接着用那一套、换上新特征
    b, env, events, led = make(clock)
    led.wear("懒洋洋大王", F1, "old", False, WALL0)
    led.describe_outfit("懒洋洋大王", "白色长斗篷", WALL0)
    led._appended.clear()  # 当作以前上线时记的
    env.outfits = [OutfitNote("懒洋洋大王", F2, "k", "new"), OutfitNote("懒洋洋大王", F2, "k", "described", "白色长斗篷")]
    b.step()
    assert [e for e in events.drain() if e.kind == "outfit"] == []
    (o,) = led.all_outfits()["懒洋洋大王"]
    assert (o["feat"], o["key"], o["desc"]) == (F2, "k", "白色长斗篷")


def test_outfits_without_ledger_no_event(clock):
    b, env, events, _ = make(clock, ledger=False)
    env.outfits = [OutfitNote("懒洋洋大王", F1, "k", "changed"), OutfitNote("懒洋洋大王", F1, "k", "described", "粉斗篷")]
    b.step()
    assert [e for e in events.drain() if e.kind == "outfit"] == [] and env.outfits == []


def test_status_shows_my_look_and_friend_look(clock):
    b, env, _, _ = make(clock)
    env.me = "白色樱花发型"
    env.near = ["懒洋洋大王", "阿花"]
    env.friend_looks = {"懒洋洋大王": "粉色长斗篷"}
    b.step()
    s = b.status()
    assert "你自己：白色樱花发型" in s and "懒洋洋大王（粉色长斗篷·" in s
    assert s.index("你自己：") < s.index("身边的好友：")
    assert "阿花（今天刚来" in s


def test_status_friend_look_without_ledger(clock):
    b, env, _, _ = make(clock, ledger=False)
    env.near = ["懒洋洋大王", "阿花"]
    env.friend_looks = {"懒洋洋大王": "粉色长斗篷"}
    assert "身边的好友：懒洋洋大王（粉色长斗篷）、阿花 / " in b.status()


def test_status_unchanged_without_appearance(clock):
    plain, look = FakeEnv(), FakeLookEnv()  # 没这些方法 / 有方法但什么都没认出来
    for env in (plain, look):
        env.near = ["懒洋洋大王"]
        env.people_list = [Person(1, "friend", "懒洋洋大王", Rect(100, 300, 90, 300), "左边", "近")]
    b1, _, ev1, _ = make(clock, env=plain)
    b2, _, ev2, _ = make(clock, env=look)
    b1.step()
    b2.step()
    assert b1.status() == b2.status() and "你自己" not in b1.status()
    assert [(e.kind, e.text) for e in ev1.drain()] == [(e.kind, e.text) for e in ev2.drain()]


def test_look_person_finds_stranger_by_id_and_maybe_friend(clock):
    env = FakeLookEnv()
    env.people_list = [
        Person(1, "stranger", None, Rect(1100, 300, 100, 260), "右边", "近", sid="陌生人A", look="白斗篷"),
        Person(2, "friend", "懒洋洋大王", Rect(300, 300, 100, 260), "左边", "近", sure=False),
    ]
    b, *_ = make(clock, env=env)
    assert b.find_person("懒洋洋大王", clock()) is None  # _locate 只认看到名字的
    _, note = b.look_person("陌生人A")
    assert "这是 陌生人A" in note["text"] and "按外观认的" not in note["text"]
    clock.advance(60)
    _, note = b.look_person("懒洋洋大王")
    assert "这是 懒洋洋大王" in note["text"] and note["text"].endswith("（没看到名字，按外观认的）")
    clock.advance(60)
    _, note = b.look_person("懒洋洋大玉")  # 聊天里 OCR 错一个字也认
    assert "按外观认的" in note["text"]
    clock.advance(60)
    with pytest.raises(ToolError, match="没找到 陌生人B"):
        b.look_person("陌生人B")


def test_look_person_prefers_named_box_over_maybe(clock):
    env = FakeLookEnv()
    env.people_list = [Person(2, "friend", "阿花", Rect(300, 300, 100, 260), "左边", "近", sure=False),
                       Person(3, "friend", "阿花", Rect(800, 300, 100, 260), "前面", "近")]
    b, *_ = make(clock, env=env)
    assert b.find_person("阿花", clock()) == Rect(800, 300, 100, 260)
    _, note = b.look_person("阿花")
    assert "按外观认的" not in note["text"]


def test_target_x_prefers_sure_people(clock):
    env = FakeLookEnv()
    env.people_list = [Person(2, "friend", "阿花", Rect(300, 300, 100, 260), "左边", "近", sure=False),
                       Person(3, "friend", "阿花", Rect(800, 300, 100, 260), "前面", "近")]
    b, *_ = make(clock, env=env)
    assert b.target_x("阿花", clock()) == (850, "body")
    env.people_list = env.people_list[:1]  # 只有像阿花的：照样能盯
    assert b.target_x("阿花", clock()) == (350, "body")
