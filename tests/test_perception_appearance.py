"""认装扮接进感知层：好友名字标签看不到时按外观认（maybe）、陌生人编号、老面孔回来。画面用彩色（人物框涂颜色）。"""

import numpy as np

from skydango.config import AppearanceConfig, EnvConfig, PerceptionConfig
from skydango.vision.appearance import AppearanceBook, ColorEmbedder
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.people import Person, describe_people
from skydango.vision.perception import PerceptionWatcher

from test_perception import FRIENDS, FakeDetector, FakeOcr, player, tag

PINK = (150, 80, 220)  # BGR：懒洋洋大王的斗篷
WHITE = (220, 220, 220)  # 点过火的陌生人
GREEN = (60, 180, 60)
R = Rect(0, 0, 10, 10)
XIAOMING, FANQIE = FRIENDS
OCR = {110: XIAOMING, 120: FANQIE}


class FakeHard:
    def __init__(self):
        self.reports = []

    def check(self, *args):
        pass

    def report(self, frame, now, reason, detail, tracks):
        self.reports.append((reason, detail))
        return True


def paint(*items):
    """items：(检测, 颜色)。在框的位置涂颜色，其余黑。"""
    img = np.zeros((1080, 1920, 3), np.uint8)
    for det, color in items:
        b = det.box
        img[b.y:b.y2, b.x:b.x2] = color
    return img


class FakeSaver:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def offer(self, track_id, who, crop, now, row):
        self.calls.append((track_id, who, crop, now, row))
        if self.fail:
            raise RuntimeError("磁盘满了")
        return True


def make(appearance=True, hardcases=None, acfg_kw=None, saver=None, **cfg):
    cfg.setdefault("stranger_after", 1.0)
    cfg.setdefault("keep", 5.0)
    det = FakeDetector()
    acfg = AppearanceConfig(enabled=True, **(acfg_kw if acfg_kw is not None else {"every": 1, "min_samples": 3}))
    extra = {}
    if appearance:
        emb = ColorEmbedder()
        extra = dict(appearance=AppearanceBook(acfg, emb.key, keep=cfg["keep"]), embedder=emb, appearance_cfg=acfg)
    w = PerceptionWatcher(
        det, FakeOcr(OCR), PerceptionConfig(**cfg), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, hardcases=hardcases, saver=saver, **extra,
    )
    return w, det


def run(w, det, items, start, end, step=0.1):
    """从 start 到 end（含）每 step 秒处理一帧同样的画面；items 空 = 什么都看不到。"""
    det.frames = [[d for d, _ in items]]
    img = paint(*[(d, c) for d, c in items if c is not None])
    t = start
    while t <= end + 1e-9:
        w.process(img, round(t, 3), panel_visible=False)
        t += step
    return round(t - step, 3)


def players(w):
    return [t for t in w.last_tracks if t.cls in ("player", "player_unlit")]


def friend_then_gone(w, det):
    """小明（粉）带标签 0~1 秒，然后看不到（镜头转了）。"""
    me = player(1000)
    run(w, det, [(me, PINK), (tag(990, 110), None)], 0.0, 1.0)
    assert w.appearance.friends[XIAOMING].n >= 3


def test_friend_survives_tag_loss_via_appearance():
    w, det = make()
    friend_then_gone(w, det)
    run(w, det, [], 1.1, 2.4)
    t = run(w, det, [(player(400), PINK)], 2.5, 3.6)  # 新轨迹、没标签
    (p,) = players(w)
    assert p.data["maybe"] == XIAOMING and not p.data.get("stranger")
    assert [p.name for p in w.people(t) if not p.sure] == [XIAOMING]
    assert w.strangers(t) == 0 and w.nearby(t + 4.0) == [XIAOMING]
    (box,) = [b for b in w.overlay(t) if b["kind"] == "maybe"]
    assert box["label"] == f"像{XIAOMING}?"
    assert w.nearest(t)[0] == XIAOMING


def test_maybe_does_not_bring_back_a_friend_who_left():
    w, det = make()
    friend_then_gone(w, det)
    run(w, det, [], 1.1, 6.9)
    t = run(w, det, [(player(400), PINK)], 7.0, 8.5)
    assert w.nearby(t) == [] and w.people(t)[0].sure is False
    assert w.strangers(t) == 0  # 也不算陌生人


def test_tag_overrides_maybe_and_reports_mismatch():
    hard = FakeHard()
    w, det = make(hardcases=hard)
    friend_then_gone(w, det)
    run(w, det, [], 1.1, 2.4)
    body = player(400)
    run(w, det, [(body, PINK)], 2.5, 3.0)
    assert players(w)[0].data["maybe"] == XIAOMING
    t = run(w, det, [(body, PINK), (tag(390, 120), None)], 3.1, 3.3)
    (p,) = players(w)
    assert "maybe" not in p.data and p.data["name"] == FANQIE
    assert hard.reports == [("appearance", f"按外观认成 {XIAOMING}，名字标签是 {FANQIE}")]
    assert [(x.name, x.sure) for x in w.people(t)] == [(FANQIE, True)]


def test_lit_stranger_gets_id_and_comes_back():
    w, det = make()
    t = run(w, det, [(player(1500), WHITE)], 0.0, 2.0)
    (p,) = players(w)
    assert p.data["sid"] == "陌生人A"
    w.appearance.set_desc("stranger", "陌生人A", "白斗篷", p.data["feat"])
    (person,) = w.people(t)
    assert (person.kind, person.sid, person.look) == ("stranger", "陌生人A", "白斗篷")
    (box,) = [b for b in w.overlay(t) if b["kind"] == "stranger"]
    assert box["label"] == "陌生人A" and box["desc"] == "白斗篷"
    assert w.pop_stranger_backs() == []
    run(w, det, [], 2.1, 8.0)  # 走开 6 秒（> keep）
    run(w, det, [(player(1200), WHITE)], 8.1, 9.6)
    assert players(w)[0].data["sid"] == "陌生人A"
    assert w.pop_stranger_backs() == ["陌生人A"]
    assert w.pop_stranger_backs() == []


def test_default_sampling_friend_never_counted_as_stranger():
    # 默认 every=3、min_samples=3，0.15 秒一帧：攒够样本要 ~1.2 秒，比 stranger_after 长，期间不能先算成陌生人
    w, det = make(acfg_kw={})
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.1, step=0.15)
    assert w.appearance.friends[XIAOMING].n >= 3
    run(w, det, [], 2.25, 3.6, step=0.15)
    det.frames = [[player(400)]]
    img = paint((player(400), PINK))
    for i in range(20):  # 3.75 ~ 6.6 秒
        t = round(3.75 + i * 0.15, 3)
        w.process(img, t, panel_visible=False)
        assert w.strangers(t) == 0, t
    assert players(w)[0].data["maybe"] == XIAOMING


def test_track_without_good_samples_still_becomes_stranger_after_grace():
    # 框太小（不到 min_height）攒不到好样本：多等 STRANGER_GRACE 秒后照样算陌生人
    w, det = make()
    small = player(1500, h=95)  # 高于 stranger_min_height（86 px），低于 min_height（108 px）
    run(w, det, [(small, WHITE)], 0.0, 2.4)
    assert w.strangers(2.4) == 0
    run(w, det, [(small, WHITE)], 2.5, 2.7)
    assert w.strangers(2.7) == 1 and "sid" not in players(w)[0].data


def test_two_similar_strangers_get_different_ids():
    w, det = make()
    run(w, det, [(player(1200), WHITE), (player(1600), WHITE)], 0.0, 2.0)
    assert sorted(p.data["sid"] for p in players(w)) == ["陌生人A", "陌生人B"]


def test_overlapping_self_is_not_learned_as_friend():
    w, det = make()
    me = Detection("self", Rect(1000, 400, 90, 220), 0.9)
    run(w, det, [(me, GREEN), (player(1040), PINK), (tag(1030, 110), None)], 0.0, 1.0)
    (p,) = players(w)
    assert p.data.get("name") == XIAOMING
    assert p.data.get("samples", 0) == 0 and XIAOMING not in w.appearance.friends
    assert w.appearance.me is None  # 团子也被好友挡着：不是好样本


def test_self_is_learned_and_looks():
    w, det = make()
    me = Detection("self", Rect(900, 500, 100, 250), 0.9)
    run(w, det, [(me, GREEN), (player(1400), PINK), (tag(1390, 110), None)], 0.0, 1.0)
    assert w.appearance.me is not None and w.appearance.me.n >= 3
    assert w.my_look() == "" and w.looks([XIAOMING]) == {}
    w.appearance.set_desc("me", "", "绿色斗篷", w.appearance.me.feat)
    w.appearance.set_desc("friend", XIAOMING, "粉色长斗篷", w.appearance.friends[XIAOMING].feat)
    assert w.my_look() == "绿色斗篷"
    assert w.looks([XIAOMING, FANQIE]) == {XIAOMING: "粉色长斗篷"}
    descs = {b["kind"]: b.get("desc") for b in w.overlay(1.0)}
    assert descs["self"] == "绿色斗篷" and descs["friend"] == "粉色长斗篷"


def test_unlit_gets_no_features():
    w, det = make()
    run(w, det, [(Detection("player_unlit", Rect(1400, 400, 90, 220), 0.9), WHITE)], 0.0, 2.0)
    (p,) = players(w)
    assert "feat" not in p.data and "sid" not in p.data and "samples" not in p.data


def test_maybe_counts_as_friend_for_talkers():
    w, det = make()
    friend_then_gone(w, det)
    run(w, det, [], 1.1, 2.4)
    body = player(400)
    bubble = Detection("typing", Rect(420, 330, 50, 40), 0.9)
    t = run(w, det, [(body, PINK), (bubble, None)], 2.5, 3.0)
    (talker,) = w.talkers(t)
    assert talker.friend and talker.name == XIAOMING


def test_maybe_dropped_after_recheck_misses():
    w, det = make()
    friend_then_gone(w, det)
    run(w, det, [], 1.1, 2.4)
    body = player(400)
    run(w, det, [(body, PINK)], 2.5, 3.0)
    assert players(w)[0].data["maybe"] == XIAOMING
    run(w, det, [(body, GREEN)], 3.1, 5.0)  # 同一条轨迹，颜色全变了
    assert "maybe" not in players(w)[0].data


def test_appearance_off_is_unchanged():
    w, det = make(appearance=False)
    t = run(w, det, [(player(1000), PINK), (tag(990, 110), None), (player(1500), WHITE)], 0.0, 2.0)
    people = w.people(t)
    assert {p.kind for p in people} == {"friend", "stranger"}
    assert all(p.sure is True and p.sid is None and p.look == "" for p in people)
    assert all("desc" not in b for b in w.overlay(t))
    assert w.pop_stranger_backs() == [] and w.my_look() == "" and w.looks([XIAOMING]) == {}
    assert all("feat" not in p.data for p in players(w))


def test_describe_people_wording():
    assert describe_people([Person(1, "friend", "小红", R, "右边", "远", sure=False),
                            Person(2, "stranger", None, R, "前面", "中", sid="陌生人A", look="白斗篷")]) \
        == "像小红（没看到名字，右边·远）、陌生人A（白斗篷，前面·中）"
    assert describe_people([Person(3, "stranger", None, R, "前面", "中", sid="陌生人B"),
                            Person(4, "friend", "小明", R, "左边", "近")]) == "陌生人B（前面·中）、小明（左边·近）"


def test_saver_gets_tagged_friend_and_stranger_samples_but_not_self():
    saver = FakeSaver()
    w, det = make(saver=saver)
    run(w, det, [(player(1000), PINK), (tag(990, 110), None), (player(1500), WHITE), (player(200), GREEN)], 0.0, 2.0)
    whos = {c[1] for c in saver.calls}
    assert XIAOMING in whos and any(x.startswith("t") for x in whos)
    _, who, crop, now, row = next(c for c in saver.calls if c[1] == XIAOMING)
    assert crop.ndim == 3 and crop.size
    assert row["tag"] == XIAOMING and row["best"] == XIAOMING and row["score"] > 0.9
    assert set(row) >= {"t", "track", "maybe", "sid", "box", "place"} and len(row["box"]) == 4


def test_saver_never_offered_for_self():
    saver = FakeSaver()
    w, det = make(saver=saver)
    run(w, det, [(Detection("self", Rect(900, 500, 120, 300), 0.9), PINK)], 0.0, 1.0)
    assert saver.calls == []


def test_saver_error_does_not_break_perception():
    saver = FakeSaver(fail=True)
    w, det = make(saver=saver)
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 1.0)
    assert saver.calls and w.appearance.friends[XIAOMING].n >= 3


# ---- 触发描述、好友装扮交给身体（Task 7） ----
class FakeWardrobe:
    """只记 request / reset，不描述。"""

    def __init__(self):
        self.requests, self.calls, self.resets = [], [], []

    def request(self, kind, who, priority, crop, feat, now):
        self.requests.append((kind, who))
        self.calls.append((kind, who, priority, crop, feat, now))
        return True

    def reset(self, kind, who):
        self.resets.append((kind, who))


def solid_feat(color):
    return ColorEmbedder().embed(np.full((220, 54, 3), color, np.uint8))


def card(color, desc):
    return [{"desc": desc, "feat": [round(float(x), 3) for x in solid_feat(color)], "key": "color-v1"}]


ME_BOX = Detection("self", Rect(900, 500, 100, 250), 0.9)


def test_me_described_once_then_on_drift():
    w, det = make(acfg_kw={"every": 1, "min_samples": 3, "redescribe_max": 1})
    fake = w.wardrobe = FakeWardrobe()
    run(w, det, [(ME_BOX, GREEN)], 0.0, 1.0)
    assert fake.requests == [("me", "")] and fake.calls[0][2] == 0  # ME 优先级
    w.on_described("me", "", "绿色斗篷", w.appearance.me.feat)
    assert w.my_look() == "绿色斗篷" and w.pop_outfits() == []  # 团子不进好友装扮队列
    run(w, det, [(ME_BOX, GREEN)], 1.1, 2.0)
    assert fake.requests == [("me", "")]  # 没换装不再描述
    run(w, det, [(ME_BOX, PINK)], 2.1, 4.0)  # 在衣柜换了装
    assert fake.requests == [("me", "")] * 2 and fake.resets == [("me", "")]
    w.on_described("me", "", "粉色斗篷", w.appearance.me.feat)
    run(w, det, [(ME_BOX, GREEN)], 4.1, 6.0)  # 又换回来：这次上线重新描述的次数用完了
    assert fake.requests == [("me", "")] * 2 and w.my_look() == "粉色斗篷"


def test_friend_without_card_is_described_and_noted_new():
    w, det = make()
    fake = w.wardrobe = FakeWardrobe()
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.0)
    notes = w.pop_outfits()
    assert [n.state for n in notes] == ["new"] and fake.requests == [("friend", XIAOMING)]
    (n,) = notes
    assert n.name == XIAOMING and n.key == "color-v1" and n.desc == ""
    assert len(n.feat) == len(w.appearance.friends[XIAOMING].feat)
    assert all(round(x, 3) == x for x in n.feat)
    assert fake.calls[0][2] == 1 and fake.calls[0][3].shape[0] > 220  # FRIEND 优先级；送的是扩过边的描述裁图
    assert w.pop_outfits() == []


def test_friend_same_outfit_uses_card_desc_without_request():
    w, det = make()
    fake = w.wardrobe = FakeWardrobe()
    w.appearance.load_cards({XIAOMING: card(PINK, "粉色长斗篷")})
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.0)
    assert [n.state for n in w.pop_outfits()] == ["same"]
    assert fake.requests == []
    assert w.looks([XIAOMING]) == {XIAOMING: "粉色长斗篷"}


def test_friend_same_outfit_without_card_desc_is_described():
    w, det = make()
    fake = w.wardrobe = FakeWardrobe()
    w.appearance.load_cards({XIAOMING: card(PINK, "")})
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.0)
    assert [n.state for n in w.pop_outfits()] == ["same"] and fake.requests == [("friend", XIAOMING)]


def test_friend_changed_outfit_is_described():
    w, det = make()
    fake = w.wardrobe = FakeWardrobe()
    w.appearance.load_cards({XIAOMING: card(GREEN, "绿斗篷")})
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.0)
    assert [n.state for n in w.pop_outfits()] == ["changed"] and fake.requests == [("friend", XIAOMING)]


def test_described_friend_pushes_note():
    w, det = make()
    w.wardrobe = FakeWardrobe()
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.0)
    F = w.appearance.friends[XIAOMING].feat
    w.on_described("friend", XIAOMING, "粉色长斗篷", F)
    note = w.pop_outfits()[-1]
    assert note.state == "described" and w.looks([XIAOMING])[XIAOMING] == "粉色长斗篷"
    assert (note.name, note.desc, note.key) == (XIAOMING, "粉色长斗篷", "color-v1")


def test_friend_changes_mid_session_noted_and_redescribed():
    w, det = make()
    fake = w.wardrobe = FakeWardrobe()
    body, label = player(1000), tag(990, 110)
    run(w, det, [(body, PINK), (label, None)], 0.0, 2.0)
    w.on_described("friend", XIAOMING, "粉色长斗篷", w.appearance.friends[XIAOMING].feat)
    w.pop_outfits()
    run(w, det, [(body, GREEN), (label, None)], 2.1, 4.0)  # 挂着标签换了衣服
    assert [n.state for n in w.pop_outfits()] == ["changed"]
    assert fake.requests == [("friend", XIAOMING)] * 2 and fake.resets == [("friend", XIAOMING)]
    assert w.appearance.friends[XIAOMING].redescribed == 1


def test_short_friend_box_waits_for_a_taller_sample():
    w, det = make()
    fake = w.wardrobe = FakeWardrobe()
    run(w, det, [(player(1000, h=150), PINK), (tag(990, 110), None)], 0.0, 1.0)  # 好样本，但不到 describe_min_height
    assert [n.state for n in w.pop_outfits()] == ["new"] and fake.requests == []
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 1.1, 2.0)
    assert fake.requests == [("friend", XIAOMING)]


def test_describe_off_still_notes_outfits():
    w, det = make(acfg_kw={"every": 1, "min_samples": 3, "describe": False})
    fake = w.wardrobe = FakeWardrobe()
    run(w, det, [(ME_BOX, GREEN), (player(1400), PINK), (tag(1390, 110), None)], 0.0, 2.0)
    assert [n.state for n in w.pop_outfits()] == ["new"] and fake.requests == []
    w2, det2 = make()  # 没挂描述器
    run(w2, det2, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.0)
    assert [n.state for n in w2.pop_outfits()] == ["new"]


def test_near_stranger_described():
    w, det = make()
    fake = w.wardrobe = FakeWardrobe()
    run(w, det, [(player(1500), WHITE)], 0.0, 2.0)
    assert fake.requests == [("stranger", "陌生人A")] and fake.calls[0][2] == 2
    w.on_described("stranger", "陌生人A", "白斗篷", w.appearance.strangers["陌生人A"].feat)
    assert w.pop_outfits() == [] and w.people(2.0)[0].look == "白斗篷"


def test_far_stranger_not_described():
    w, det = make()
    fake = w.wardrobe = FakeWardrobe()
    big_me = Detection("self", Rect(700, 300, 200, 600), 0.9)  # 团子框高 600：220 高的人算远
    run(w, det, [(big_me, GREEN), (player(1500), WHITE)], 0.0, 2.0)
    assert players(w)[0].data["sid"] == "陌生人A"
    assert [r for r in fake.requests if r[0] == "stranger"] == []
