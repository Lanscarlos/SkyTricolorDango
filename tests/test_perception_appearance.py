"""认装扮接进感知层：好友名字标签看不到时按外观认（maybe）、陌生人编号、老面孔回来。画面用彩色（人物框涂颜色）。"""

import numpy as np

from skydango.config import AppearanceConfig, EnvConfig, PerceptionConfig
from skydango.vision.appearance import AppearanceBook, ColorEmbedder, DinoGuard
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.embed import unit
from skydango.vision.people import Person, describe_people
from skydango.vision.perception import DANGO_LOOK_HOLD, PerceptionWatcher

from test_appearance import _basis
from test_perception import FRIENDS, Clock, FakeDetector, FakeOcr, player, tag

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


class FakeDino:
    size = 224

    def __init__(self):
        self.calls = 0

    def embed(self, img):
        self.calls += 1
        v = np.zeros(8, np.float32)
        v[3] = 1.0
        return v


class FakeSaver:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def offer(self, track_id, who, crop, now, row):
        self.calls.append((track_id, who, crop, now, row))
        if self.fail:
            raise RuntimeError("磁盘满了")
        return True


def make(appearance=True, hardcases=None, acfg_kw=None, saver=None, clock=None, dino=None, **cfg):
    cfg.setdefault("stranger_after", 1.0)
    cfg.setdefault("keep", 5.0)
    det = FakeDetector()
    acfg = AppearanceConfig(enabled=True, **(acfg_kw if acfg_kw is not None else {"every": 1, "min_samples": 3, "outfit_change": True}))
    extra = {"clock": clock} if clock is not None else {}
    if appearance:
        emb = ColorEmbedder()
        extra |= dict(appearance=AppearanceBook(acfg, emb.key, keep=cfg["keep"]), embedder=emb, appearance_cfg=acfg, dino=dino)
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
    w, det = make(acfg_kw={"every": 1, "min_samples": 3, "redescribe_max": 1, "outfit_change": True})
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
    F = w.appearance.friends[XIAOMING].feat.copy()
    F[int(np.argmax(F))] *= 0.9  # 描述时的特征和现在的平均特征差一点（还是同一套）
    F = unit(F)
    w.on_described("friend", XIAOMING, "粉色长斗篷", F)
    note = w.pop_outfits()[-1]
    assert note.state == "described" and w.looks([XIAOMING])[XIAOMING] == "粉色长斗篷"
    assert (note.name, note.desc, note.key) == (XIAOMING, "粉色长斗篷", "color-v1")
    assert note.feat == [round(float(x), 3) for x in F]  # 用描述时的特征，不是现在的
    assert note.feat != [round(float(x), 3) for x in w.appearance.friends[XIAOMING].feat]


class BusyWardrobe(FakeWardrobe):
    """同一个身份在排队 / 描述中时拒绝（同真的 Wardrobe）。"""

    def __init__(self):
        super().__init__()
        self.busy = set()

    def request(self, kind, who, priority, crop, feat, now):
        if (kind, who) in self.busy:
            return False
        self.busy.add((kind, who))
        return super().request(kind, who, priority, crop, feat, now)


def test_stale_description_after_outfit_change_is_dropped():
    w, det = make()
    fake = w.wardrobe = BusyWardrobe()
    body, label = player(1000), tag(990, 110)
    run(w, det, [(body, PINK), (label, None)], 0.0, 2.0)
    assert [n.state for n in w.pop_outfits()] == ["new"] and fake.requests == [("friend", XIAOMING)]
    run(w, det, [(body, GREEN), (label, None)], 2.1, 4.0)  # 请求还在排着，人换了装
    assert fake.requests == [("friend", XIAOMING)]
    fake.busy.clear()  # 上一套（粉）的描述回来了
    w.on_described("friend", XIAOMING, "粉色长斗篷", fake.calls[0][4])
    assert w.pop_outfits() == [] and w.looks([XIAOMING]) == {}
    run(w, det, [(body, GREEN), (label, None)], 4.1, 5.0)
    assert fake.requests == [("friend", XIAOMING)] * 2  # 现在这一套接着请求
    assert unit(fake.calls[1][4]) @ solid_feat(GREEN) > 0.9
    assert w.appearance.friends[XIAOMING].redescribed == 0 and w.pop_outfits() == []  # 没多出 changed
    w.on_described("friend", XIAOMING, "绿斗篷", fake.calls[1][4])
    assert [n.state for n in w.pop_outfits()] == ["described"] and w.looks([XIAOMING]) == {XIAOMING: "绿斗篷"}


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
    w, det = make(acfg_kw={"every": 1, "min_samples": 3, "describe": False, "outfit_change": True})
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


# ---- 终审修正 ----
class Vanishing(dict):
    """身体线程读的同时感知线程刚好 pop 掉：get 还拿到了值，再用 [] 取就 KeyError。"""

    def __getitem__(self, k):
        if k in ("maybe", "sid", "name"):
            raise KeyError(k)
        return super().__getitem__(k)


def test_people_and_overlay_read_each_key_once():
    w, det = make()
    friend_then_gone(w, det)
    run(w, det, [], 1.1, 2.4)
    items = [(player(400), PINK), (player(1500), WHITE), (player(1000), GREEN), (tag(990, 120), None)]
    t = run(w, det, items, 2.5, 4.6)
    who = {p.data.get("maybe") or p.data.get("name") or p.data.get("sid") for p in players(w)}
    assert who == {XIAOMING, FANQIE, "陌生人A"}
    book = w.appearance
    book.set_desc("friend", XIAOMING, "粉色长斗篷", book.friends[XIAOMING].feat)
    book.set_desc("friend", FANQIE, "绿斗篷", book.friends[FANQIE].feat)
    book.set_desc("stranger", "陌生人A", "白斗篷", book.strangers["陌生人A"].feat)
    for p in players(w):
        p.data = Vanishing(p.data)
    got = {(p.kind, p.name, p.sid, p.sure, p.look) for p in w.people(t)}
    assert got == {("friend", XIAOMING, None, False, ""), ("friend", FANQIE, None, True, ""),
                   ("stranger", None, "陌生人A", True, "白斗篷")}
    boxes = {(b["kind"], b["label"], b.get("desc")) for b in w.overlay(t) if b["kind"] in ("maybe", "friend", "stranger")}
    assert boxes == {("maybe", f"像{XIAOMING}?", "粉色长斗篷"), ("friend", FANQIE, "绿斗篷"),
                     ("stranger", "陌生人A", "白斗篷")}


def test_hold_longer_than_keep_does_not_report_stranger_back():
    clock = Clock()
    w, det = make(clock=clock)
    run(w, det, [(player(1500), WHITE)], 0.0, 2.0)
    assert players(w)[0].data["sid"] == "陌生人A"
    clock.t = 2.0
    w.hold("camera")  # 转了 7 秒圈（> keep），人一直没走
    clock.t = 9.0
    w.release("camera")
    run(w, det, [], 9.1, 11.0)  # 旧轨迹过了 track_buffer 删掉
    run(w, det, [(player(800), WHITE)], 11.1, 12.5)  # 镜头转回来：换了条轨迹
    assert players(w)[0].data["sid"] == "陌生人A"
    assert w.pop_stranger_backs() == []


def change_twice(w, det):
    """卡里是粉的那一套（带描述）：粉 → 绿 → 又换回粉，每段 2 秒挂着标签。返回每段之后的装扮状态。"""
    w.appearance.load_cards({XIAOMING: card(PINK, "粉色长斗篷")})
    body, label = player(1000), tag(990, 110)
    out = []
    for color, (a, b) in ((PINK, (0.0, 2.0)), (GREEN, (2.1, 4.0)), (PINK, (4.1, 6.0))):
        run(w, det, [(body, color), (label, None)], a, b)
        out.append([n.state for n in w.pop_outfits()])
    return out


def test_describe_off_notes_every_mid_session_change():
    w, det = make(acfg_kw={"every": 1, "min_samples": 3, "describe": False, "outfit_change": True})
    fake = w.wardrobe = FakeWardrobe()
    assert change_twice(w, det) == [["same"], ["changed"], ["changed"]]
    assert fake.requests == [] and w._describe_want == set()
    assert w.appearance.friends[XIAOMING].redescribed == 2


def test_no_wardrobe_notes_every_mid_session_change():
    w, det = make()  # 没挂描述器
    assert change_twice(w, det) == [["same"], ["changed"], ["changed"]]
    assert w._describe_want == set()


def test_change_again_while_redescription_pending():
    w, det = make()
    fake = w.wardrobe = BusyWardrobe()  # 描述器一直没回（或者放弃了）：want 一直挂着
    assert change_twice(w, det) == [["same"], ["changed"], ["changed"]]
    assert fake.requests == [("friend", XIAOMING)]  # 第一次换装的请求还排着，后面的被拒


def test_one_outfit_change_is_noted_once_while_average_catches_up():
    w, det = make(acfg_kw={"every": 1, "min_samples": 3, "describe": False, "redescribe_max": 3, "outfit_change": True})
    w.appearance.load_cards({XIAOMING: card(PINK, "粉色长斗篷")})
    body, label = player(1000), tag(990, 110)
    run(w, det, [(body, PINK), (label, None)], 0.0, 2.0)
    w.pop_outfits()
    run(w, det, [(body, GREEN), (label, None)], 2.1, 8.0)  # 平均特征慢慢从粉挪到绿：只算一次换装
    assert [n.state for n in w.pop_outfits()] == ["changed"]
    assert w.appearance.friends[XIAOMING].redescribed == 1


# ---- [appearance] outfit_change = false（默认）：不判换装，每次上线描述一次 ----
OFF = {"every": 1, "min_samples": 3}  # outfit_change 用默认（关）


def test_outfit_change_off_friend_described_each_session_and_never_changed():
    w, det = make(acfg_kw=OFF)
    fake = w.wardrobe = FakeWardrobe()
    w.appearance.load_cards({XIAOMING: card(GREEN, "绿斗篷")})  # 卡里是上次的绿斗篷，今天穿粉的
    body, label = player(1000), tag(990, 110)
    run(w, det, [(body, PINK), (label, None)], 0.0, 2.0)
    assert [n.state for n in w.pop_outfits()] == ["same"]  # 不说"换了"：只更新最近那一套
    assert fake.requests == [("friend", XIAOMING)]  # 卡里有描述也照样描述一次（今天这身）
    assert w.looks([XIAOMING]) == {}  # 描述回来之前不拿上次的旧描述说事
    w.on_described("friend", XIAOMING, "粉色长斗篷", w.appearance.friends[XIAOMING].feat)
    assert [n.state for n in w.pop_outfits()] == ["described"] and w.looks([XIAOMING]) == {XIAOMING: "粉色长斗篷"}
    run(w, det, [(body, GREEN), (label, None)], 2.1, 6.0)  # 上线中途颜色变了：不判换装、不重新描述
    assert w.pop_outfits() == [] and fake.requests == [("friend", XIAOMING)] and fake.resets == []
    assert w.appearance.friends[XIAOMING].redescribed == 0


def test_outfit_change_off_friend_without_card_is_new():
    w, det = make(acfg_kw=OFF)
    fake = w.wardrobe = FakeWardrobe()
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.0)
    assert [n.state for n in w.pop_outfits()] == ["new"] and fake.requests == [("friend", XIAOMING)]


def test_outfit_change_off_me_described_once():
    w, det = make(acfg_kw=OFF)
    fake = w.wardrobe = FakeWardrobe()
    run(w, det, [(ME_BOX, GREEN)], 0.0, 1.0)
    w.on_described("me", "", "绿色斗篷", w.appearance.me.feat)
    run(w, det, [(ME_BOX, PINK)], 1.1, 4.0)
    assert fake.requests == [("me", "")] and fake.resets == [] and w.my_look() == "绿色斗篷"


def test_outfit_change_off_keeps_description_even_if_features_moved():
    w, det = make(acfg_kw=OFF)
    w.wardrobe = FakeWardrobe()
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 2.0)
    w.pop_outfits()
    w.on_described("friend", XIAOMING, "粉色长斗篷", solid_feat(GREEN))  # 颜色特征飘得厉害：不当成上一套的描述丢掉
    assert [n.state for n in w.pop_outfits()] == ["described"] and w.looks([XIAOMING]) == {XIAOMING: "粉色长斗篷"}


def test_features_store_sample_on_track():
    dino = FakeDino()
    w, det = make(dino=DinoGuard(dino))
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 1.0)
    (p,) = players(w)
    smp = p.data["sample"]
    assert smp.dino is not None and smp.color.shape == p.data["feat"].shape and not smp.pinned
    assert p.data["samples"] >= 3 and dino.calls >= 3


def test_features_without_dino_have_color_only():
    w, det = make()
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 1.0)
    (p,) = players(w)
    assert p.data["sample"].dino is None and p.data["samples"] >= 3


def test_dango_and_self_tracks_get_samples_once_per_frame():
    dino = FakeDino()
    w, det = make(dino=DinoGuard(dino))
    me = Detection("self", Rect(550, 470, 165, 395), 0.9)
    on_me = Detection("player", Rect(549, 468, 287, 399), 0.8)
    run(w, det, [(me, PINK), (on_me, PINK)], 0.0, 0.5)
    dangos = [t for t in w.last_tracks if t.data.get("dango")]
    selfs = [t for t in w.last_tracks if t.cls == "self"]
    assert len(dangos) == 1 and len(selfs) == 1
    assert dangos[0].data["sample"].dino is not None and selfs[0].data["sample"].dino is not None
    assert dino.calls == dangos[0].data["samples"] + selfs[0].data["samples"]  # 每次好样本只算一次，没重复处理


# ---- 学和认：像团子 / 像小明 / 可能是小明（身份底库 spec 2026-10-03 §3） ----
KW = {"match": 0.88, "unsure": 0.83, "min_samples": 3, "every": 1}
DANGO = (60, 180, 60)  # 团子的颜色（BGR）
DARK = (30, 90, 30)  # 好友：和团子同色相、同饱和度，只是暗——颜色特征分不开（余弦 1），DINOv2 分得开
SELF_AT = Detection("self", Rect(900, 500, 100, 250), 0.9)


class ColorDino:
    """假 DINOv2：按裁图中心的颜色查表（粉 → _basis(0)、暗绿 → _basis(2)、团子色 → _basis(4)），别的一律 _basis(1)。"""

    size = 224

    def __init__(self):
        self.table = {PINK: _basis(0), DARK: _basis(2), DANGO: _basis(4)}

    def embed(self, img):
        h, w = img.shape[:2]
        return self.table.get(tuple(int(x) for x in img[h // 2, w // 2]), _basis(1)).copy()


def make_g(dino=True, **cfg):
    return make(acfg_kw=dict(KW), dino=DinoGuard(ColorDino()) if dino else None, **cfg)


def striped(det, white_rows=3, period=8):
    """粉色里每 period 行掺 white_rows 行白：和纯粉的颜色余弦约 0.85（落在 unsure ~ match 之间）。"""
    img = paint((det, PINK))
    b = det.box
    for y in range(b.y, b.y2):
        if (y - b.y) % period < white_rows:
            img[y, b.x:b.x2] = WHITE
    return img


def run_img(w, det, dets, img, start, end, step=0.1):
    det.frames = [list(dets)]
    t = start
    while t <= end + 1e-9:
        w.process(img, round(t, 3), panel_visible=False)
        t += step
    return round(t - step, 3)


def newest(gallery):
    return max(s.t for s in gallery.samples)


def test_learns_friend_only_on_tagged_frame():
    w, det = make_g()
    body = player(1000)
    run(w, det, [(body, PINK), (tag(990, 110), None)], 0.0, 1.0)
    prof = w.appearance.friends[XIAOMING]
    n, last = prof.n, newest(prof.gallery)
    assert n >= 3 and last == 1.0
    run(w, det, [(body, PINK)], 1.1, 3.0)  # 同一条轨迹，名字标签淡掉了：身份照旧，但不再学
    (p,) = players(w)
    assert p.data["name"] == XIAOMING and p.data["samples"] > n
    assert prof.n == n and newest(prof.gallery) == last


def test_dango_marked_player_track_feeds_dango_gallery():
    w, det = make_g()
    me = Detection("self", Rect(550, 470, 165, 395), 0.9)
    on_me = Detection("player", Rect(549, 468, 287, 399), 0.8)  # 团子身上的 player 框（同一个身体）
    run(w, det, [(me, DANGO), (on_me, DANGO)], 0.0, 0.5)
    n = w.appearance.me.n
    run(w, det, [(on_me, DANGO)], 0.6, 2.0)  # YOLO 不出 self 了：只剩 _mark_dango 认出来的 player 框
    (p,) = [t for t in w.last_tracks if t.cls == "player"]
    assert p.data.get("dango")
    assert w.appearance.me.n > n + 10  # 这 1.5 秒每帧都学进团子底库
    assert newest(w.appearance.me.gallery) == 2.0
    assert w.appearance.friends == {} and w.strangers(2.0) == 0


def dango_elsewhere(w, det, extra=()):
    """0~1 秒：团子（self 框）和小明（暗绿、挂标签）都在；镜头转开 1 秒，之后画面左边一个团子色的人（不在团子的位置上）。"""
    run(w, det, [(SELF_AT, DANGO), (player(1400), DARK), (tag(1390, 110), None)], 0.0, 1.0)
    assert w.appearance.me is not None and XIAOMING in w.appearance.friends
    run(w, det, [], 1.1, 2.0)
    return run(w, det, [(player(300), DANGO), *extra], 2.1, 5.1)


def test_dango_look_blocks_maybe_and_stranger():
    w, det = make_g()
    t = dango_elsewhere(w, det)
    (p,) = players(w)
    assert t - p.data["dango_look"] < 0.2 and not p.data.get("dango")
    assert not p.data.get("maybe") and not p.data.get("unsure")  # 颜色和小明一样，但 DINOv2 更像团子
    assert not p.data.get("stranger") and w.strangers(t) == 0
    assert w.people(t) == [] and w.nearest(t) is None
    (box,) = [b for b in w.overlay(t) if b["kind"] == "dango_look"]
    assert box["label"] == "像团子"
    assert w._catalog_who(p).kind == "self"


def test_dango_look_expires():
    w, det = make_g()
    t0 = dango_elsewhere(w, det)
    (p,) = players(w)
    body = Detection("player", p.box, 0.9)
    t = run(w, det, [(body, WHITE)], t0 + 0.1, t0 + DANGO_LOOK_HOLD - 0.4)  # 换成白色：不像团子了，但还在 HOLD 里
    assert players(w)[0].id == p.id and not p.data.get("stranger")
    t = run(w, det, [(body, WHITE)], t + 0.1, t0 + DANGO_LOOK_HOLD + 0.5)
    assert players(w)[0].id == p.id and p.data.get("stranger") is True
    assert [x.kind for x in w.people(t)] == ["stranger"]


def unsure_scene(w, det, until):
    """小明（粉、挂标签）0~1 秒，走开；2.5 秒起画面左边来了个粉里掺白的人（像小明，但拿不准）。"""
    run(w, det, [(player(1000), PINK), (tag(990, 110), None)], 0.0, 1.0)
    run(w, det, [], 1.1, 2.4)
    body = player(300)
    return run_img(w, det, [body], striped(body), 2.5, until), body


def test_striped_person_lands_in_unsure_band():
    cos = solid_feat(PINK) @ ColorEmbedder().embed(striped(player(300))[400:620, 318:372])
    assert 0.83 <= cos < 0.88


def test_unsure_holds_stranger_then_gives_up():
    w, det = make_g(dino=False)
    wait = w.appearance_cfg.unsure_wait + w.call_window
    t, body = unsure_scene(w, det, 2.5 + wait - 0.3)
    (p,) = players(w)
    name, since = p.data["unsure"]
    assert name == XIAOMING and since <= 2.8
    assert not p.data.get("maybe") and not p.data.get("stranger") and w.strangers(t) == 0
    (person,) = w.people(t)
    assert (person.kind, person.name, person.sure, person.unsure) == ("friend", XIAOMING, False, True)
    (box,) = [b for b in w.overlay(t) if b["kind"] == "unsure"]
    assert box["label"] == f"可能是{XIAOMING}?"
    assert XIAOMING not in w.nearby(t)  # 可能是：不刷新在场
    t = run_img(w, det, [body], striped(body), t + 0.1, since + wait + 0.5)
    assert players(w)[0].id == p.id
    assert p.data.get("unsure_miss") is True and "unsure" not in p.data
    assert p.data.get("stranger") is True and [x.kind for x in w.people(t)] == ["stranger"]
    t = run_img(w, det, [body], striped(body), t + 0.1, t + 3.0)
    assert "unsure" not in p.data  # 以后不再进"可能是"


def test_no_friend_gallery_goes_straight_to_stranger():
    w, det = make_g()
    w.appearance.load_cards({XIAOMING: card(WHITE, "白斗篷")})  # 关系卡里的旧特征一模一样：也不拿来认
    body = player(1500)
    det.frames = [[body]]
    img = paint((body, WHITE))
    for i in range(16):  # 0 ~ 1.5 秒
        t = round(i * 0.1, 3)
        w.process(img, t, panel_visible=False)
        (p,) = players(w)
        assert "unsure" not in p.data and "maybe" not in p.data and "dango_look" not in p.data, t
        assert p.data.get("stranger") is (t >= 1.0), t  # 到 stranger_after 就判，不多等 unsure_wait
    assert p.data["sid"] == "陌生人A"


def test_tag_clears_unsure_and_dango_look():
    w, det = make_g(dino=False)
    t, body = unsure_scene(w, det, 4.0)
    (p,) = players(w)
    assert p.data.get("unsure")
    label = tag(290, 110)
    t = run_img(w, det, [body, label], striped(body), t + 0.1, t + 0.3)
    assert p.data.get("name") == XIAOMING and "unsure" not in p.data and "dango_look" not in p.data
    assert [(x.name, x.sure, x.unsure) for x in w.people(t)] == [(XIAOMING, True, False)]

    w, det = make_g()
    t = dango_elsewhere(w, det)
    (p,) = players(w)
    assert p.data.get("dango_look")
    t = run(w, det, [(player(300), DANGO), (tag(290, 120), None)], t + 0.1, t + 0.3)
    assert p.data.get("name") == FANQIE and "dango_look" not in p.data and "unsure" not in p.data
    assert [(x.name, x.sure) for x in w.people(t)] == [(FANQIE, True)]


def test_unsure_tag_shown_elsewhere_drops_unsure():
    w, det = make_g(dino=False)
    t, body = unsure_scene(w, det, 4.0)
    (p,) = players(w)
    assert p.data.get("unsure")
    real = player(1500)  # 真的小明在别处亮出了名字
    img = striped(body)
    img[real.box.y:real.box.y2, real.box.x:real.box.x2] = PINK
    run_img(w, det, [body, real, tag(1490, 110)], img, t + 0.1, t + 0.3)
    assert "unsure" not in p.data and not p.data.get("unsure_miss")


def test_describe_unsure_person():
    text = describe_people([Person(1, "friend", "小明", R, "左边", "近", sure=False, unsure=True),
                            Person(2, "friend", "小红", R, "右边", "远", sure=False)])
    assert text == "可能是小明（没看到名字，左边·近）、像小红（没看到名字，右边·远）"


def test_unnamed_skips_dango_look_and_unsure():
    w, det = make_g()
    stranger = player(800)  # 离小明走开的地方够远：不会被按位置接回成他
    t = dango_elsewhere(w, det, extra=[(stranger, WHITE)])
    assert players(w)[0].data.get("dango_look") or players(w)[1].data.get("dango_look")
    assert w.unnamed(t) == 1  # 只数白色那个陌生人

    w, det = make_g(dino=False)
    t, body = unsure_scene(w, det, 4.0)
    assert players(w)[0].data.get("unsure") and w.unnamed(t) == 0
    t = run_img(w, det, [body], striped(body), t + 0.1, 2.5 + w.appearance_cfg.unsure_wait + w.call_window + 0.8)
    assert players(w)[0].data.get("unsure_miss") and w.unnamed(t) == 1


def test_appearance_identify_off_never_sets_new_keys():
    w, det = make(appearance=False)
    run(w, det, [(SELF_AT, DANGO), (player(300), DANGO), (player(1500), WHITE)], 0.0, 3.0)
    for p in players(w):
        assert not {"dango_look", "unsure", "unsure_miss", "sample"} & set(p.data)
