import json

import numpy as np
import pytest

from skydango.config import AppearanceConfig, Config
from skydango.vision.appearance import AppearanceBook, ColorEmbedder, CropSaver, describe_crop, good_crop, make_embedder
from skydango.vision.bubbles import Rect
from skydango.vision.embed import cosine, unit

# BGR 颜色
WHITE = (235, 235, 235)
BROWN_HAIR = (30, 45, 70)
PINK = (150, 100, 230)
PINK_DARKER = (75, 50, 115)
BLUE = (230, 110, 50)


def person(color_head, color_body, h=300, w=120):
    """合成"人"：上 30% 一种颜色、下 70% 另一种。"""
    img = np.zeros((h, w, 3), np.uint8)
    cut = int(h * 0.3)
    img[:cut] = color_head
    img[cut:] = color_body
    return img


def frame_with_person(box: Rect, size=(1080, 1920)):
    f = np.full((*size, 3), 90, np.uint8)
    f[box.y : box.y2, box.x : box.x2] = person(WHITE, PINK, box.h, box.w)
    return f


def test_color_embedder_separates_outfits():
    e = ColorEmbedder()
    pink, pink2, blue = e.embed(person(WHITE, PINK)), e.embed(person(WHITE, PINK, h=280)), e.embed(person(WHITE, BLUE))
    assert cosine(pink, pink2) > 0.95 and cosine(pink, blue) < 0.7
    assert abs(float(np.linalg.norm(pink)) - 1.0) < 1e-5


def test_color_embedder_head_matters():
    e = ColorEmbedder()
    assert cosine(e.embed(person(WHITE, PINK)), e.embed(person(BROWN_HAIR, PINK))) < 0.9


def test_same_color_different_brightness_is_close():  # 已知限制：亮度不同分不开
    e = ColorEmbedder()
    assert cosine(e.embed(person(WHITE, PINK)), e.embed(person(WHITE, PINK_DARKER))) > 0.85


def test_good_crop_rejects_small_and_overlap():
    f = frame_with_person(Rect(800, 400, 120, 300))
    assert good_crop(f, Rect(800, 400, 120, 300), [], [], 0.10, 0.2).shape[1] == 72  # 中间 60% 宽
    assert good_crop(f, Rect(800, 400, 30, 80), [], [], 0.10, 0.2) is None
    assert good_crop(f, Rect(800, 400, 120, 300), [Rect(830, 420, 120, 300)], [], 0.10, 0.2) is None
    assert good_crop(f, Rect(100, 400, 120, 300), [], [Rect(0, 0, 640, 900)], 0.10, 0.2) is None


def test_good_crop_rejects_box_mostly_covered_by_bigger_one():
    f = frame_with_person(Rect(800, 400, 120, 300))
    big = Rect(700, 300, 400, 600)  # IoU 很小，但把小框整个盖住
    assert good_crop(f, Rect(800, 400, 120, 300), [big], [], 0.10, 0.2) is None


def test_describe_crop_pads_and_clamps():
    f = frame_with_person(Rect(800, 400, 120, 300))
    c = describe_crop(f, Rect(800, 400, 120, 300))
    assert c.shape[:2] == (390, 156)  # 四周各扩 15%
    edge = describe_crop(f, Rect(0, 0, 100, 200))
    assert edge.shape[:2] == (230, 115)  # 夹到画面内


def test_make_embedder():
    cfg = Config().appearance
    assert isinstance(make_embedder(cfg), ColorEmbedder)
    assert ColorEmbedder().key == "color-v1"
    cfg.model = "foo.bin"
    with pytest.raises(ValueError):
        make_embedder(cfg)


def test_config_has_appearance_defaults():
    c = Config().appearance
    assert (c.enabled, c.model, c.match, c.card_match, c.margin, c.changed) == (False, "color", 0.85, 0.92, 0.05, 0.40)
    assert c.outfit_change is False  # 颜色特征判换装太不稳（10-01 标定），默认不判
    assert (c.size, c.norm, c.device, c.quota_wait, c.retry_after, c.describe_timeout) == (224, "imagenet", "cpu", 600.0, 60.0, 60.0)


# ---- AppearanceBook：特征直接用手写单位向量，不经裁图 ----
def _basis(i, n=8):
    v = np.zeros(n, np.float32)
    v[i] = 1.0
    return v


V_PINK, V_WHITE, V_BLUE = _basis(0), _basis(1), _basis(2)


def near(v, s=0.95):
    """和 v 余弦相似度为 s 的单位向量（垂直分量放在最后一维）。"""
    perp = _basis(7)
    return unit(s * v + np.sqrt(1 - s * s) * perp)


def book(keep=5.0, **kw):
    return AppearanceBook(AppearanceConfig(**kw), "color-v1", keep=keep)


def outfit(feat, key="color-v1", desc=""):
    return {"desc": desc, "feat": [float(x) for x in feat], "key": key, "first": "2026-10-01", "last": "2026-10-01"}


def test_learn_averages_and_assigns_maybe():
    b = book(match=0.85, margin=0.05)
    b.learn("friend", "小明", V_PINK, 0.0)
    assert b.assign_friends({7: near(V_PINK)}, exclude=set()) == {7: "小明"}


def test_learn_moving_average_and_crops():
    b = book(ema=0.5)
    p = b.learn("friend", "小明", V_PINK, 0.0, crop=(100, np.zeros((2, 2, 3), np.uint8)))
    assert p.n == 1 and np.allclose(p.feat, V_PINK)
    p = b.learn("friend", "小明", V_WHITE, 1.0)
    assert p.n == 2 and np.allclose(p.feat, unit(0.5 * V_PINK + 0.5 * V_WHITE), atol=1e-5)
    assert p.updated == 1.0 and len(p.crops) == 1
    for h in range(10):
        b.learn("friend", "小明", V_PINK, 2.0, crop=(h, np.zeros((1, 1, 3), np.uint8)))
    assert len(p.crops) == 5 and b.best_crop("friend", "小明") is not None
    assert b.best_crop("friend", "没有") is None


def test_name_shown_elsewhere_is_excluded():
    b = book()
    b.learn("friend", "小明", V_PINK, 0.0)
    assert b.assign_friends({7: V_PINK}, exclude={"小明"}) == {}


def test_two_tracks_one_name_only_best_wins():
    b = book()
    b.learn("friend", "小明", V_PINK, 0.0)
    assert b.assign_friends({1: near(V_PINK, 0.9), 2: near(V_PINK, 0.97)}, set()) == {2: "小明"}


def test_no_maybe_when_two_friends_look_alike():
    b = book()
    b.learn("friend", "小明", V_PINK, 0.0)
    b.learn("friend", "小红", near(V_PINK, 0.99), 0.0)
    assert b.assign_friends({1: V_PINK}, set()) == {}


def test_card_only_friend_needs_stricter_threshold():
    b = book(match=0.85, card_match=0.92)
    b.load_cards({"小明": [outfit(V_PINK, key="color-v1")]})
    assert b.assign_friends({1: near(V_PINK, 0.88)}, set()) == {}
    assert b.assign_friends({1: near(V_PINK, 0.95)}, set()) == {1: "小明"}


def test_card_feature_with_other_key_is_ignored():
    b = book()
    b.load_cards({"小明": [outfit(V_PINK, key="other", desc="粉斗篷")]})
    assert b.assign_friends({1: V_PINK}, set()) == {} and b.card_state("小明") == "new"
    assert b.look("friend", "小明") == "粉斗篷"


def test_load_cards_takes_last_outfit():
    b = book()
    b.load_cards({"小明": [outfit(V_BLUE, desc="蓝"), outfit(V_PINK, desc="粉")]})
    assert b.look("friend", "小明") == "粉"
    assert b.assign_friends({1: V_PINK}, set()) == {1: "小明"}


def test_still_like_prefers_learned_over_card():
    b = book(match=0.85, card_match=0.92)
    b.load_cards({"小明": [outfit(V_PINK)]})
    assert b.still_like(near(V_PINK, 0.95), "小明") and not b.still_like(near(V_PINK, 0.88), "小明")
    b.learn("friend", "小明", V_BLUE, 0.0)  # 学到的是蓝：以它为准
    assert b.still_like(near(V_BLUE, 0.9), "小明") and not b.still_like(V_PINK, "小明")
    assert not b.still_like(V_PINK, "没有这人")


def test_stranger_ids_reuse_new_back_and_forget():
    b = book(stranger_forget=1800, keep=5.0)
    assert b.stranger_id(V_WHITE, 0.0) == ("陌生人A", False)
    assert b.stranger_id(V_BLUE, 1.0) == ("陌生人B", False)
    assert b.stranger_id(near(V_WHITE), 2.0) == ("陌生人A", False)
    assert b.stranger_id(near(V_WHITE), 60.0) == ("陌生人A", True)  # 离开超过 keep 又认回来
    b.forget(60.0 + 1801)
    assert b.stranger_id(V_WHITE, 2000.0)[0] == "陌生人C"  # 字母不复用


def test_stranger_id_skips_ids_held_by_other_tracks():
    b = book(keep=5.0)
    assert b.stranger_id(V_WHITE, 0.0) == ("陌生人A", False)
    b.stranger_id(V_BLUE, 0.5)  # 陌生人B：第二像的候选也不该被拿来顶替
    assert b.stranger_id(near(V_WHITE), 1.0, exclude={"陌生人A"}) == ("陌生人C", False)  # 最像的被占着：新编号
    assert b.stranger_id(near(V_WHITE), 2.0, exclude={"陌生人B"})[0] in ("陌生人A", "陌生人C")


def test_stranger_letters_roll_over_after_z():
    b = book()
    ids = [b.stranger_id(np.eye(40, dtype=np.float32)[i], float(i))[0] for i in range(28)]
    assert ids[0] == "陌生人A" and ids[25] == "陌生人Z" and ids[26] == "陌生人AA" and ids[27] == "陌生人AB"


def test_card_state_and_drift():
    b = book(changed=0.70)
    b.load_cards({"小明": [outfit(V_PINK)]})
    b.learn("friend", "小明", near(V_PINK, 0.95), 0.0)
    assert b.card_state("小明") == "same"
    b.learn("friend", "小红", V_BLUE, 0.0)
    assert b.card_state("小红") == "new"
    b.learn("friend", "小白", V_WHITE, 0.0)
    b.load_cards({"小白": [outfit(V_BLUE)]})
    assert b.card_state("小白") == "changed"
    m = book(ema=1.0, changed=0.70)  # ema = 1：学一次就整个换掉
    m.learn("me", "", V_WHITE, 0.0)
    m.set_desc("me", "", "白斗篷", V_WHITE)
    assert not m.drifted("me", "") and m.look("me", "") == "白斗篷"
    m.learn("me", "", V_BLUE, 1.0)
    assert m.drifted("me", "")


def test_shift_moves_stranger_seen_but_not_past_now():
    b = book(keep=5.0)
    b.stranger_id(V_WHITE, 2.0)
    b.stranger_id(V_BLUE, 0.0)
    b.shift(7.0, 8.0)  # 感知暂停了 7 秒
    assert b.strangers["陌生人A"].seen == 8.0 and b.strangers["陌生人B"].seen == 7.0
    assert b.stranger_id(near(V_WHITE), 10.0) == ("陌生人A", False)  # 暂停的时间不算"走开"


def test_marked_change_settles_on_new_outfit_before_drifting_again():
    b = book(ema=0.2, changed=0.70, match=0.85)
    b.learn("friend", "小明", V_PINK, 0.0)
    b.set_desc("friend", "小明", "粉斗篷", V_PINK)
    while not b.drifted("friend", "小明"):
        b.learn("friend", "小明", V_BLUE, 1.0)
    b.mark_changed("friend", "小明")
    assert not b.drifted("friend", "小明")  # 平均特征还在往蓝挪：不再算一次
    for _ in range(30):
        b.learn("friend", "小明", V_BLUE, 2.0)
        assert not b.drifted("friend", "小明")
    for _ in range(30):  # 稳在蓝上之后又换成白：照样算
        b.learn("friend", "小明", V_WHITE, 3.0)
    assert b.drifted("friend", "小明")


CROP = person(WHITE, PINK)


def test_crop_saver_rate_limits_per_track_and_caps(tmp_path):
    s = CropSaver(tmp_path, save_every=2.0, save_max=3)
    assert s.offer(1, "小明", CROP, 0.0, {"t": 0.0}) and not s.offer(1, "小明", CROP, 1.0, {})
    assert s.offer(2, "t2", CROP, 1.0, {}) and s.offer(1, "小明", CROP, 2.5, {}) and not s.offer(3, "t3", CROP, 9.0, {})
    assert len(list((tmp_path / "crops").rglob("*.jpg"))) == 3
    rows = [json.loads(l) for l in (tmp_path / "appearance.jsonl").read_text("utf-8").splitlines()]
    assert rows[0]["file"].startswith("crops/小明/") and rows[0]["t"] == 0.0


def test_crop_saver_sanitizes_names(tmp_path):
    CropSaver(tmp_path, 2.0, 10).offer(1, 'a:b*?', CROP, 0.0, {})
    assert (tmp_path / "crops" / "a_b__").is_dir()


def test_crop_saver_folder_names_safe_on_windows():
    from skydango.vision.appearance import _folder_name
    assert [_folder_name(n) for n in ("", ".", "..", "  ", "a. ", "CON", "con", "Nul", "com1", "LPT9", "aux.txt")] \
        == ["_", "_", "_", "_", "a", "_CON", "_con", "_Nul", "_com1", "_LPT9", "_aux.txt"]
    assert [_folder_name(n) for n in ("小明", "COM10", "CONSOLE", "a:b*?")] == ["小明", "COM10", "CONSOLE", "a_b__"]


def test_crop_saver_failed_write_rate_limits(tmp_path, monkeypatch):
    import skydango.vision.appearance as appearance
    calls = []

    def boom(*a, **kw):
        calls.append(a)
        raise OSError("写不进去")

    monkeypatch.setattr(appearance, "imwrite", boom)
    s = CropSaver(tmp_path, save_every=2.0, save_max=10)
    with pytest.raises(OSError):
        s.offer(1, "小明", CROP, 0.0, {})
    assert s.offer(1, "小明", CROP, 1.0, {}) is False and len(calls) == 1  # 失败也隔 save_every 才再试


def test_crop_saver_same_millisecond_does_not_overwrite(tmp_path):
    s = CropSaver(tmp_path, 0.0, 10, wall=lambda: 100.0)
    assert s.offer(1, "小明", CROP, 0.0, {}) and s.offer(1, "小明", CROP, 1.0, {})
    assert len(list((tmp_path / "crops" / "小明").glob("*.jpg"))) == 2


def test_book_best_friend():
    cfg = AppearanceConfig(enabled=True)
    b = AppearanceBook(cfg, "k")
    e = ColorEmbedder()
    assert b.best_friend(e.embed(person(WHITE, PINK))) == (None, 0.0)
    b.learn("friend", "小明", e.embed(person(WHITE, PINK)), 0.0)
    b.learn("friend", "番茄", e.embed(person(WHITE, BLUE)), 0.0)
    name, score = b.best_friend(e.embed(person(WHITE, PINK)))
    assert name == "小明" and score > 0.95
