import numpy as np
import pytest

from skydango.config import Config
from skydango.vision.appearance import ColorEmbedder, describe_crop, good_crop, make_embedder
from skydango.vision.bubbles import Rect
from skydango.vision.embed import cosine

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
    assert (c.enabled, c.model, c.match, c.card_match, c.margin, c.changed) == (False, "color", 0.85, 0.92, 0.05, 0.70)
    assert (c.size, c.norm, c.device, c.quota_wait, c.retry_after, c.describe_timeout) == (224, "imagenet", "cpu", 600.0, 60.0, 60.0)
