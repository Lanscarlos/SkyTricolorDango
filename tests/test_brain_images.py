import base64

import numpy as np
import pytest

from skydango.brain.images import crop_view, difference, fit, image_block, is_black, label_note, thumb


def frame(color=(60, 90, 40)):
    return np.full((1080, 1920, 3), color, np.uint8)


def test_fit_scales_1080p_to_720p_but_never_up():
    assert fit(frame(), (1280, 720)).shape == (720, 1280, 3)
    small = np.zeros((360, 640, 3), np.uint8)
    assert fit(small, (1280, 720)) is small


def test_image_block_is_base64_jpeg():
    block = image_block(frame())
    data = base64.standard_b64decode(block["source"]["data"])
    assert block["type"] == "image" and block["source"]["media_type"] == "image/jpeg" and data[:2] == b"\xff\xd8"


def test_crop_view_maps_thumbnail_coords_to_full_frame():
    img = frame()
    img[300:400, 600:800] = 255
    crop, area = crop_view(img, 400, 200, 200, 100, view_w=1280, max_side=800)
    assert area == (600, 300, 300, 150)
    assert crop.shape == (150, 300, 3) and crop[:100, :200].min() == 255


def test_crop_view_downscales_big_regions_and_rejects_outside():
    crop, _ = crop_view(frame(), 0, 0, 1280, 720, view_w=1280, max_side=800)
    assert max(crop.shape[:2]) == 800
    with pytest.raises(ValueError):
        crop_view(frame(), 1300, 0, 50, 50, view_w=1280, max_side=800)


def test_scene_change_and_black_screen():
    a, b = thumb(frame((60, 90, 40))), thumb(frame((200, 180, 170)))
    assert a.shape == (36, 64)
    assert difference(a, a) == 0.0 and difference(a, b) > 0.25
    assert is_black(frame((5, 5, 5))) and not is_black(frame())


def test_label_note_maps_names_into_thumbnail():
    note = label_note({"懒洋洋大王": (1320, 300, 160, 44, 5.0)}, scale=1280 / 1920)
    assert "懒洋洋大王：名字在 (933, 200)" in note
    assert label_note({}, 1.0).startswith("图里没认出好友的名字")
