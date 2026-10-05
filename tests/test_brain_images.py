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


class OverlayEnv:
    def __init__(self, boxes):
        self.boxes = boxes

    def overlay(self, now):
        return list(self.boxes)


def test_scene_note_lists_people_by_kind_and_scales():
    from skydango.brain.images import scene_note

    env = OverlayEnv([
        {"x": 1580, "y": 560, "w": 100, "h": 140, "kind": "unlit", "label": "陌生人（没点火）"},
        {"x": 1170, "y": 440, "w": 96, "h": 140, "kind": "friend", "label": "懒洋洋大王"},
        {"x": 900, "y": 640, "w": 120, "h": 130, "kind": "self", "label": "团子"},
        {"x": 400, "y": 520, "w": 100, "h": 130, "kind": "stranger", "label": "陌生人"},
        {"x": 1170, "y": 380, "w": 96, "h": 40, "kind": "name", "label": "懒洋洋大王"},
        {"x": 10, "y": 10, "w": 50, "h": 50, "kind": "typing", "label": "正在输入"},
        {"x": 10, "y": 10, "w": 50, "h": 50, "kind": "player", "label": ""},
    ])
    assert scene_note(env, 0.0, 2 / 3).splitlines() == [
        "画面里认出的人（坐标按这张图）：", "- 懒洋洋大王：(812, 340) 附近", "- 陌生人（没点火，黑影）：(1087, 420) 附近",
        "- 陌生人：(300, 390) 附近", "- 团子（就是“你”自己）：(640, 470) 附近", "没列出的人都叫“陌生人”。"]


def test_scene_note_lists_maybe_after_friends():
    from skydango.brain.images import scene_note

    env = OverlayEnv([
        {"x": 400, "y": 520, "w": 100, "h": 130, "kind": "stranger", "label": "陌生人A"},
        {"x": 1500, "y": 440, "w": 100, "h": 140, "kind": "maybe", "label": "像懒洋洋大王?"},
        {"x": 1100, "y": 440, "w": 100, "h": 140, "kind": "friend", "label": "番茄炒蛋盖饭"},
    ])
    assert scene_note(env, 0.0, 1.0).splitlines()[1:4] == [
        "- 番茄炒蛋盖饭：(1150, 510) 附近", "- 像懒洋洋大王（没看到名字）：(1550, 510) 附近", "- 陌生人：(450, 585) 附近"]


def test_scene_note_name_only_and_empty():
    from skydango.brain.images import scene_note

    env = OverlayEnv([{"x": 300, "y": 90, "w": 120, "h": 45, "kind": "name", "label": "番茄炒蛋盖饭"}])
    assert "- 番茄炒蛋盖饭：头顶名字在 (360, 90)，人在名字下方" in scene_note(env, 0.0, 1.0)
    assert scene_note(OverlayEnv([]), 0.0, 1.0) == "画面里没认出人（可能被挡住、离得远，或者没人）\n没列出的人都叫“陌生人”。"


def test_scene_note_lists_objects():
    from skydango.brain.images import scene_note
    from skydango.vision.bubbles import Rect
    from skydango.vision.people import Thing

    class ThingEnv(OverlayEnv):
        def __init__(self, boxes, things):
            super().__init__(boxes)
            self.things = things

        def objects(self, now):
            return list(self.things)

    note = scene_note(ThingEnv([], [Thing(1, "bench", Rect(400, 500, 20, 120), "左边", "近")]), 0.0, 0.5)
    assert "画面里认出的东西（坐标按这张图）：\n- 座位：(205, 280) 附近" in note
    assert note.endswith("没列出的东西按你自己看到的说。")
    empty = scene_note(ThingEnv([], []), 0.0, 0.5)
    assert "认出的东西" not in empty and "没列出的东西" not in empty


def test_scene_note_lists_icons():
    from skydango.brain.images import scene_note
    from skydango.vision.icons_map import Icon
    from skydango.vision.track import Rect

    class IconEnv(OverlayEnv):
        def icons(self, now):
            return [Icon(1, "sit", "bench", "坐下", Rect(1000, 500, 100, 100), "右边")]

    plain = scene_note(OverlayEnv([]), 0.0, 0.5)
    text = scene_note(IconEnv([]), 0.0, 0.5)
    assert text.startswith(plain)
    assert text[len(plain):] == "\n画面里认出的图标（坐标按这张图）：\n- 坐下：(525, 275) 附近\n图标只说明那里能互动。"
