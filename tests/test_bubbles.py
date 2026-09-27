import numpy as np

from conftest import H, W, scene
from skydango.config import BubbleConfig
from skydango.vision.bubbles import find_bubbles, roi_rect
from skydango.vision.ocr import OcrLine, join_lines
from skydango.vision.bubbles import Rect


def test_finds_bubbles_and_ignores_stars():
    img = scene([(400, 200, 300, 50), (800, 420, 220, 46)])
    rects = find_bubbles(img, BubbleConfig())
    assert len(rects) == 2
    first, second = rects
    assert abs(first.x - 400) <= 3 and abs(first.y - 200) <= 3
    assert abs(second.w - 220) <= 4


def test_empty_white_region_is_not_a_bubble():
    img = scene()
    img[100:300, 100:500] = 250  # 大片白色（云 / 雪地），里面没有字
    assert find_bubbles(img, BubbleConfig()) == []


def test_huge_bright_area_filtered_by_size():
    img = scene([(10, 10, int(W * 0.9), int(H * 0.5))])
    assert find_bubbles(img, BubbleConfig()) == []


def test_roi_limits_search_and_keeps_full_frame_coordinates():
    img = scene([(400, 100, 300, 50), (400, 650, 300, 50)])
    rects = find_bubbles(img, BubbleConfig(), roi=[0.0, 0.0, 1.0, 0.8])
    assert len(rects) == 1 and rects[0].y < 200
    img2 = scene([(900, 300, 250, 50)])
    rects2 = find_bubbles(img2, BubbleConfig(), roi=[0.5, 0.2, 1.0, 1.0])
    assert len(rects2) == 1 and abs(rects2[0].x - 900) <= 3


def test_roi_rect():
    r = roi_rect([0.25, 0.5, 0.75, 1.0], 1000, 800)
    assert (r.x, r.y, r.w, r.h) == (250, 400, 500, 400)


def test_join_lines_reading_order():
    lines = [
        OcrLine("一起去霞谷吗", 0.9, Rect(130, 20, 160, 34)),
        OcrLine("第二行", 0.9, Rect(20, 70, 80, 34)),
        OcrLine("你好呀", 0.9, Rect(20, 22, 90, 34)),
    ]
    assert join_lines(lines) == "你好呀一起去霞谷吗第二行"
    assert join_lines([]) == ""


def test_join_lines_drops_duplicate_punctuation_and_spaces_latin():
    lines = [
        OcrLine("你好呀，", 0.99, Rect(18, 15, 93, 36)),
        OcrLine("一起去霞谷吗", 0.99, Rect(127, 13, 163, 40)),
        OcrLine("，", 0.79, Rect(101, 37, 10, 8)),
    ]
    assert join_lines(lines) == "你好呀，一起去霞谷吗"
    latin = [OcrLine("hello", 0.9, Rect(0, 0, 50, 20)), OcrLine("sky", 0.9, Rect(60, 0, 30, 20))]
    assert join_lines(latin) == "hello sky"
