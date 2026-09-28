import numpy as np

from skydango.chat.memory import MemoryStore
from skydango.chat.responder import build_system_prompt
from skydango.config import EnvConfig, ReplyConfig
from skydango.vision.bubbles import Rect
from skydango.vision.env import EnvWatcher
from skydango.vision.ocr import OcrLine


class FakeOcr:
    def __init__(self, texts):
        self.texts = list(texts)  # 每次 recognize 依次返回一组文字
        self.shapes = []

    def recognize(self, img):
        self.shapes.append(img.shape[:2])
        texts = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return [OcrLine(t, 0.99, Rect(10, 10, 100, 40)) for t in texts]


def frame():
    return np.zeros((1080, 1920, 3), np.uint8)


def watcher(ocr, names=("懒洋洋大王", "番茄炒蛋盖饭"), **cfg):
    return EnvWatcher(ocr, EnvConfig(**cfg), lambda: list(names), log_roi=[0.0, 0.0, 0.335, 0.855], background=False)


def test_sees_friends_by_name_label():
    w = watcher(FakeOcr([["懒洋洋大王", "●", "2级"]]))
    w.observe(frame(), 0.0, panel_visible=True)
    assert w.nearby(1.0) == ["懒洋洋大王"]
    assert "懒洋洋大王" in w.describe(1.0) and "番茄炒蛋盖饭" not in w.describe(1.0)


def test_ocr_typos_still_match_and_noise_is_ignored():
    w = watcher(FakeOcr([["懒洋详大王", "番茄炒蛋盖", "云", "X"]]))  # 名字标签偶尔少字、错字
    w.observe(frame(), 0.0, panel_visible=False)
    assert w.nearby(0.0) == ["懒洋洋大王", "番茄炒蛋盖饭"]


def test_scans_only_every_interval_and_forgets_after_keep():
    ocr = FakeOcr([["懒洋洋大王"], []])
    w = watcher(ocr, interval=5.0, keep=30.0)
    w.observe(frame(), 0.0, panel_visible=False)
    w.observe(frame(), 2.0, panel_visible=False)  # 没到间隔：不扫
    assert len(ocr.shapes) == 1
    w.observe(frame(), 6.0, panel_visible=False)  # 扫了，这次没看到名字
    assert len(ocr.shapes) == 2
    assert w.nearby(20.0) == ["懒洋洋大王"]  # 30 秒内看到过就算在旁边（标签会被挡住、会闪）
    assert w.nearby(31.0) == []


def test_panel_area_is_skipped_when_open():
    ocr = FakeOcr([[]])
    w = watcher(ocr, interval=0.0)
    w.observe(frame(), 0.0, panel_visible=True)
    w.observe(frame(), 1.0, panel_visible=False)
    (h1, w1), (h2, w2) = ocr.shapes
    assert w1 < w2  # 面板开着时不扫左边被挡住的那块
    assert h1 == h2 < 1080  # 底部的输入栏也不扫


def test_recognizes_place_names():
    w = watcher(FakeOcr([["云野", "懒洋洋大王"]]))
    w.observe(frame(), 100.0, panel_visible=False)
    assert w.place == "云野"
    assert "云野" in w.describe(101.0)


def test_describe_is_empty_before_anything_seen():
    w = watcher(FakeOcr([[]]))
    assert w.describe(0.0) == ""


def test_env_goes_into_system_prompt():
    prompt = build_system_prompt(ReplyConfig(), env="旁边能看到：懒洋洋大王")
    assert "## 现在的环境" in prompt and "旁边能看到：懒洋洋大王" in prompt
    assert "## 现在的环境" not in build_system_prompt(ReplyConfig())


def test_friend_names_from_friends_file(tmp_path):
    (tmp_path / "friends.md").write_text(
        "# 好友资料\n\n## 懒洋洋大王\n- 本名卡洛\n\n## 番茄炒蛋盖饭\n- 老登\n", encoding="utf-8"
    )
    assert MemoryStore(tmp_path).friend_names() == ["懒洋洋大王", "番茄炒蛋盖饭"]


def test_agent_feeds_frames_to_env_watcher():
    from conftest import FakeDevice

    from skydango.agent import Agent
    from skydango.config import Config

    cfg = Config()
    seen = []

    class SpyEnv:
        def observe(self, frame, now, panel_visible):
            seen.append((now, panel_visible))

    class NoRead:
        panel_closed_since = None

        def read(self, frame, now):
            return []

    t = [5.0]
    agent = Agent(cfg, FakeDevice([frame()]), NoRead(), None, None, None, clock=lambda: t[0], sleep=lambda s: None, env=SpyEnv())
    agent.step()
    assert seen == [(5.0, True)]


def test_scan_spots_interaction_requests_below_friend_labels():
    """顺带看好友名字下方圆圈里的图标：换成牵手等图标就是有请求。"""
    import cv2

    from skydango.game.social import IconClassifier, load_icons

    img = np.full((1080, 1920, 3), (60, 90, 40), np.uint8)
    icon = cv2.imread("assets/social/hand.png")
    cx, label_top, label_h = 1400, 300, 44
    cy = label_top + round(2.23 * label_h)
    img[cy - 40 : cy + 40, cx - 40 : cx + 40] = icon

    class LabelOcr:
        def recognize(self, region):
            # 名字标签在整张图 (1400-80, 300)；扫描区域从 (0, 0) 开始，面板关着
            return [OcrLine("懒洋洋大王", 0.99, Rect(cx - 80, label_top, 160, label_h))]

    w = EnvWatcher(LabelOcr(), EnvConfig(), lambda: ["懒洋洋大王"], log_roi=[0.0, 0.0, 0.335, 0.855],
                   background=False, icons=IconClassifier(load_icons("assets/social")), icon_offset=2.23)
    w.observe(img, 5.0, panel_visible=False)
    req = w.requests["懒洋洋大王"]
    assert req.kind == "hand" and abs(req.pos[0] - cx) <= 3 and abs(req.pos[1] - cy) <= 3 and req.seen_at == 5.0

    img[cy - 40 : cy + 40, cx - 40 : cx + 40] = cv2.imread("assets/social/star.png")  # 取消了
    w.observe(img, 10.0, panel_visible=False)
    assert "懒洋洋大王" not in w.requests


def test_scan_records_label_positions_and_circle_state():
    import cv2

    from skydango.game.social import IconClassifier, load_icons
    from skydango.vision.bubbles import roi_rect

    img = np.full((1080, 1920, 3), (60, 90, 40), np.uint8)
    cx, top, h = 1400, 300, 44
    cy = top + round(2.23 * h)
    img[cy - 40 : cy + 40, cx - 40 : cx + 40] = cv2.imread("assets/social/star.png")
    left = roi_rect([0.0, 0.0, 0.335, 0.855], 1920, 1080).x2  # 面板开着：扫描区域从面板右边开始

    class LabelOcr:
        def recognize(self, region):
            return [OcrLine("懒洋洋大王", 0.99, Rect(cx - 80 - left, top, 160, h))]  # OCR 给的是区域内坐标

    w = EnvWatcher(LabelOcr(), EnvConfig(), lambda: ["懒洋洋大王"], log_roi=[0.0, 0.0, 0.335, 0.855],
                   background=False, icons=IconClassifier(load_icons("assets/social")), icon_offset=2.23)
    w.observe(img, 5.0, panel_visible=True)
    assert w.labels["懒洋洋大王"] == (cx - 80, top, 160, h, 5.0)  # 换回整张图的坐标
    assert w.circles["懒洋洋大王"] == ("star", 5.0)

    img[cy - 40 : cy + 40, cx - 40 : cx + 40] = (60, 90, 40)  # 圆圈没了（牵着手时就是这样）
    w.observe(img, 10.0, panel_visible=True)
    assert w.circles["懒洋洋大王"] == (None, 10.0)


def test_env_watcher_hold_is_noop():
    w = watcher(FakeOcr([]))
    w.hold("camera")
    with w.held("blackout"):
        assert not w.paused
    w.release("camera")
    assert not w.paused


def test_match_names_filters_score_and_fuzzy_matches():
    from skydango.vision.env import match_names

    lines = [
        OcrLine("懒洋洋大玉", 0.95, Rect(0, 0, 10, 10)),  # OCR 错一个字也算
        OcrLine("番茄炒蛋盖饭", 0.5, Rect(0, 0, 10, 10)),  # 置信度太低
        OcrLine("好", 0.99, Rect(0, 0, 10, 10)),
    ]
    assert [n for n, _ in match_names(lines, ["懒洋洋大王", "番茄炒蛋盖饭"], 0.9)] == ["懒洋洋大王"]
