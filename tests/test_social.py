import cv2
import numpy as np
from conftest import FakeDevice

from skydango.config import SocialConfig
from skydango.game.social import IconClassifier, Request, SocialHandler, load_icons

ICONS = load_icons("assets/social")  # 从真机录像里截的圆圈图标（80×80）


def paste(frame, kind, center):
    icon = cv2.imread(f"assets/social/{kind}.png")
    h, w = icon.shape[:2]
    x, y = center[0] - w // 2, center[1] - h // 2
    frame[y : y + h, x : x + w] = icon
    return frame


def scene(kind=None, center=(1400, 400)):
    frame = np.full((1080, 1920, 3), (60, 90, 40), np.uint8)  # 暗绿色草地
    if kind:
        paste(frame, kind, center)
    return frame


def test_classifier_tells_request_kinds_apart():
    clf = IconClassifier(ICONS)
    for kind in ("star", "hand", "hug", "highfive"):
        region = scene(kind)[400 - 56 : 400 + 56, 1400 - 56 : 1400 + 56]
        assert clf.classify(region)[0] == kind
    assert clf.classify(scene()[344:456, 1344:1456])[0] is None  # 什么都没有


def test_classifier_finds_icon_near_expected_position():
    clf = IconClassifier(ICONS)
    found = clf.find(scene("hand", (1460, 430)), "hand", (1400, 400))  # 人走了一点
    assert found is not None and abs(found[0] - 1460) <= 3 and abs(found[1] - 430) <= 3
    assert clf.find(scene("hug", (1460, 430)), "hand", (1400, 400)) is None


def draw_ring(frame):
    """触屏模式时左下角的摇杆圈（实测中心约 (230, 847)，半径约 40）。"""
    cv2.circle(frame, (230, 847), 40, (150, 150, 150), 4)
    return frame


class AcceptDevice(FakeDevice):
    """模拟真机（2026-09-27 实测）：
    - 键盘模式下第一下触摸只切到触屏模式（左下角出现摇杆圈），还会关掉聊天面板；
    - 触屏模式下点中圆圈：团子走过去（图标在动，站得近时图标不动），然后圆圈消失；
    - 牵上之后再点一下圆圈会松手（这就是要避免的）。
    """

    def __init__(self, kind="hand", touch=False, walk=True, walk_frames=2):
        self.kind = kind
        self.touch = touch
        self.walk = walk
        self.walk_frames = walk_frames
        self.hits = 0  # 真正生效的点击次数
        self.frames_since_hit = 0
        self.panel = True
        super().__init__([scene(kind)])

    def tap(self, x, y):
        super().tap(x, y)
        self.panel = False  # 点屏幕会关掉聊天面板
        if not self.touch:
            self.touch = True
            return
        self.hits += 1

    def hw_key(self, code):
        super().hw_key(code)
        if code == 46:
            self.panel = not self.panel

    def screenshot(self):
        if self.hits == 0:
            frame = scene(self.kind)
        else:
            self.frames_since_hit += 1
            if self.frames_since_hit <= self.walk_frames:
                frame = scene(self.kind, (1430, 420) if self.walk else (1400, 400))
            else:
                frame = scene()  # 牵上了：圆圈没了
        return draw_ring(frame) if self.touch else frame


def handler(device, **cfg):
    t = [0.0]

    def sleep(s):
        t[0] += s

    h = SocialHandler(
        device, SocialConfig(**cfg), IconClassifier(ICONS), friends=lambda: ["懒洋洋大王"],
        sleep=sleep, clock=lambda: t[0], panel_visible=lambda frame: device.panel,
    )
    return h, t


def taps(device):
    return [c for c in device.calls if c[0] == "tap"]


def test_keyboard_mode_wakes_with_one_tap_then_taps_once_more():
    device = AcceptDevice()
    h, _ = handler(device)
    assert h.accept(Request("懒洋洋大王", "hand", (1400, 400), 0.0)) is True
    assert len(taps(device)) == 2 and device.hits == 1  # 一下切模式，一下真正点中
    assert device.panel is True  # 点屏幕关掉的聊天面板重新打开了


def test_touch_mode_taps_only_once():
    device = AcceptDevice(touch=True)
    h, _ = handler(device)
    assert h.accept(Request("懒洋洋大王", "hand", (1400, 400), 0.0)) is True
    assert len(taps(device)) == 1 and device.hits == 1


def test_never_taps_again_when_friend_is_close_and_icon_does_not_move():
    """实测断手：好友站得近，团子几乎不用走，图标原地多停了一会儿，旧逻辑以为被吞了又点一下 → 松手。"""
    device = AcceptDevice(touch=True, walk=False, walk_frames=4)
    h, _ = handler(device)
    assert h.accept(Request("懒洋洋大王", "hand", (1400, 400), 0.0)) is True
    assert device.hits == 1


def test_gives_up_without_extra_taps_when_nothing_happens():
    device = AcceptDevice(touch=True, walk=False, walk_frames=999)
    h, _ = handler(device)
    assert h.accept(Request("懒洋洋大王", "hand", (1400, 400), 0.0)) is False
    assert device.hits == 1


def test_handle_accepts_friend_requests_with_cooldown_and_skips_strangers():
    device = AcceptDevice()
    h, t = handler(device)
    requests = {"懒洋洋大王": Request("懒洋洋大王", "hand", (1400, 400), 0.0)}
    assert h.handle(requests, now=0.5) == ["懒洋洋大王:hand"]
    assert "牵手" in h.describe(1.0) and "懒洋洋大王" in h.describe(1.0)
    assert h.handle(requests, now=5.0) == []  # 冷却中
    assert h.handle({"路人": Request("路人", "hand", (1400, 400), 5.0)}, now=5.5) == []  # 陌生人的牵手不接


def test_handle_waits_while_typing():
    device = AcceptDevice()
    device.shown = True  # 输入框开着：正在打字，点屏幕会打断
    h, _ = handler(device)
    assert h.handle({"懒洋洋大王": Request("懒洋洋大王", "hand", (1400, 400), 0.0)}, now=0.5) == []
    assert device.calls == []


def test_stale_requests_are_ignored():
    device = AcceptDevice()
    h, _ = handler(device)
    assert h.handle({"懒洋洋大王": Request("懒洋洋大王", "hand", (1400, 400), 0.0)}, now=30.0) == []


class BrokenImeDevice(FakeDevice):
    """实测 18:32：adb 起不来（0xC0000142），查输入框每次都失败。"""

    def __init__(self):
        super().__init__([scene("hand")])
        self.ime_calls = 0

    def ime_shown(self):
        self.ime_calls += 1
        raise RuntimeError("adb 失败 (3221225794)")


def test_adb_error_backs_off_instead_of_retrying_every_frame():
    dev = BrokenImeDevice()
    h = SocialHandler(dev, SocialConfig(), IconClassifier(ICONS), lambda: ["懒洋洋大王"])
    reqs = {"懒洋洋大王": Request("懒洋洋大王", "hand", (1400, 400), 0.0)}
    assert h.handle(reqs, 0.0) == []
    assert h.handle(reqs, 0.15) == [] and h.handle(reqs, 5.0) == []
    assert dev.ime_calls == 1  # 退避期间不再碰 adb（旧逻辑每 0.15 s 起一个 adb 进程）
    reqs = {"懒洋洋大王": Request("懒洋洋大王", "hand", (1400, 400), 10.5)}
    h.handle(reqs, 10.5)  # 退避（10 s）过了，再试一次
    assert dev.ime_calls == 2


def test_policy_overrides_config():
    h = SocialHandler(FakeDevice([scene()]), SocialConfig(), IconClassifier(ICONS), lambda: ["懒洋洋大王", "番茄炒蛋盖饭"])
    hand = Request("懒洋洋大王", "hand", (0, 0), 0.0)
    piggy = Request("番茄炒蛋盖饭", "piggyback", (0, 0), 0.0)
    assert h.allowed(hand) and h.allowed(piggy)
    h.set_policy("*", "piggyback", False)
    assert not h.allowed(piggy) and h.allowed(hand)
    h.set_policy("番茄炒蛋盖饭", "*", True)  # 点名的比“所有好友”优先
    assert h.allowed(piggy)
    assert not h.allowed(Request("路人", "hug", (0, 0), 0.0))
    h.set_policy("stranger", "hug", True)
    assert h.allowed(Request("路人", "hug", (0, 0), 0.0))
    text = h.describe_policy()
    assert text.startswith("好友默认接受：牵手") and "所有好友的背背：不接" in text


def test_classifier_knows_stranger_and_status_icons():
    """2026-09-28 录像 / 用户截图里截的：火焰（陌生人举着蜡烛走过来要点火）、陌生人平时、眼睛（在看留影 / 听音乐）、共享空间。"""
    clf = IconClassifier(ICONS)
    for kind in ("candle", "stranger", "eye", "shared"):
        region = scene(kind)[400 - 56 : 400 + 56, 1400 - 56 : 1400 + 56]
        assert clf.classify(region)[0] == kind, kind


def test_only_real_requests_count():
    from skydango.game.social import is_request

    assert all(is_request(k) for k in ("candle", "hand", "hug", "highfive", "piggyback"))
    assert not any(is_request(k) for k in ("star", "stranger", "eye", "shared", None))
