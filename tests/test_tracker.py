from skydango.chat.tracker import SeenTracker, SelfFilter, normalize, similar


def test_normalize():
    assert normalize("你好 呀！ Hi") == "你好呀hi"
    assert normalize("ＡＢＣ，") == "abc"


def test_similar_handles_ocr_jitter_and_partial():
    assert similar("一起去霞谷吗", "一起去霞各吗", 0.8)
    assert similar("你好呀一起去霞谷吗", "你好呀一起去霞谷", 0.8)
    assert not similar("你好", "再见", 0.8)


def test_tracker_dedupes_while_visible_and_expires():
    t = SeenTracker(ttl=10, threshold=0.8)
    assert t.observe("你好呀", 0)
    assert not t.observe("你好呀", 1)
    assert not t.observe("你好呀!", 9)  # 持续可见会刷新时间
    assert not t.observe("你好呀", 18)
    assert t.observe("你好呀", 40)  # 很久之后又说了一遍，算新的
    assert t.observe("另一句话", 40)


def test_self_filter_with_prefix():
    f = SelfFilter(window=60, threshold=0.8, prefix="【AI】")
    f.remember("【AI】好呀，一起走", 0)
    assert f.is_self("【AI】好呀，一起走", 5)
    assert f.is_self("AI好呀一起走", 5)  # OCR 丢了括号
    assert not f.is_self("好呀，一起走", 100)  # 过了窗口
    assert not f.is_self("去哪里", 5)


# ---- 视觉 Tracker.open_low ----
def _det(cls, score, x=100, y=100, w=60, h=160):
    from skydango.vision.bubbles import Rect
    from skydango.vision.detect import Detection
    return Detection(cls, Rect(x, y, w, h), score)


def test_open_low_creates_provisional_track():
    from skydango.vision.track import Tracker
    tr = Tracker(open_low=frozenset({"player", "player_unlit"}))
    out = tr.update([], 1.0, low=[_det("player", 0.3)])
    assert len(out) == 1
    assert out[0].strong_last == float("-inf")
    tid = out[0].id
    out = tr.update([_det("player", 0.8)], 1.1)
    assert [t.id for t in out] == [tid]
    assert out[0].strong_last == 1.1


def test_open_low_ignores_other_classes():
    from skydango.vision.track import Tracker
    tr = Tracker(open_low=frozenset({"player", "player_unlit"}))
    assert tr.update([], 1.0, low=[_det("name_tag", 0.3)]) == []
    assert tr.tracks == {}


def test_default_still_only_extends():
    from skydango.vision.track import Tracker
    tr = Tracker()
    assert tr.update([], 1.0, low=[_det("player", 0.3)]) == []
    assert tr.tracks == {}


def test_open_low_order_extended_before_opened():
    from skydango.vision.track import Tracker
    tr = Tracker(open_low=frozenset({"player"}))
    tr.update([_det("player", 0.8, x=100)], 1.0)
    out = tr.update([], 1.1, low=[_det("player", 0.3, x=600), _det("player", 0.3, x=100)])
    assert [t.box.x for t in out] == [100, 600]
    assert out[0].strong_last == 1.0 and out[1].strong_last == float("-inf")
