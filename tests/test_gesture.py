import numpy as np
import pytest

from skydango.config import GestureConfig
from skydango.imageio import imread, imwrite
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.gesture import (
    ClipBuffer,
    OnnxGestureClassifier,
    eligible,
    evaluate,
    extract_clips,
    load_clip,
    person_crop,
)


def test_person_crop_is_square_and_padded():
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[400:620, 1000:1090] = 200
    crop = person_crop(img, Rect(1000, 400, 90, 220), 112)
    assert crop.shape == (112, 112, 3)
    assert crop[0, 0].max() == 0 and crop[56, 56].min() == 200  # 四周放宽了一点，中间是人


def test_person_crop_at_the_edge_does_not_crash():
    img = np.zeros((1080, 1920, 3), np.uint8)
    assert person_crop(img, Rect(0, 0, 30, 60), 112).shape == (112, 112, 3)


def test_clip_buffer_keeps_fps_and_last_frames():
    buf = ClipBuffer(16, 8.0)
    for i in range(15):
        buf.push(i * 0.125, np.full((4, 4, 3), i, np.uint8))
    buf.push(15 * 0.125 - 0.1, np.full((4, 4, 3), 99, np.uint8))  # 离上一张太近：丢掉
    assert not buf.ready()
    buf.push(15 * 0.125, np.full((4, 4, 3), 15, np.uint8))
    buf.push(16 * 0.125, np.full((4, 4, 3), 16, np.uint8))
    assert buf.ready() and len(buf.clip()) == 16
    assert [int(f[0, 0, 0]) for f in buf.clip()] == list(range(1, 17))


def test_eligible_only_named_friends_near_or_middle_in_the_center():
    ref = 200.0
    center = Rect(900, 400, 90, 200)
    assert eligible("懒洋洋大王", center, 1920, ref, 0.8, 0.4)
    assert eligible("懒洋洋大王", Rect(900, 400, 90, 100), 1920, ref, 0.8, 0.4)  # 中
    assert not eligible(None, center, 1920, ref, 0.8, 0.4)  # 不知道是谁（陌生人）
    assert not eligible("懒洋洋大王", Rect(900, 400, 30, 60), 1920, ref, 0.8, 0.4)  # 远
    assert not eligible("懒洋洋大王", Rect(100, 400, 90, 200), 1920, ref, 0.8, 0.4)  # 偏到左边
    assert not eligible("懒洋洋大王", Rect(1700, 400, 90, 200), 1920, ref, 0.8, 0.4)


class PixelClassifier:
    """按片段第一帧的亮度"认"动作：0 → none，100 → wave，200 → bow；亮度 150 → bow 但没把握。"""

    def classify(self, clip):
        v = float(clip[0].mean())
        if v > 175:
            return "bow", 0.95
        if v > 125:
            return "bow", 0.5
        if v > 50:
            return "wave", 0.95
        return "none", 0.99


def write_clip(folder, value, n=16):
    folder.mkdir(parents=True)
    for i in range(n):
        imwrite(folder / f"{i:02d}.png", np.full((32, 32, 3), value, np.uint8))


def test_evaluate_precision_and_recall(tmp_path):
    for i, v in enumerate((100, 100, 100, 0)):  # 挥手：3 个认对，1 个漏了
        write_clip(tmp_path / "wave" / f"c{i}", v)
    write_clip(tmp_path / "none" / "c0", 100)  # 站着被认成挥手：错报
    write_clip(tmp_path / "none" / "c1", 0)
    write_clip(tmp_path / "bow" / "c0", 200)
    write_clip(tmp_path / "bow" / "c1", 150)  # 认成鞠躬但没把握：算没报
    (tmp_path / "_unlabeled").mkdir()
    r = evaluate(tmp_path, PixelClassifier(), GestureConfig())
    assert r["wave"] == {"tp": 3, "fp": 1, "fn": 1, "precision": 0.75, "recall": 0.75}
    assert r["bow"] == {"tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5}
    assert r["all"]["tp"] == 4 and r["all"]["precision"] == pytest.approx(0.8)
    assert r["all"]["recall"] == pytest.approx(4 / 6, abs=1e-3) and r["clips"] == 8


def test_load_clip_sorts_frames(tmp_path):
    write_clip(tmp_path / "c", 7, n=3)
    clip = load_clip(tmp_path / "c")
    assert len(clip) == 3 and int(clip[0][0, 0, 0]) == 7


class WalkingDetector:
    """一个人慢慢往右走；每帧还有一个名字标签（不该被切片段）。"""

    def __init__(self):
        self.i = 0

    def detect(self, img):
        self.i += 1
        return [Detection("player", Rect(800 + self.i * 2, 400, 90, 220), 0.9),
                Detection("name_tag", Rect(800, 330, 110, 44), 0.9)]


def test_extract_clips_cuts_16_frame_windows_per_track(tmp_path):
    frames = [(i * 0.125, np.zeros((1080, 1920, 3), np.uint8)) for i in range(40)]
    n = extract_clips(frames, WalkingDetector(), tmp_path, GestureConfig(), conf=0.35)
    assert n == 2
    clips = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(clips) == 2 and all(len(list(c.glob("*.jpg"))) == 16 for c in clips)
    assert ":" not in clips[0].name and "t0.00s" in clips[0].name
    assert imread(next(clips[0].glob("*.jpg"))).shape == (112, 112, 3)


def _mean_model(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("clip", TensorProto.FLOAT, [1, 16, 3, 112, 112])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("ReduceMean", ["clip"], ["logits"], axes=[1, 3, 4], keepdims=0)
    model = helper.make_model(helper.make_graph([node], "g", [inp], [out]), opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_onnx_gesture_classifier_softmax_over_labels(tmp_path):
    pytest.importorskip("onnxruntime")
    _mean_model(tmp_path / "g.onnx")
    clf = OnnxGestureClassifier(str(tmp_path / "g.onnx"), ["none", "wave", "bow"])
    blue = np.zeros((112, 112, 3), np.uint8)
    blue[..., 0] = 255  # BGR 的蓝 → RGB 第 3 个通道 → 第 3 个标签
    label, prob = clf.classify([blue] * 16)
    assert label == "bow" and 1 / 3 < prob < 1


def test_onnx_gesture_classifier_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        OnnxGestureClassifier(str(tmp_path / "nope.onnx"), ["none", "wave"])


# ---- 评审暂缓项：片段不能跨断档拼接、类别数启动时就检查 ----
def test_clip_buffer_restarts_after_a_gap():
    buf = ClipBuffer(16, 8.0)
    for i in range(6):
        buf.push(i * 0.125, np.full((4, 4, 3), i, np.uint8))
    for i in range(99, 109):  # 断了十几秒再回来
        buf.push(i * 0.125, np.full((4, 4, 3), i, np.uint8))
    assert not buf.ready() and len(buf.clip()) == 10
    assert int(buf.clip()[0][0, 0, 0]) == 99


def test_clip_buffer_tolerates_jittery_body_frames():
    buf = ClipBuffer(16, 8.0)
    t = 0.0
    for i in range(16):  # 身体约 0.15 s 一帧，偶尔 0.3 s
        buf.push(t, np.zeros((4, 4, 3), np.uint8))
        t += 0.3 if i % 5 == 4 else 0.15
    assert buf.ready()


def test_onnx_gesture_classifier_checks_label_count_up_front(tmp_path):
    pytest.importorskip("onnxruntime")
    _mean_model(tmp_path / "g.onnx")  # 输出 3 类
    with pytest.raises(ValueError, match="3"):
        OnnxGestureClassifier(str(tmp_path / "g.onnx"), ["none", "wave"])
