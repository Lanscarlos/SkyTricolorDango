import json

import numpy as np

from skydango.config import IconsConfig, PerceptionConfig
from skydango.imageio import imwrite
from skydango.vision.detect import Detection
from skydango.vision.icon_eval import cut, evaluate
from skydango.vision.track import Rect


class FakeTemplate:
    min_score = 0.75

    def classify(self, region, scales=None):
        return "sit", 0.9


class FakeGallery:
    match = 0.75

    def classify(self, crop):
        return "sit", 0.8


def _frame(tmp_path):
    img = np.full((1080, 1920, 3), 90, np.uint8)
    p = tmp_path / "images" / "a.jpg"
    imwrite(p, img)
    return p


def _detect(img):
    return [
        Detection("player", Rect(300, 500, 90, 220), 0.9),
        Detection("social_ring", Rect(320, 380, 80, 80), 0.9),    # 下面有人：不进
        Detection("social_ring", Rect(1200, 300, 80, 80), 0.9),   # 下面空：地图上
    ]


def test_evaluate_only_map_rings(tmp_path):
    p = _frame(tmp_path)
    out = tmp_path / "out"
    res = evaluate([p], _detect, FakeTemplate(), FakeGallery(), IconsConfig(), out)
    assert res["rings"] == 2 and res["map"] == 1 and res["agree"] == 1 and res["recall"] is None
    rows = [json.loads(s) for s in (out / "crops.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["owner"] == "map" and rows[0]["template"] == "sit"
    assert (out / "report.md").exists()
    assert list(out.glob("template-*.jpg")) and list(out.glob("dino-*.jpg"))
    assert (out / "owners" / "a.jpg").exists()


def test_evaluate_recall(tmp_path):
    p = _frame(tmp_path)
    labels = tmp_path / "labels"
    labels.mkdir()
    cls = PerceptionConfig().classes
    idx = cls.index("social_ring")
    (labels / "a.txt").write_text(f"{idx} {1240/1920} {340/1080} {80/1920} {80/1080}\n", encoding="utf-8")
    res = evaluate([p], _detect, FakeTemplate(), None, IconsConfig(), tmp_path / "o", labels=labels)
    assert res["recall"] == 1.0


def test_cut_numbering(tmp_path):
    img = np.full((600, 800, 3), 128, np.uint8)
    shot = tmp_path / "s.png"
    imwrite(shot, img)
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "sit.png").write_bytes(b"x")
    made = cut(shot, "sit", Rect(300, 200, 80, 80), tmp_path / "refs", tpl)
    names = sorted(p.name for p in made)
    assert "sit-2.png" in names and any(n.endswith(".jpg") for n in names)
    assert (tpl / "sit-2.png").exists() and (tmp_path / "refs" / "sit").is_dir()
    made2 = cut(shot, "sit", Rect(300, 200, 80, 80), tmp_path / "refs", tpl)
    assert any(p.name == "sit-3.png" for p in made2)


def test_icon_eval_cli_refuses_without_template_and_gallery(monkeypatch, tmp_path):
    """没有模板也没有 DINOv2 底库：直接退出说清楚，不出一份全是不认识的报告。"""
    import argparse

    import pytest

    import skydango.vision.detect as detect
    from skydango import cli
    from skydango.config import Config

    monkeypatch.setattr(detect, "make_detector", lambda *a, **k: object())
    _frame(tmp_path)
    cfg = Config()
    cfg.social.enabled = False
    cfg.icons.dino = str(tmp_path / "missing.onnx")
    args = argparse.Namespace(source=str(tmp_path / "images"), labels=None, model=None, output=str(tmp_path / "out"))
    with pytest.raises(SystemExit) as exc:
        cli._perception_icon_eval(cfg, args)
    assert "没有模板" in str(exc.value) and "DINOv2" in str(exc.value)
    assert not (tmp_path / "out").exists()
