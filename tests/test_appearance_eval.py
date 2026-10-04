"""认装扮离线标定（spec 2026-10-01-appearance §10）：相似度分布、建议值、藏标签重放、报告。"""

from types import SimpleNamespace

import numpy as np
import pytest

from skydango.config import AppearanceConfig
from skydango.vision.appearance_eval import Harvester, pair_scores, replay, report_md, suggest


def e(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    v[i] = 1.0
    return v


def test_suggest_threshold_from_distributions():
    s = suggest(same=[0.9] * 100, diff=[0.5] * 98 + [0.8, 0.95])
    assert s["match"] == pytest.approx(0.81) and s["margin"] == 0.05
    assert s["changed"] == pytest.approx(0.9)
    assert s["same_n"] == 100 and s["diff_n"] == 100
    assert "p5" in s["same_q"] and "p98" in s["diff_q"]


def test_suggest_changed_is_low_quantile_of_same():
    same = [i / 100 for i in range(1, 101)]  # 0.01 ~ 1.00
    s = suggest(same=same, diff=[0.1] * 50)
    assert s["changed"] == pytest.approx(float(np.percentile(same, 5)), abs=0.01)
    assert s["match"] == pytest.approx(0.11)


def test_suggest_without_samples():
    s = suggest(same=[], diff=[])
    assert s["match"] is None and s["changed"] is None and s["margin"] == 0.05
    assert s["same_n"] == 0 and s["diff_n"] == 0


def test_pair_scores_counts_and_caps():
    tracks = {"A": [e(0)] * 3, "B": [e(1)] * 2}
    same, diff = pair_scores(tracks)
    assert len(same) == 3 + 1 and all(v == pytest.approx(1.0) for v in same)
    assert len(diff) == 6 and all(v == pytest.approx(0.0) for v in diff)
    big = {"A": [e(0)] * 60, "B": [e(1)] * 60}
    same, diff = pair_scores(big)
    assert len(same) == 2 * (50 * 49 // 2)  # 同一身份最多 50 个样本
    assert len(diff) == 200  # 每对身份最多 200 对
    assert pair_scores(big) == (same, diff)  # 固定随机种子


def test_replay_counts_right_wrong_missed():
    seq = []
    for t in range(20):
        if t < 10:
            seq.append((float(t), "小明", e(0)))
        seq.append((float(t), "小红", e(1)))
    seq += [
        (16.0, "t7", e(0)),  # 撞衫的陌生人，藏标签时被认成小明 → 接错
        (17.0, "小刚", e(2)),  # 没学过的好友 → 漏接
        (18.0, "t9", e(3)),  # 谁也不像的陌生人 → 没认成好友（对）
    ]
    r = replay(sorted(seq, key=lambda x: x[0]), AppearanceConfig(), hide_every=5.0)
    # 0~5 s、10~15 s 露着标签学，5~10 s、15~20 s 藏掉标签判
    assert (r["right"], r["wrong"], r["missed"]) == (15, 1, 1)
    assert r["stranger_ok"] == 1
    assert r["wrong_rate"] == pytest.approx(1 / 16)


def test_report_mentions_tighten_when_wrong_rate_high():
    base = {"frames": 100, "identities": {"小明": 30, "t3": 10}, "suggest": suggest([0.9] * 10, [0.5] * 10),
            "current": {"match": 0.85, "changed": 0.7, "margin": 0.05}}
    bad = report_md({**base, "replay": {"right": 90, "wrong": 10, "missed": 5, "stranger_ok": 3, "wrong_rate": 0.1}})
    assert "往严了调" in bad and "小明" in bad
    good = report_md({**base, "replay": {"right": 99, "wrong": 1, "missed": 5, "stranger_ok": 3, "wrong_rate": 0.01}})
    assert "往严了调" not in good


def _track(tid, cls="player", name=None, feat=None, samples=0, last=1.0):
    data = {}
    if name:
        data["name"] = name
    if feat is not None:
        data["feat"], data["samples"] = feat, samples
    return SimpleNamespace(id=tid, cls=cls, last=last, data=data)


def test_harvester_takes_new_samples_and_names_tracks():
    h = Harvester()
    h.add([_track(1, feat=e(0), samples=1), _track(2, feat=e(1), samples=1), _track(3, cls="self", feat=e(2), samples=1)], 1.0)
    h.add([_track(1, name="小明", feat=e(0), samples=1, last=2.0)], 2.0)  # 样本数没变：不重复收
    h.add([_track(1, name="小明", feat=e(0), samples=2, last=3.0), _track(2, feat=e(1), samples=2, last=2.0)], 3.0)  # 2 这一帧没看到
    ids = h.by_identity()
    assert set(ids) == {"小明", "t2"}  # 轨迹后来挂上了名字：之前的样本也算他的；团子不收
    assert len(ids["小明"]) == 2 and len(ids["t2"]) == 1
    assert [x[1] for x in h.sequence()] == ["小明", "t2", "小明"]


@pytest.mark.parametrize("gallery", [False, True])
def test_cli_appearance_eval_writes_report(tmp_path, monkeypatch, gallery):
    """录像上跑感知层（强制打开认装扮、--embed 覆盖特征）：收到样本，写 report.md / summary.json。"""
    import json

    from conftest import FakeOcr

    from skydango import cli
    from skydango.imageio import imwrite
    from skydango.vision.bubbles import Rect
    from skydango.vision.detect import Detection

    class OnePlayer:
        providers = ["Fake"]
        imgsz = 960

        def detect(self, img):
            return [Detection("player", Rect(800, 300, 200, 500), 0.9)]

    src = tmp_path / "rec"
    src.mkdir()
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    frame[300:800, 800:1000] = (40, 60, 200)
    for i in range(12):
        imwrite(src / f"{i:04d}_{i * 0.5:06.2f}s.jpg", frame)
    monkeypatch.setattr("skydango.vision.detect.make_detector", lambda *a, **k: OnePlayer())
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda *a, **k: FakeOcr())
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: []))
    out = tmp_path / "out"
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "appearance-eval", str(src), "--embed", "color", "-o", str(out)]
             + (["--gallery"] if gallery else []))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 12 and summary["embed"] == "color-v1"
    assert len(summary["identities"]) == 1 and next(iter(summary["identities"])).startswith("t")
    assert next(iter(summary["identities"].values())) >= 3
    assert summary["current"]["match"] == AppearanceConfig().match
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "# 认装扮离线标定" in report
    assert ("gallery" in summary) == gallery and ("底库模式" in report) == gallery  # 不带 --gallery：和以前一样


# ---- 底库模式（身份底库 spec 2026-10-03 §7） ----
def _mk_samples():
    """两个好友 + 团子：同身份颜色余弦 0.95、跨身份 0.6；DINOv2 同理。"""
    from skydango.vision.gallery import Sample

    dim = 128
    r = 0.6 / 0.95
    s = float(np.sqrt(1 - r * r))

    def unit_(v):
        return (v / np.linalg.norm(v)).astype(np.float32)

    def base(kind):
        v = np.zeros(dim, np.float32)
        if kind == "A":
            v[0] = 1
        elif kind == "B":
            v[0], v[1] = r, s
        else:
            v[0], v[2] = r, s
        return v

    k = [3]

    def sample(kind, t):
        n = np.zeros(dim, np.float32)
        n[k[0]] = 1
        k[0] += 1
        v = unit_(np.sqrt(0.95) * base(kind) + np.sqrt(0.05) * n)
        return Sample(color=v, dino=v.copy(), t=t, h=0.3)

    seq = []
    for t in range(40):
        seq.append((float(t), "小明", sample("A", float(t))))
        seq.append((float(t), "小红", sample("B", float(t))))
    dango = [(float(t), sample("D", float(t))) for t in range(40)]
    return seq, dango


def test_gallery_eval_suggests_thresholds():
    from skydango.vision.appearance_eval import gallery_eval

    seq, dango = _mk_samples()
    g = gallery_eval(seq, dango)
    assert 0.6 < g["match"] <= 0.95
    assert g["unsure"] is not None and g["unsure"] <= g["match"]
    assert 0.6 < g["dango_match"] <= 0.95
    queries = g["queries"]
    assert queries > 0 and sum(g["bands"].values()) == queries
    assert set(g["bands"]) == {"maybe", "unsure", "below"}
    assert g["bands"]["maybe"] == queries  # 同身份 0.95 ≥ 建议的 match
    assert g["recall"]["match"] == pytest.approx(1.0) and g["recall"]["dango"] == pytest.approx(1.0)


def test_gallery_eval_untagged_and_empty():
    from skydango.vision.appearance_eval import gallery_eval

    seq, dango = _mk_samples()
    only_t = [(t, f"t{i % 3}", s) for i, (t, _, s) in enumerate(seq)]  # 全是没名字的轨迹：不当身份
    g = gallery_eval(only_t, [])
    assert g["match"] is None and g["bands"] == {"maybe": 0, "unsure": 0, "below": 0} and g["queries"] == 0
    assert g["dango_match"] is None and g["recall"]["dango"] is None
    g = gallery_eval([], [])
    assert g["match"] is None


def test_gallery_eval_dango_gallery_is_first_20_seconds():
    from skydango.vision.appearance_eval import gallery_eval

    seq, dango = _mk_samples()
    g = gallery_eval(seq, dango)
    assert g["dango_queries"] == 20  # t = 0..19 当底库，20..39 当查询


def test_report_has_gallery_section_only_with_gallery():
    from skydango.vision.appearance_eval import gallery_eval

    seq, dango = _mk_samples()
    base = {"frames": 10, "identities": {"小明": 3}, "suggest": suggest([0.9] * 10, [0.5] * 10),
            "current": {"match": 0.88, "changed": 0.4, "margin": 0.05, "unsure": 0.83, "dango_match": 0.80}}
    assert "底库模式" not in report_md(base)
    md = report_md({**base, "gallery": gallery_eval(seq, dango)})
    assert "底库模式" in md and "`unsure`" in md and "`dango_match`" in md


def test_harvester_gallery_collects_fresh_samples_and_dango():
    from skydango.vision.gallery import Sample

    def smp(t):
        return Sample(color=e(0), dino=e(1), t=t, h=0.3)

    def tr(tid, cls="player", name=None, sample=None, last=1.0, dango=False):
        t = _track(tid, cls=cls, name=name, last=last)
        if sample is not None:
            t.data["sample"] = sample
        if dango:
            t.data["dango"] = True
        return t

    h = Harvester()
    h.add([tr(1, name="小明", sample=smp(1.0)), tr(2, sample=smp(0.5)),  # 2 的样本是旧的：不收
           tr(3, cls="self", sample=smp(1.0)), tr(4, dango=True, sample=smp(1.0))], 1.0, gallery=True)
    h.add([tr(1, name="小明", sample=smp(2.0), last=2.0)], 2.0, gallery=True)
    seq = h.gallery_sequence()
    assert [(t, who) for t, who, _ in seq] == [(1.0, "小明"), (2.0, "小明")]
    assert [t for t, _ in h.dango_samples()] == [1.0]  # 同一帧团子的 self 框和 player 框只算一个
    h2 = Harvester()
    h2.add([tr(1, name="小明", sample=smp(1.0))], 1.0)  # 默认不收 Sample
    assert h2.gallery_sequence() == [] and h2.dango_samples() == []
