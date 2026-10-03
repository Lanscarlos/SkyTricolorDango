import numpy as np
import pytest

from skydango.vision.embed import unit
from skydango.vision.gallery import DUP, Gallery, Sample


def _basis(i, n=8):
    v = np.zeros(n, np.float32)
    v[i] = 1.0
    return v


V_PINK, V_WHITE, V_BLUE = _basis(0), _basis(1), _basis(2)


def near(v, s=0.95):
    """和 v 余弦相似度为 s 的单位向量（垂直分量放在最后一维）。"""
    perp = _basis(7)
    return unit(s * v + np.sqrt(1 - s * s) * perp)


def S(color, t=0.0, dino=None, h=300.0, pinned=False):
    return Sample(color=color, dino=dino, t=t, h=h, pinned=pinned)


def test_dup_constant():
    assert DUP == 0.97


def test_add_dedups_and_refreshes_time():
    g = Gallery(10)
    g.add(S(V_PINK, t=0))
    g.add(S(near(V_PINK, 0.99), t=5))
    assert len(g) == 1
    assert g.samples[0].t == 5


def test_best_is_max_over_samples():
    g = Gallery(10)
    g.add(S(V_PINK))
    g.add(S(V_BLUE))
    assert g.best(V_BLUE, "color") == pytest.approx(1.0)
    assert g.best(near(V_PINK, 0.9), "color") == pytest.approx(0.9, abs=1e-3)


def test_best_dino_none_when_missing():
    g = Gallery(10)
    assert g.best(V_PINK, "color") is None  # 空底库
    g.add(S(V_PINK, dino=None))
    assert g.best(V_PINK, "dino") is None  # 样本没有 dino
    assert g.best(None, "dino") is None  # 没有查询特征
    g.add(S(V_BLUE, dino=V_WHITE))
    assert g.best(V_WHITE, "dino") == pytest.approx(1.0)


def test_evicts_most_redundant_unpinned():
    g = Gallery(3)
    a, a2 = V_PINK, near(V_PINK, 0.95)
    g.add(S(a, t=0))
    g.add(S(a2, t=1))
    g.add(S(V_WHITE, t=2))
    g.add(S(V_BLUE, t=3))
    assert len(g) == 3
    colors = [s.color for s in g.samples]
    assert not any(np.array_equal(c, a) for c in colors)  # 旧的那张被挤掉
    assert any(np.array_equal(c, a2) for c in colors)


def test_pinned_never_evicted():
    g = Gallery(2)
    g.add(S(V_PINK, pinned=True))
    g.add(S(V_WHITE))
    g.add(S(V_BLUE))
    assert len(g) == 2
    assert g.pinned_count == 1
    assert any(np.array_equal(s.color, V_PINK) for s in g.samples)


def test_all_pinned_allows_overflow():
    g = Gallery(2)
    for v in (V_PINK, V_WHITE, V_BLUE):
        g.add(S(v, pinned=True))
    assert len(g) == 3


def test_samples_is_a_copy():
    g = Gallery(5)
    g.add(S(V_PINK))
    g.samples.clear()
    assert len(g) == 1
