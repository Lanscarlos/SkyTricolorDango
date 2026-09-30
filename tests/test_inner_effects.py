import pytest
from skydango.inner.effects import NEUTRAL, Effects, effects


def test_effects_table():
    assert effects("平常", "精神") == Effects() == NEUTRAL
    assert effects("开心", "还行") == Effects(quota=1.5, addressed=1.5, wander=0.8)
    assert effects("烦", "精神") == Effects(quota=0.5, addressed=0.0, wander=1.2)
    assert effects("低落", "困") == Effects(quota=0.25, idle=0.6, slow=True, wander=3.0)  # 0.5 × 0.5
    assert effects("平常", "有点累").quota == 0.75
    assert effects("烦", "困").quota == 0.25  # 下限
    assert effects("开心", "困") == Effects(quota=0.75, addressed=1.5, idle=0.6, slow=True, wander=1.6)
    assert effects("不认识", "不认识") == Effects()


def test_wander_multiplier():  # 空闲注意力随意看的间隔倍数：困了看得少，和反射的 idle 方向相反
    from skydango.inner.effects import effects as fx

    assert fx("开心", "精神").wander == pytest.approx(0.8)
    assert fx("平常", "困").wander == pytest.approx(2.0)
    assert fx("低落", "有点累").wander == pytest.approx(1.95)
    assert fx("烦", "还行").wander == pytest.approx(1.2)
    assert fx("奇怪", "奇怪").wander == 1.0
