from skydango.inner.effects import NEUTRAL, Effects, effects


def test_effects_table():
    assert effects("平常", "精神") == Effects() == NEUTRAL
    assert effects("开心", "还行") == Effects(quota=1.5, addressed=1.5)
    assert effects("烦", "精神") == Effects(quota=0.5, addressed=0.0)
    assert effects("低落", "困") == Effects(quota=0.25, idle=0.6, slow=True)  # 0.5 × 0.5
    assert effects("平常", "有点累").quota == 0.75
    assert effects("烦", "困").quota == 0.25  # 下限
    assert effects("开心", "困") == Effects(quota=0.75, addressed=1.5, idle=0.6, slow=True)
    assert effects("不认识", "不认识") == Effects()
