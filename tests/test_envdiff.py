from skydango.vision.envdiff import EnvSnapshot, diff_line, snapshot


def test_diff_line_lists_changes_in_fixed_order():
    a = EnvSnapshot(frozenset({"小明", "小红"}), 0, "云野")
    b = EnvSnapshot(frozenset({"小明", "阿白"}), 2, "雨林")
    assert diff_line(a, b) == "（这之间：阿白来了；小红走了；陌生人 0→2 个；到了雨林）"


def test_diff_line_empty_when_nothing_changed_or_no_before():
    a = EnvSnapshot(frozenset({"小明"}), 1, None)
    assert diff_line(a, a) == ""
    assert diff_line(None, a) == ""


def test_place_disappearing_is_not_reported():  # 地名提示过期不等于离开
    assert diff_line(EnvSnapshot(frozenset(), 0, "云野"), EnvSnapshot(frozenset(), 0, None)) == ""


def test_snapshot_works_without_perception():  # EnvWatcher 没有 strangers()
    env = type("E", (), {"nearby": lambda self, now: ["小明"], "place": None})()
    assert snapshot(env, 0.0) == EnvSnapshot(frozenset({"小明"}), 0, None)
