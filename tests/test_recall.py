import time

import pytest

from skydango.chat.memory import MemoryStore
from skydango.chat.recall import recall

DAY = 86400.0
NOW = time.mktime((2026, 9, 29, 21, 0, 0, 0, 0, -1))


def store_with(tmp_path, turns):
    """turns：(几天前, 听到的, 团子说的)"""
    store = MemoryStore(tmp_path)
    for ago, heard, said in turns:
        store.history.append("新的聊天消息：\n" + heard, said, NOW - ago * DAY)
    return store


def test_keyword_finds_the_turn_with_date_and_both_sides(tmp_path):
    store = store_with(tmp_path, [
        (1, "懒洋洋大王：「明天跑暴风眼吗」", "行啊，明天晚上"),
        (1, "番茄炒蛋盖饭：「好饿」", "吃点东西"),
    ])
    out = recall(store, query="暴风眼", now=NOW)
    assert "懒洋洋大王：「明天跑暴风眼吗」 → 我：行啊，明天晚上" in out
    assert "9月28日 21:00" in out
    assert "新的聊天消息" not in out and "好饿" not in out


def test_keywords_match_what_dango_said(tmp_path):
    store = store_with(tmp_path, [(2, "懒洋洋大王：「周末干嘛」", "我要去霞谷看赛跑")])
    assert "霞谷看赛跑" in recall(store, query="霞谷", now=NOW)


def test_more_keyword_hits_rank_first_and_results_are_capped(tmp_path):
    turns = [(3 - i * 0.1, f"甲：「跑图{i}」", "好") for i in range(10)]  # i 越大越新
    turns.append((5, "乙：「去暴风眼跑图吗」", "去"))
    store = store_with(tmp_path, turns)
    out = recall(store, query="暴风眼 跑图", now=NOW, limit=3)
    assert "去暴风眼跑图吗" in out  # 两个词都中，虽然最旧也排进来
    assert out.count(" → 我：") == 3
    assert "跑图9" in out and "跑图8" in out and "跑图0" not in out  # 其余按新旧


def test_results_are_shown_oldest_first(tmp_path):
    store = store_with(tmp_path, [(1, "甲：「新的暴风眼」", "嗯"), (3, "甲：「旧的暴风眼」", "嗯")])
    out = recall(store, query="暴风眼", now=NOW)
    assert out.index("旧的暴风眼") < out.index("新的暴风眼")


def test_who_filters_turns_and_can_be_used_alone(tmp_path):
    store = store_with(tmp_path, [
        (1, "懒洋洋大王：「我加班」", "辛苦啦"),
        (1, "番茄炒蛋盖饭：「我也加班」", "你们都好惨"),
        (2, "番茄炒蛋盖饭：「在吗」", "在，懒洋洋大王刚走"),
    ])
    out = recall(store, query="加班", who="懒洋洋", now=NOW)
    assert "我加班" in out and "我也加班" not in out
    alone = recall(store, who="懒洋洋", now=NOW)
    assert "我加班" in alone and "懒洋洋大王刚走" in alone  # 对他说过的 / 提到他的也算


def test_days_limits_how_far_back(tmp_path):
    store = store_with(tmp_path, [(1, "甲：「近的暴风眼」", "嗯"), (20, "甲：「远的暴风眼」", "嗯")])
    out = recall(store, query="暴风眼", now=NOW)  # 默认 14 天
    assert "近的暴风眼" in out and "远的暴风眼" not in out
    assert "远的暴风眼" in recall(store, query="暴风眼", days=30, now=NOW)


def test_days_is_clamped(tmp_path):
    store = store_with(tmp_path, [(200, "甲：「很远的暴风眼」", "嗯")])
    assert "很远的暴风眼" not in recall(store, query="暴风眼", days=1000, now=NOW)  # 最多 90 天
    assert "最近 1 天" in recall(store, query="暴风眼", days=0, now=NOW)


def test_long_turns_are_truncated(tmp_path):
    store = store_with(tmp_path, [(1, "甲：「暴风眼" + "啊" * 500 + "」", "嗯")])
    line = next(l for l in recall(store, query="暴风眼", now=NOW).splitlines() if "甲：" in l)
    assert len(line) <= 220 and line.endswith("…")


def test_notes_and_inbox_lines_are_included(tmp_path):
    store = store_with(tmp_path, [])
    store.write_notes("## 懒洋洋大王\n- 约好10月3日跑暴风眼\n- 喜欢樱花发型")
    store.add_memos(["番茄炒蛋盖饭也想去暴风眼"])
    out = recall(store, query="暴风眼", now=NOW)
    assert "笔记里" in out and "约好10月3日跑暴风眼" in out and "番茄炒蛋盖饭也想去暴风眼" in out
    assert "樱花" not in out


def test_nothing_found_says_so_and_warns_not_to_make_it_up(tmp_path):
    store = store_with(tmp_path, [(1, "甲：「好饿」", "吃点东西")])
    out = recall(store, query="暴风眼", now=NOW)
    assert "没找到" in out and "共 1 轮" in out and "别顺着" in out


def test_needs_query_or_who(tmp_path):
    with pytest.raises(ValueError):
        recall(store_with(tmp_path, []), query="  ", who="", now=NOW)
