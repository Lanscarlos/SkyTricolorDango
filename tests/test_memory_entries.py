from datetime import date

import pytest

from skydango.chat.entries import (
    Entry,
    archive_line,
    check_rewrite,
    entry_line,
    parse_inbox,
    parse_notes,
    pinned_view,
    render_notes,
    memory_stats,
    parse_archive,
    render_shown,
    section_lines,
    settle,
    show,
    split_pinned,
)

NOTES = """## 番茄炒蛋盖饭
- 养仓鼠，叫芝麻球 〔2026-10-31｜长期｜仓鼠.名字〕
- 下周去青岛出差三天 〔2026-10-31｜出差〕
- 新耳机是粉色猫耳造型 〔2026-10-31〕

## 自己
- 无翼，跑图容易出事
"""


def test_parse_fields():
    es = parse_notes(NOTES)
    assert [(e.section, e.text, e.date, e.pinned, e.key) for e in es] == [
        ("番茄炒蛋盖饭", "养仓鼠，叫芝麻球", date(2026, 10, 31), True, "仓鼠.名字"),
        ("番茄炒蛋盖饭", "下周去青岛出差三天", date(2026, 10, 31), False, "出差"),
        ("番茄炒蛋盖饭", "新耳机是粉色猫耳造型", date(2026, 10, 31), False, ""),
        ("自己", "无翼，跑图容易出事", None, True, ""),  # 没方括号 = 长期、日期不详
    ]


def test_render_roundtrip():
    assert render_notes(parse_notes(NOTES)) == NOTES


def test_bad_bracket_is_text():
    (e,) = parse_notes("## 小明\n- 去了雨林 〔昨天｜长期〕\n")
    assert e.text == "去了雨林 〔昨天｜长期〕" and e.date is None and e.pinned


def test_other_brackets_are_text():
    es = parse_notes("## 小明\n- 去了雨林 [2026-10-31]\n- 爱吃辣【长期】\n- 换发型 〔2026-10-30|发型〕\n")
    assert [e.text for e in es] == ["去了雨林 [2026-10-31]", "爱吃辣【长期】", "换发型 〔2026-10-30|发型〕"]
    assert all(e.pinned and e.date is None for e in es)


def test_parse_inbox_speaker():
    es = parse_inbox("- 番茄炒蛋盖饭：煮了酸辣粉 〔2026-10-31〕\n- 我：不会吹萨克斯 〔2026-10-31｜长期〕\n- 随手一句\n")
    assert [(e.section, e.text, e.pinned) for e in es] == [
        ("番茄炒蛋盖饭", "煮了酸辣粉", False), ("自己", "不会吹萨克斯", True), ("", "随手一句", True)]


@pytest.mark.parametrize("d,want", [
    (date(2026, 10, 31), "今天"), (date(2026, 10, 30), "昨天"), (date(2026, 10, 21), "10 天前"),
    (date(2026, 10, 2), "10 月 2 日"), (date(2026, 9, 26), "上个月 09-26"), (date(2026, 3, 1), "3 月 1 日"),
    (date(2025, 9, 17), "去年 09-17"), (date(2024, 9, 17), "2024 年 9 月 17 日"),
])
def test_show_relative(d, want):
    e = Entry("小明", "去了雨林", d, False, "", "")
    assert show(e, date(2026, 10, 31)) == f"去了雨林（{want}）"


def test_show_future_date():
    e = Entry("小明", "约好跑图", date(2026, 11, 30), False, "", "")
    assert show(e, date(2026, 10, 31)) == "约好跑图（11 月 30 日）"


def test_show_unknown_date():
    assert show(Entry("自己", "无翼", None, True, "", ""), date(2026, 10, 31)) == "无翼"


# ---- Task 2：取代、补日期、淡出、把关、常驻切分 ----
T = date(2026, 10, 31)


def E(sec, text, d=None, pinned=False, key=""):
    return Entry(sec, text, d, pinned, key, entry_line(Entry(sec, text, d, pinned, key, ""))[2:])


def test_same_key_keeps_newest():
    old = [E("番茄", "早饭吃榴莲包子", date(2026, 10, 24), key="早饭")]
    new = [E("番茄", "早饭吃榴莲包子", date(2026, 10, 24), key="早饭"), E("番茄", "早饭吃油条", T, key="早饭")]
    keep, gone = settle(old, new, T, 30)
    assert [e.text for e in keep] == ["早饭吃油条"]
    assert gone == [(new[0], "被取代：早饭吃油条")]


def test_same_key_tie_keeps_later_written():
    new = [E("番茄", "早饭吃包子", T, key="早饭"), E("番茄", "早饭吃油条", T, key="早饭")]
    keep, gone = settle([], new, T, 30)
    assert [e.text for e in keep] == ["早饭吃油条"] and gone[0][0].text == "早饭吃包子"


def test_same_key_other_section_untouched():
    keep, gone = settle([], [E("番茄", "a", T, key="早饭"), E("小明", "b", T, key="早饭")], T, 30)
    assert len(keep) == 2 and gone == []


def test_fill_missing_date_from_similar_old():
    old = [E("番茄", "养仓鼠，叫芝麻球", date(2026, 10, 1), True)]
    keep, _ = settle(old, [E("番茄", "养仓鼠叫芝麻球", None, True)], T, 30)
    assert keep[0].date == date(2026, 10, 1)


def test_no_fill_when_not_similar():
    keep, _ = settle([E("番茄", "养仓鼠", date(2026, 10, 1), True)], [E("番茄", "要去青岛出差", None)], T, 30)
    assert keep[0].date is None


def test_recent_expires_pinned_stays():
    new = [E("番茄", "在追长风渡", date(2026, 9, 30)), E("番茄", "喜欢云野", date(2026, 1, 1), True)]
    keep, gone = settle([], new, T, 30)
    assert [e.text for e in keep] == ["喜欢云野"] and gone[0][1] == "近况过期"


def test_future_date_not_expired():
    keep, gone = settle([], [E("番茄", "约好跑图", date(2026, 11, 30))], T, 30)
    assert len(keep) == 1 and gone == []


def test_split_pinned_overflow_oldest_first_self_last():
    es = [E("自己", "x" * 600, None, True), E("番茄", "y" * 600, date(2026, 1, 1), True),
          E("番茄", "z" * 600, date(2026, 10, 1), True), E("番茄", "近况", T)]
    pinned, moved = split_pinned(es, T, 1500)
    assert [e.text[0] for e in moved] == ["y"] and {e.text[0] for e in pinned} == {"x", "z"}


def test_split_pinned_self_moved_last():
    es = [E("自己", "x" * 600, None, True), E("番茄", "y" * 600, date(2026, 1, 1), True)]
    pinned, moved = split_pinned(es, T, 700)
    assert [e.text[0] for e in moved] == ["y"] and [e.text[0] for e in pinned] == ["x"]
    pinned, moved = split_pinned(es, T, 100)
    assert [e.text[0] for e in moved] == ["y", "x"] and pinned == []


def test_check_rewrite_rejects_shrink_and_garbage():
    old = "## 番茄\n" + "".join(f"- 第{i}条 〔2026-10-31〕\n" for i in range(10))
    assert check_rewrite(old, "## 番茄\n- 第1条 〔2026-10-31〕\n") is not None
    assert check_rewrite(old, "乱七八糟\n" * 20 + "## 番茄\n- 一条\n") is not None
    assert check_rewrite(old, old) is None
    assert check_rewrite("## 番茄\n- 一条\n", "## 番茄\n") is None  # 旧的不到 6 条不管骤减


def test_archive_line():
    e = E("番茄", "在追长风渡", date(2026, 9, 30))
    assert archive_line(e, T, "近况过期") == "- 〔挪走 2026-10-31：近况过期〕在追长风渡 〔2026-09-30〕"


# ---- Task 4：常驻切分 / 全部渲染 ----
def test_pinned_view_splits():
    text, pool = pinned_view(NOTES, "- 番茄炒蛋盖饭：煮了酸辣粉 〔2026-10-31〕\n", date(2026, 10, 31), 1500)
    assert "养仓鼠，叫芝麻球（今天）" in text and "无翼" in text
    assert "青岛" not in text and "酸辣粉" not in text and "〔" not in text
    assert {e.text for e in pool} == {"下周去青岛出差三天", "新耳机是粉色猫耳造型", "煮了酸辣粉"}


def test_pinned_view_inbox_pinned_merges_into_section():
    text, pool = pinned_view(NOTES, "- 番茄炒蛋盖饭：爱喝奶茶 〔2026-10-31｜长期〕\n", date(2026, 10, 31), 1500)
    assert text.count("## 番茄炒蛋盖饭") == 1 and "爱喝奶茶（今天）" in text and pool and all(not e.pinned for e in pool)


def test_pinned_view_overflow_goes_to_pool():
    text, pool = pinned_view(NOTES, "", date(2026, 10, 31), 30)
    assert "养仓鼠" not in text and any(e.text == "养仓鼠，叫芝麻球" for e in pool)


def test_render_shown_all_entries():
    out = render_shown(NOTES, "- 我：不会吹萨克斯 〔2026-10-30｜长期〕\n", date(2026, 10, 31))
    assert "青岛出差三天（今天）" in out and "不会吹萨克斯（昨天）" in out and "〔" not in out


# ---- Task 7 ----
def test_reflect_notes_by_section():  # 现有 bug 的回归：条目行里不带名字
    lines = section_lines("## 番茄\n- 养仓鼠 〔2026-10-31〕\n## 小明\n- 腿摔了\n", "", ["番茄"], date(2026, 10, 31))
    assert lines == ["番茄：养仓鼠（今天）"]


def test_section_lines_limit_per_person_and_inbox():
    notes = "## 番茄\n" + "".join(f"- 第{i}条 〔2026-10-31〕\n" for i in range(15))
    lines = section_lines(notes, "- 番茄：随手记的 〔2026-10-30〕\n- 小明：别人的\n", ["番茄"], date(2026, 10, 31))
    assert len(lines) == 10 and lines[0] == "番茄：第0条（今天）"


def test_parse_archive():
    ((e, why),) = parse_archive("- 〔挪走 2026-10-31：近况过期〕在追长风渡 〔2026-09-30〕\n乱写的一行\n")
    assert (e.text, e.date, why) == ("在追长风渡", date(2026, 9, 30), "近况过期")


def test_memory_stats():
    n, chars, pool = memory_stats(NOTES, "- 番茄炒蛋盖饭：煮了酸辣粉 〔2026-10-31〕\n", date(2026, 10, 31), 1500)
    assert n == 2 and chars > 0 and pool == 3


# ---- 11-01 沙盒验收后补的：第一次重写别把日期都改成今天、删掉的条目进 archive ----
N1 = date(2026, 11, 1)


def test_legacy_stamped_today_takes_date_from_text():
    old = [E("自己", "2026年10月4日：YOLO 一直把卡洛认成陌生人", None, True)]
    new = [E("自己", "2026年10月4日：YOLO 一直把卡洛认成陌生人", N1, True)]
    keep, gone = settle(old, new, N1, 30)
    assert keep[0].date == date(2026, 10, 4) and gone == []
    assert keep[0].raw == "2026年10月4日：YOLO 一直把卡洛认成陌生人 〔2026-10-04｜长期〕"


def test_legacy_stamped_today_without_text_date_stays_unknown():
    old = [E("自己", "喜欢樱花头，觉得正太头挺可爱", None, True)]
    keep, _ = settle(old, [E("自己", "喜欢樱花头，觉得正太头挺可爱", N1, True)], N1, 30)
    assert keep[0].date is None and keep[0].pinned
    assert entry_line(keep[0]) == "- 喜欢樱花头，觉得正太头挺可爱"  # 没方括号 = 长期、日期不详，和原来一样


def test_text_date_beats_today_stamp():  # 随手记里的「2026-10-26 头疼」被模型盖上了今天
    keep, _ = settle([], [E("懒洋洋大王", "2026-10-26 头疼，在烦心某事", N1)], N1, 30)
    assert keep[0].date == date(2026, 10, 26)
    keep, _ = settle([], [E("自己", "10月26日约好一起看云没去成", N1, key="看云")], N1, 30)
    assert keep[0].date == date(2026, 10, 26)  # 没写年份按今年


def test_text_date_kept_when_mentioned_again_today():
    again = [E("懒洋洋大王", "2026-10-26 说的头疼今天还疼", N1)]
    keep, _ = settle([], [E("懒洋洋大王", "2026-10-26 说的头疼今天还疼", N1)], N1, 30, inbox=again)
    assert keep[0].date == N1


def test_bracketless_line_with_text_date_gets_that_date_and_stays_pinned():
    keep, gone = settle([], [E("懒洋洋大王", "2026年10月26日头疼，在烦心某事", None, True),
                             E("自己", "2026年9月30日和卡洛互相表白喜欢对方", None, True)], N1, 30)
    assert [(e.date, e.pinned) for e in keep] == [(date(2026, 10, 26), True), (date(2026, 9, 30), True)]
    assert keep[0].raw == "2026年10月26日头疼，在烦心某事 〔2026-10-26｜长期〕" and gone == []


def test_today_stamp_without_any_date_in_text_untouched():
    keep, _ = settle([], [E("番茄", "早饭吃油条", N1, key="早饭")], N1, 30)
    assert keep[0].date == N1


def test_dropped_old_entries_go_to_archive():
    old = [E("自己", "无翼，跑图容易出事", None, True), E("自己", "2026年10月22日：被饶总说在卡洛头上", None, True),
           E("自己", "把飞舞理解成夸我像蝴蝶", date(2026, 10, 26))]
    new = [E("自己", "无翼，跑图容易出事", None, True)]
    keep, gone = settle(old, new, N1, 30)
    assert [e.text for e in keep] == ["无翼，跑图容易出事"]
    assert [(e.text, why) for e, why in gone] == [
        ("2026年10月22日：被饶总说在卡洛头上", "整理时删掉"), ("把飞舞理解成夸我像蝴蝶", "整理时删掉")]


def test_dropped_old_entry_with_same_key_is_superseded():
    old = [E("番茄", "早饭吃榴莲包子", date(2026, 10, 24), key="早饭")]
    keep, gone = settle(old, [E("番茄", "早饭吃油条", N1, key="早饭")], N1, 30)
    assert gone == [(old[0], "被取代：早饭吃油条")]


def test_reworded_old_entry_not_counted_as_dropped():
    old = [E("懒洋洋大王", "2026年10月26日头疼，在烦心某事", None, True)]
    _, gone = settle(old, [E("懒洋洋大王", "2026-10-26 头疼，在烦心某事", date(2026, 10, 26))], N1, 30)
    assert gone == []


def test_moved_to_other_section_not_counted_as_dropped():
    old = [E("", "番茄养了仓鼠叫芝麻球", None, True)]
    _, gone = settle(old, [E("番茄", "番茄养了仓鼠叫芝麻球", N1, True)], N1, 30)
    assert gone == []


def test_parse_inbox_reflection_points_line():  # 下线反思写进 inbox 的「某天的要点：」
    es = parse_inbox("- 2026年11月2日（周一） 的要点：零要求夸和损分两条消息发\n")
    assert [(e.section, e.text, e.date, e.pinned) for e in es] == [
        ("自己", "零要求夸和损分两条消息发", date(2026, 11, 2), False)]
