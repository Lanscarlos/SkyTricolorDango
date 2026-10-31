from datetime import date

import pytest

from skydango.chat.entries import Entry, parse_inbox, parse_notes, render_notes, show

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
