import tomllib

import pytest

from skydango.console.tomlfile import dumps, read, write_atomic

DATA = {
    "device": {"serial": 'emu "5554"\\x', "auto_connect": True, "adb_timeout": 10.0},
    "vision": {"bubble": {"min_value": 215}},
    "reply": {"friends": {"好友 昵称": "本名\n小明"}},
    "emotes": {"extra": ["拥抱", "鞠躬"]},
    "console": {"duration": 0.0, "brain": False},
}


def test_dumps_round_trips_through_tomllib():
    assert tomllib.loads(dumps(DATA)) == DATA


def test_dumps_puts_scalars_before_subtables():  # TOML 要求：表头之后的键归那张表
    text = dumps({"a": {"x": 1, "b": {"y": 2}}})
    assert text.index("x = 1") < text.index("[a.b]")
    assert tomllib.loads(text) == {"a": {"x": 1, "b": {"y": 2}}}


def test_dumps_rejects_unsupported_types():
    with pytest.raises(ValueError, match="None"):
        dumps({"a": {"x": None}})


def test_read_missing_file_is_empty(tmp_path):
    assert read(tmp_path / "nope.toml") == {}


def test_read_broken_file_names_the_file(tmp_path):
    p = tmp_path / "console.toml"
    p.write_text("[a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="console.toml 读不出来"):
        read(p)


def test_write_atomic_replaces_and_leaves_no_temp(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text("old", encoding="utf-8")
    write_atomic(p, "new = 1\n")
    assert p.read_text(encoding="utf-8") == "new = 1\n"
    assert [f.name for f in tmp_path.iterdir()] == ["x.toml"]


def test_dumps_escapes_delete_char():  # U+007F：json.dumps 不转义，tomllib 却不认
    data = {"reply": {"disclosure_prefix": "a\x7fb"}}
    assert tomllib.loads(dumps(data)) == data


def test_dumps_number_lists():  # 单价 [命中, 没命中, 输出]（spec 2026-10-06-model-usage §3.1）
    data = {"providers": {"deepseek": {"peak": 2.0, "prices": {"deepseek-flash": [0.02, 1, 4.0]}}}}
    assert tomllib.loads(dumps(data)) == data
    with pytest.raises(ValueError):
        dumps({"a": [True, 1]})
