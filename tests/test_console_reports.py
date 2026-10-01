"""回放报告的列出 / 读取（console-redesign Task 3）。"""

from __future__ import annotations

import os

import pytest

from skydango.console.reports import MAX_REPORT, list_reports, read_report


def _write(path, text, mtime):
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_list_newest_first_only_md(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    _write(d / "放鸽子-20261001-113000.md", "# a", mtime=100)
    _write(d / "a-b-剧本-20261001-120000-2.md", "# b", mtime=200)
    _write(d / "notes.txt", "x", mtime=300)
    rs = list_reports(d)
    assert [r["name"] for r in rs] == ["a-b-剧本-20261001-120000-2.md", "放鸽子-20261001-113000.md"]
    assert rs[0]["scenario"] == "a-b-剧本" and rs[1]["scenario"] == "放鸽子"


def test_scenario_falls_back_to_stem(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    _write(d / "手写的.md", "x", mtime=1)
    assert list_reports(d)[0]["scenario"] == "手写的"


def test_missing_dir_is_empty(tmp_path):
    assert list_reports(tmp_path / "nope") == []


@pytest.mark.parametrize("name", ["../x.md", ".." + chr(92) + "x.md", "sub/x.md", "", "x.txt", "不存在.md", "..", "%2e%2e%2fx.md"])
def test_read_refuses_bad_names(tmp_path, name):
    d = tmp_path / "reports"
    d.mkdir()
    (tmp_path / "x.md").write_text("secret", encoding="utf-8")
    assert read_report(d, name)[0] == 404


def test_read_ok_and_too_big(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    (d / "a-20261001-000000.md").write_text("# 标题\n正文", encoding="utf-8")
    assert read_report(d, "a-20261001-000000.md") == (200, {"name": "a-20261001-000000.md", "text": "# 标题\n正文"})
    (d / "big.md").write_bytes(b"x" * (MAX_REPORT + 1))
    assert read_report(d, "big.md")[0] == 413
