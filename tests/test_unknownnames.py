import json

import numpy as np

from skydango.vision.unknownnames import UnknownNames, collect


def img():
    return np.zeros((20, 60, 3), np.uint8)


def test_add_counts_similar_names_once_and_writes_jsonl(tmp_path):
    folder = tmp_path / "unknown_names"
    u = UnknownNames(folder, lambda: ["懒洋洋大王"], wall=lambda: 0.0)
    u.add("星星小铺", img())
    u.add("星星小铺", img())
    u.add("星星小铺子", img())  # OCR 多读了一个字：算同一个
    assert list(u.entries) == ["星星小铺"] and u.entries["星星小铺"]["count"] == 3
    rows = [json.loads(line) for line in (folder / "names.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["name"] == "星星小铺" and rows[0]["count"] == 3
    assert (folder / rows[0]["image"]).exists() and ":" not in rows[0]["image"]


def test_add_ignores_names_that_now_match_a_friend(tmp_path):
    u = UnknownNames(tmp_path / "u", lambda: ["懒洋洋大王"])
    u.add("懒洋洋大王", img())
    assert u.entries == {}


def _run(root, name, rows):
    folder = root / name / "unknown_names"
    folder.mkdir(parents=True)
    (folder / "names.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def test_collect_merges_runs_sorts_by_count_and_drops_known_friends(tmp_path):
    _run(tmp_path, "20260928-100000-dry", [
        {"name": "甲", "count": 2, "first": "2026-09-28 10:00:01", "last": "2026-09-28 10:05:00", "image": "001.jpg"},
        {"name": "乙", "count": 9, "first": "2026-09-28 10:00:01", "last": "2026-09-28 10:05:00", "image": "002.jpg"},
    ])
    _run(tmp_path, "20260928-110000-live", [
        {"name": "甲", "count": 2, "first": "2026-09-28 11:00:01", "last": "2026-09-28 11:09:00", "image": "001.jpg"},
        {"name": "丙", "count": 1, "first": "2026-09-28 11:00:01", "last": "2026-09-28 11:01:00", "image": "002.jpg"},
    ])
    (tmp_path / "not-a-run").mkdir()
    out = collect(tmp_path, 5, ["乙"])
    assert [r["name"] for r in out] == ["甲", "丙"]
    assert out[0]["count"] == 4 and out[0]["last"] == "2026-09-28 11:09:00"
    assert out[0]["image"] == str(tmp_path / "20260928-110000-live" / "unknown_names" / "001.jpg")
    assert [r["name"] for r in collect(tmp_path, 1, [])] == ["甲", "丙"]  # 只看最近一次
    assert collect(tmp_path / "nope", 5, []) == []
