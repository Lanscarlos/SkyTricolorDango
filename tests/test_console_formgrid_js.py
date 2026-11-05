"""外形页「网格」的纯逻辑（console/static/formgrid.js）：用 node require 跑，没有 node 就跳过。"""
import importlib.resources
import json
import shutil
import subprocess

import pytest

STATIC = importlib.resources.files("skydango.console") / "static"


def _call(expr: str):
    node = shutil.which("node") or pytest.skip("没有 node")
    js = f"const G=require({json.dumps(str(STATIC / 'formgrid.js'))});process.stdout.write(JSON.stringify({expr}))"
    return json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)


def test_person_prob():
    assert _call("G.personProb({lit: 0.5, unlit: 0.25, spirit: 0.125, shared: 0.0625, morph: 0.0625, not_person: 0})") == 1.0
    assert _call("G.personProb({lit: 0.5, not_person: 0.5})") == 0.5  # 缺的类按 0
    assert _call("G.personProb({})") is None and _call("G.personProb(undefined)") is None


def test_grid_items_order_and_filter():
    items = [{"crop": c} for c in ("d", "c", "b", "a", "e", "f")]
    scores = {"a": {"lit": 0.9}, "b": {"lit": 0.1}, "c": {"lit": 0.1}, "e": {"lit": 0.5}}
    out = _call(f"G.gridItems({json.dumps(items)}, c => c.crop !== 'e', new Set(['a']), {json.dumps(scores)}).map(c => c.crop)")
    # e 不匹配、a 核过了；b / c 同分按裁图名；没分数的 d / f 排最后
    assert out == ["b", "c", "d", "f"]


def test_page():
    assert _call("G.PAGE") == 36
    lst = [{"crop": str(i)} for i in range(40)]
    assert len(_call(f"G.page({json.dumps(lst)}, 0)")) == 36
    assert [c["crop"] for c in _call(f"G.page({json.dumps(lst)}, 1)")] == ["36", "37", "38", "39"]
    assert _call(f"G.page({json.dumps(lst)}, 2)") == []
    assert _call("G.pageCount(40)") == 2 and _call("G.pageCount(0)") == 0 and _call("G.pageCount(36)") == 1


def test_toggle_cycles():
    out = _call("""(() => {
      let m = new Map();
      m = G.toggle(m, 'a', 'not_person'); const s1 = m.get('a');
      m = G.toggle(m, 'a', 'discard'); const s2 = m.get('a');
      m = G.toggle(m, 'a', 'discard'); const s3 = m.has('a');
      m = G.toggle(m, 'a', 'not_person'); m = G.toggle(m, 'a', 'not_person');
      return [s1, s2, s3, m.has('a')];
    })()""")
    assert out == ["not_person", "discard", False, False]


def test_toggle_does_not_mutate():
    assert _call("(() => { const m = new Map([['a', 'discard']]); G.toggle(m, 'a', 'not_person'); return m.get('a'); })()") == "discard"


def test_submission_defaults_person():
    items = [{"crop": "a"}, {"crop": "b"}, {"crop": "c"}]
    out = _call(f"G.submission({json.dumps(items)}, new Map([['b', 'not_person'], ['c', 'discard'], ['zzz', 'discard']]))")
    assert out == {"person": ["a"], "not_person": ["b"], "discard": ["c"]}


def test_read_mode_defaults_single():
    assert _call("G.readMode(null)") == "single"
    assert _call("G.readMode({getItem: () => { throw new Error('x') }})") == "single"
    assert _call("G.readMode({getItem: () => null})") == "single"
    assert _call("G.readMode({getItem: () => 'grid'})") == "grid"
    assert _call("G.readMode({getItem: () => 'weird'})") == "single"
    assert _call("(() => { G.writeMode({setItem: () => { throw new Error('x') }}, 'grid'); return 1; })()") == 1
    assert _call("(() => { const m = {}; G.writeMode({setItem: (k, v) => { m[k] = v }}, 'grid'); return m; })()") == {"skydango.form.mode": "grid"}
