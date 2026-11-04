"""外形页「按轨迹」的纯逻辑（console/static/formgroup.js）：用 node require 跑，没有 node 就跳过。"""
import importlib.resources
import json
import shutil
import subprocess

import pytest

STATIC = importlib.resources.files("skydango.console") / "static"


def _call(expr: str):
    node = shutil.which("node") or pytest.skip("没有 node")
    js = f"const F=require({json.dumps(str(STATIC / 'formgroup.js'))});process.stdout.write(JSON.stringify({expr}))"
    return json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)


def item(crop, where="_unlabeled", confirmed=None):
    d = {"crop": crop, "where": where}
    if confirmed is not None:
        d["confirmed"] = confirmed
    return d


def test_build_groups_filters_and_singletons():
    items = [item("a"), item("b", "lit"), item("c"), item("d"), item("e", "lit"), item("f")]
    groups = [["a", "b", "c"], ["e"], ["gone", "f"]]
    out = _call(f"F.buildGroups({json.dumps(items)}, {json.dumps(groups)}, c => c.where === '_unlabeled').map(g => ({{key: g.key, members: g.members.map(c => c.crop), shown: g.shown.map(c => c.crop)}}))")
    assert out == [
        {"key": "a", "members": ["a", "b", "c"], "shown": ["a", "c"]},  # b 不匹配 → 只显示 2 张
        {"key": "f", "members": ["f"], "shown": ["f"]},  # 组里不存在的名字略过；e 全不匹配 → 整组不出现
        {"key": "d", "members": ["d"], "shown": ["d"]},  # 没进任何组的各自一组
    ]


def test_targets_respects_excluded():
    g = {"shown": [{"crop": "a"}, {"crop": "b"}, {"crop": "c"}]}
    assert _call(f"F.targets({json.dumps(g)}, new Set(['b']))") == ["a", "c"]
    assert _call(f"F.targets({json.dumps(g)}, new Set())") == ["a", "b", "c"]


def test_center_index():
    assert [_call(f"F.centerIndex({n})") for n in (1, 2, 5)] == [0, 0, 2]


def test_focus_match():
    forms = ["not_person", "lit", "unlit"]
    focus = "new Set(['a','b','c'])"
    assert _call(f"F.focusMatch({json.dumps(item('a'))}, {focus}, {json.dumps(forms)})") is True  # 待确认
    assert _call(f"F.focusMatch({json.dumps(item('b', 'lit', True))}, {focus}, {json.dumps(forms)})") is False  # 已确认
    assert _call(f"F.focusMatch({json.dumps(item('c', 'lit', False))}, {focus}, {json.dumps(forms)})") is True  # 导入未确认
    assert _call(f"F.focusMatch({json.dumps(item('x'))}, {focus}, {json.dumps(forms)})") is False  # 不在名单
    assert _call(f"F.focusMatch({json.dumps(item('a', '_discard'))}, {focus}, {json.dumps(forms)})") is False


def test_read_by_track_defaults_on():
    assert _call("F.readByTrack({getItem() { throw new Error('x'); }})") is True
    assert _call("F.readByTrack({getItem() { return null; }})") is True
    assert _call("F.readByTrack(null)") is True
    assert _call("F.readByTrack({getItem(k) { return k === 'skydango.form.byTrack' ? '0' : null; }})") is False
    assert _call("F.readByTrack({getItem() { return '1'; }})") is True


def test_write_by_track_swallows_errors():
    assert _call("(() => { const m = {}; F.writeByTrack({setItem(k, v) { m[k] = v; }}, false); return m; })()") == {"skydango.form.byTrack": "0"}
    assert _call("(() => { F.writeByTrack({setItem() { throw new Error('x'); }}, true); return 1; })()") == 1
