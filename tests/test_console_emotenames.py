import json

import cv2
import numpy as np
import pytest

from skydango.config import Config
from skydango.console.emotenames import EmoteNames, check_name, references
from skydango.imageio import imread, imwrite
from test_console_server import GOOD, make_server, request, upstream  # noqa: F401


def _icon(kind: str) -> np.ndarray:
    """深色格子上的米白色剪影（和动作列表里截下来的一样），几种形状互相不像。"""
    img = np.full((100, 100, 3), 40, np.uint8)
    w = (235, 235, 235)
    if kind == "bar":
        cv2.rectangle(img, (15, 45), (85, 55), w, -1)
    elif kind == "pole":
        cv2.rectangle(img, (45, 15), (55, 85), w, -1)
    elif kind == "ring":
        cv2.circle(img, (50, 50), 30, w, 6)
    elif kind == "ell":
        cv2.rectangle(img, (20, 15), (30, 85), w, -1)
        cv2.rectangle(img, (20, 75), (80, 85), w, -1)
    elif kind == "dots":
        cv2.circle(img, (25, 25), 9, w, -1)
        cv2.circle(img, (75, 75), 9, w, -1)
    elif kind == "tri":
        cv2.fillPoly(img, [np.array([[50, 15], [85, 85], [15, 85]])], w)
    return img


KINDS = ["bar", "pole", "ring", "ell", "dots", "tri"]
PNG = bytes([0x89]) + b"PNG"


def make_lib(tmp_path, named=None):
    """扫描图标 scan/001~006 = KINDS；named = {形状: 名字} 放进图标库（从扫描图复制的，带一点噪声和位移，像旧的截图）。"""
    lib = tmp_path / "emotes"
    (lib / "scan").mkdir(parents=True)
    for i, k in enumerate(KINDS, 1):
        imwrite(lib / "scan" / f"{i:03d}.png", _icon(k))
    imwrite(lib / "scan" / "_sheet.png", _icon("bar"))
    for k, name in (named or {}).items():
        img = np.roll(_icon(k), 3, axis=1)
        imwrite(lib / f"{name}.png", img)
    return lib


def by_id(state, i):
    return next(x for x in state["items"] if x["id"] == i)


def test_state_missing_scan(tmp_path):
    s = EmoteNames(tmp_path / "emotes").state()
    assert s["ok"] is False and "emotes scan" in s["text"]


def test_state_matches_by_silhouette_not_number(tmp_path):
    lib = make_lib(tmp_path, {"ring": "鞠躬", "dots": "害羞"})
    s = EmoteNames(lib).state()
    assert s["ok"] and s["total"] == 6 and s["named"] == 2
    assert [x["id"] for x in s["items"]] == ["001", "002", "003", "004", "005", "006"]  # _sheet 不算
    assert by_id(s, "003")["name"] == "鞠躬" and by_id(s, "005")["name"] == "害羞"
    assert by_id(s, "001")["name"] is None
    assert s["orphans"] == []


def test_state_orphan_library_icon(tmp_path):
    lib = make_lib(tmp_path)
    img = np.full((100, 100, 3), 40, np.uint8)
    cv2.putText(img, "X", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (235, 235, 235), 8)
    imwrite(lib / "怪动作.png", img)
    imwrite(lib / "_旧.png", _icon("bar"))  # 下划线开头的图标库不读
    s = EmoteNames(lib).state()
    assert s["named"] == 0 and s["orphans"] == ["怪动作"]


def test_each_library_icon_matches_one(tmp_path):
    lib = make_lib(tmp_path, {"ring": "鞠躬"})
    imwrite(lib / "scan" / "007.png", _icon("ring"))  # “最近使用”里的重复
    s = EmoteNames(lib).state()
    assert sum(1 for x in s["items"] if x["name"] == "鞠躬") == 1


def test_name_copies_scan_icon(tmp_path):
    lib = make_lib(tmp_path)
    code, body = EmoteNames(lib).name("002", "  挥手 ")
    assert code == 200 and body["ok"] and body["name"] == "挥手"
    assert (lib / "挥手.png").is_file() and (lib / "scan" / "002.png").is_file()
    assert by_id(EmoteNames(lib).state(), "002")["name"] == "挥手"


def test_rename_and_clear(tmp_path):
    lib = make_lib(tmp_path, {"ring": "鞠躬"})
    e = EmoteNames(lib)
    code, body = e.name("003", "深鞠躬")
    assert code == 200 and body["old"] == "鞠躬"
    assert (lib / "深鞠躬.png").is_file() and not (lib / "鞠躬.png").exists()
    assert np.array_equal(imread(lib / "深鞠躬.png"), np.roll(_icon("ring"), 3, axis=1))  # 改名不换图
    code, body = e.clear("003")
    assert code == 200 and body["old"] == "深鞠躬"
    assert not (lib / "深鞠躬.png").exists() and (lib / "_removed" / "深鞠躬.png").is_file()
    assert by_id(e.state(), "003")["name"] is None
    e.name("003", "深鞠躬")
    assert e.clear("003")[0] == 200  # 再清一次：_removed 里已有同名的也不覆盖
    assert len(list((lib / "_removed").glob("*.png"))) == 2


def test_same_name_is_noop(tmp_path):
    lib = make_lib(tmp_path, {"ring": "鞠躬"})
    code, body = EmoteNames(lib).name("003", "鞠躬")
    assert code == 200 and body["ok"] and (lib / "鞠躬.png").is_file()


def test_clear_unnamed_and_unknown_id(tmp_path):
    lib = make_lib(tmp_path)
    e = EmoteNames(lib)
    assert e.clear("001")[0] == 409
    assert e.name("999", "挥手")[0] == 404
    assert e.name("../emotes/scan/001", "挥手")[0] == 404
    assert e.icon("_sheet") is None and e.icon("..\\x") is None
    assert e.icon("001")[:4] == PNG


def test_duplicate_name_rejected(tmp_path):
    lib = make_lib(tmp_path, {"ring": "鞠躬"})
    code, body = EmoteNames(lib).name("001", "鞠躬")
    assert code == 409 and "重名" in body["text"]
    assert not (lib / "scan" / "001.png").samefile(lib / "鞠躬.png")


@pytest.mark.parametrize("bad", ["", "   ", "一二三四五六七八九十一二三", "a/b", "a\\b", "a:b", "问号?", "_私有", ".隐藏",
                                 "CON", "nul", "com1", "尾巴.", "<x>", "a\tb"])
def test_check_name_rejects(bad):
    assert check_name(bad)[1] is not None


@pytest.mark.parametrize("good", ["鞠躬", "欢呼2级", "张臂踢腿", "一二三四五六七八九十一二"])
def test_check_name_accepts(good):
    assert check_name(good) == (good, None)


def test_references():
    cfg = Config()
    cfg.social.after_light = "鞠躬"
    cfg.emotes.extra = ["鞠躬", "挥手"]
    cfg.reflex.addressed = ["挥手"]
    cfg.reflex.idle = ["挥手"]
    cfg.reflex.return_map = {"wave": "挥手"}
    assert references(cfg, "鞠躬") == ["social.after_light", "emotes.extra", "gesture.names.bow"]
    assert references(cfg, "挥手") == ["emotes.extra", "gesture.names.wave", "reflex.addressed", "reflex.idle", "reflex.return_map.wave"]
    assert references(cfg, "没人用") == []


def test_rename_reports_references(tmp_path):
    lib = make_lib(tmp_path, {"ring": "鞠躬"})
    cfg = Config()
    code, body = EmoteNames(lib, cfg).name("003", "深鞠躬")
    assert code == 200 and "social.after_light" in body["refs"]
    EmoteNames(lib, cfg).name("003", "鞠躬")
    code, body = EmoteNames(lib, cfg).clear("003")
    assert code == 200 and body["refs"] == ["social.after_light", "gesture.names.bow"]
    assert EmoteNames(lib, cfg).name("001", "新名字")[1]["refs"] == []


# ---- HTTP ----
def test_http_endpoints(tmp_path, upstream, monkeypatch):  # noqa: F811
    import urllib.request

    make_lib(tmp_path, {"ring": "鞠躬"})  # 默认 [wheel] library_dir = "emotes"，相对当前目录
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    s = make_server(tmp_path, upstream)
    try:
        u = s.url
        st, d = request(u + "api/emotes/state")
        assert st == 200 and d["ok"] and d["named"] == 1 and d["total"] == 6
        with urllib.request.urlopen(u + "api/emotes/icon?id=003", timeout=10) as r:
            assert r.headers["Content-Type"] == "image/png" and r.read()[:4] == PNG
        assert request(u + "api/emotes/icon?id=..%2Fscan%2F001")[0] == 404
        assert request(u + "api/emotes/state", headers={"Host": f"evil.com:{s.port}"})[0] == 403
        body = json.dumps({"id": "001", "name": "横杠"}).encode()
        assert request(u + "api/emotes/name", body, {"Content-Type": "application/json"})[0] == 403
        st, d = request(u + "api/emotes/name", body, GOOD)
        assert st == 200 and d["name"] == "横杠" and (tmp_path / "emotes" / "横杠.png").is_file()
        st, d = request(u + "api/emotes/clear", json.dumps({"id": "001"}).encode(), GOOD)
        assert st == 200 and d["old"] == "横杠"
        st, d = request(u + "api/emotes/name", json.dumps({"id": "003", "name": "a/b"}).encode(), GOOD)
        assert st == 400 and not d["ok"]
    finally:
        s.stop()


def test_page_has_names_tab():
    import importlib.resources

    static = importlib.resources.files("skydango.console") / "static"
    page = (static / "console.html").read_text(encoding="utf-8")
    assert 'data-tab="names"' in page
    for id_ in ("en-cols", "en-empty", "en-grid", "en-filter", "en-count", "en-bar", "en-big", "en-form", "en-name", "en-save", "en-save-t",
                "en-clear", "en-refs", "en-lib", "en-orphans", "en-title", "en-big-note"):
        assert f'id="{id_}"' in page, id_
    assert page.index('src="console/static/emotenames.js"') < page.index('src="console/static/labeling.js"')
    js = (static / "emotenames.js").read_text(encoding="utf-8")
    for api in ("api/emotes/state", "api/emotes/icon", "api/emotes/name", "api/emotes/clear"):
        assert api in js
    assert "innerHTML" not in js  # 名字一律 textContent
    lab = (static / "labeling.js").read_text(encoding="utf-8")
    assert 'if (LB.tab === "names") return;' in lab  # 动作名页开着时 1~9 / 回车不去标动作片段
