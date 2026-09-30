"""内心页的接口（spec 2026-09-30-inner-viewer §2）：inner/api.py 的解析 + viewer 的 /inner、/inner/forget。"""

import json

import pytest
from test_viewer import GOOD, request, viewer

from skydango.inner.api import EMPTY_SNAPSHOT, KINDS, forget_result, parse_forget


def test_parse_forget_good():
    assert KINDS == ("catchphrase", "joke", "opinion")
    assert parse_forget({"kind": "catchphrase", "text": " 害 "}) == ("catchphrase", "害", "", "")
    assert parse_forget({"kind": "joke", "text": "梗", "who": "小明"}) == ("joke", "梗", "小明", "")
    assert parse_forget({"kind": "opinion", "text": "丑", "topic": "雨林"}) == ("opinion", "丑", "", "雨林")
    assert parse_forget({"kind": "opinion", "topic": "雨林"}) == ("opinion", "", "", "雨林")


@pytest.mark.parametrize("req", [
    {"kind": "xx", "text": "害"}, {"text": "害"}, {"kind": "catchphrase"}, {"kind": "catchphrase", "text": "  "},
    {"kind": "joke", "text": 5}, {"kind": "opinion", "text": "丑"}, [], "害",
])
def test_parse_forget_bad(req):
    with pytest.raises(ValueError):
        parse_forget(req)


def test_forget_result_and_empty():
    assert forget_result("") == {"ok": True}
    assert forget_result("找不到这条（可能已经淡出了）") == {"ok": False, "error": "找不到这条（可能已经淡出了）"}
    assert EMPTY_SNAPSHOT["running"] is False and EMPTY_SNAPSHOT["mood"] is None and EMPTY_SNAPSHOT["log"] == []
    assert EMPTY_SNAPSHOT["persona"] is None and EMPTY_SNAPSHOT["wants"] == [] and EMPTY_SNAPSHOT["soft"] == []


# ---- viewer ----
SNAP = {"running": True, "at": 1.0, "mood": {"level": "开心", "text": "好"}, "energy": None, "grudge": None,
        "wants": [], "soft": [], "persona": None, "log": []}


def inner_viewer(forget_reply=""):
    v = viewer()
    calls = []
    v.inner = lambda: SNAP

    def forget(k, t, w, tp):
        calls.append((k, t, w, tp))
        if forget_reply is None:
            raise RuntimeError("身体已经停了")
        return forget_reply

    v.forget = forget
    return v, v.start(), calls


def test_inner_404_without_body():
    v = viewer()
    url = v.start()
    try:
        assert request(url + "inner")[0] == 404
        assert request(url + "inner/forget", b"{}", GOOD)[0] == 404
    finally:
        v.stop()


def test_inner_get():
    v, url, _ = inner_viewer()
    port = v._server.server_address[1]
    try:
        assert request(url + "inner") == (200, SNAP)
        assert request(url + "inner", None, {"Host": f"evil.example:{port}"})[0] == 403
    finally:
        v.stop()


def test_inner_forget_ok_and_reason():
    v, url, calls = inner_viewer()
    body = json.dumps({"kind": "joke", "text": "梗", "who": "小明"}).encode()
    try:
        assert request(url + "inner/forget", body, GOOD) == (200, {"ok": True})
        assert calls == [("joke", "梗", "小明", "")]
    finally:
        v.stop()
    v, url, _ = inner_viewer("找不到这条（可能已经淡出了）")
    try:
        assert request(url + "inner/forget", body, GOOD) == (200, {"ok": False, "error": "找不到这条（可能已经淡出了）"})
    finally:
        v.stop()


def test_inner_forget_refuses():
    v, url, calls = inner_viewer()
    port = v._server.server_address[1]
    body = json.dumps({"kind": "catchphrase", "text": "害"}).encode()
    try:
        assert request(url + "inner/forget", body, {"Content-Type": "application/json"})[0] == 403  # 缺头
        assert request(url + "inner/forget", body, {**GOOD, "Host": f"evil.example:{port}"})[0] == 403  # 非本机
        assert request(url + "inner/forget", "不是json".encode(), GOOD)[0] == 400
        assert request(url + "inner/forget", json.dumps({"kind": "xx", "text": "害"}).encode(), GOOD)[0] == 400
        assert calls == []
    finally:
        v.stop()


def test_inner_forget_body_gone_503():
    v, url, _ = inner_viewer(None)
    try:
        code, data = request(url + "inner/forget", json.dumps({"kind": "catchphrase", "text": "害"}).encode(), GOOD)
        assert code == 503 and data == {"ok": False, "error": "团子正忙，稍后再试"}
    finally:
        v.stop()
