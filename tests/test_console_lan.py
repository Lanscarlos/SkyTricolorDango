"""管理面板开放给局域网（[console] host = "0.0.0.0"）：要设面板密码，本机直接用，别的设备先登录；Host 只多认 IP 写法、不认域名。"""

import http.client
import json
import urllib.parse

from skydango.config import ConsoleConfig
from skydango.console.server import COOKIE, ConsoleServer, auth_token, bind_host
from skydango.console.settings import SettingsStore
from skydango.vision.viewer import is_local_host, post_guard
from test_console_server import GOOD, FakeRunner, upstream  # noqa: F401

PW = '[env]\nSKYDANGO_CONSOLE_PASSWORD = "芝麻开门"\n'


def make(tmp_path, upstream, host="0.0.0.0", secrets=PW, remote=True):
    if secrets:
        (tmp_path / "secrets.toml").write_text(secrets, encoding="utf-8")
    store = SettingsStore(tmp_path / "config.toml", environ={})
    srv = ConsoleServer(tmp_path / "config.toml", store, FakeRunner(), port=0, child_port=upstream.server_address[1],
                        device_checks=lambda cfg, make: [], make_device=lambda cfg: None, find_spec=lambda name: object(),
                        host=host)
    srv.login_delay = 0
    if remote:
        srv.trusted_client = lambda ip: False  # 测试只能从本机连，假装是别的设备
    srv.start()
    return srv


def call(srv, method, path, body=b"", headers=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    h = {"Host": host or f"192.168.1.5:{srv.port}", **(headers or {})}
    conn.request(method, path, body=body or None, headers=h)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, dict(r.getheaders()), data


def login(srv, password):
    body = urllib.parse.urlencode({"password": password}).encode()
    return call(srv, "POST", "/login", body, {"Content-Type": "application/x-www-form-urlencoded"})


def test_default_host_is_loopback():
    assert ConsoleConfig().host == "127.0.0.1"


def test_bind_host_needs_password_for_lan():
    assert bind_host("127.0.0.1", "") == ("127.0.0.1", None)
    assert bind_host("0.0.0.0", "pw") == ("0.0.0.0", None)
    host, note = bind_host("0.0.0.0", "")
    assert host == "127.0.0.1" and "密码" in note and "SKYDANGO_CONSOLE_PASSWORD" in note


def test_ip_host_only_when_allowed():
    assert not is_local_host("192.168.1.5:19390", 19390)
    assert is_local_host("192.168.1.5:19390", 19390, allow_ip=True)
    assert is_local_host("[fe80::1]:19390", 19390, allow_ip=True)
    assert not is_local_host("evil.com:19390", 19390, allow_ip=True)  # 域名照样拒（防 DNS 重绑定）
    assert not is_local_host("192.168.1.5:80", 19390, allow_ip=True)
    headers = {"Host": "192.168.1.5:19390", "X-Skydango": "1", "Content-Type": "application/json", "Content-Length": "2"}
    assert post_guard(headers, 19390, 100)[0] == 403
    assert post_guard(headers, 19390, 100, allow_ip=True) is None


def test_default_server_listens_on_loopback_and_refuses_ip_host(tmp_path, upstream):  # noqa: F811
    srv = make(tmp_path, upstream, host="127.0.0.1", remote=False)
    try:
        assert srv._server.server_address[0] == "127.0.0.1"
        assert call(srv, "GET", "/api/state")[0] == 403
    finally:
        srv.stop()


def test_lan_without_password_falls_back_to_loopback(tmp_path, upstream):  # noqa: F811
    srv = make(tmp_path, upstream, secrets="", remote=False)
    try:
        assert srv._server.server_address[0] == "127.0.0.1" and "密码" in srv.lan_note
    finally:
        srv.stop()


def test_lan_binds_all_and_local_client_needs_no_login(tmp_path, upstream):  # noqa: F811
    srv = make(tmp_path, upstream, remote=False)
    try:
        assert srv._server.server_address[0] == "0.0.0.0" and srv.lan_note is None
        assert call(srv, "GET", "/api/state")[0] == 200  # 本机连的，IP 写法的 Host 也认
        assert call(srv, "GET", "/api/state", host=f"evil.com:{srv.port}")[0] == 403
    finally:
        srv.stop()


def test_remote_must_log_in(tmp_path, upstream):  # noqa: F811
    srv = make(tmp_path, upstream)
    try:
        st, h, _ = call(srv, "GET", "/")
        assert st == 303 and h["Location"] == "/login"
        assert call(srv, "GET", "/api/state")[0] == 401
        assert call(srv, "GET", "/console/static/common.js")[0] == 401
        assert call(srv, "POST", "/api/settings", b"{}", {**GOOD, "Content-Length": "2"})[0] == 401
        st, _, page = call(srv, "GET", "/login")
        assert st == 200 and "密码" in page.decode("utf-8") and "<script" not in page.decode("utf-8")
    finally:
        srv.stop()


def test_wrong_password_rejected(tmp_path, upstream):  # noqa: F811
    srv = make(tmp_path, upstream)
    try:
        st, h, page = login(srv, "不对")
        assert st == 401 and "Set-Cookie" not in h and "密码不对" in page.decode("utf-8")
        assert login(srv, "芝麻开门")[0] == 303
        # 登录也只认 IP / 本机写法的 Host
        body = urllib.parse.urlencode({"password": "芝麻开门"}).encode()
        assert call(srv, "POST", "/login", body, {"Content-Type": "application/x-www-form-urlencoded"},
                    host=f"evil.com:{srv.port}")[0] == 403
    finally:
        srv.stop()


def test_login_sets_cookie_that_unlocks(tmp_path, upstream):  # noqa: F811
    srv = make(tmp_path, upstream)
    try:
        st, h, _ = login(srv, "芝麻开门")
        assert st == 303 and h["Location"] == "/"
        cookie = h["Set-Cookie"]
        assert cookie.startswith(f"{COOKIE}={auth_token('芝麻开门')}") and "HttpOnly" in cookie and "SameSite=Strict" in cookie
        jar = {"Cookie": f"{COOKIE}={auth_token('芝麻开门')}"}
        assert call(srv, "GET", "/", headers=jar)[0] == 200
        st, _, data = call(srv, "GET", "/api/state", headers=jar)
        assert st == 200 and json.loads(data)
        st, _, _ = call(srv, "POST", "/api/run/stop", b"{}", {**GOOD, **jar, "Content-Length": "2"})
        assert st != 401 and st != 403
        assert call(srv, "GET", "/api/state", headers={"Cookie": f"{COOKIE}=fake"})[0] == 401
    finally:
        srv.stop()


def test_changed_password_logs_everyone_out(tmp_path, upstream):  # noqa: F811
    srv = make(tmp_path, upstream)
    try:
        jar = {"Cookie": f"{COOKIE}={auth_token('芝麻开门')}"}
        assert call(srv, "GET", "/api/state", headers=jar)[0] == 200
        (tmp_path / "secrets.toml").write_text('[env]\nSKYDANGO_CONSOLE_PASSWORD = "新的"\n', encoding="utf-8")
        assert call(srv, "GET", "/api/state", headers=jar)[0] == 401
        (tmp_path / "secrets.toml").write_text("", encoding="utf-8")  # 运行中把密码删了：别的设备一律不让进
        assert call(srv, "GET", "/api/state", headers=jar)[0] == 403
        assert login(srv, "")[0] == 403
    finally:
        srv.stop()


def test_trusted_client_is_loopback_only(tmp_path, upstream):  # noqa: F811
    store = SettingsStore(tmp_path / "config.toml", environ={})
    srv = ConsoleServer(tmp_path / "config.toml", store, FakeRunner(), port=0, child_port=upstream.server_address[1])
    assert srv.trusted_client("127.0.0.1") and srv.trusted_client("::1") and srv.trusted_client("::ffff:127.0.0.1")
    assert not srv.trusted_client("192.168.1.5") and not srv.trusted_client("fe80::1") and not srv.trusted_client("乱写")


def test_lan_urls_skip_loopback(monkeypatch):
    import skydango.console.server as mod

    monkeypatch.setattr(mod.socket, "getaddrinfo", lambda *a: [(0, 0, 0, "", ("127.0.0.1", 0)), (0, 0, 0, "", ("192.168.1.5", 0))])
    assert mod.lan_urls(19390) == ["http://192.168.1.5:19390/"]
    monkeypatch.setattr(mod.socket, "getaddrinfo", lambda *a: (_ for _ in ()).throw(OSError("没网")))
    assert mod.lan_urls(19390) == []
