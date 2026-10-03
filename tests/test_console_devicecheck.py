import subprocess

import numpy as np

from skydango.config import Config
from skydango.console.devicecheck import run_checks
from skydango.device.adb import AdbError


def _maybe_raise(value):
    if isinstance(value, Exception):
        raise value
    return value


class Dev:
    """假设备：每个方法的返回值 / 异常由构造参数决定。"""

    def __init__(self, devices=("emulator-5554",), shot=None, ime="com.android.adbkeyboard/.AdbIME",
                 focus="  mCurrentFocus=Window{1 u0 com.netease.sky.vivo/com.tgc.sky.GameActivity}", key="/dev/input/event4"):
        self._devices, self._shot, self._ime, self._focus, self._key = devices, shot, ime, focus, key
        self.capture, self._capture_failed = None, False
        self.connected = False

    def connect(self):
        self.connected = True
        return "connected"

    def devices(self):
        return list(_maybe_raise(self._devices))

    def screenshot(self):
        return _maybe_raise(self._shot if self._shot is not None else np.zeros((1080, 1920, 3), np.uint8))

    def current_ime(self):
        return _maybe_raise(self._ime)

    def shell(self, *args):
        assert args == ("dumpsys", "window")
        return "  mFocusedApp=x\n" + _maybe_raise(self._focus)

    def _key_device(self):
        return _maybe_raise(self._key)


def ok_run(cmd, **kw):
    return subprocess.CompletedProcess(cmd, 0, stdout=b"Android Debug Bridge version 1.0.41\nVersion 34", stderr=b"")


def cfg(tmp_path):
    adb = tmp_path / "adb.exe"
    adb.write_text("x")
    c = Config()
    c.device.adb_path, c.device.serial = str(adb), "emulator-5554"
    return c


def test_all_green(tmp_path):
    checks = run_checks(cfg(tmp_path), lambda c: Dev(), run=ok_run)
    assert [c.key for c in checks] == ["adb", "online", "screen", "ime", "game", "keys"]
    assert [c.status for c in checks] == ["ok"] * 6, [(c.key, c.detail) for c in checks]
    assert checks[0].detail.startswith("Android Debug Bridge") and checks[2].data["thumb"]
    assert "1920×1080" in checks[2].detail and "adb screencap" in checks[2].detail and checks[5].detail == "/dev/input/event4"


def test_network_serial_connects_first(tmp_path):
    c = cfg(tmp_path)
    c.device.serial = "127.0.0.1:16384"
    dev = Dev(devices=("127.0.0.1:16384",))
    checks = run_checks(c, lambda c: dev, run=ok_run)
    assert dev.connected and checks[1].status == "ok"


def test_missing_adb_skips_rest(tmp_path):
    c = Config()
    c.device.adb_path = str(tmp_path / "nope.exe")
    checks = run_checks(c, lambda c: Dev(), run=ok_run, which=lambda p: None)
    assert [x.status for x in checks] == ["fail"] + ["skip"] * 5
    assert "adb.exe" in checks[0].hint and checks[1].detail == "前面一项没通过"


def test_adb_that_does_not_run_fails(tmp_path):
    def broken(cmd, **kw):
        raise OSError("不是可执行文件")

    checks = run_checks(cfg(tmp_path), lambda c: Dev(), run=broken)
    assert checks[0].status == "fail" and "不是可执行文件" in checks[0].detail


def test_wrong_serial_lists_devices(tmp_path):
    checks = run_checks(cfg(tmp_path), lambda c: Dev(devices=("127.0.0.1:16384",)), run=ok_run)
    assert checks[1].status == "fail" and checks[1].data["devices"] == ["127.0.0.1:16384"]
    assert [x.status for x in checks[2:]] == ["skip"] * 4


def test_warnings(tmp_path):
    dev = Dev(shot=np.zeros((720, 1280, 3), np.uint8), ime="com.baidu.input/.Ime", focus="mCurrentFocus=Window{launcher}",
              key=AdbError("没找到"))
    checks = run_checks(cfg(tmp_path), lambda c: dev, run=ok_run)
    assert [x.status for x in checks[2:]] == ["warn"] * 4 and "1280×720" in checks[2].detail


def test_exception_in_one_step_is_a_fail_not_a_crash(tmp_path):
    checks = run_checks(cfg(tmp_path), lambda c: Dev(shot=AdbError("adb 超时")), run=ok_run)
    assert checks[2].status == "fail" and "超时" in checks[2].detail and checks[3].status == "ok"


def test_device_factory_error_is_online_fail(tmp_path):
    def boom(c):
        raise AdbError("adb 失败")

    checks = run_checks(cfg(tmp_path), boom, run=ok_run)
    assert checks[1].status == "fail" and "adb 失败" in checks[1].detail


def test_capture_handle_is_closed(tmp_path):  # MuMu 截图句柄每次检测都新建，用完要关
    class Capture:
        closed = False

        def close(self):
            Capture.closed = True

    dev = Dev()
    dev.capture = Capture()
    run_checks(cfg(tmp_path), lambda c: dev, run=ok_run)
    assert Capture.closed


class ImeDev(Dev):
    def __init__(self, current="com.sohu.inputmethod.sogou/.SogouIME"):
        super().__init__(ime=current)
        self.closed, self.sets = False, []

        class Capture:
            def close(inner):
                self.closed = True

        self.capture = Capture()

    def list_imes(self):
        return ["com.sohu.inputmethod.sogou/.SogouIME", "com.android.adbkeyboard/.AdbIME"]

    def set_ime(self, ime_id):
        self.sets.append(ime_id)
        self._ime = ime_id


def test_ime_state_lists_with_labels(tmp_path):
    from skydango.console.devicecheck import ime_state

    dev = ImeDev()
    res = ime_state(cfg(tmp_path), lambda c: dev)
    assert res["ok"] and res["current"] == "com.sohu.inputmethod.sogou/.SogouIME"
    assert res["imes"] == [
        {"id": "com.sohu.inputmethod.sogou/.SogouIME", "label": "搜狗输入法", "adb": False},
        {"id": "com.android.adbkeyboard/.AdbIME", "label": "ADBKeyboard（团子打字用）", "adb": True},
    ]
    assert dev.closed  # MuMu 截图句柄用完就关


def test_ime_state_switches(tmp_path):
    from skydango.console.devicecheck import ime_state

    dev = ImeDev()
    res = ime_state(cfg(tmp_path), lambda c: dev, "com.android.adbkeyboard/.AdbIME")
    assert dev.sets == ["com.android.adbkeyboard/.AdbIME"] and res["current"] == "com.android.adbkeyboard/.AdbIME"


def test_ime_state_refuses_unknown_ime(tmp_path):  # 只切装了的（别把任意字符串塞进 adb shell）
    from skydango.console.devicecheck import ime_state

    dev = ImeDev()
    res = ime_state(cfg(tmp_path), lambda c: dev, "x; reboot")
    assert not res["ok"] and dev.sets == [] and "没装" in res["text"]


def test_ime_state_reports_errors(tmp_path):
    from skydango.console.devicecheck import ime_state

    def broken(c):
        raise AdbError("adb 失败")

    res = ime_state(cfg(tmp_path), broken)
    assert not res["ok"] and "adb 失败" in res["text"]
