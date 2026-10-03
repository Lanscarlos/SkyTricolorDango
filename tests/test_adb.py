import base64
import struct
import subprocess

import numpy as np
import pytest

from skydango.device.adb import AdbDevice, AdbError, parse_raw_screencap


def raw(w, h, header=16, rgba=(10, 20, 30, 255)):
    head = struct.pack("<III", w, h, 1) + (b"\0" * (header - 12))
    return head + bytes(rgba) * (w * h)


@pytest.mark.parametrize("header", [12, 16])
def test_parse_raw_screencap(header):
    img = parse_raw_screencap(raw(4, 3, header))
    assert img.shape == (3, 4, 3)
    assert tuple(img[0, 0]) == (30, 20, 10)  # RGBA -> BGR


def test_parse_raw_rejects_garbage():
    with pytest.raises(AdbError):
        parse_raw_screencap(b"\x89PNG....")
    with pytest.raises(AdbError):
        parse_raw_screencap(struct.pack("<III", 4, 3, 1) + b"\0" * 10)


class Recorder:
    def __init__(self, stdout=b"", returncode=0):
        self.cmds = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, cmd, capture_output, timeout):
        self.cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, b"boom")


def test_input_text_uses_base64_broadcast():
    rec = Recorder()
    dev = AdbDevice("127.0.0.1:16384", "adb", runner=rec)
    dev.input_text("你好 'hi'")
    cmd = rec.cmds[-1]
    assert cmd[:3] == ["adb", "-s", "127.0.0.1:16384"]
    assert cmd[3:8] == ["shell", "am", "broadcast", "-a", "ADB_INPUT_B64"]
    assert base64.b64decode(cmd[-1]).decode() == "你好 'hi'"


def test_tap_and_editor_action():
    rec = Recorder()
    dev = AdbDevice("emulator-5554", runner=rec)
    dev.tap(10.6, 20)
    dev.editor_action(4)
    assert rec.cmds[0][-4:] == ["input", "tap", "10", "20"]
    assert rec.cmds[1][-5:] == ["-a", "ADB_EDITOR_CODE", "--ei", "code", "4"]


def test_screenshot_raw_path():
    rec = Recorder(stdout=raw(8, 6))
    img = AdbDevice("x:1", runner=rec).screenshot()
    assert img.shape == (6, 8, 3)
    assert rec.cmds[0][-2:] == ["exec-out", "screencap"]


def test_errors_surface():
    rec = Recorder(returncode=1)
    with pytest.raises(AdbError, match="boom"):
        AdbDevice("x:1", runner=rec).tap(1, 1)


def test_enable_keyboard_requires_install():
    rec = Recorder(stdout=b"com.other/.Ime\n")
    with pytest.raises(AdbError, match="ADBKeyboard"):
        AdbDevice("x:1", runner=rec).enable_adb_keyboard()


GETEVENT = b"""add device 1: /dev/input/event5
  name:     "Xiaomi Joystick"
  events:
    KEY (0001): BTN_A                 BTN_B
add device 2: /dev/input/event4
  name:     "Xiaomi Input"
  events:
    KEY (0001): KEY_ESC               KEY_1                 KEY_ENTER*
    ABS (0003): ABS_MT_POSITION_X     : value 0, min 0, max 1080
add device 3: /dev/input/event0
  name:     "Power Button"
  events:
    KEY (0001): KEY_POWER
"""


class ShellRunner(Recorder):
    """getevent 返回设备列表，dumpsys 返回输入法状态，其它命令返回空。"""

    def __init__(self, ime_shown=False):
        super().__init__()
        self.ime_shown = ime_shown

    def __call__(self, cmd, capture_output, timeout):
        self.cmds.append(cmd)
        out = b""
        if "getevent" in cmd:
            out = GETEVENT
        elif "dumpsys" in cmd:
            out = f"  mShowRequested=true mInputShown={str(self.ime_shown).lower()}\n".encode()
        return subprocess.CompletedProcess(cmd, 0, out, b"")


def test_hw_key_finds_keyboard_device_and_sends_events():
    rec = ShellRunner()
    dev = AdbDevice("emulator-5554", runner=rec)
    dev.hw_key(28)
    dev.hw_key(28)
    getevents = [c for c in rec.cmds if "getevent" in c]
    assert len(getevents) == 1  # 设备只探测一次
    script = rec.cmds[-1][-1]
    assert "sendevent /dev/input/event4 1 28 1" in script
    assert "sendevent /dev/input/event4 1 28 0" in script
    assert script.index("1 28 1") < script.rindex("1 28 0")


def test_hw_key_uses_configured_device():
    rec = ShellRunner()
    AdbDevice("x:1", key_device="/dev/input/event9", runner=rec).hw_key(28)
    assert not any("getevent" in c for c in rec.cmds)
    assert "sendevent /dev/input/event9 1 28 1" in rec.cmds[-1][-1]


def test_hw_key_without_keyboard_device():
    rec = Recorder(stdout=b'add device 1: /dev/input/event0\n  name: "Power Button"\n')
    with pytest.raises(AdbError, match="KEY_ENTER"):
        AdbDevice("x:1", runner=rec).hw_key(28)


@pytest.mark.parametrize("shown", [True, False])
def test_ime_shown(shown):
    assert AdbDevice("x:1", runner=ShellRunner(ime_shown=shown)).ime_shown() is shown


def test_find_mumu_new_and_old_layout(tmp_path):
    from skydango.device.mumu import DLL_NAME, find_mumu

    new = tmp_path / "MuMu"
    shell = new / "nx_device" / "12.0" / "shell"
    (shell / "sdk").mkdir(parents=True)
    (shell / "sdk" / DLL_NAME).write_bytes(b"")
    assert find_mumu(str(shell / "adb.exe")) == (new, shell / "sdk" / DLL_NAME)

    old = tmp_path / "MuMuPlayer-12.0"
    (old / "shell" / "sdk").mkdir(parents=True)
    (old / "shell" / "sdk" / DLL_NAME).write_bytes(b"")
    assert find_mumu(str(old / "shell" / "adb.exe")) == (old, old / "shell" / "sdk" / DLL_NAME)

    assert find_mumu("adb") is None  # PATH 里的 adb：不是 MuMu 自带的


class FlakyCapture:
    def __init__(self):
        self.fail = False

    def screenshot(self):
        if self.fail:
            raise RuntimeError("MuMu 截图失败")
        return np.zeros((3, 4, 3), np.uint8)


def test_screenshot_prefers_fast_capture_and_falls_back_to_adb():
    rec = Recorder(stdout=raw(4, 3))
    cap = FlakyCapture()
    dev = AdbDevice("emulator-5554", runner=rec, capture=cap)
    assert dev.screenshot().shape == (3, 4, 3)
    assert rec.cmds == []  # 没走 adb
    cap.fail = True
    img = dev.screenshot()
    assert tuple(img[0, 0]) == (30, 20, 10) and rec.cmds[-1][-1] == "screencap"
    cap.fail = False
    dev.screenshot()
    assert len(rec.cmds) == 1  # 恢复后又用回快速截图


def test_hw_key_hold_sleeps_on_device():
    rec = ShellRunner()
    dev = AdbDevice("emulator-5554", runner=rec)
    dev.hw_key_hold(106, 0.05)
    sends = [c for c in rec.cmds if "sendevent" in c[-1]]
    assert len(sends) == 1  # 一条命令：sleep 在模拟器里，不受 adb 往返影响
    script = sends[0][-1]
    parts = [
        "sendevent /dev/input/event4 1 106 0",
        "sendevent /dev/input/event4 1 106 1",
        "sleep 0.05",
        "sendevent /dev/input/event4 1 106 0",
    ]
    pos = 0
    for part in parts:
        i = script.find(part, pos)
        assert i >= 0, (part, script)
        pos = i + len(part)


def test_base_hw_key_hold_releases_on_error(monkeypatch):
    """默认实现：按下 + sleep + 抬起，sleep 被打断也要抬起。"""
    import skydango.device.base as base

    calls = []

    class Dev:
        hw_key_hold = base.Device.hw_key_hold

        def hw_key_down(self, code):
            calls.append(("down", code))

        def hw_key_up(self, code):
            calls.append(("up", code))

    def boom(s):
        raise KeyboardInterrupt

    monkeypatch.setattr(base.time, "sleep", boom)
    with pytest.raises(KeyboardInterrupt):
        Dev().hw_key_hold(105, 0.03)
    assert calls == [("down", 105), ("up", 105)]


class ImeRunner(Recorder):
    """ime list 返回装了的输入法，settings get 返回当前输入法。"""

    def __init__(self, current, installed):
        super().__init__()
        self.current, self.installed = current, installed

    def __call__(self, cmd, capture_output, timeout):
        self.cmds.append(cmd)
        out = b""
        if cmd[-4:] == ["ime", "list", "-a", "-s"]:
            out = ("\n".join(self.installed) + "\n").encode()
        elif "default_input_method" in cmd:
            out = (self.current + "\n").encode()
        elif cmd[-3:-1] == ["ime", "set"]:
            self.current = cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, out, b"")


SOGOU = "com.sohu.inputmethod.sogou/.SogouIME"
ADBK = "com.android.adbkeyboard/.AdbIME"


def test_list_and_set_ime():
    rec = ImeRunner(SOGOU, [SOGOU, ADBK])
    dev = AdbDevice("x:1", runner=rec)
    assert dev.list_imes() == [SOGOU, ADBK]
    dev.set_ime(ADBK)
    assert ["ime", "enable", ADBK] == rec.cmds[-2][-3:]  # 没启用的输入法 set 不上：先 enable
    assert dev.current_ime() == ADBK


def test_enable_adb_keyboard_needs_it_installed():
    dev = AdbDevice("x:1", runner=ImeRunner(SOGOU, [SOGOU]))
    with pytest.raises(AdbError, match="ADBKeyboard"):
        dev.enable_adb_keyboard()
