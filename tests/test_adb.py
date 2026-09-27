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
