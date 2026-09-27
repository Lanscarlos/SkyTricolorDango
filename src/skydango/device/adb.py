"""通过 adb 控制 MuMu 模拟器（也适用于任何开了 adb 的安卓设备）。"""

from __future__ import annotations

import base64
import logging
import struct
import subprocess
from collections.abc import Callable, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

Runner = Callable[..., subprocess.CompletedProcess]


class AdbError(RuntimeError):
    pass


def parse_raw_screencap(data: bytes) -> np.ndarray:
    """解析 `screencap`（不带 -p）输出的原始 RGBA 数据，比 PNG 快得多。

    头部：width, height, format 各 4 字节；Android 9+ 还多一个 colorspace（共 16 字节）。
    """
    if len(data) < 12:
        raise AdbError(f"screencap 输出太短（{len(data)} 字节）")
    width, height, fmt = struct.unpack_from("<III", data, 0)
    pixel_bytes = width * height * 4
    header = len(data) - pixel_bytes
    if width == 0 or height == 0 or header not in (12, 16):
        raise AdbError(f"无法识别的 screencap 格式: {width}x{height} fmt={fmt} len={len(data)}")
    if fmt != 1:  # PIXEL_FORMAT_RGBA_8888
        log.warning("screencap 像素格式为 %s，按 RGBA 解析，颜色可能不对", fmt)
    rgba = np.frombuffer(data, dtype=np.uint8, count=pixel_bytes, offset=header)
    return cv2.cvtColor(rgba.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)


class AdbDevice:
    def __init__(
        self,
        serial: str,
        adb_path: str = "adb",
        timeout: float = 10.0,
        ime_id: str = "com.android.adbkeyboard/.AdbIME",
        key_device: str = "",
        runner: Runner = subprocess.run,
    ) -> None:
        self.serial = serial
        self.adb_path = adb_path
        self.timeout = timeout
        self.ime_id = ime_id
        self.key_device = key_device  # 为空时自动探测
        self._run = runner

    # ---- 基础 ----
    def _adb(self, args: Sequence[str], *, serial: bool = True, timeout: float | None = None) -> bytes:
        cmd = [self.adb_path]
        if serial and self.serial:
            cmd += ["-s", self.serial]
        cmd += list(args)
        try:
            proc = self._run(cmd, capture_output=True, timeout=timeout or self.timeout)
        except FileNotFoundError as exc:
            raise AdbError(f"找不到 adb：{self.adb_path}（可以用 MuMu 安装目录 shell\\adb.exe）") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"adb 超时: {' '.join(cmd)}") from exc
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace").strip()
            raise AdbError(f"adb 失败 ({proc.returncode}): {' '.join(cmd)}\n{err}")
        return proc.stdout or b""

    def shell(self, *args: str) -> str:
        return self._adb(["shell", *args]).decode("utf-8", "replace").strip()

    def connect(self) -> str:
        """网络设备（host:port）需要先 adb connect。"""
        if ":" not in self.serial:
            return "usb device, skip connect"
        out = self._adb(["connect", self.serial], serial=False).decode("utf-8", "replace").strip()
        if "cannot" in out or "failed" in out:
            raise AdbError(out)
        return out

    def devices(self) -> list[str]:
        out = self._adb(["devices"], serial=False).decode("utf-8", "replace")
        return [line.split()[0] for line in out.splitlines()[1:] if line.strip().endswith("device")]

    # ---- 画面 ----
    def screenshot(self) -> np.ndarray:
        data = self._adb(["exec-out", "screencap"])
        try:
            return parse_raw_screencap(data)
        except AdbError:
            log.debug("raw screencap 解析失败，回退到 PNG", exc_info=True)
        png = self._adb(["exec-out", "screencap", "-p"])
        img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise AdbError("screencap -p 解码失败")
        return img

    # ---- 输入 ----
    def tap(self, x: int, y: int) -> None:
        self.shell("input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self.shell("input", "swipe", *(str(int(v)) for v in (x1, y1, x2, y2, duration_ms)))

    def key(self, keycode: int) -> None:
        self.shell("input", "keyevent", str(int(keycode)))

    def input_text(self, text: str) -> None:
        # 走 base64，避开 Windows 命令行和 adb shell 的中文 / 引号转义问题
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self.shell("am", "broadcast", "-a", "ADB_INPUT_B64", "--es", "msg", b64)

    def clear_text(self) -> None:
        self.shell("am", "broadcast", "-a", "ADB_CLEAR_TEXT")

    def editor_action(self, code: int) -> None:
        self.shell("am", "broadcast", "-a", "ADB_EDITOR_CODE", "--ei", "code", str(int(code)))

    def _find_key_device(self) -> str:
        """找一个带 KEY_ENTER 的输入设备（MuMu 里是 "Xiaomi Input"，键盘透传走的就是它）。"""
        out = self.shell("getevent", "-pl")
        current = ""
        for line in out.splitlines():
            if line.startswith("add device"):
                current = line.split(":", 1)[1].strip()
            elif current and "KEY_ENTER" in line:
                return current
        raise AdbError("没找到带 KEY_ENTER 的输入设备，可以在 device.key_device 里手动指定（adb shell getevent -pl 查看）")

    def _key_device(self) -> str:
        if not self.key_device:
            self.key_device = self._find_key_device()
        return self.key_device

    def hw_key_down(self, code: int) -> None:
        dev, code = self._key_device(), int(code)
        self.shell(f"sendevent {dev} 1 {code} 0; sendevent {dev} 0 0 0; sendevent {dev} 1 {code} 1; sendevent {dev} 0 0 0")

    def hw_key_up(self, code: int) -> None:
        dev, code = self._key_device(), int(code)
        self.shell(f"sendevent {dev} 1 {code} 0; sendevent {dev} 0 0 0")

    def hw_key(self, code: int) -> None:
        # adb shell 属于 input 组，可以直接写 /dev/input/eventX，游戏会当成实体键盘。
        # 先补一个抬起：之前的按键如果漏了抬起事件，按下会被当成重复而没反应。
        dev, code = self._key_device(), int(code)
        self.shell(
            f"sendevent {dev} 1 {code} 0; sendevent {dev} 0 0 0; "
            f"sendevent {dev} 1 {code} 1; sendevent {dev} 0 0 0; sleep 0.05; "
            f"sendevent {dev} 1 {code} 0; sendevent {dev} 0 0 0"
        )

    def ime_shown(self) -> bool:
        out = self.shell("dumpsys", "input_method")
        return "mInputShown=true" in out

    # ---- 输入法 ----
    def current_ime(self) -> str:
        return self.shell("settings", "get", "secure", "default_input_method")

    def enable_adb_keyboard(self) -> None:
        installed = self.shell("ime", "list", "-a", "-s")
        if self.ime_id not in installed.split():
            raise AdbError(
                "模拟器里没有安装 ADBKeyboard。下载 https://github.com/senzhk/ADBKeyBoard/releases "
                "里的 apk，拖进 MuMu 安装后重试。"
            )
        self.shell("ime", "enable", self.ime_id)
        self.shell("ime", "set", self.ime_id)

    def reset_ime(self) -> None:
        self.shell("ime", "reset")
