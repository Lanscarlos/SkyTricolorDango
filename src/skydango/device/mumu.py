"""MuMu 12 原生截图：通过 MuMu 自带的 external_renderer_ipc.dll 直接读模拟器画面。

实测每张约 9 ms（adb screencap 约 400 ms），而且不走 adb，不会遇到 adb 截图偶尔超时的问题。
只读画面，不改游戏；输入还是走 adb。仅限 Windows + MuMu 12。
"""

from __future__ import annotations

import ctypes
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DLL_NAME = "external_renderer_ipc.dll"


class MumuError(RuntimeError):
    pass


def find_mumu(adb_path: str) -> tuple[Path, Path] | None:
    """从 MuMu 自带 adb 的路径推出（安装目录, dll 路径）。

    新版目录：<安装目录>/nx_device/12.0/shell/adb.exe，dll 在 shell/sdk/ 下；
    旧版目录：<安装目录>/shell/adb.exe，dll 同样在 shell/sdk/ 下。
    """
    adb = Path(adb_path)
    dll = adb.parent / "sdk" / DLL_NAME
    if not dll.is_file():
        return None
    for parent in adb.parents:
        if (parent / "nx_device").is_dir() or (parent / "nx_main").is_dir():
            return parent, dll
    return adb.parent.parent, dll  # 旧版：shell 的上一级就是安装目录


class MumuCapture:
    def __init__(self, install_dir: str | Path, dll_path: str | Path, instance: int = 0, package: str = "") -> None:
        self.install_dir = str(install_dir)
        self.instance = instance
        self.package = package.encode()
        try:
            lib = ctypes.CDLL(str(dll_path))
        except OSError as exc:
            raise MumuError(f"加载 {dll_path} 失败: {exc}") from exc
        lib.nemu_connect.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
        lib.nemu_connect.restype = ctypes.c_int
        lib.nemu_disconnect.argtypes = [ctypes.c_int]
        lib.nemu_capture_display.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
        ]
        lib.nemu_capture_display.restype = ctypes.c_int
        if hasattr(lib, "nemu_get_display_id"):
            lib.nemu_get_display_id.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            lib.nemu_get_display_id.restype = ctypes.c_int
        self._lib = lib
        self._handle = 0
        self._display = 0
        self._buf: ctypes.Array | None = None
        self._lock = threading.Lock()

    def _connect(self) -> None:
        self._handle = self._lib.nemu_connect(self.install_dir, self.instance)
        if not self._handle:
            raise MumuError(f"连接 MuMu 失败（安装目录 {self.install_dir}，实例 {self.instance}）")
        self._display = 0
        if self.package and hasattr(self._lib, "nemu_get_display_id"):
            # 应用分身 / 多窗口时游戏可能不在 0 号屏；取不到就用 0
            self._display = max(0, self._lib.nemu_get_display_id(self._handle, self.package, 0))

    def close(self) -> None:
        if self._handle:
            self._lib.nemu_disconnect(self._handle)
            self._handle = 0

    def screenshot(self) -> np.ndarray:
        # 截图缓冲区是共用的：身体线程和感知线程（perception.capture = "own"）可能同时截，排队来
        with self._lock:
            return self._screenshot()

    def _screenshot(self) -> np.ndarray:
        if not self._handle:
            self._connect()
        w, h = ctypes.c_int(), ctypes.c_int()
        if self._buf is None:
            if self._lib.nemu_capture_display(self._handle, self._display, 0, ctypes.byref(w), ctypes.byref(h), None):
                self.close()
                raise MumuError("MuMu 截图失败（取尺寸）")
            self._buf = (ctypes.c_ubyte * (w.value * h.value * 4))()
        ret = self._lib.nemu_capture_display(
            self._handle, self._display, len(self._buf), ctypes.byref(w), ctypes.byref(h), self._buf
        )
        if ret or w.value * h.value * 4 != len(self._buf):
            self._buf = None  # 分辨率变了或者连接断了：下次重新取尺寸、重连
            self.close()
            raise MumuError(f"MuMu 截图失败（返回 {ret}）")
        rgba = np.frombuffer(self._buf, np.uint8).reshape(h.value, w.value, 4)
        return cv2.cvtColor(rgba[::-1], cv2.COLOR_RGBA2BGR)  # 画面是上下颠倒的
