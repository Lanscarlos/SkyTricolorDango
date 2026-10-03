"""设备页的「开始检测」：adb、设备在线、截图、输入法、游戏在前台、模拟键盘，逐项给结论和怎么办；还有「输入法」一栏手动切换。

团子没运行时才能用（运行时设备归身体线程独占）；前面的项没过，依赖它的后面几项标"跳过"。
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from ..config import Config
from ..device.ime import ime_label

THUMB_WIDTH = 480
GAME_PACKAGE = "com.netease.sky"
LABELS = {"adb": "adb", "online": "设备在线", "screen": "截图", "ime": "输入法", "game": "游戏在前台", "keys": "模拟键盘"}


@dataclass
class Check:
    key: str
    label: str
    status: str  # ok / fail / warn / skip
    detail: str = ""
    hint: str = ""
    data: dict = field(default_factory=dict)


def _thumb(img) -> str:
    height, width = img.shape[:2]
    small = cv2.resize(img, (THUMB_WIDTH, max(1, round(height * THUMB_WIDTH / width))))
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else ""


def _check_adb(cfg: Config, run, which) -> Check:
    path = cfg.device.adb_path
    if not Path(path).is_file() and not which(path):
        return Check("adb", LABELS["adb"], "fail", f"找不到 {path}", "在设置页填 MuMu 安装目录下的 shell\\adb.exe")
    proc = run([path, "version"], capture_output=True, timeout=cfg.device.adb_timeout)
    out = (proc.stdout or b"").decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        return Check("adb", LABELS["adb"], "fail", err or out or f"退出码 {proc.returncode}", "在设置页填 MuMu 安装目录下的 shell\\adb.exe")
    return Check("adb", LABELS["adb"], "ok", out.splitlines()[0] if out else "能运行")


def _check_online(cfg: Config, dev) -> Check:
    serial = cfg.device.serial
    if cfg.device.auto_connect and ":" in serial:
        dev.connect()
    found = dev.devices()
    if serial in found:
        return Check("online", LABELS["online"], "ok", serial)
    seen = "、".join(found) or "一个都没有"
    return Check("online", LABELS["online"], "fail", f"{serial} 不在线；adb 看到的设备：{seen}",
                 "点下面的设备名填进设置" if found else "先打开 MuMu 模拟器", {"devices": found})


def _check_screen(dev) -> Check:
    start = time.perf_counter()
    img = dev.screenshot()
    ms = (time.perf_counter() - start) * 1000
    height, width = img.shape[:2]
    how = "MuMu 原生" if dev.capture is not None and not dev._capture_failed else "adb screencap"
    detail = f"{width}×{height}，{how}，{ms:.0f} ms"
    if (width, height) != (1920, 1080):
        return Check("screen", LABELS["screen"], "warn", detail, "坐标按 1920×1080 标定：把 MuMu 分辨率设成 1920×1080 横屏",
                     {"thumb": _thumb(img)})
    return Check("screen", LABELS["screen"], "ok", detail, data={"thumb": _thumb(img)})


def _check_ime(cfg: Config, dev) -> Check:
    ime = dev.current_ime()
    if ime == cfg.device.ime_id:
        return Check("ime", LABELS["ime"], "ok", ime)
    return Check("ime", LABELS["ime"], "warn", f"现在是 {ime_label(ime, cfg.device.ime_id) if ime else '（空）'}",
                 "团子启动时会自动切到 ADBKeyboard、停下时切回来；也可以在下面「输入法」一栏手动切")


def _check_game(dev) -> Check:
    focus = next((line.strip() for line in dev.shell("dumpsys", "window").splitlines() if "mCurrentFocus" in line), "")
    if GAME_PACKAGE in focus:
        return Check("game", LABELS["game"], "ok", focus)
    return Check("game", LABELS["game"], "warn", focus or "看不出前台是什么", "在模拟器里打开光遇")


def _check_keys(dev) -> Check:
    return Check("keys", LABELS["keys"], "ok", dev._key_device())


def _safe(key: str, step: Callable[[], Check]) -> Check:
    try:
        return step()
    except Exception as exc:  # 某一项出错只算这一项没过
        return Check(key, LABELS[key], "fail", str(exc) or type(exc).__name__)


def _skipped(keys) -> list[Check]:
    return [Check(key, LABELS[key], "skip", "前面一项没通过") for key in keys]


def run_checks(
    cfg: Config, make_device: Callable[[Config], Any], run=subprocess.run, which=shutil.which
) -> list[Check]:
    checks = [_safe("adb", lambda: _check_adb(cfg, run, which))]
    if checks[0].status == "fail":
        return checks + _skipped(["online", "screen", "ime", "game", "keys"])
    dev = None

    def online() -> Check:
        nonlocal dev
        dev = make_device(cfg)
        return _check_online(cfg, dev)

    try:
        checks.append(_safe("online", online))
        if checks[1].status == "fail":
            return checks + _skipped(["screen", "ime", "game", "keys"])
        checks.append(_safe("screen", lambda: _check_screen(dev)))
        checks.append(_safe("ime", lambda: _check_ime(cfg, dev)))
        checks.append(_safe("game", lambda: _check_game(dev)))
        keys = _safe("keys", lambda: _check_keys(dev))
        if keys.status == "fail":  # 找不到键盘设备不影响读聊天，只是按不了实体键
            keys.status, keys.hint = "warn", "可以在 config.toml 的 device.key_device 里手动指定（adb shell getevent -pl 查看）"
        checks.append(keys)
        return checks
    finally:
        _close(dev)


def _close(dev) -> None:
    close = getattr(getattr(dev, "capture", None), "close", None)
    if close is not None:  # MuMu 原生截图的句柄：每次检测都新建，用完就关
        try:
            close()
        except Exception:
            pass


def ime_state(cfg: Config, make_device: Callable[[Config], Any], set_to: str | None = None) -> dict:
    """设备页「输入法」一栏：读当前输入法和装了的输入法；set_to 不为空时先切过去（只能切装了的）。"""
    dev = None
    try:
        dev = make_device(cfg)
        installed = dev.list_imes()
        if set_to:
            if set_to not in installed:
                return {"ok": False, "text": f"模拟器里没装 {set_to}"}
            dev.set_ime(set_to)
        adb = cfg.device.ime_id
        return {"ok": True, "current": dev.current_ime(), "adb": adb,
                "imes": [{"id": i, "label": ime_label(i, adb), "adb": i == adb} for i in installed]}
    except Exception as exc:
        return {"ok": False, "text": str(exc) or type(exc).__name__}
    finally:
        _close(dev)
