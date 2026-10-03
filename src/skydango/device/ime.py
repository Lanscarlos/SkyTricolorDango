"""输入法：团子打字靠 ADBKeyboard（广播输入中文），用户自己打字用 MuMu 里的搜狗输入法。

`run` 启动时切到 ADBKeyboard、停下时切回来（`[device] switch_ime`）；管理面板设备页也能手动切。
出错只记日志：切不过去团子打不了中文，但别的照常。
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def is_sogou(ime_id: str) -> bool:
    return "sogou" in ime_id.lower()


def ime_label(ime_id: str, adb_id: str) -> str:
    if ime_id == adb_id:
        return "ADBKeyboard（团子打字用）"
    if is_sogou(ime_id):
        return "搜狗输入法"
    return ime_id


def restore_target(start: str, adb_id: str, installed: list[str], user: str = "") -> str:
    """停下时切回哪个：配置里指定的（装了才算）> 启动时的那个 > 启动时已经是 ADBKeyboard（上次被强杀没切回）就找搜狗；都没有 = 不动。"""
    if user and user in installed:
        return user
    if user:
        log.warning("[device] user_ime = %s 没装在模拟器里，照启动时的输入法切回", user)
    if start and start != adb_id:
        return start
    return next((i for i in installed if is_sogou(i)), "")


class ImeSwitch:
    def __init__(self, device, adb_id: str, user: str = "") -> None:
        self.device, self.adb_id, self.user = device, adb_id, user
        self.back = ""  # 停下时切回哪个；空 = 不动

    def start(self) -> None:
        if not hasattr(self.device, "list_imes"):  # 测试的假设备、沙盒
            return
        try:
            current, installed = self.device.current_ime(), self.device.list_imes()
            if self.adb_id not in installed:
                log.warning("模拟器里没装 ADBKeyboard（%s），团子打不了中文；输入法不动", self.adb_id)
                return
            self.back = restore_target(current, self.adb_id, installed, self.user)
            if current != self.adb_id:
                self.device.set_ime(self.adb_id)
                log.info("输入法：%s → ADBKeyboard（停下时切回）", ime_label(current, self.adb_id) if current else "（空）")
            elif self.back:
                log.info("输入法已经是 ADBKeyboard；停下时切到 %s", ime_label(self.back, self.adb_id))
        except Exception:
            log.exception("切输入法出错，团子可能打不了中文")

    def stop(self) -> None:
        back, self.back = self.back, ""
        if not back:
            return
        try:
            if self.device.current_ime() != back:
                self.device.set_ime(back)
                log.info("输入法切回 %s", ime_label(back, self.adb_id))
        except Exception:
            log.exception("输入法没切回 %s：在管理面板设备页或 python -m skydango ime off 切", ime_label(back, self.adb_id))
