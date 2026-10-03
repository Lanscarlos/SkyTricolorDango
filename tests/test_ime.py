import logging

from skydango.device.ime import ImeSwitch, ime_label, restore_target

ADBK = "com.android.adbkeyboard/.AdbIME"
SOGOU = "com.sohu.inputmethod.sogou/.SogouIME"
LATIN = "com.android.inputmethod.latin/.LatinIME"


class Dev:
    def __init__(self, current, installed=(SOGOU, ADBK), fail=None):
        self.current, self.installed, self.fail = current, list(installed), fail or set()
        self.sets = []

    def _maybe(self, name):
        if name in self.fail:
            raise RuntimeError(f"{name} 坏了")

    def current_ime(self):
        self._maybe("current")
        return self.current

    def list_imes(self):
        self._maybe("list")
        return list(self.installed)

    def set_ime(self, ime_id):
        self._maybe("set")
        self.sets.append(ime_id)
        self.current = ime_id


def test_labels():
    assert "ADBKeyboard" in ime_label(ADBK, ADBK)
    assert ime_label(SOGOU, ADBK) == "搜狗输入法"
    assert ime_label("com.example/.X", ADBK) == "com.example/.X"


def test_restore_target():
    assert restore_target(SOGOU, ADBK, [SOGOU, ADBK]) == SOGOU  # 启动时的那个
    assert restore_target(LATIN, ADBK, [SOGOU, LATIN, ADBK]) == LATIN
    # 启动时已经是 ADBKeyboard（上次被强杀没切回）：找搜狗
    assert restore_target(ADBK, ADBK, [LATIN, SOGOU, ADBK]) == SOGOU
    assert restore_target(ADBK, ADBK, [LATIN, ADBK]) == ""  # 没有搜狗就不动
    assert restore_target(SOGOU, ADBK, [SOGOU, LATIN, ADBK], user=LATIN) == LATIN  # 配置里指定的优先
    assert restore_target(SOGOU, ADBK, [SOGOU, ADBK], user="com.gone/.X") == SOGOU  # 指定的没装：照自动


def test_start_switches_from_sogou_and_stop_switches_back():
    dev = Dev(SOGOU)
    ime = ImeSwitch(dev, ADBK)
    ime.start()
    assert dev.current == ADBK
    ime.stop()
    assert dev.current == SOGOU
    ime.stop()  # 再停一次不动
    assert dev.sets == [ADBK, SOGOU]


def test_already_adb_keyboard_switches_to_sogou_on_stop():
    dev = Dev(ADBK)
    ime = ImeSwitch(dev, ADBK)
    ime.start()
    assert dev.sets == []
    ime.stop()
    assert dev.sets == [SOGOU]


def test_stop_skips_when_already_back():  # 用户中途自己切回去了
    dev = Dev(SOGOU)
    ime = ImeSwitch(dev, ADBK)
    ime.start()
    dev.current = SOGOU
    ime.stop()
    assert dev.sets == [ADBK]


def test_adb_keyboard_missing_leaves_it_alone(caplog):
    dev = Dev(SOGOU, installed=[SOGOU])
    ime = ImeSwitch(dev, ADBK)
    with caplog.at_level(logging.WARNING):
        ime.start()
        ime.stop()
    assert dev.sets == []
    assert "ADBKeyboard" in caplog.text


def test_errors_are_only_logged(caplog):
    dev = Dev(SOGOU, fail={"set"})
    ime = ImeSwitch(dev, ADBK)
    with caplog.at_level(logging.ERROR):
        ime.start()
        ime.stop()
    assert "输入法" in caplog.text


def test_device_without_ime_support_is_skipped():  # 测试里的假设备、沙盒
    ime = ImeSwitch(object(), ADBK)
    ime.start()
    ime.stop()


def test_cmd_ime_off_switches_to_sogou(monkeypatch, capsys):
    from types import SimpleNamespace

    from skydango import cli
    from skydango.config import Config

    dev = Dev(ADBK, installed=[LATIN, SOGOU, ADBK])
    monkeypatch.setattr(cli, "_device", lambda cfg: dev)
    cli.cmd_ime(Config(), SimpleNamespace(action="off"))
    assert dev.sets == [SOGOU] and "SogouIME" in capsys.readouterr().out
