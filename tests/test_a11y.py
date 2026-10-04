import hashlib
import io
import json
import subprocess
import threading
from pathlib import Path

import pytest

from skydango.device.a11y import A11yError, A11yReader, Node, Snapshot, parse_line, split_speaker

LINE = json.dumps({
    "t": 1759539000000,
    "pkg": "com.netease.sky.vivo",
    "nodes": [
        {"text": "团子早上好 - 懒洋洋大王", "desc": None, "cls": "TextView", "id": None, "b": [21, 937, 321, 975], "v": 1},
        {"text": "..... - 陌生人", "desc": None, "cls": "TextView", "id": None, "b": [21, 0, 160, -884], "v": 0},
        {"text": None, "desc": "设置", "cls": "Button", "id": "settings", "b": [1800, 20, 1900, 100], "v": 1},
    ],
}, ensure_ascii=False)


def test_parse_line():
    snap = parse_line(LINE.encode("utf-8"), at=12.5)
    assert snap.at == 12.5
    assert snap.package == "com.netease.sky.vivo"
    assert snap.nodes[0] == Node("团子早上好 - 懒洋洋大王", None, "TextView", None, (21, 937, 321, 975), True)
    assert snap.nodes[1].visible is False
    assert snap.nodes[2].text == "" and snap.nodes[2].desc == "设置"
    assert snap.texts() == ["团子早上好 - 懒洋洋大王"]  # 默认只要看得见、有文字的
    assert snap.texts(visible=False) == ["团子早上好 - 懒洋洋大王", "..... - 陌生人"]


def test_parse_line_without_window_and_garbage():
    snap = parse_line('{"t":1,"pkg":null,"nodes":[]}', at=0.0)
    assert snap.package is None and snap.nodes == ()
    assert parse_line(b"Aborted\n", at=0.0) is None
    assert parse_line(b"", at=0.0) is None


@pytest.mark.parametrize("text, expected", [
    ("团子早上好 - 懒洋洋大王", ("团子早上好", "懒洋洋大王")),
    ("a - b - 小明", ("a - b", "小明")),  # 话里带分隔符：说话人取最后一段
    ("懒洋洋大王", None),
    ("聊天……", None),
])
def test_split_speaker(text, expected):
    assert split_speaker(text) == expected


class FakeRun:
    """假的 subprocess.run：按命令里的关键字回话。"""

    def __init__(self, remote_md5="", dump=b""):
        self.cmds = []
        self.remote_md5 = remote_md5
        self.dump = dump

    def __call__(self, cmd, capture_output=True, timeout=None):
        self.cmds.append(cmd)
        out = b""
        if "md5sum" in cmd[-1]:
            out = f"{self.remote_md5}  /data/local/tmp/skydango-a11y.jar\n".encode()
        elif cmd[-1].endswith(" dump"):
            out = self.dump
        return subprocess.CompletedProcess(cmd, 0, out, b"")


def make_jar(tmp_path):
    jar = tmp_path / "x.jar"
    jar.write_bytes(b"dex")
    return jar, hashlib.md5(b"dex").hexdigest()


def test_push_only_when_changed(tmp_path):
    jar, md5 = make_jar(tmp_path)
    run = FakeRun(remote_md5="other")
    reader = A11yReader("adb", "emulator-5554", jar=jar, run=run)
    reader.ensure_pushed()
    assert run.cmds[-1][:4] == ["adb", "-s", "emulator-5554", "push"]

    run = FakeRun(remote_md5=md5)
    A11yReader("adb", "emulator-5554", jar=jar, run=run).ensure_pushed()
    assert not any("push" in c for c in run.cmds)


def test_dump(tmp_path):
    jar, md5 = make_jar(tmp_path)
    run = FakeRun(remote_md5=md5, dump=LINE.encode("utf-8") + b"\n")
    snap = A11yReader("adb", "emulator-5554", jar=jar, run=run, clock=lambda: 3.0).dump()
    assert snap.at == 3.0 and len(snap.nodes) == 3
    assert run.cmds[-1][3] == "exec-out"
    assert "app_process" in run.cmds[-1][4] and "--nice-name=skydango-a11y" in run.cmds[-1][4]


class FakeProc:
    def __init__(self, lines, stderr=b""):
        self._lines = lines
        self.stdout = self
        self.stderr = io.BytesIO(stderr)
        self.done = threading.Event()
        self.killed = False

    def __iter__(self):
        yield from self._lines
        self.done.set()

    def readline(self):  # pragma: no cover - 只为看起来像文件
        return b""

    def poll(self):
        return 0 if self.done.is_set() else None

    def terminate(self):
        self.killed = True

    kill = terminate

    def wait(self, timeout=None):
        return 0


def test_on_line_gets_raw_snapshot_lines(tmp_path):
    jar, md5 = make_jar(tmp_path)
    proc = FakeProc([LINE.encode("utf-8") + b"\n", b"garbage\n"])
    got = []
    reader = A11yReader("adb", "emulator-5554", jar=jar, run=FakeRun(remote_md5=md5),
                        popen=lambda cmd, stdout, stderr: proc, on_line=got.append)
    reader.start()
    assert proc.done.wait(2)
    reader._thread.join(2)
    assert got == [LINE.encode("utf-8")]  # 只有解析成功的快照，原样、去掉行尾换行
    reader.stop()


def test_watch_keeps_latest_and_stops(tmp_path):
    jar, md5 = make_jar(tmp_path)
    run = FakeRun(remote_md5=md5)
    second = LINE.replace("团子早上好", "你好团子").encode("utf-8")
    proc = FakeProc([LINE.encode("utf-8") + b"\n", b"garbage\n", second + b"\n"])
    popen_cmds = []

    def popen(cmd, stdout, stderr):
        popen_cmds.append(cmd)
        return proc

    clock = iter([1.0, 2.0, 3.0, 4.0, 5.0])
    reader = A11yReader("adb", "emulator-5554", jar=jar, run=run, popen=popen, clock=lambda: next(clock))
    reader.start()
    assert proc.done.wait(2)
    reader._thread.join(2)
    snap = reader.latest()
    assert snap.texts()[0] == "你好团子 - 懒洋洋大王"
    assert reader.count == 2
    assert popen_cmds[0][4].endswith(" watch 150")
    assert any("pkill" in c[-1] for c in run.cmds)  # 起之前先清掉上次留下的
    reader.stop()
    assert proc.killed
    assert sum("pkill" in c[-1] for c in run.cmds) == 2  # 停的时候再清一次


def test_latest_respects_max_age(tmp_path):
    reader = A11yReader("adb", "x", jar=tmp_path / "x.jar", clock=lambda: 10.0)
    assert reader.latest() is None
    reader._latest = Snapshot(at=8.0, device_ms=0, package="p", nodes=())
    assert reader.latest(max_age=3) is not None
    assert reader.latest(max_age=1) is None


def test_watch_records_error_from_stdout(tmp_path):
    # adb exec-out 没有单独的 stderr：设备端的报错（客户端的 "error: …"、被系统杀掉时 sh 打的 "Killed"）都在 stdout 里
    jar, md5 = make_jar(tmp_path)
    proc = FakeProc([LINE.encode("utf-8") + b"\n", b"Killed \n"], stderr=b"error: device offline\n")
    reader = A11yReader("adb", "x", jar=jar, run=FakeRun(remote_md5=md5), popen=lambda *a, **k: proc)
    reader.start()
    reader._thread.join(2)
    assert not reader.alive
    assert "Killed" in reader.error and "device offline" in reader.error


def test_watch_survives_odd_lines(tmp_path):
    jar, md5 = make_jar(tmp_path)
    odd = b'{"t":1,"pkg":"p","nodes":[1, 2]}\n'  # nodes 里不是对象
    proc = FakeProc([odd, LINE.encode("utf-8") + b"\n"])
    reader = A11yReader("adb", "x", jar=jar, run=FakeRun(remote_md5=md5), popen=lambda *a, **k: proc)
    reader.start()
    reader._thread.join(2)
    assert reader.count == 1


def test_start_twice_stops_the_old_one(tmp_path):
    jar, md5 = make_jar(tmp_path)
    procs = [FakeProc([]), FakeProc([])]
    it = iter(procs)
    reader = A11yReader("adb", "x", jar=jar, run=FakeRun(remote_md5=md5), popen=lambda *a, **k: next(it))
    reader.start()
    reader.start()
    assert procs[0].killed and not procs[1].killed
    reader.stop()


class FailingPush(FakeRun):
    def __call__(self, cmd, capture_output=True, timeout=None):
        res = super().__call__(cmd, capture_output, timeout)
        if "push" in cmd:
            return subprocess.CompletedProcess(cmd, 1, b"", b"adb: error: failed to get feature set: device offline")
        return res


def test_push_failure_raises(tmp_path):
    jar, _ = make_jar(tmp_path)
    with pytest.raises(A11yError, match="device offline"):
        A11yReader("adb", "x", jar=jar, run=FailingPush(remote_md5="")).ensure_pushed()


def test_dump_without_snapshot_keeps_reason(tmp_path):
    jar, md5 = make_jar(tmp_path)
    run = FakeRun(remote_md5=md5, dump=b"error: java.lang.IllegalStateException: already registered\n")
    reader = A11yReader("adb", "x", jar=jar, run=run)
    assert reader.dump() is None
    assert "already registered" in reader.error


def test_committed_jar_has_dex():
    import zipfile

    from skydango.device.a11y import JAR

    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(root / JAR) as z:
        assert "classes.dex" in z.namelist()
