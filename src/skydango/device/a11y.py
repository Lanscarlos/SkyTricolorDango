"""读游戏的无障碍节点：光遇把聊天记录、头顶名字、气泡等 UI 文字镜像成了原生 TextView（读屏用），
文字和屏幕坐标都拿得到，不用 OCR。

设备上跑的是 `assets/a11y/skydango-a11y.jar`（源码 `android/a11y/`，`python android/a11y/build.py` 编译）：
shell 用户经 app_process 起一个 UiAutomation 连接（和系统的 uiautomator 一样），每个快照一行 JSON。
只往 /data/local/tmp 推一个文件，不装 App、不改系统设置。

- 同一时间设备上只能有一个 UiAutomation 连接：开着它时 `uiautomator dump / events`、appium 之类连不上，反过来也一样
  （10-04 实测：第二个连上来的 connect 抛 `IllegalStateException: … already registered!`，进了 `error`；`start()` 会先清掉自己上次留下的）
- App 能察觉有无障碍客户端连着（`AccessibilityManager`），这一点和别的读法一样
- `adb exec-out` 没有单独的 stderr：设备端的报错都混在 stdout 里，不是 JSON 的行收进 `error`
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

JAR = Path("assets/a11y/skydango-a11y.jar")
REMOTE = "/data/local/tmp/skydango-a11y.jar"
NAME = "skydango-a11y"  # 设备上的进程名（app_process --nice-name），清理残留用
SEP = " - "  # 聊天行：「内容 - 说话人」


class A11yError(RuntimeError):
    pass


@dataclass(frozen=True)
class Node:
    text: str
    desc: str | None
    cls: str | None
    view_id: str | None
    box: tuple[int, int, int, int]  # 屏幕坐标 左, 上, 右, 下（滚出去的会被裁成奇怪的值）
    visible: bool


@dataclass(frozen=True)
class Snapshot:
    at: float  # 本机收到的时间（clock，默认 time.monotonic）
    device_ms: int  # 设备上的时间
    package: str | None  # 当前窗口的包名；拿不到窗口时为 None
    nodes: tuple[Node, ...]

    def texts(self, visible: bool = True) -> list[str]:
        """有文字的节点的文字；visible=True 时只要看得见的（聊天面板里滚出去的历史行不要）。"""
        return [n.text for n in self.nodes if n.text and (n.visible or not visible)]


def parse_line(line: bytes | str, at: float) -> Snapshot | None:
    """解析客户端打的一行；不是快照（报错、空行）返回 None。"""
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    try:
        data = json.loads(line)
        nodes = tuple(
            Node(n.get("text") or "", n.get("desc"), n.get("cls"), n.get("id"), tuple(n["b"]), bool(n.get("v")))
            for n in data["nodes"]
        )
        return Snapshot(at, int(data.get("t", 0)), data.get("pkg"), nodes)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def split_speaker(text: str) -> tuple[str, str] | None:
    """「内容 - 说话人」→（内容，说话人）；话里带分隔符时说话人取最后一段。不是聊天行返回 None。"""
    if SEP not in text:
        return None
    content, speaker = text.rsplit(SEP, 1)
    return content, speaker


class A11yReader:
    """设备上常驻一个客户端，后台线程收快照，`latest()` 取最近一份；`dump()` 单读一次（约 1 秒，多半花在起 JVM 上）。

    客户端每秒至少出一份（心跳），常驻时取快照要传 `max_age`（比如 2 秒）：客户端死了以后最后一份会一直留着。
    """

    def __init__(
        self,
        adb_path: str,
        serial: str,
        jar: Path | str = JAR,
        interval_ms: int = 150,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.adb = [adb_path, "-s", serial] if serial else [adb_path]
        self.jar = Path(jar)
        self.interval_ms = interval_ms
        self._run = run
        self._popen = popen
        self._clock = clock
        self._proc = None
        self._thread: threading.Thread | None = None
        self._latest: Snapshot | None = None
        self.count = 0  # 收到几份快照
        self.error = ""  # 出错的原因：客户端输出里不是快照的行（设备端报错）+ adb 自己的 stderr

    def _cmd(self, mode: str) -> str:
        return f"CLASSPATH={REMOTE} app_process /system/bin --nice-name={NAME} skydango.A11y {mode}"

    def _shell(self, script: str, timeout: float = 10) -> bytes:
        return self._run(self.adb + ["shell", script], capture_output=True, timeout=timeout).stdout or b""

    def ensure_pushed(self) -> None:
        """设备上的 jar 和本地不一样（或没有）才推；推不上去抛 A11yError（不然会悄悄跑设备上的旧版本）。"""
        local = hashlib.md5(self.jar.read_bytes()).hexdigest()
        remote = self._shell(f"md5sum {REMOTE} 2>/dev/null").decode("utf-8", "replace").split()
        if not remote or remote[0] != local:
            res = self._run(self.adb + ["push", str(self.jar), REMOTE], capture_output=True, timeout=30)
            if res.returncode != 0:
                err = (res.stderr or res.stdout or b"").decode("utf-8", "replace").strip()
                raise A11yError(f"推不上无障碍客户端：{err}")

    def _kill_remote(self) -> None:
        # --nice-name 改的是 cmdline（comm 是 "main"，所以 -x 匹配不上）；锚定开头，不然连执行 pkill 的那个 sh -c 一起杀
        self._shell(f"pkill -f '^{NAME}'; true", timeout=3)

    def dump(self) -> Snapshot | None:
        """单读一次；读不到返回 None，原因在 `error`。"""
        self.ensure_pushed()
        out = self._run(self.adb + ["exec-out", self._cmd("dump")], capture_output=True, timeout=15)
        other = []
        for line in (out.stdout or b"").splitlines():
            snap = parse_line(line, self._clock())
            if snap is not None:
                self.error = ""
                return snap
            if line.strip():
                other.append(line.decode("utf-8", "replace").strip())
        other.append((out.stderr or b"").decode("utf-8", "replace").strip())
        self.error = "\n".join(s for s in other if s)[-500:] or "客户端没有输出"
        return None

    # ---- 常驻 ----
    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._proc is not None:
            self.stop()
        self.ensure_pushed()
        self._kill_remote()  # 上次没收好留下的（同一时间只能有一个连接）
        self.error = ""
        self._proc = self._popen(self.adb + ["exec-out", self._cmd(f"watch {self.interval_ms}")],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._pump, args=(self._proc,), name="a11y", daemon=True)
        self._thread.start()

    def _pump(self, proc) -> None:
        other: collections.deque[str] = collections.deque(maxlen=20)  # 不是快照的行：设备端的报错
        try:
            for line in proc.stdout:
                snap = parse_line(line, self._clock())
                if snap is None:
                    if line.strip():
                        other.append(line.decode("utf-8", "replace").strip())
                elif proc is self._proc:  # 停掉之后别再往里写
                    self._latest = snap
                    self.count += 1
            if proc.stderr:
                other.append(proc.stderr.read().decode("utf-8", "replace").strip())
        except Exception as exc:
            other.append(f"读取出错：{exc!r}")
        self.error = "\n".join(s for s in other if s)[-500:]
        if proc is self._proc:  # 不是 stop() 叫停的
            log.warning("无障碍客户端退出了：%s", self.error or "（没有输出）")

    def latest(self, max_age: float | None = None) -> Snapshot | None:
        snap = self._latest
        if snap is None or (max_age is not None and self._clock() - snap.at > max_age):
            return None
        return snap

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)
        try:
            self._kill_remote()  # adb 断开后设备上的进程下次写 stdout 时才退（最多 1 秒心跳），这里直接清掉
        except Exception:
            log.debug("清理设备上的无障碍客户端失败", exc_info=True)

    def __enter__(self) -> A11yReader:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
