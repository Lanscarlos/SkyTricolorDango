"""团子的 HTTP 接口（给管理面板「真机团子」页用）：画面 + 识别框（好友名字、陌生人、黑影、团子、互动圆圈、
聊天面板、新消息）+ 状态、大脑时间线、聊天记录、手动控制、内心、退出。自己没有网页（2026-10-04 删了，
spec docs/superpowers/specs/2026-10-04-console-attach-design.md），框和中文字由管理面板的 stage.js 画。

- 只用标准库起 HTTP 服务，不加依赖
- 画面、框、状态放在同一份快照里（/snapshot 长轮询），框不会和画面错位
- 身体每圈调 update() 只是记下最新一帧和框（很便宜）；有人在看时才压 JPEG，同一帧只压一次
- 只监听 127.0.0.1：画面里有好友昵称和聊天
- /status 带上 run_info（这次 run 的身份）：管理面板靠它认出团子、接管终端起的团子
- 大脑模式时 brain 是 brain.trace.BrainTrace：/brain 长轮询给大脑控制台
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..config import ViewerConfig
from .bubbles import Rect
from .people import describe_things

log = logging.getLogger(__name__)

MESSAGE_KEEP = 3.0  # 新消息的框留几秒
MESSAGE_CHARS = 24  # 新消息的标签最多显示几个字
WAIT = 2.0  # /snapshot 没有新帧时最多等几秒
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")  # Host 头只认这些（防 DNS 重绑定）
MAX_BODY = 4096  # /control 请求体上限（字节）


class _Server(ThreadingHTTPServer):
    # Windows 上 SO_REUSEADDR 允许两个进程绑同一个端口：第二个 view 不报“端口被占用”，浏览器可能连到旧的那个
    allow_reuse_address = os.name != "nt"


def is_local_host(host_header: str, port: int) -> bool:
    """Host 头必须是本机地址 + 这个服务的端口（防 DNS 重绑定：恶意域名解析到 127.0.0.1）。管理面板共用。"""
    host = host_header or ""
    if host.startswith("["):
        name, _, rest = host[1:].partition("]")
        got = rest.removeprefix(":")
    else:
        name, _, got = host.rpartition(":")
    return name.lower() in LOCAL_HOSTS and got == str(port)


def post_guard(headers, port: int, max_body: int) -> tuple[int, str] | None:
    """POST 的防护：本机 Host、带 X-Skydango 头、JSON、不超过 max_body（别的网页借浏览器发不过来：要先预检，我们不应答）。
    不通过返回 (状态码, 原因)，通过返回 None；不读请求体。管理面板共用。"""
    ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if not is_local_host(headers.get("Host") or "", port) or headers.get("X-Skydango") != "1" or ctype != "application/json":
        return 403, "只接受本机网页发来的请求"
    try:
        length = int(headers.get("Content-Length") or -1)
    except ValueError:
        length = -1
    if not 0 <= length <= max_body:
        return 413, f"请求太大（上限 {max_body} 字节）"
    return None


def brain_body(trace, after: int, wait: float = WAIT) -> bytes:
    """/brain：大脑时间线长轮询的响应体。errors="replace"：截断在 emoji 中间的半个代理字符不能让整条时间线卡在"连不上"。"""
    return json.dumps(trace.since(after, wait), ensure_ascii=False).encode(errors="replace")


def usage_response(get: Callable[[], dict]) -> tuple[int, dict]:
    """/usage：UsageMeter.snapshot()；出错 500（团子 / 沙盒共用）。"""
    try:
        return 200, {"ok": True, **get()}
    except Exception as exc:
        log.exception("取模型用量出错")
        return 500, {"ok": False, "text": f"取模型用量出错：{exc}"}


def inner_response(inner: Callable[[], dict]) -> tuple[int, dict]:
    """/inner：取内心快照（经身体线程）；身体超时 / 已经停了 → 503。"""
    try:
        return 200, inner()
    except Exception as exc:
        log.debug("取内心快照失败：%s", exc)
        return 503, {"ok": False, "error": "团子正忙，稍后再试"}


def forget_response(forget: Callable[[str, str, str, str], str], raw: bytes) -> tuple[int, dict]:
    """/inner/forget：删一条性格条目：解析 → 交给身体线程 → {"ok": true} 或原因。"""
    from ..inner.api import forget_result, parse_forget

    try:
        kind, text, who, topic = parse_forget(json.loads(raw.decode("utf-8")))
    except ValueError as exc:  # 坏 JSON、类别不认识：不交给身体
        return 400, {"ok": False, "error": str(exc)}
    try:
        reason = forget(kind, text, who, topic)
    except Exception as exc:  # 身体超时 / 已经停了
        log.debug("删性格条目失败：%s", exc)
        return 503, {"ok": False, "error": "团子正忙，稍后再试"}
    return 200, forget_result(reason)


class JsonHandler(BaseHTTPRequestHandler):
    """viewer 和沙盒共用的请求处理小工具：回 JSON、读 after、读掉没读的请求体、校验 Host。"""

    def _local_host(self) -> bool:
        return is_local_host(self.headers.get("Host") or "", self.server.server_address[1])

    def _drain(self) -> None:
        """没读的请求体读掉（最多 64 KB），否则直接关连接浏览器那边会报连接被重置、看不到状态码。"""
        try:
            left = min(int(self.headers.get("Content-Length") or 0), 65536)
            if left > 0:
                self.rfile.read(left)
        except (ValueError, OSError):
            pass

    def _json(self, code: int, data) -> None:
        self._send(code, "application/json; charset=utf-8", json.dumps(data, ensure_ascii=False).encode(errors="replace"))

    def _after(self, url) -> int:
        try:
            return int(parse_qs(url.query).get("after", ["0"])[0] or 0)
        except ValueError:
            return 0

    def _wait(self, url, cap: float = 2.0) -> float:
        """长轮询最多等几秒：query 里的 wait，夹到 [0, cap]；坏值用 cap。"""
        try:
            value = float(parse_qs(url.query).get("wait", [cap])[0])
        except ValueError:
            return cap
        return max(0.0, min(cap, value)) if value == value else cap

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            if code != 204:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if code != 204:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args) -> None:  # 别刷屏
        pass


def _box(r: Rect, kind: str, label: str) -> dict:
    return {"x": int(r.x), "y": int(r.y), "w": int(r.w), "h": int(r.h), "kind": kind, "label": label}


BUTTON_KINDS = {"retreat": "button_ok", "allow": "button_ok", "other": "button_ask", "never": "button_never"}


def panel_boxes(watcher) -> list[dict]:
    """面板识别（vision.panels.PanelWatcher）认出的面板和按钮的框；聊天记录面板另外画（panel_box）。"""
    out = []
    for p in watcher.state.others():
        kind = "panel_unknown" if p.name == "unknown" else ("panel_ok" if p.verified else "panel_new")
        out.append(_box(p.box, kind, p.label))
        reading = watcher.readings.get(p.name)
        if reading is not None and reading.panel.box == p.box:
            out += [_box(b.box, BUTTON_KINDS[b.kind], b.text) for b in reading.buttons]
    return out


class Viewer:
    def __init__(self, cfg: ViewerConfig) -> None:
        self.cfg = cfg
        self._cond = threading.Condition()
        self._encode = threading.Lock()  # 多个浏览器同时要同一帧：只压一次
        self._seq = 0
        self._frame: np.ndarray | None = None
        self._boxes: list[dict] = []
        self._info: dict = {}
        self._last = float("-inf")
        self._cache: tuple[int, bytes] | None = None  # (seq, 压好的 JSON)
        self._messages: deque[tuple[float, dict]] = deque()
        self._server: ThreadingHTTPServer | None = None
        self.brain = None  # brain.trace.BrainTrace：有它网页才显示大脑时间线
        self.control = None  # brain.manual.ManualControl：有它网页才显示手动控制栏（只在本机模式挂）
        self.chat = None  # brain.transcript.Transcript：真机聊天记录（管理面板真机页用），只在大脑真机模式挂
        self.on_shutdown: Callable[[], None] | None = None  # POST /shutdown 时调（cli 里设成 interrupt_main，走 Ctrl+C 的收尾）
        # 内心页（spec 2026-09-30-inner-viewer §2）：cli 在 run --view 时经 body.call 挂上；None 时 /inner、/inner/forget 回 404
        self.inner: Callable[[], dict] | None = None
        self.forget: Callable[[str, str, str, str], str] | None = None
        self.usage: Callable[[], dict] | None = None  # 模型用量（spec 2026-10-06-model-usage §6.1）：UsageMeter.snapshot
        self._updated: float | None = None  # 最近一帧记下时的 time.monotonic()，/status 算 age
        # 这次 run 的身份（pid、运行目录、live / dry……，见 cli._run_info）：/status 带上它，管理面板才认得出是团子
        self.run_info: dict | None = None
        self.frames = 0  # 更新了多少帧（测试 / 统计用）
        self.encodes = 0  # 压了多少次 JPEG（测试用）

    # ---- 身体这边 ----
    def update(
        self,
        frame: np.ndarray,
        now: float,
        env=None,  # EnvWatcher / PerceptionWatcher：有 overlay() 就画它的框
        panel: Rect | None = None,  # 聊天记录面板开着时的区域
        messages=(),  # 这一圈读到的新消息（chat.reader.Message，box 是整张截图坐标）
        info: dict | None = None,  # 右侧状态栏额外显示的内容
        source: str = "实时截图",
        panels=None,  # vision.panels.PanelWatcher：画面板和按钮

    ) -> bool:
        """记下这一帧；超过 viewer.fps 的帧直接丢掉（新消息照样记着）。返回这一帧有没有被记下。"""
        for m in messages:
            text = f"{m.speaker}：{m.text}" if getattr(m, "speaker", "") else m.text
            self._messages.append((now, _box(m.box, "message", text[:MESSAGE_CHARS])))
        while self._messages and now - self._messages[0][0] > MESSAGE_KEEP:
            self._messages.popleft()
        if now - self._last < 1.0 / max(self.cfg.fps, 0.1):
            return False
        self._last = now
        boxes: list[dict] = []
        if panel is not None:
            boxes.append(_box(panel, "panel", "聊天记录面板"))
        state: dict = {"来源": source}
        if env is None:
            state["识别"] = "没开（[env] enabled = false）"
        else:
            try:
                boxes += env.overlay(now) if hasattr(env, "overlay") else []
                state.update(describe_env(env, now))
            except Exception:
                log.debug("取识别框出错", exc_info=True)
        if panels is not None:
            try:
                boxes += panel_boxes(panels)
                state["开着的面板"] = "、".join(p.describe() for p in panels.state.others()) or "没有"
            except Exception:
                log.debug("取面板框出错", exc_info=True)
        boxes += [box for _, box in self._messages]
        state.update(info or {})
        with self._cond:
            self._seq += 1
            self._frame, self._boxes, self._info = frame, boxes, state
            self._updated = time.monotonic()
            self.frames += 1
            self._cond.notify_all()
        return True

    # ---- 浏览器这边 ----
    def status(self) -> dict:
        """只有状态、不带图（管理面板真机团子页每 2 秒拉一次）。age：离最近一帧多少秒，没帧是 None。"""
        with self._cond:
            age = None if self._updated is None else max(0.0, time.monotonic() - self._updated)
            out = {"seq": self._seq, "age": age, "info": dict(self._info)}
        if self.run_info is not None:  # 管理面板认团子靠它（spec 2026-10-04-console-attach §1）
            out["run"] = dict(self.run_info)
        return out

    def snapshot(self, after: int = 0, timeout: float = WAIT) -> bytes | None:
        """等到有比 after 新的一帧（最多 timeout 秒），返回 JSON；没有新帧返回 None。"""
        with self._cond:
            if after > self._seq:  # 浏览器记的序号比这边大：程序重启过，从头来
                after = 0
            self._cond.wait_for(lambda: self._seq > after and self._frame is not None, timeout)
            if self._seq <= after or self._frame is None:
                return None
            seq, frame, boxes, info = self._seq, self._frame, self._boxes, self._info
        with self._encode:
            if self._cache is not None and self._cache[0] >= seq:
                return self._cache[1]
            body = self._render(seq, frame, boxes, info)
            if body is not None:
                self._cache = (seq, body)
            return body

    def _render(self, seq: int, frame: np.ndarray, boxes: list[dict], info: dict) -> bytes | None:
        height, width = frame.shape[:2]
        scale = min(1.0, self.cfg.width / width)
        view = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.quality)])
        if not ok:
            return None
        self.encodes += 1
        return json.dumps(
            {
                "seq": seq,
                "width": width,
                "height": height,
                "image": base64.b64encode(jpg.tobytes()).decode(),
                "boxes": boxes,
                "info": info,
            },
            ensure_ascii=False,
        ).encode()

    def start(self) -> str:
        """起网页服务（后台线程），返回地址。端口被占用时抛 OSError。"""
        viewer = self

        class Handler(JsonHandler):
            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                if not self._local_host():  # 画面里有好友昵称、大脑时间线里有聊天原话：防 DNS 重绑定
                    self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                elif url.path == "/snapshot":
                    body = viewer.snapshot(self._after(url))
                    if body is None:
                        self._send(204, "application/json", b"")
                    else:
                        self._send(200, "application/json; charset=utf-8", body)
                elif url.path == "/status":
                    if not self._local_host():
                        self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                    else:
                        self._json(200, viewer.status())
                elif url.path == "/control/options" and viewer.control is not None:
                    if not self._local_host():
                        self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                    else:
                        self._json(200, viewer.control.options())
                elif url.path == "/inner" and viewer.inner is not None:
                    if not self._local_host():
                        self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                        return
                    self._json(*inner_response(viewer.inner))
                elif url.path == "/usage" and viewer.usage is not None:
                    if not self._local_host():
                        self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                        return
                    self._json(*usage_response(viewer.usage))
                elif url.path == "/chat" and viewer.chat is not None:
                    if not self._local_host():
                        self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                    else:
                        version, lines = viewer.chat.wait_since(self._after(url), self._wait(url, WAIT))
                        self._json(200, {"v": version, "lines": lines})
                elif url.path == "/brain" and viewer.brain is not None:
                    self._send(200, "application/json; charset=utf-8", brain_body(viewer.brain, self._after(url)))
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                """手动控制 / 退出：只收本机（Host）、带 X-Skydango 头的 JSON（见 post_guard）。"""
                path = urlparse(self.path).path
                if not ((path == "/control" and viewer.control is not None) or (path == "/shutdown" and viewer.on_shutdown is not None)
                        or (path == "/inner/forget" and viewer.forget is not None)):
                    self._drain()
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                refused = post_guard(self.headers, self.server.server_address[1], MAX_BODY)
                if refused is not None:
                    self._drain()
                    self._json(refused[0], {"ok": False, "text": refused[1]})
                    return
                length = int(self.headers.get("Content-Length"))
                if path == "/shutdown":
                    self.rfile.read(length)
                    viewer.on_shutdown()  # interrupt_main 只是让主线程收到 KeyboardInterrupt，这里照样能回应
                    self._json(200, {"ok": True, "text": "正在退出"})
                    return
                if path == "/inner/forget":
                    self._json(*forget_response(viewer.forget, self.rfile.read(length)))
                    return
                try:
                    req = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(req, dict):
                        raise ValueError("请求要是一个 JSON 对象")
                    result = viewer.control.run(str(req.get("action") or ""), req.get("args") or {})
                except ValueError as exc:  # 坏 JSON、参数不对：不交给身体
                    self._json(400, {"ok": False, "text": str(exc)})
                    return
                self._json(200, result)

        self._server = _Server(("127.0.0.1", self.cfg.port), Handler)  # 只听本机：画面里有好友昵称和聊天
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="viewer", daemon=True).start()
        return self.url

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2] if self._server else ("127.0.0.1", self.cfg.port)
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        return f"http://{host}:{port}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def panel_box(vision, reader, frame: np.ndarray) -> Rect | None:
    """聊天记录面板开着时它的区域（整张截图坐标）；不是 log 模式或面板关着返回 None。"""
    from .bubbles import roi_rect

    if vision.mode != "log" or reader.panel_closed_since is not None:
        return None
    height, width = frame.shape[:2]
    return roi_rect(vision.log_roi, width, height)


def describe_env(env, now: float) -> dict:
    """识别器的状态 → 右侧状态栏。"""
    from ..game.social import KIND_NAMES

    yolo = hasattr(env, "strangers")
    interval = getattr(getattr(env, "cfg", None), "interval", None)
    ocr = f"整图 OCR（每 {interval:g} 秒一次）" if isinstance(interval, (int, float)) else "整图 OCR"
    out: dict = {"识别": "YOLO 感知层" if yolo else ocr}
    out["身边的好友"] = env.nearby(now) or "没看到"
    look = env.my_look() if hasattr(env, "my_look") else ""
    if look:
        out["团子穿着"] = look
    if yolo:
        n = env.strangers(now)
        dark = env.unlit(now) if hasattr(env, "unlit") else 0
        out["陌生人"] = f"{n} 个" + (f"（{dark} 个没点火）" if dark else "")
    if hasattr(env, "objects"):
        out["附近的东西"] = describe_things(env.objects(now)) or "没有"
    requests = [f"{r.name}：{KIND_NAMES.get(r.kind, r.kind)}" for r in list(dict(env.requests).values())]
    out["互动请求"] = requests or "没有"
    timings = getattr(env, "timings", None)
    if timings:
        recent = list(timings)[-30:]
        out["检测耗时"] = f"{sum(d for d, _ in recent) / len(recent):.1f} ms（整帧 {sum(t for _, t in recent) / len(recent):.1f} ms）"
    return out
