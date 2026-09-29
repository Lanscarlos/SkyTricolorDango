"""识别过程可视化：本机开一个网页，实时显示身体看到的画面 + 识别框（好友名字、陌生人、黑影、团子、互动圆圈、
聊天面板、新消息）+ 右侧状态。浏览器打开 http://127.0.0.1:<端口> 就能看。设计见 docs/superpowers/specs/2026-09-28-viewer-design.md。

- 只用标准库起 HTTP 服务，不加依赖；框和中文字在浏览器里画（OpenCV 画不了中文）
- 画面、框、状态放在同一份快照里（/snapshot 长轮询），框不会和画面错位
- 身体每圈调 update() 只是记下最新一帧和框（很便宜）；有浏览器在看时才压 JPEG，同一帧只压一次
- 只监听 127.0.0.1：画面里有好友昵称和聊天，别开到局域网
- `run --brain --view` 时 brain 是 brain.trace.BrainTrace：/brain 长轮询给画面下方的大脑时间线，和画面分开
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
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")  # 只有这些地址才开手动控制、才收 /control 请求
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
        self.on_shutdown: Callable[[], None] | None = None  # POST /shutdown 时调（cli 里设成 interrupt_main，走 Ctrl+C 的收尾）
        self._updated: float | None = None  # 最近一帧记下时的 time.monotonic()，/status 算 age
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
        """只有状态、不带图（管理面板总览每 2 秒拉一次）。age：离最近一帧多少秒，没帧是 None。"""
        with self._cond:
            age = None if self._updated is None else max(0.0, time.monotonic() - self._updated)
            return {"seq": self._seq, "age": age, "info": dict(self._info)}

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

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                if url.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())
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
                elif url.path == "/brain" and viewer.brain is not None:
                    # errors="replace"：截断在 emoji 中间的半个代理字符不能让整条时间线卡在"连不上"
                    body = json.dumps(viewer.brain.since(self._after(url), WAIT), ensure_ascii=False).encode(errors="replace")
                    self._send(200, "application/json; charset=utf-8", body)
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                """手动控制 / 退出：只收本机（Host）、带 X-Skydango 头的 JSON（见 post_guard）。"""
                path = urlparse(self.path).path
                if not ((path == "/control" and viewer.control is not None) or (path == "/shutdown" and viewer.on_shutdown is not None)):
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
                try:
                    req = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(req, dict):
                        raise ValueError("请求要是一个 JSON 对象")
                    result = viewer.control.run(str(req.get("action") or ""), req.get("args") or {})
                except ValueError as exc:  # 坏 JSON、参数不对：不交给身体
                    self._json(400, {"ok": False, "text": str(exc)})
                    return
                self._json(200, result)

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

        self._server = _Server((self.cfg.host, self.cfg.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="viewer", daemon=True).start()
        return self.url

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2] if self._server else (self.cfg.host, self.cfg.port)
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


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>团子看到的</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--text:#e6e8ee;--muted:#8b93a7;--line:#2a2f3a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}
main{display:flex;gap:12px;padding:12px 16px;align-items:flex-start}
#stage{flex:1;min-width:0}canvas{width:100%;height:auto;display:block;border-radius:8px;background:#000}
aside{width:300px;flex:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
h1{font-size:15px;margin:0 0 8px}dl{margin:0}dt{color:var(--muted);font-size:12px;margin-top:8px}dd{margin:0;word-break:break-all}
dd ul{margin:0;padding-left:16px}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;color:var(--muted);font-size:12px}
button{background:#232833;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer;font:inherit}
.legend{display:flex;flex-wrap:wrap;gap:6px 10px;margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
#status{margin-left:auto}#status.off{color:#f87171}
#control{margin-top:12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px}#control[hidden]{display:none}
#control .row{display:flex;gap:6px 8px;align-items:center;flex-wrap:wrap;margin:6px 0;font-size:13px}
#control .row>b{width:3em;flex:none;color:var(--muted);font-weight:600}
#control input[type=text]{flex:1;min-width:160px}#control input[type=number]{width:4em}
#control input[type=text],#control input[type=number],#control select{background:#0f1115;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:3px 6px;font:inherit}
#control button:disabled,#control select:disabled,#control input:disabled{opacity:.45;cursor:not-allowed}
#ctl-warn{color:#facc15;font-size:12px}#ctl-count,#ctl-pick-tip{color:var(--muted);font-size:12px}
#ctl-pick.on{border-color:#f472b6;color:#f472b6}#ctl-busy{margin-left:auto;color:#60a5fa;font-size:12px}
#ctl-log{list-style:none;margin:6px 0 0;padding:6px 0 0;border-top:1px solid var(--line);font:12px/1.6 ui-monospace,Consolas,monospace}
#ctl-log li{word-break:break-all}#ctl-log .ok{color:#3ddc84}#ctl-log .bad{color:#f87171}#ctl-log .t{color:var(--muted)}
#brain{margin-top:12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px}#brain[hidden]{display:none}
.bhead{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding-bottom:6px;font-size:13px}
.bhead label{margin-left:auto;color:var(--muted);font-size:12px;display:flex;gap:4px;align-items:center;cursor:pointer}
#brain-state.warn{color:#facc15}#brain-state.bad{color:#f87171}
.turn{border-top:1px solid var(--line)}.turn[hidden]{display:none}
.turn>summary{cursor:pointer;padding:5px 0;display:flex;gap:6px 12px;flex-wrap:wrap;font-size:13px;list-style:none}
.turn>summary::-webkit-details-marker{display:none}.turn>summary::before{content:"▶";color:var(--muted);font-size:10px;line-height:20px}
.turn[open]>summary::before{content:"▼"}.turn>summary .t{color:var(--muted);font-variant-numeric:tabular-nums}
.turn>summary .n{color:var(--muted)}.turn.err>summary{color:#f87171}.turn.live>summary{color:#60a5fa}
.tbody{padding:2px 0 10px 16px}.step{margin:6px 0;border-left:3px solid var(--line);padding-left:8px}
.step h4{margin:0;font-size:12px;font-weight:600;color:var(--muted)}
.step pre{margin:2px 0 0;white-space:pre-wrap;word-break:break-all;font:12px/1.5 ui-monospace,Consolas,monospace}
.k-thinking{border-color:#a78bfa}.k-text{border-color:#3ddc84}.k-tool{border-color:#60a5fa}.k-result{border-color:#6b7280}
.k-result.error{border-color:#facc15}.k-result.error pre{color:#facc15}.k-end{border-color:#8b93a7}.k-end.error pre{color:#f87171}
.step button{margin-top:4px;font-size:12px;padding:1px 8px}
@media (max-width:900px){main{flex-direction:column}aside{width:100%}}
</style></head><body><main>
<div id="stage"><div class="bar"><button id="pause">暂停</button><button id="boxes">隐藏框</button><button id="save">存图</button>
<span id="status">连接中…</span></div><canvas id="c"></canvas>
<section id="control" hidden><div class="bhead"><b>手动控制</b><span id="ctl-warn" hidden>手动操作会真的在游戏里执行（大脑是 dry-run 也一样）</span><span id="ctl-busy"></span></div>
<div class="row" id="ctl-say"><b>说话</b><input type="text" id="ctl-say-text" placeholder="让团子说一句…"><span id="ctl-count"></span><button id="ctl-say-go">说</button></div>
<div class="row" id="ctl-emote"><b>动作</b><select id="ctl-emote-name"></select><label><input type="checkbox" id="ctl-force">牵着手也做（会松手）</label><button id="ctl-emote-go">做</button></div>
<div class="row" id="ctl-camera"><b>视角</b><button data-cam="left">左转</button><button data-cam="right">右转</button><button data-cam="up">抬头</button><button data-cam="down">低头</button><button data-cam="zoom_in">拉近</button><button data-cam="zoom_out">拉远</button>
步数<input type="number" id="ctl-steps" value="1" min="1"><button id="ctl-reset">复位</button><button id="ctl-around">环视一圈</button></div>
<div class="row"><b>看人</b><button id="ctl-pick">在画面上选人</button><span id="ctl-pick-tip" class="n"></span></div>
<div class="row" id="ctl-track"><b>盯人</b><input type="text" id="ctl-track-name" placeholder="好友名字"><button id="ctl-track-pick">在画面上选</button>
秒<input type="number" id="ctl-track-sec" value="30" min="1"><button id="ctl-track-go">盯</button><button id="ctl-stop">停下</button><span id="ctl-track-tip" class="n"></span></div>
<div class="row" id="ctl-panels" hidden><b>面板</b><button id="ctl-panel-read">读面板</button><button id="ctl-panel-close">关面板</button></div>
<ul id="ctl-log"></ul></section>
<section id="brain" hidden><div class="bhead"><b>大脑</b><span id="brain-state">连接中…</span>
<label><input type="checkbox" id="brain-acted">只看做了事的轮次</label></div><div id="brain-turns"></div></section></div>
<aside><h1>团子看到的</h1><dl id="info"></dl><div class="legend" id="legend"></div></aside>
</main><script>
const COLORS={friend:"#3ddc84",name:"#3ddc84",tag:"#facc15",stranger:"#ff9f43",unlit:"#a78bfa",player:"#60a5fa",self:"#cbd5e1",
ring:"#22d3ee",request:"#f43f5e",panel:"#6b7280",message:"#f472b6",typing:"#e879f9",
bench:"#1d4ed8",bonfire:"#ea580c",instrument:"#fda4af",spirit:"#ffffff",
panel_ok:"#3b82f6",panel_new:"#facc15",panel_unknown:"#ef4444",button_ok:"#22c55e",button_ask:"#9ca3af",button_never:"#dc2626"};
const NAMES={friend:"好友",tag:"没认出的名字",stranger:"陌生人",unlit:"没点火",player:"没判定的人",self:"团子",ring:"互动圆圈",
request:"互动请求",panel:"聊天面板",message:"新消息",typing:"正在输入",
bench:"座位",bonfire:"篝火",instrument:"乐器",spirit:"先祖",
panel_ok:"面板（已核对）",panel_new:"面板（未核对）",panel_unknown:"不认识的面板",button_ok:"能按",button_ask:"要放行",button_never:"不能按"};
const $=id=>document.getElementById(id),c=$("c"),ctx=c.getContext("2d"),img=new Image();
let seq=0,paused=false,running=false,showBoxes=true,last=null,times=[];
$("legend").innerHTML=Object.entries(NAMES).map(([k,v])=>`<span><i style="background:${COLORS[k]}"></i>${v}</span>`).join("");
$("pause").onclick=e=>{paused=!paused;e.target.textContent=paused?"继续":"暂停";if(!paused)loop()};
$("boxes").onclick=e=>{showBoxes=!showBoxes;e.target.textContent=showBoxes?"隐藏框":"显示框";if(last)draw(last)};
$("save").onclick=()=>{if(!last)return;const a=document.createElement("a");
  a.download=`dango-${new Date().toISOString().replace(/[-:T]/g,"").slice(0,14)}.png`;a.href=c.toDataURL("image/png");a.click()};
function draw(s){
  c.width=img.naturalWidth;c.height=img.naturalHeight;ctx.drawImage(img,0,0);drawMark(s);
  if(!showBoxes)return;const k=c.width/s.width,fs=Math.max(12,Math.round(c.width/80));
  ctx.font=`${fs}px system-ui,"Microsoft YaHei",sans-serif`;ctx.textBaseline="middle";
  for(const b of s.boxes){const col=COLORS[b.kind]||"#fff",x=b.x*k,y=b.y*k,w=b.w*k,h=b.h*k;
    ctx.strokeStyle=col;ctx.lineWidth=b.kind==="request"?4:2;ctx.setLineDash(b.kind.startsWith("panel")?[8,5]:[]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
    const t=(b.label||"")+(b.score!==undefined?` ${b.score.toFixed(2)}`:"");if(!t)continue;
    const below=b.kind==="ring"||b.kind==="request",tw=ctx.measureText(t).width+8,th=fs+6,ty=(below||y-th<0)?y+h:y-th;
    ctx.fillStyle=col;ctx.fillRect(x,ty,tw,th);ctx.fillStyle="#0b0d12";ctx.fillText(t,x+4,ty+th/2);}
}
function info(s){const dl=$("info");dl.innerHTML="";
  for(const [k,v] of Object.entries(s.info)){const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=k;
    if(Array.isArray(v)&&v.length>1){const ul=document.createElement("ul");for(const x of v){const li=document.createElement("li");li.textContent=x;ul.append(li)}dd.append(ul)}
    else dd.textContent=Array.isArray(v)?(v[0]??"—"):String(v);
    dl.append(dt,dd);}}
function status(t,off){const el=$("status");el.textContent=t;el.className=off?"off":""}
async function loop(){
  if(running)return;running=true;
  while(!paused){
    try{const r=await fetch(`snapshot?after=${seq}`,{cache:"no-store"});if(r.status===204)continue;if(!r.ok)throw new Error(r.status);
      const s=await r.json();if(paused)break;seq=s.seq;
      await new Promise((ok,bad)=>{img.onload=ok;img.onerror=bad;img.src="data:image/jpeg;base64,"+s.image});last=s;draw(s);info(s);
      const now=performance.now();times.push(now);times=times.filter(t=>now-t<2000);
      status(`${(times.length/2).toFixed(1)} 帧/秒 · ${s.width}×${s.height}`);
    }catch(e){status("连不上（程序停了？）",true);await new Promise(r=>setTimeout(r,1000));}
  }
  running=false;
}
loop();
// ---- control ----
const CAM={left:"左转",right:"右转",up:"抬头",down:"低头",zoom_in:"拉近",zoom_out:"拉远"};
const K={opts:null,busy:false,picking:false,trackPick:false,mark:null};
function nameAt(boxes,x,y){let best=null,area=Infinity;for(const b of boxes){if(!b.label||(b.kind!=="friend"&&b.kind!=="name"))continue;const tag=b.kind==="name",x1=tag?b.x-b.w:b.x,w=tag?b.w*3:b.w,h=tag?b.h*7:b.h;/* 名字标签：人在它正下方（宽 3 倍、连标签 7 倍高，同身体 _below_tag） */if(x<x1||x>=x1+w||y<b.y||y>=b.y+h)continue;if(w*h<area){best=b.label;area=w*h}}return best}
function toFrame(clientX,clientY,rect,width,height){return [Math.round((clientX-rect.left)*width/rect.width),Math.round((clientY-rect.top)*height/rect.height)]}
function controlLine(action,args,res){const a=args||{};let what;
  if(action==="say")what=`说「${a.text}」`;else if(action==="emote")what=`动作「${a.name}」${a.force?"（松手也做）":""}`;
  else if(action==="camera")what=`${CAM[a.action]||a.action} ×${a.steps}`;else if(action==="camera_reset")what="复位";
  else if(action==="look_around")what="环视一圈";else if(action==="panel_read")what="读面板";else if(action==="panel_close")what="关面板";else if(action==="check_friend")what=`看人 (${a.x}, ${a.y})`;
  else if(action==="track")what=`盯着${a.name}（${a.seconds} 秒）`;else if(action==="stop_task")what="停下";else what=action;
  return `${what} → ${res.text}`}
function drawMark(s){if(!K.mark||!s)return;const k=c.width/s.width,x=K.mark[0]*k,y=K.mark[1]*k,r=Math.max(12,c.width/60);
  ctx.strokeStyle="#f472b6";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-r,y);ctx.lineTo(x+r,y);ctx.moveTo(x,y-r);ctx.lineTo(x,y+r);ctx.stroke()}
function ctlApply(){const o=K.opts;if(!o)return;
  $("ctl-warn").hidden=!o.dry_run;
  const sel=$("ctl-emote-name"),keep=sel.value;
  if([...sel.options].map(x=>x.value).join("|")!==o.emotes.join("|")){  // 列表变了才重建：定时刷新时别把打开的下拉框关掉
    while(sel.firstChild)sel.firstChild.remove();
    for(const n of o.emotes){const op=el("option","",n);op.value=n;sel.append(op)}if(o.emotes.includes(keep))sel.value=keep}
  $("ctl-steps").max=o.max_steps;ctlCount();ctlLock()}
function ctlLock(){const o=K.opts||{emotes:[],camera:[]},b=K.busy;
  $("ctl-say-text").disabled=b;$("ctl-say-go").disabled=b||!$("ctl-say-text").value.trim();
  $("ctl-emote-name").disabled=$("ctl-emote-go").disabled=$("ctl-force").disabled=b||!o.emotes.length;
  for(const x of document.querySelectorAll("#ctl-camera button,#ctl-steps"))x.disabled=b||!o.camera.length;
  const pick=$("ctl-pick");pick.disabled=b||!o.friend_check;pick.title=o.friend_check?"":"[friend_check] enabled = false";
  $("ctl-pick-tip").textContent=o.friend_check?(K.picking?"点一下画面上的人":""):"没开（[friend_check] enabled = false）";
  pick.className=K.picking?"on":"";
  const tr=!!o.track;$("ctl-track-name").disabled=$("ctl-track-sec").disabled=$("ctl-track-pick").disabled=b||!tr;
  $("ctl-track-go").disabled=b||!tr||!$("ctl-track-name").value.trim();$("ctl-stop").disabled=b;
  $("ctl-track-sec").max=o.max_track_seconds||60;$("ctl-track-pick").className=K.trackPick?"on":"";
  if(!tr)$("ctl-track-tip").textContent="要开感知层（[perception]）和镜头";
  else if(K.trackPick)$("ctl-track-tip").textContent="点一下画面上的好友";
  $("ctl-panels").hidden=!o.panels;$("ctl-panel-read").disabled=$("ctl-panel-close").disabled=b||!o.panels}
function ctlCount(){const n=[...$("ctl-say-text").value.trim()].length,max=K.opts?K.opts.max_chars:0;
  $("ctl-count").textContent=`${n} / ${max}`;ctlLock()}
function ctlLog(action,args,res){const li=el("li");li.append(el("span","t",new Date().toTimeString().slice(0,8)+" "),el("span",res.ok?"ok":"bad",controlLine(action,args,res)));
  const ul=$("ctl-log");ul.prepend(li);while(ul.children.length>10)ul.lastChild.remove()}
async function ctlOptions(retry){
  try{const r=await fetch("control/options",{cache:"no-store"});
    if(r.status===404){if(retry)setTimeout(()=>ctlOptions(true),3000);return}  // 身体还没建好：3 秒后再试
    if(!r.ok)throw new Error(r.status);K.opts=await r.json();$("control").hidden=false;ctlApply();
  }catch(e){if(retry)setTimeout(()=>ctlOptions(true),3000)}}
async function ctlSend(action,args){if(K.busy)return;K.busy=true;ctlLock();let res;
  $("ctl-busy").textContent=`正在做：${controlLine(action,args,{text:"…"}).split(" → ")[0]}…（身体排队执行，环视要几十秒）`;
  try{const r=await fetch("control",{method:"POST",headers:{"Content-Type":"application/json","X-Skydango":"1"},body:JSON.stringify({action,args})});
    res=await r.json().catch(()=>({ok:false,text:`HTTP ${r.status}`}));if(!r.ok&&res.ok===undefined)res={ok:false,text:`HTTP ${r.status}`};
  }catch(e){res={ok:false,text:"连不上（程序停了？）"}}
  ctlLog(action,args,res);$("ctl-busy").textContent="";K.busy=false;ctlLock();await ctlOptions(false);return res}
// 动作刚做完有冷却（能做的列表暂时变空）：定时刷新，不用等下一次操作
setInterval(()=>{if(K.opts&&!K.busy)ctlOptions(false)},5000);
$("ctl-say-text").oninput=ctlCount;
$("ctl-say-text").onkeydown=e=>{if(e.key==="Enter"&&!e.isComposing)$("ctl-say-go").click()};
$("ctl-say-go").onclick=async()=>{const t=$("ctl-say-text").value.trim();if(!t)return;const res=await ctlSend("say",{text:t});if(res&&res.ok){$("ctl-say-text").value="";ctlCount()}};
$("ctl-emote-go").onclick=()=>ctlSend("emote",{name:$("ctl-emote-name").value,force:$("ctl-force").checked});
for(const x of document.querySelectorAll("#ctl-camera button[data-cam]"))
  x.onclick=()=>{const max=K.opts?K.opts.max_steps:4,n=Math.min(max,Math.max(1,parseInt($("ctl-steps").value,10)||1));$("ctl-steps").value=n;ctlSend("camera",{action:x.dataset.cam,steps:n})};
$("ctl-reset").onclick=()=>ctlSend("camera_reset",{});
$("ctl-around").onclick=()=>ctlSend("look_around",{});
$("ctl-panel-read").onclick=()=>ctlSend("panel_read",{});
$("ctl-panel-close").onclick=()=>ctlSend("panel_close",{});
$("ctl-pick").onclick=()=>{K.picking=!K.picking;K.trackPick=false;K.mark=null;ctlLock();if(last)draw(last)};
$("ctl-track-name").oninput=ctlLock;
$("ctl-track-pick").onclick=()=>{K.trackPick=!K.trackPick;K.picking=false;$("ctl-track-tip").textContent="";ctlLock()};
$("ctl-track-go").onclick=()=>{const name=$("ctl-track-name").value.trim();if(!name)return;
  const max=K.opts&&K.opts.max_track_seconds||60,n=Math.min(max,Math.max(1,parseInt($("ctl-track-sec").value,10)||30));
  $("ctl-track-sec").value=n;ctlSend("track",{name,seconds:n})};
$("ctl-stop").onclick=()=>ctlSend("stop_task",{});
c.addEventListener("click",e=>{if(!K.trackPick||!last)return;  // 盯人：按快照里的框认出点的是谁，填进名字
  const [x,y]=toFrame(e.clientX,e.clientY,c.getBoundingClientRect(),last.width,last.height),name=nameAt(last.boxes||[],x,y);
  K.trackPick=false;$("ctl-track-tip").textContent=name?"":"那里没认出好友的名字，换个地方点，或者直接输入";
  if(name)$("ctl-track-name").value=name;ctlLock()});
c.addEventListener("click",e=>{if(!K.picking||!last)return;
  const [x,y]=toFrame(e.clientX,e.clientY,c.getBoundingClientRect(),last.width,last.height);K.mark=[x,y];draw(last);
  setTimeout(async()=>{const go=confirm(`点 (${x}, ${y}) 这个人？`);K.picking=false;ctlLock();
    if(go)await ctlSend("check_friend",{x,y});K.mark=null;if(last)draw(last)},30)});
ctlOptions(true);
// ---- brain ----
const B={turns:new Map(),els:new Map(),version:0,boot:null,want:new Map(),acted:false};
const REASONS={events:"新消息 / 事件",background:"周围的变化",heartbeat:"心跳",farewell:"退出前总结",outside:"轮外"};
const ICONS={thinking:"💭 思考",text:"💬 说",tool:"🔧 调用",result:"↩ 返回"};
const FOLD=10;
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function kilo(n){return n==null?"—":n>=1000?(n/1000).toFixed(1)+"k":String(n)}
function hms(t){return t?new Date(t*1000).toTimeString().slice(0,8):"--:--:--"}
// 点过的按用户的来；没点过的：进行中的和最新一轮展开（收尾时不自己收起），下一轮开始后收起
function wantOpen(t,want,newestId){return want!==undefined?want:(t.end===null||t.id===newestId)}
function acted(t){return t.end===null||!!t.error||t.steps.some(s=>s.kind==="tool"||s.kind==="text")}
function toolCounts(t){const c={};for(const n of t.tools)c[n]=(c[n]||0)+1;
  const s=Object.entries(c).map(([n,k])=>`${n}×${k}`).join(" ");return s||(t.steps.some(x=>x.kind==="text")?"（只说了话）":"（什么都没做）")}
function tokens(r){if(!r)return "";const k=r.tokens||{};return `in ${kilo(k.input)} · out ${kilo(k.output)}`+(k.thinking?` · 思考 ${kilo(k.thinking)}`:"")}
function stepTitle(s){return s.kind==="tool"?`${ICONS.tool} ${s.name}`:ICONS[s.kind]+(s.kind==="result"&&s.error?"（出错 / 被拒绝）":"")}
function stepText(s){return s.kind==="tool"?JSON.stringify(s.input,null,2):(s.text||"")}
function endText(t){const r=t.result,k=(r&&r.tokens)||{},parts=[];
  if(r)parts.push(`${r.subtype??"—"} · ${r.num_turns??"—"} 步`);
  if(t.seconds!=null)parts.push(`${t.seconds.toFixed(1)}s`);
  if(r)parts.push(`输入 ${kilo(k.input)} · 输出 ${kilo(k.output)} · 缓存读 ${kilo(k.cache_read)} · 缓存写 ${kilo(k.cache_write)} · 思考 ${kilo(k.thinking)}`);
  if(r&&r.cost!=null)parts.push(`参考 $${r.cost.toFixed(4)}`);
  return (t.error?`失败：${t.error}\\n`:"")+parts.join(" · ")}
function copyText(t){const lines=[`[${hms(t.start)}] ${REASONS[t.reason]||t.reason}`,"== 收到 ==",t.prompt];
  for(const s of t.steps)lines.push(`== ${stepTitle(s)} ==`,stepText(s));
  if(t.end!==null)lines.push("== 结果 ==",endText(t));return lines.join("\\n")}
async function copy(t,btn){const text=copyText(t);
  try{await navigator.clipboard.writeText(text)}catch(e){const a=el("textarea");a.value=text;document.body.append(a);a.select();document.execCommand("copy");a.remove()}
  btn.textContent="已复制";setTimeout(()=>btn.textContent="复制这一轮",1200)}
function block(kind,title,text,error){const d=el("div",`step k-${kind}${error?" error":""}`);d.append(el("h4","",title));
  const lines=text.split("\\n"),pre=el("pre");d.append(pre);
  if(lines.length<=FOLD){pre.textContent=text;return d}
  let full=false;const btn=el("button","");
  const show=()=>{pre.textContent=full?text:lines.slice(0,FOLD).join("\\n")+"\\n…";btn.textContent=full?"收起":`展开全部（${lines.length} 行）`};
  btn.onclick=()=>{full=!full;show()};show();d.append(btn);return d}
function newest(){return Math.max(0,...B.turns.keys())}
function turnEl(t){const live=t.end===null,d=el("details",`turn${t.error?" err":""}${live?" live":""}`);
  d.open=wantOpen(t,B.want.get(t.id),newest());
  const sm=el("summary");sm.append(el("span","t",hms(t.start)),el("span","",REASONS[t.reason]||t.reason));
  if(t.error)sm.append(el("span","",`失败：${t.error}`));
  else if(live)sm.append(el("span","",`进行中…${t.tools.length?" "+toolCounts(t):""}`));
  else sm.append(el("span","",toolCounts(t)),el("span","n",t.seconds!=null?`${t.seconds.toFixed(1)}s`:""),el("span","n",tokens(t.result)));
  sm.onclick=()=>B.want.set(t.id,!d.open);d.append(sm);
  const body=el("div","tbody");
  if(t.reason!=="outside")body.append(block("prompt","收到",t.prompt||""));
  for(const s of t.steps)body.append(block(s.kind,stepTitle(s),stepText(s),s.kind==="result"&&s.error));
  if(!live)body.append(block("end","结果",endText(t),!!t.error));
  const btn=el("button","","复制这一轮");btn.onclick=()=>copy(t,btn);body.append(btn);
  d.append(body);d.hidden=B.acted&&!acted(t);return d}
function brainState(st){const e=$("brain-state");let text=`${st.model??"?"} / ${st.effort??"?"} · 已醒 ${st.turns??0} 轮 · `,cls="";
  if(st.offline){text+="已转备用回复（DeepSeek）";cls="bad"}
  else if(st.retry_in!=null){text+=`连续失败 ${st.failures} 次，${Math.ceil(st.retry_in)} 秒后重试`;cls="warn"}
  else text+="在线";e.textContent=text;e.className=cls}
function brainRender(changed){const box=$("brain-turns");
  for(const t of changed){const old=B.els.get(t.id),neu=turnEl(t);if(old)old.replaceWith(neu);B.els.set(t.id,neu)}
  const top=newest();for(const [id,e] of B.els)if(!B.want.has(id))e.open=wantOpen(B.turns.get(id),undefined,top);
  const ids=[...B.turns.keys()].sort((a,b)=>(a===0)-(b===0)||b-a);
  ids.forEach((id,i)=>{const e=B.els.get(id);if(box.children[i]!==e)box.insertBefore(e,box.children[i]||null)})}
function brainMerge(d){
  if((B.boot&&d.boot!==B.boot)||d.version<B.version){  // 程序重启过：这次的增量不可信，清空后从头拉
    B.turns.clear();for(const e of B.els.values())e.remove();B.els.clear();B.want.clear();B.boot=d.boot;B.version=0;return}
  B.boot=d.boot;B.version=d.version;for(const t of d.turns)B.turns.set(t.id,t);
  if(d.oldest!=null)for(const id of [...B.turns.keys()])if(id!==0&&id<d.oldest){B.turns.delete(id);B.els.get(id)?.remove();B.els.delete(id)}
  brainState(d.state||{});brainRender(d.turns.filter(t=>B.turns.has(t.id)))}
$("brain-acted").onchange=e=>{B.acted=e.target.checked;for(const [id,x] of B.els)x.hidden=B.acted&&!acted(B.turns.get(id))};
async function brainLoop(){let seen=false;
  while(true){
    try{const r=await fetch(`brain?after=${B.version}`,{cache:"no-store"});
      if(r.status===404&&!seen)return;if(!r.ok)throw new Error(r.status);
      const d=await r.json();seen=true;$("brain").hidden=false;brainMerge(d);
    }catch(e){if(seen){const s=$("brain-state");s.textContent="连不上（程序停了？）";s.className="bad"}await new Promise(r=>setTimeout(r,1000))}
  }
}
brainLoop();
</script></body></html>
"""
