"""识别过程可视化：本机开一个网页，实时显示身体看到的画面 + 识别框（好友名字、陌生人、黑影、团子、互动圆圈、
聊天面板、新消息）+ 右侧状态。浏览器打开 http://127.0.0.1:<端口> 就能看。设计见 docs/superpowers/specs/2026-09-28-viewer-design.md。

- 只用标准库起 HTTP 服务，不加依赖；框和中文字在浏览器里画（OpenCV 画不了中文）
- 画面、框、状态放在同一份快照里（/snapshot 长轮询），框不会和画面错位
- 身体每圈调 update() 只是记下最新一帧和框（很便宜）；有浏览器在看时才压 JPEG，同一帧只压一次
- 只监听 127.0.0.1：画面里有好友昵称和聊天，别开到局域网
"""

from __future__ import annotations

import base64
import json
import logging
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

log = logging.getLogger(__name__)

MESSAGE_KEEP = 3.0  # 新消息的框留几秒
MESSAGE_CHARS = 24  # 新消息的标签最多显示几个字
WAIT = 2.0  # /snapshot 没有新帧时最多等几秒


def _box(r: Rect, kind: str, label: str) -> dict:
    return {"x": int(r.x), "y": int(r.y), "w": int(r.w), "h": int(r.h), "kind": kind, "label": label}


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
        boxes += [box for _, box in self._messages]
        state.update(info or {})
        with self._cond:
            self._seq += 1
            self._frame, self._boxes, self._info = frame, boxes, state
            self.frames += 1
            self._cond.notify_all()
        return True

    # ---- 浏览器这边 ----
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
                    try:
                        after = int(parse_qs(url.query).get("after", ["0"])[0] or 0)
                    except ValueError:
                        after = 0
                    body = viewer.snapshot(after)
                    if body is None:
                        self._send(204, "application/json", b"")
                    else:
                        self._send(200, "application/json; charset=utf-8", body)
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")

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

        self._server = ThreadingHTTPServer((self.cfg.host, self.cfg.port), Handler)
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
@media (max-width:900px){main{flex-direction:column}aside{width:100%}}
</style></head><body><main>
<div id="stage"><div class="bar"><button id="pause">暂停</button><button id="boxes">隐藏框</button><button id="save">存图</button>
<span id="status">连接中…</span></div><canvas id="c"></canvas></div>
<aside><h1>团子看到的</h1><dl id="info"></dl><div class="legend" id="legend"></div></aside>
</main><script>
const COLORS={friend:"#3ddc84",name:"#3ddc84",tag:"#facc15",stranger:"#ff9f43",unlit:"#a78bfa",player:"#60a5fa",self:"#cbd5e1",
ring:"#22d3ee",request:"#f43f5e",panel:"#6b7280",message:"#f472b6"};
const NAMES={friend:"好友",tag:"没认出的名字",stranger:"陌生人",unlit:"没点火",player:"没判定的人",self:"团子",ring:"互动圆圈",
request:"互动请求",panel:"聊天面板",message:"新消息"};
const $=id=>document.getElementById(id),c=$("c"),ctx=c.getContext("2d"),img=new Image();
let seq=0,paused=false,running=false,showBoxes=true,last=null,times=[];
$("legend").innerHTML=Object.entries(NAMES).map(([k,v])=>`<span><i style="background:${COLORS[k]}"></i>${v}</span>`).join("");
$("pause").onclick=e=>{paused=!paused;e.target.textContent=paused?"继续":"暂停";if(!paused)loop()};
$("boxes").onclick=e=>{showBoxes=!showBoxes;e.target.textContent=showBoxes?"隐藏框":"显示框";if(last)draw(last)};
$("save").onclick=()=>{if(!last)return;const a=document.createElement("a");
  a.download=`dango-${new Date().toISOString().replace(/[-:T]/g,"").slice(0,14)}.png`;a.href=c.toDataURL("image/png");a.click()};
function draw(s){
  c.width=img.naturalWidth;c.height=img.naturalHeight;ctx.drawImage(img,0,0);
  if(!showBoxes)return;const k=c.width/s.width,fs=Math.max(12,Math.round(c.width/80));
  ctx.font=`${fs}px system-ui,"Microsoft YaHei",sans-serif`;ctx.textBaseline="middle";
  for(const b of s.boxes){const col=COLORS[b.kind]||"#fff",x=b.x*k,y=b.y*k,w=b.w*k,h=b.h*k;
    ctx.strokeStyle=col;ctx.lineWidth=b.kind==="request"?4:2;ctx.setLineDash(b.kind==="panel"?[8,5]:[]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
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
    try{const r=await fetch(`/snapshot?after=${seq}`,{cache:"no-store"});if(r.status===204)continue;if(!r.ok)throw new Error(r.status);
      const s=await r.json();if(paused)break;seq=s.seq;
      await new Promise((ok,bad)=>{img.onload=ok;img.onerror=bad;img.src="data:image/jpeg;base64,"+s.image});last=s;draw(s);info(s);
      const now=performance.now();times.push(now);times=times.filter(t=>now-t<2000);
      status(`${(times.length/2).toFixed(1)} 帧/秒 · ${s.width}×${s.height}`);
    }catch(e){status("连不上（程序停了？）",true);await new Promise(r=>setTimeout(r,1000));}
  }
  running=false;
}
loop();
</script></body></html>
"""
