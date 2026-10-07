"""把身体的工具通过本机 MCP 服务交给 Claude Code（大脑）：只监听 127.0.0.1、随机端口、后台线程跑。

工具先过 ToolBox（参数校验、每轮计数），再经 Body.call() 在身体线程执行。同步的工具函数 mcp 会放到线程池里跑。
"""

from __future__ import annotations

import base64
import logging
import socket
import threading
import time

import uvicorn
from mcp.server.mcpserver import Image, MCPServer
from mcp.types import CallToolResult, TextContent

from .tools import CALL_DESCRIPTION, INTROSPECT_DESCRIPTION, JAB_NOTE, ToolBox, descriptions

log = logging.getLogger(__name__)


def to_mcp(out) -> list:
    """ToolBox 的结果（文字，或图片块 + 文字块的列表）→ MCP 工具返回值。"""
    if isinstance(out, str):
        return [out]
    items: list = []
    for block in out:
        if block.get("type") == "image":
            items.append(Image(data=base64.standard_b64decode(block["source"]["data"]), format="jpeg"))
        elif block.get("type") == "text":
            items.append(block["text"])
    return items


def build_server(toolbox: ToolBox) -> MCPServer:
    srv = MCPServer("sky", instructions="光遇里的身体：看画面、说话、做动作、转视角。")

    DESCRIPTIONS = descriptions(hasattr(getattr(toolbox.body, "env", None), "sweep"))

    def call(tool: str, **args):
        out, err = toolbox.run(tool, args)
        if err:  # 错误文字原样给模型，让它换个办法
            return CallToolResult(content=[TextContent(type="text", text=str(out))], is_error=True)
        return to_mcp(out)

    # 参数写在函数签名里（MCP 用它生成给模型看的参数说明），注册顺序 = DESCRIPTIONS 的顺序
    @srv.tool(name="look", description=DESCRIPTIONS["look"])
    def look(image: bool = False):
        return call("look", image=image)

    @srv.tool(name="look_at", description=DESCRIPTIONS["look_at"])
    def look_at(x: int, y: int, w: int, h: int):
        return call("look_at", x=x, y=y, w=w, h=h)

    @srv.tool(name="look_person", description=DESCRIPTIONS["look_person"])
    def look_person(name: str):
        return call("look_person", name=name)

    @srv.tool(name="look_around", description=DESCRIPTIONS["look_around"])
    def look_around():
        return call("look_around")

    if toolbox.calling:  # 按 Q 喊一声：开关打开、有感知层才有（关掉时工具列表逐字照旧）
        @srv.tool(name="call", description=CALL_DESCRIPTION)
        def call_out():
            return call("call")

    @srv.tool(name="status", description=DESCRIPTIONS["status"])
    def status():
        return call("status")

    @srv.tool(name="chat_log", description=DESCRIPTIONS["chat_log"])
    def chat_log(n: int = 20):
        return call("chat_log", n=n)

    @srv.tool(name="recall", description=DESCRIPTIONS["recall"])
    def recall(query: str = "", who: str = "", days: int = 14):
        return call("recall", query=query, who=who, days=days)

    if toolbox.jab:  # [pacing] manner：大脑自己标这句贱不贱（关着时工具列表逐字照旧）
        @srv.tool(name="say", description=f"{DESCRIPTIONS['say']} {JAB_NOTE}")
        def say(text: str, jab: bool = False):
            return call("say", text=text, jab=jab)
    else:
        @srv.tool(name="say", description=DESCRIPTIONS["say"])
        def say(text: str):
            return call("say", text=text)

    @srv.tool(name="emote", description=DESCRIPTIONS["emote"])
    def emote(name: str):
        return call("emote", name=name)

    @srv.tool(name="set_request_policy", description=DESCRIPTIONS["set_request_policy"])
    def set_request_policy(who: str, kind: str, accept: bool):
        return call("set_request_policy", who=who, kind=kind, accept=accept)

    @srv.tool(name="camera", description=DESCRIPTIONS["camera"])
    def camera(action: str, steps: int = 1):
        return call("camera", action=action, steps=steps)

    @srv.tool(name="camera_reset", description=DESCRIPTIONS["camera_reset"])
    def camera_reset():
        return call("camera_reset")

    @srv.tool(name="attention", description=DESCRIPTIONS["attention"])
    def attention(mode: str, focus: str = ""):
        return call("attention", mode=mode, focus=focus)

    @srv.tool(name="move", description=DESCRIPTIONS["move"])
    def move(direction: str, steps: int = 1, force: bool = False):
        return call("move", direction=direction, steps=steps, force=force)

    @srv.tool(name="check_friend", description=DESCRIPTIONS["check_friend"])
    def check_friend(x: int, y: int):
        return call("check_friend", x=x, y=y)

    @srv.tool(name="track", description=DESCRIPTIONS["track"])
    def track(name: str, seconds: int = 30):
        return call("track", name=name, seconds=seconds)

    @srv.tool(name="find", description=DESCRIPTIONS["find"])
    def find(name: str, seconds: int = 30):
        return call("find", name=name, seconds=seconds)

    @srv.tool(name="stop_task", description=DESCRIPTIONS["stop_task"])
    def stop_task():
        return call("stop_task")

    @srv.tool(name="panel_read", description=DESCRIPTIONS["panel_read"])
    def panel_read(image: bool = False):
        return call("panel_read", image=image)

    @srv.tool(name="panel_press", description=DESCRIPTIONS["panel_press"])
    def panel_press(button: str):
        return call("panel_press", button=button)

    @srv.tool(name="panel_close", description=DESCRIPTIONS["panel_close"])
    def panel_close():
        return call("panel_close")

    if toolbox.backstage:  # 幕后：开关打开才有，放最后（TOOL_NAMES 的顺序不变）
        @srv.tool(name="introspect", description=INTROSPECT_DESCRIPTION)
        def introspect(topic: str):
            return call("introspect", topic=topic)

    return srv


def _free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class SkyServer:
    def __init__(self, toolbox: ToolBox, host: str = "127.0.0.1") -> None:
        self.port = _free_port(host)
        self.url = f"http://{host}:{self.port}/mcp"
        app = build_server(toolbox).streamable_http_app()
        self._server = uvicorn.Server(uvicorn.Config(app, host=host, port=self.port, log_level="warning"))
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 10.0) -> None:
        logging.getLogger("mcp").setLevel(logging.WARNING)  # 每个请求都打 INFO，太吵
        self._thread = threading.Thread(target=self._server.run, name="mcp", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("MCP 服务没起来")
            time.sleep(0.05)
        log.info("MCP 服务：%s", self.url)

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(5)
