"""大脑的工具：给 Claude 的定义 + 执行（参数逐个校验，交给身体在身体线程里做）。出错不抛异常，原因交给大脑。"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .body import REQUEST_KINDS, ToolError
from .camera import KEYS

log = logging.getLogger(__name__)

EMPTY = {"type": "object", "properties": {}}

TOOLS = [
    {
        "name": "look",
        "description": "截一张现在的游戏画面（缩到 1280×720），附带身体认出的好友名字和它们在图里的位置。5 秒内只能看一次。",
        "input_schema": EMPTY,
    },
    {
        "name": "look_at",
        "description": "放大看画面的一块（看小图标、装扮细节）。坐标按 look 返回的 1280×720 图给：左上角 x、y，宽 w，高 h。",
        "input_schema": {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in ("x", "y", "w", "h")},
            "required": ["x", "y", "w", "h"],
        },
    },
    {
        "name": "status",
        "description": "看身体现在的状态：面板和输入框开没开、身边有谁、是不是牵着手、镜头、能做的动作、互动规则、刚说过的话。",
        "input_schema": EMPTY,
    },
    {
        "name": "chat_log",
        "description": "看最近 n 条聊天记录（含你自己说的，标成“我”）。",
        "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "description": "条数，1~50，默认 20"}}},
    },
    {
        "name": "say",
        "description": "在游戏里发一句话。一次一句，口语，短。",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {
        "name": "emote",
        "description": "做一个动作（只能用 status 里“能做的动作”列出的）。牵着手时会被拦下，确定要松手才传 force=true。",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "force": {"type": "boolean"}},
            "required": ["name"],
        },
    },
    {
        "name": "set_request_policy",
        "description": (
            "改互动请求的规则（本次运行有效）。who：好友昵称，或 \"*\" 表示所有好友、\"stranger\" 表示陌生人；"
            "kind：hand 牵手 / hug 拥抱 / highfive 击掌 / piggyback 背背 / candle 点火 / \"*\" 所有；accept：接不接。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "who": {"type": "string"},
                "kind": {"type": "string", "enum": list(REQUEST_KINDS)},
                "accept": {"type": "boolean"},
            },
            "required": ["who", "kind", "accept"],
        },
    },
    {
        "name": "camera",
        "description": (
            "转视角 / 缩放：left、right 左右转（每步约 45°），up 往上看，down 往下看，zoom_in 拉近，zoom_out 拉远。"
            "steps 1~4，默认 1。转之前身体会关掉聊天记录面板，转完再打开。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": list(KEYS)}, "steps": {"type": "integer"}},
            "required": ["action"],
        },
    },
    {
        "name": "camera_reset",
        "description": "把镜头转回原位（按之前转过的反着转回去）。",
        "input_schema": EMPTY,
    },
]

ACTIONS = {"say", "emote", "set_request_policy", "camera", "camera_reset"}  # 算“做了事”的工具（心跳退档用）

_MISSING = object()


def _get(args: dict, key: str, default, kind: type, label: str):
    value = args.get(key, default)
    if value is _MISSING:
        raise ToolError(f"缺少参数 {key}")
    if (kind is int and isinstance(value, bool)) or not isinstance(value, kind):
        raise ToolError(f"参数 {key} 应该是{label}")
    return value


def _int(args: dict, key: str, default=_MISSING) -> int:
    return _get(args, key, default, int, "整数")


def _str(args: dict, key: str, default=_MISSING) -> str:
    return _get(args, key, default, str, "字符串")


def _bool(args: dict, key: str, default=_MISSING) -> bool:
    return _get(args, key, default, bool, " true / false")


class ToolBox:
    def __init__(self, body) -> None:
        self.body = body

    def run(self, name: str, args: dict) -> tuple[str | list[dict], bool]:
        """执行一个工具，返回 (tool_result 的 content, 是不是出错)。"""
        try:
            fn = self._bind(name, args or {})
            return self.body.call(fn), False
        except ToolError as exc:
            return str(exc), True
        except Exception as exc:
            log.exception("工具 %s 出错", name)
            return f"出错了：{exc}", True

    def _bind(self, name: str, a: dict) -> Callable[[], object]:
        b = self.body
        if name == "look":
            return b.look
        if name == "look_at":
            x, y, w, h = _int(a, "x"), _int(a, "y"), _int(a, "w"), _int(a, "h")
            return lambda: b.look_at(x, y, w, h)
        if name == "status":
            return b.status
        if name == "chat_log":
            n = _int(a, "n", 20)
            return lambda: b.chat_log(n)
        if name == "say":
            text = _str(a, "text")
            return lambda: b.say(text)
        if name == "emote":
            emote, force = _str(a, "name"), _bool(a, "force", False)
            return lambda: b.emote(emote, force)
        if name == "set_request_policy":
            who, kind, accept = _str(a, "who"), _str(a, "kind"), _bool(a, "accept")
            return lambda: b.set_policy(who, kind, accept)
        if name == "camera":
            action, steps = _str(a, "action"), _int(a, "steps", 1)
            return lambda: b.camera_move(action, steps)
        if name == "camera_reset":
            return b.camera_reset
        raise ToolError(f"没有这个工具：{name}")
