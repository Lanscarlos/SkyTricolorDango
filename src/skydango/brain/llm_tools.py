"""OpenAI function-calling 的工具 schema 单一来源（DeepSeek 备用大脑用）。

参数表 PARAMETERS 要和 tools.py 的 `_bind`、mcp_server.py 的 `@srv.tool` 签名三处保持同步：
改参数时一起改。
"""

from .tools import CALL_DESCRIPTION, INTROSPECT_DESCRIPTION, descriptions

BLIND_EXCLUDE = {"look", "look_at", "look_person", "look_around", "check_friend"}

# name -> [(参数名, JSON 类型, 默认值)]；默认值 None 表示必填（进 required）
PARAMETERS = {
    "look": [("image", "boolean", False)],
    "look_at": [("x", "integer", None), ("y", "integer", None), ("w", "integer", None), ("h", "integer", None)],
    "look_person": [("name", "string", None)],
    "look_around": [],
    "call": [],
    "status": [],
    "chat_log": [("n", "integer", 20)],
    "recall": [("query", "string", ""), ("who", "string", ""), ("days", "integer", 14)],
    "say": [("text", "string", None)],
    "emote": [("name", "string", None)],
    "set_request_policy": [("who", "string", None), ("kind", "string", None), ("accept", "boolean", None)],
    "camera": [("action", "string", None), ("steps", "integer", 1)],
    "camera_reset": [],
    "attention": [("mode", "string", None), ("focus", "string", "")],
    "move": [("direction", "string", None), ("steps", "integer", 1), ("force", "boolean", False)],
    "check_friend": [("x", "integer", None), ("y", "integer", None)],
    "track": [("name", "string", None), ("seconds", "integer", 30)],
    "find": [("name", "string", None), ("seconds", "integer", 30)],
    "stop_task": [],
    "panel_read": [("image", "boolean", False)],
    "panel_press": [("button", "string", None)],
    "panel_close": [],
    "introspect": [("topic", "string", None)],
}


def _function(name: str, description: str, params: list[tuple[str, str, object]]) -> dict:
    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, ptype, default in params:
        prop: dict[str, object] = {"type": ptype}
        if default is None:
            required.append(pname)
        else:
            prop["default"] = default
        properties[pname] = prop
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def openai_tools(toolbox, *, blind: bool = True) -> list[dict]:
    """DeepSeek 备用大脑的工具 schema。blind=True 时去掉视觉 / 点人工具，panel_read 不带 image。"""
    sweep = hasattr(toolbox.body.env, "sweep")
    desc = descriptions(sweep)
    tools = []
    for name in desc:  # descriptions() 的顺序就是注册顺序
        if blind and name in BLIND_EXCLUDE:
            continue
        params = PARAMETERS[name]
        if blind and name == "panel_read":
            params = [p for p in params if p[0] != "image"]
        tools.append(_function(name, desc[name], params))
    if toolbox.calling:
        tools.append(_function("call", CALL_DESCRIPTION, PARAMETERS["call"]))
    if toolbox.backstage:
        tools.append(_function("introspect", INTROSPECT_DESCRIPTION, PARAMETERS["introspect"]))
    return tools
