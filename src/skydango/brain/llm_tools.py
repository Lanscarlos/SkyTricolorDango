"""OpenAI function-calling 的工具 schema 单一来源（OpenAI 兼容的大脑 ToolLoopBrain 用）。

参数表 PARAMETERS 要和 tools.py 的 `_bind`、mcp_server.py 的 `@srv.tool` 签名三处保持同步：
改参数时一起改。
"""

from .tools import CALL_DESCRIPTION, INTROSPECT_DESCRIPTION, JAB_NOTE, descriptions

# 会返回图的工具：OpenAI 兼容的大脑看不了图，这几个交给眼睛代看，多一个可选参数 question（想看清什么）
QUESTION_TOOLS = ("look", "look_at", "look_person", "check_friend", "panel_read")
QUESTION = ("question", "string", "")

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


def openai_tools(toolbox, sighted: bool = False) -> list[dict]:
    """OpenAI 兼容大脑的工具 schema：工具全给，会返回图的几个多一个 question（眼睛代看时问什么）。
    sighted：大脑能直接看图（[brain.vision]）：图不交给眼睛，question 没人回答，不给（11-02 真机：问了「我脚下是什么」没人答）；
    look 一律给图，去掉 image 参数，说明换成 SIGHTED_DESCRIPTIONS。"""
    sweep = hasattr(toolbox.body.env, "sweep")
    desc = descriptions(sweep, sighted)
    tools = []
    for name in desc:  # descriptions() 的顺序就是注册顺序
        params = [] if sighted and name == "look" else PARAMETERS[name]
        params = params + ([QUESTION] if not sighted and name in QUESTION_TOOLS else [])
        text = desc[name]
        if name == "say" and getattr(toolbox, "jab", False):  # [pacing] manner：大脑自己标这句贱不贱
            params = params + [("jab", "boolean", False)]
            text = f"{text} {JAB_NOTE}"
        tools.append(_function(name, text, params))
    if toolbox.calling:
        tools.append(_function("call", CALL_DESCRIPTION, PARAMETERS["call"]))
    if toolbox.backstage:
        tools.append(_function("introspect", INTROSPECT_DESCRIPTION, PARAMETERS["introspect"]))
    return tools
