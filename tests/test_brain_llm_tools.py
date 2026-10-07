from types import SimpleNamespace

from skydango.brain.llm_tools import openai_tools


def fake_toolbox(backstage=False, calling=False):
    return SimpleNamespace(backstage=backstage, calling=calling, body=SimpleNamespace(env=None))


def _tool(name, tb=None):
    return next(t for t in openai_tools(tb or fake_toolbox()) if t["function"]["name"] == name)


def test_openai_tools_all_present_with_question():
    names = [t["function"]["name"] for t in openai_tools(fake_toolbox())]
    assert set(names) >= {"look", "look_at", "look_person", "look_around", "check_friend", "say", "panel_read"}
    lp = _tool("look_person")["function"]["parameters"]
    assert "question" in lp["properties"] and lp["required"] == ["name"]
    pr = _tool("panel_read")["function"]["parameters"]["properties"]
    assert "image" in pr and "question" in pr
    assert "question" not in _tool("say")["function"]["parameters"]["properties"]


def test_backstage_and_call_flags_add_tools():
    tb = fake_toolbox(backstage=True, calling=True)
    names = [t["function"]["name"] for t in openai_tools(tb)]
    assert "introspect" in names and "call" in names


def test_required_and_defaults():
    say = next(t for t in openai_tools(fake_toolbox()) if t["function"]["name"] == "say")
    assert say["function"]["parameters"]["required"] == ["text"]
    cl = next(t for t in openai_tools(fake_toolbox()) if t["function"]["name"] == "chat_log")
    assert cl["function"]["parameters"]["properties"]["n"]["default"] == 20  # OpenAI 用 "default"，非 required


def test_say_jab_only_when_on():
    """say 标 jab（spec 2026-10-07-chat-pacing §2）：只在 [pacing] manner 开着时有，关着逐字照旧。"""
    from skydango.brain.tools import DESCRIPTIONS, JAB_NOTE

    off = _tool("say")
    assert off["function"]["parameters"] == {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    assert off["function"]["description"] == DESCRIPTIONS["say"]
    tb = fake_toolbox()
    tb.jab = True
    on = _tool("say", tb)
    assert on["function"]["parameters"]["properties"]["jab"] == {"type": "boolean", "default": False}
    assert on["function"]["parameters"]["required"] == ["text"]
    assert on["function"]["description"] == DESCRIPTIONS["say"] + " " + JAB_NOTE
