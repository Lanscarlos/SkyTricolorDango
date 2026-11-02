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


def test_sighted_tools_no_question_and_look_always_image():
    """大脑能直接看图：图不走眼睛代看，question 没人回答，不给；look 一律给图，不要 image 参数。"""
    tools = {t["function"]["name"]: t for t in openai_tools(fake_toolbox(), sighted=True)}
    for name in ("look", "look_at", "look_person", "check_friend", "panel_read"):
        assert "question" not in tools[name]["function"]["parameters"]["properties"]
    assert tools["look"]["function"]["parameters"]["properties"] == {}
    assert "眼睛" not in tools["look"]["function"]["description"] and "编号" in tools["look"]["function"]["description"]
    assert "1280" not in tools["look_at"]["function"]["description"]
    assert "image" in tools["panel_read"]["function"]["parameters"]["properties"]  # 面板细读默认给文字，照旧


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
