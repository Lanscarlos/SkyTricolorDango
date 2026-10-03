from types import SimpleNamespace

from skydango.brain.llm_tools import BLIND_EXCLUDE, openai_tools


def fake_toolbox(backstage=False, calling=False):
    return SimpleNamespace(backstage=backstage, calling=calling, body=SimpleNamespace(env=None))


def test_blind_excludes_vision_tools():
    tools = openai_tools(fake_toolbox(), blind=True)
    names = [t["function"]["name"] for t in tools]
    assert not (set(names) & BLIND_EXCLUDE)
    assert set(names) >= {"say", "emote", "status", "camera", "track", "panel_read", "panel_press"}


def test_sighted_keeps_vision_tools():
    names = [t["function"]["name"] for t in openai_tools(fake_toolbox(), blind=False)]
    assert set(names) >= {"look", "look_person", "check_friend"}


def test_panel_read_has_no_image_when_blind():
    pr = next(t for t in openai_tools(fake_toolbox(), blind=True) if t["function"]["name"] == "panel_read")
    assert "image" not in pr["function"]["parameters"]["properties"]


def test_backstage_and_call_flags_add_tools():
    tb = fake_toolbox(backstage=True, calling=True)
    names = [t["function"]["name"] for t in openai_tools(tb, blind=True)]
    assert "introspect" in names and "call" in names


def test_required_and_defaults():
    say = next(t for t in openai_tools(fake_toolbox(), blind=True) if t["function"]["name"] == "say")
    assert say["function"]["parameters"]["required"] == ["text"]
    cl = next(t for t in openai_tools(fake_toolbox(), blind=True) if t["function"]["name"] == "chat_log")
    assert cl["function"]["parameters"]["properties"]["n"]["default"] == 20  # OpenAI 用 "default"，非 required
