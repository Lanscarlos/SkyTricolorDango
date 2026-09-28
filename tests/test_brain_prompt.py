from skydango.brain.prompt import SUMMARY_REQUEST, brain_prompt, memory_prompt, static_prompt
from skydango.chat.memory import MemoryStore
from skydango.config import ReplyConfig


def test_static_prompt_keeps_identity_rules_and_explains_tools():
    text = static_prompt(ReplyConfig(max_chars=40))
    assert "不要说“我是真人”" in text and "老实承认是 AI" in text  # 身份底线不能丢
    assert "不会发进游戏" in text  # 普通文字是想法，只有 say 才说话
    for tool in ("say", "look", "look_at", "look_around", "emote", "set_request_policy", "camera"):
        assert tool in text
    assert "image=true" in text and "40 个字" in text


def test_memory_prompt_uses_files(tmp_path):
    store = MemoryStore(tmp_path)
    (tmp_path / "profile.md").write_text("我是团子", encoding="utf-8")
    (tmp_path / "friends.md").write_text("## 懒洋洋大王\n- 叫他懒懒", encoding="utf-8")
    store.add_memos(["懒洋洋大王 10月3日考试"])
    text = memory_prompt(ReplyConfig(), store)
    assert text.startswith("我是团子") and "叫他懒懒" in text and "10月3日考试" in text


def test_brain_prompt_is_persona_then_rules():
    text = brain_prompt(ReplyConfig(persona="小团子"), None)
    assert text.startswith("小团子") and text.index("小团子") < text.index("## 你在做什么")


def test_summary_request_asks_for_summary_without_tools():
    assert "经过" in SUMMARY_REQUEST and "不要调用工具" in SUMMARY_REQUEST


def test_brain_prompt_quick_around():
    assert "要十几秒" in brain_prompt(ReplyConfig(), None)
    quick = brain_prompt(ReplyConfig(), None, quick_around=True)
    assert "要十几秒" not in quick and "几秒就好" in quick


def test_brain_prompt_mentions_gestures_and_following_by_holding_hands():
    text = static_prompt(ReplyConfig())
    assert "对你挥手" in text and "回礼" in text
    assert "牵我一下" in text and "带着你走" in text
    assert "跟着别人走" not in text  # 以前说"不能跟着别人走"：现在能靠牵手跟
