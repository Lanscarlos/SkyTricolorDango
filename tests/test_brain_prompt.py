from skydango.brain.prompt import COMPACT_REQUEST, memory_prompt, static_prompt
from skydango.chat.memory import MemoryStore
from skydango.config import ReplyConfig


def test_static_prompt_keeps_identity_rules_and_explains_tools():
    text = static_prompt(ReplyConfig(max_chars=40))
    assert "不要说“我是真人”" in text and "老实承认是 AI" in text  # 身份底线不能丢
    assert "不会发进游戏" in text  # 普通文字是想法，只有 say 才说话
    for tool in ("say", "look", "look_at", "emote", "set_request_policy", "camera"):
        assert tool in text
    assert "40 个字" in text


def test_memory_prompt_uses_files(tmp_path):
    store = MemoryStore(tmp_path)
    (tmp_path / "profile.md").write_text("我是团子", encoding="utf-8")
    (tmp_path / "friends.md").write_text("## 懒洋洋大王\n- 叫他懒懒", encoding="utf-8")
    store.add_memos(["懒洋洋大王 10月3日考试"])
    text = memory_prompt(ReplyConfig(), store)
    assert text.startswith("我是团子") and "叫他懒懒" in text and "10月3日考试" in text


def test_memory_prompt_without_store_uses_persona():
    assert memory_prompt(ReplyConfig(persona="小团子"), None).startswith("小团子")


def test_compact_request_asks_for_summary_without_tools():
    assert "摘要" in COMPACT_REQUEST and "不要调用工具" in COMPACT_REQUEST
