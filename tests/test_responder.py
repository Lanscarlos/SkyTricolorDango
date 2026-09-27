from skydango.chat.reader import Message
from skydango.chat.responder import SKIP_TOKEN, Responder, build_system_prompt, clean_reply
from skydango.config import ReplyConfig
from skydango.vision.bubbles import Rect


class ScriptedLlm:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, [dict(m) for m in messages]))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def msg(text):
    return Message(text, Rect(0, 0, 1, 1), 0.0)


def test_clean_reply():
    assert clean_reply("  回复：“好呀，一起去！”\n（解释……）", 40) == "好呀，一起去！"
    assert clean_reply("<SKIP>", 40) is None
    assert clean_reply("", 40) is None
    assert clean_reply("一" * 50, 10) == "一" * 10


def test_system_prompt_contains_rules():
    prompt = build_system_prompt(ReplyConfig(max_chars=30))
    assert "30" in prompt and SKIP_TOKEN in prompt and "AI" in prompt


def test_reply_history_alternates_and_trims():
    llm = ScriptedLlm(["你好呀", "<skip>", "去霞谷吧"])
    r = Responder(llm, ReplyConfig(history_turns=2))
    assert r.reply([msg("你好")]) == "你好呀"
    assert r.reply([msg("啊啊啊")]) is None
    assert r.reply([msg("去哪"), msg("一起吗")]) == "去霞谷吧"
    roles = [m["role"] for m in llm.calls[-1][1]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert "「去哪」\n「一起吗」" in llm.calls[-1][1][-1]["content"]
    assert len(r.history) == 4  # history_turns=2


def test_llm_failure_returns_none():
    r = Responder(ScriptedLlm([RuntimeError("network")]), ReplyConfig())
    assert r.reply([msg("在吗")]) is None
    assert r.reply([]) is None


def test_echo_client_quotes_messages_with_speaker():
    from skydango.chat.llm import EchoClient
    from skydango.chat.reader import Message
    from skydango.chat.responder import format_incoming
    from skydango.vision.bubbles import Rect

    msgs = [Message("看？", Rect(0, 0, 1, 1), 0.0, "懒洋洋大王"), Message("嗯", Rect(0, 0, 1, 1), 0.0)]
    assert EchoClient().complete("", [{"role": "user", "content": format_incoming(msgs)}]) == "收到：看？ / 嗯"


def test_clean_reply_drops_claims_of_being_human():
    """提示词里写了不许说自己是真人，但实测模型三次里有两次还是会说：输出端再拦一道。"""
    for bad in ["行行行，那我是真人还不行吗", "我不是AI啦", "我才不是机器人", "我就是个真人", "我是人类好吗"]:
        assert clean_reply(bad, 40) is None, bad
    for ok in ["哈哈哈你才是ai", "你猜呗", "不装了，我是ai", "你是真人吗", "真人快打"]:
        assert clean_reply(ok, 40) == ok, ok
