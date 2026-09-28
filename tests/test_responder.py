from skydango.chat.reader import Message
from skydango.chat.responder import SKIP_TOKEN, Reply, Responder, build_system_prompt, clean_reply, parse_reply
from skydango.config import ReplyConfig
from skydango.vision.envdiff import EnvSnapshot
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
    assert r.reply([msg("你好")]).text == "你好呀"
    assert r.reply([msg("啊啊啊")]) is None
    assert r.reply([msg("去哪"), msg("一起吗")]).text == "去霞谷吧"
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


EMOTES = ["鞠躬", "害羞", "欢呼"]


def test_parse_reply_splits_emote_tag():
    assert parse_reply("[害羞]哪有啦", 40, EMOTES) == Reply("哪有啦", "害羞")
    assert parse_reply("【欢呼】好耶", 40, EMOTES) == Reply("好耶", "欢呼")
    assert parse_reply("哪有啦 [害羞]", 40, EMOTES) == Reply("哪有啦", "害羞")
    assert parse_reply("[鞠躬]", 40, EMOTES) == Reply(None, "鞠躬")
    assert parse_reply("[鞠躬]\n晚安", 40, EMOTES) == Reply("晚安", "鞠躬")  # 标签单独一行
    assert parse_reply("回复：“[害羞]哪有啦”", 40, EMOTES) == Reply("哪有啦", "害羞")
    assert parse_reply("好呀", 40, EMOTES) == Reply("好呀", None)
    assert parse_reply("<skip>", 40, EMOTES) is None


def test_parse_reply_drops_unknown_emote_but_keeps_text():
    assert parse_reply("[跳舞]来了", 40, EMOTES) == Reply("来了", None)
    assert parse_reply("[跳舞]", 40, EMOTES) is None
    assert parse_reply("[害羞]哪有啦", 40, []) == Reply("哪有啦", None)  # 这一轮不让做动作


def test_parse_reply_claim_of_human_drops_emote_too():
    assert parse_reply("[害羞]我是真人啦", 40, EMOTES) is None


def test_parse_reply_truncates_only_text():
    assert parse_reply("[欢呼]" + "一" * 50, 10, EMOTES) == Reply("一" * 10, "欢呼")


def test_reply_render():
    assert Reply("哪有啦", "害羞").render("【AI】") == "[害羞]【AI】哪有啦"
    assert Reply(None, "鞠躬").render("【AI】") == "[鞠躬]"
    assert Reply("好呀").render() == "好呀"


def test_prompt_emote_section_only_when_available():
    plain = build_system_prompt(ReplyConfig())
    assert "## 动作" not in plain and "做动作" in plain  # 没动作可用：还是“动不了”
    with_emotes = build_system_prompt(ReplyConfig(), emotes=["鞠躬", "害羞"])
    assert "## 动作" in with_emotes and "鞠躬、害羞" in with_emotes and "[动作名]" in with_emotes


def test_responder_asks_available_emotes_each_turn_and_keeps_tag_in_history():
    llm = ScriptedLlm(["[害羞]哪有啦", "嗯嗯"])
    offers = [["害羞"], []]
    r = Responder(llm, ReplyConfig(), available_emotes=lambda: offers.pop(0))
    assert r.reply([msg("你真可爱")]) == Reply("哪有啦", "害羞")
    assert "## 动作" in llm.calls[0][0]
    assert r.reply([msg("真的")]) == Reply("嗯嗯", None)
    assert "## 动作" not in llm.calls[1][0]
    assert llm.calls[1][1][1] == {"role": "assistant", "content": "[害羞]哪有啦"}


def test_claims_human_filter_catches_variants():
    from skydango.chat.responder import clean_reply

    for bad in ("我当然是真人", "我可是活人", "当然不是AI啦", "真人一个", "我是真人", "我不是机器人哦"):
        assert clean_reply(bad, 40) is None, bad
    for ok in ("你才是AI吧", "哈哈哈随你怎么想", "我是AI", "在呢"):
        assert clean_reply(ok, 40) == ok, ok


def test_env_change_is_prefixed_to_user_turn_and_kept_in_history():
    snaps = iter([EnvSnapshot(frozenset(), 0, None), EnvSnapshot(frozenset({"小明"}), 0, None)])
    llm = ScriptedLlm(["嗨", "来啦"])
    r = Responder(llm, ReplyConfig(), env_snapshot=lambda: next(snaps))
    r.reply([msg("在吗")])
    assert "这之间" not in llm.calls[0][1][-1]["content"]  # 第一轮没有"之前"
    r.reply([msg("我来了")])
    assert llm.calls[1][1][-1]["content"].startswith("（这之间：小明来了）\n")
    assert r.history[-2]["content"].startswith("（这之间：小明来了）")
