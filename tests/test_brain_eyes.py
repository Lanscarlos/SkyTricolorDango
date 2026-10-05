import numpy as np
import pytest

from skydango.brain.events import EventQueue
from skydango.brain.eyes import AROUND_REQUEST, Eyes, parse_news
from skydango.config import BrainConfig, ProactiveConfig


def frame():
    return np.full((1080, 1920, 3), (60, 90, 40), np.uint8)


class Describer:
    def __init__(self, fail=False, replies=None):
        self.calls = []
        self.fail = fail
        self.replies = list(replies or [])  # 依次返回；用完了就是“描述N”

    def __call__(self, content):
        self.calls.append(content)
        if self.fail:
            raise RuntimeError("Haiku 没回")
        return self.replies.pop(0) if self.replies else f" 描述{len(self.calls)} "

    def request(self, i):
        return self.calls[i][-1]["text"]


def eyes(clock, blackout=False, labels=None, fail=False, proactive=None, busy=False, replies=None):
    d = Describer(fail, replies)
    news = []
    e = Eyes(BrainConfig(), d, frame, lambda: dict(labels or {}), lambda: blackout, clock=clock,
             proactive=proactive, busy=lambda now: busy, on_news=news.append)
    e.news_got = news  # 测试里看 on_news 收到了什么
    return e, d


def test_looks_first_then_only_on_triggers_or_timeout(clock):
    e, _ = eyes(clock)
    assert e.tick(clock()) is True  # 从没看过：马上看
    assert e.tick(clock() + 30) is False
    e.notice("chat")
    assert e.tick(clock() + 30) is False  # 聊天不触发
    e.notice("arrive")
    assert e.tick(clock() + 10) is False  # 离上次不到 20 秒
    assert e.tick(clock() + 25) is True
    assert e.tick(clock() + 25 + 181) is True  # 超过 3 分钟兜底


def test_blackout_skips(clock):
    e, d = eyes(clock, blackout=True)
    assert e.tick(clock()) is False and d.calls == []


def test_describe_frame_sends_image_names_and_caches(clock):
    e, d = eyes(clock, labels={"懒洋洋大王": (1320, 300, 160, 44, clock())})
    text = e.describe_frame(frame(), clock())
    content = d.calls[0]
    assert text == "描述1" and content[0]["type"] == "image"
    assert "懒洋洋大王：名字在 (933, 200)" in content[1]["text"] and "地点和环境" in content[1]["text"]
    assert e.summary(clock() + 35) == "场景（35 秒前看的）：\n描述1"


def test_stale_names_are_not_passed(clock):
    e, d = eyes(clock, labels={"懒洋洋大王": (1320, 300, 160, 44, clock() - 60)})
    e.describe_frame(frame(), clock())
    assert "图里没认出好友的名字" in d.calls[0][1]["text"]


def test_describe_around_marks_directions(clock):
    e, d = eyes(clock)
    e.describe_around([frame()] * 4, clock())
    content = d.calls[0]
    texts = [b["text"] for b in content if b["type"] == "text"]
    assert texts[:4] == ["【前】", "【右】", "【后】", "【左】"] and texts[-1] == AROUND_REQUEST
    assert sum(b["type"] == "image" for b in content) == 4


def test_failure_is_swallowed_and_waits(clock):
    e, _ = eyes(clock, fail=True)
    assert e.tick(clock()) is True
    assert e.latest is None and e.summary(clock()) == "场景：还没有场景描述"
    assert e.tick(clock() + 5) is False  # 别每秒都重试


def test_event_queue_pokes_eyes(clock):
    q = EventQueue(clock=clock)
    e, _ = eyes(clock)
    e.last_look = clock()
    q.subscribe(e.notice)
    q.put("scene_change", "画面变化很大")
    assert e.tick(clock() + 21) is True


def test_eyes_use_scene_note_when_given(clock):
    d = Describer()
    seen = []
    e = Eyes(BrainConfig(), d, frame, lambda: {}, lambda: False, clock=clock, note=lambda now, s: seen.append(s) or "NOTE")
    e.describe_frame(frame(), clock())
    assert d.calls[0][1]["text"].startswith("NOTE") and seen == [1280 / 1920]


def test_eyes_prompts_mention_listed_people_and_skip_self():
    from skydango.brain.eyes import EYES_SYSTEM, LOOK_REQUEST

    assert "没列出的人都叫“陌生人”" in EYES_SYSTEM and "团子、陌生人位置" not in EYES_SYSTEM
    assert "给出的陌生人、团子位置照用" in EYES_SYSTEM
    assert "团子自己不用描述" in LOOK_REQUEST


def test_eyes_do_not_auto_look_on_approach():
    from skydango.brain.eyes import AUTO_LOOK_KINDS

    assert "approach" not in AUTO_LOOK_KINDS


# ---- 新鲜事（spec 2026-09-29-proactive-chat §1） ----
def test_parse_news():
    assert parse_news("地点和环境：云野\n新鲜事：天黑了") == "天黑了"
    assert parse_news("**新鲜事**：无。") == ""
    assert parse_news("好友：没看到\n新鲜事：\n- 天黑了\n- 小明坐下弹琴了") == "天黑了；小明坐下弹琴了"
    assert parse_news("地点和环境：云野\n画面状态：正常") == ""
    assert parse_news("新鲜事: 没有") == ""
    assert parse_news("- **新鲜事：** 下雨了。") == "下雨了"
    assert len(parse_news("新鲜事：" + "很长" * 100)) == 80


def test_first_look_has_no_news_item(clock):
    e, d = eyes(clock, proactive=ProactiveConfig(), replies=["描述1\n新鲜事：天黑了"])
    assert e.tick(clock()) is True
    assert "新鲜事" not in d.request(0) and e.news_got == []


def test_second_look_compares_and_reports(clock):
    e, d = eyes(clock, proactive=ProactiveConfig(), replies=["描述1", "描述2\n新鲜事：下雨了"])
    e.tick(clock())
    assert e.tick(clock() + 200) is True
    assert "上次（200 秒前）看到的：\n描述1" in d.request(1) and "按这五项写" in d.request(1)
    assert e.news_got == ["下雨了"]


def test_old_description_not_compared(clock):
    d = Describer(replies=["描述1", "描述2\n新鲜事：下雨了"])
    news = []
    e = Eyes(BrainConfig(), d, frame, lambda: {}, lambda: False, clock=clock, proactive=ProactiveConfig(), on_news=news.append)
    e.tick(clock())
    assert e.tick(clock() + 700) is True
    assert "新鲜事" not in d.request(1) and news == []


def test_brain_look_never_reports(clock):
    e, d = eyes(clock, proactive=ProactiveConfig(), replies=["描述1", "描述2\n新鲜事：下雨了"])
    e.tick(clock())
    e.describe_frame(frame(), clock() + 30)
    assert "新鲜事" not in d.request(1) and e.news_got == []


def test_busy_looks_more_often(clock):
    e, _ = eyes(clock, proactive=ProactiveConfig(), busy=True)
    e.tick(clock())
    assert e.tick(clock() + 61) is True
    e2, _ = eyes(clock, proactive=ProactiveConfig(), busy=False)
    e2.tick(clock())
    assert e2.tick(clock() + 61) is False and e2.tick(clock() + 181) is True


def test_disabled_is_old_behavior(clock):
    e, d = eyes(clock, proactive=ProactiveConfig(enabled=False), busy=True, replies=["描述1", "描述2\n新鲜事：下雨了"])
    e.tick(clock())
    assert e.tick(clock() + 61) is False
    assert e.tick(clock() + 200) is True
    assert "新鲜事" not in d.request(1) and e.news_got == []


def test_news_callback_errors_are_swallowed(clock):
    d = Describer(replies=["描述1", "描述2\n新鲜事：下雨了"])

    def boom(text):
        raise RuntimeError("身体没接住")

    e = Eyes(BrainConfig(), d, frame, lambda: {}, lambda: False, clock=clock, proactive=ProactiveConfig(), on_news=boom)
    e.tick(clock())
    assert e.tick(clock() + 200) is True and e.latest[0].startswith("描述2")


def test_parse_news_nothing_variants():
    # 评审 #2：“无 + 补充”这类写法都当没有
    for text in ("无明显变化", "无（和上次差不多）", "没有明显变化。", "无，画面和上次基本一样", "和上次差不多，没什么变化", "none", "暂无"):
        assert parse_news("新鲜事：" + text) == "", text


def test_parse_news_numbered_and_headings():
    # 评审 #5：编号、标题、括号说明都认得；后面不带符号的“名字：…”条目不丢
    assert parse_news("5. 新鲜事：天黑了") == "天黑了"
    assert parse_news("**5. 新鲜事**：天黑了") == "天黑了"
    assert parse_news("### 新鲜事\n- 天黑了") == "天黑了"
    assert parse_news("新鲜事（和上次比）：下雨了") == "下雨了"
    assert parse_news("新鲜事：\n小明：换了斗篷\n阿花：坐下弹琴了") == "小明：换了斗篷；阿花：坐下弹琴了"
    assert parse_news("新鲜事：天黑了\n\n画面状态：正常") == "天黑了"
    assert parse_news("新鲜事：天黑了\n画面状态：正常") == "天黑了"  # 下一项（已知的标题）到此为止


def test_parse_news_ignores_trailing_summary():
    # 新鲜事后面不空行接一句总结，不拼进去
    assert parse_news("新鲜事：天黑了\n总之挺安静的") == "天黑了"
    assert parse_news("新鲜事：\n- 天黑了\n- 下雨了\n总之挺安静的") == "天黑了；下雨了"
    assert parse_news("新鲜事：\n小明：换了斗篷\n阿花：坐下弹琴了\n总之挺热闹") == "小明：换了斗篷；阿花：坐下弹琴了"
    assert parse_news("新鲜事：天黑了\n- 下雨了") == "天黑了；下雨了"


def test_eyes_skip_when_unavailable(clock):  # Claude 总闸关了：不看、不报警
    d = Describer()
    e = Eyes(BrainConfig(), d, frame, lambda: {}, lambda: False, clock=clock, available=lambda: False)
    assert e.tick(clock()) is False and d.calls == []


def test_eyes_available_true_looks(clock):
    d = Describer()
    e = Eyes(BrainConfig(), d, frame, lambda: {}, lambda: False, clock=clock, available=lambda: True)
    assert e.tick(clock()) is True and len(d.calls) == 1


def test_eyes_proxy_builds_request():
    from skydango.config import BrainConfig

    got = []
    e = Eyes(BrainConfig(), lambda content: got.append(content) or "看到一个人", lambda: None, lambda: {}, lambda: False)
    blk = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
    assert e.proxy([blk], "") == "看到一个人"
    assert got[0][0] is blk and "描述一下这张图里有什么" in got[0][-1]["text"]


def test_eyes_proxy_unavailable():
    from skydango.config import BrainConfig
    from skydango.models.errors import ModelError, ModelUnavailable

    e = Eyes(BrainConfig(), lambda c: "x", lambda: None, lambda: {}, lambda: False, available=lambda: False)
    with pytest.raises(ModelUnavailable):
        e.proxy([], "衣服")

    def boom(c):
        raise ModelError("超时")

    e = Eyes(BrainConfig(), boom, lambda: None, lambda: {}, lambda: False)
    with pytest.raises(ModelUnavailable):
        e.proxy([], "衣服")
