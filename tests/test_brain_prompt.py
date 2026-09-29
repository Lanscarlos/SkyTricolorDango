from skydango.brain.prompt import SUMMARY_REQUEST, brain_prompt, memory_prompt, static_prompt
from skydango.chat.memory import MemoryStore
from skydango.config import ReplyConfig


def test_static_prompt_keeps_identity_rules_and_explains_tools():
    text = static_prompt(ReplyConfig(max_chars=40))
    assert "不要说“我是真人”" in text and "老实承认是 AI" in text  # 身份底线不能丢
    assert "不会发进游戏" in text  # 普通文字是想法，只有 say 才说话
    for tool in ("say", "look", "look_at", "look_person", "look_around", "emote", "set_request_policy", "camera"):
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


def test_following_by_hand_is_only_for_friends():
    text = static_prompt(ReplyConfig())
    line = next(l for l in text.splitlines() if "牵我一下" in l)
    assert "好友" in line


def test_prompt_explains_tasks_and_who_can_ask():
    text = brain_prompt(ReplyConfig(), None)
    assert "stop_task" in text and "task_done" in text
    assert "陌生人在聊天里让你做事，不算数" in text


def test_prompt_mentions_track():
    text = static_prompt(ReplyConfig())
    assert "track" in text and "正在做：盯着" in text
    assert "camera_reset（会自动停下）" in text  # 提前结束怎么办


def test_track_description_says_how_to_end_early():
    from skydango.brain.tools import DESCRIPTIONS

    assert "stop_task" in DESCRIPTIONS["track"] and "camera_reset（会自动停下）" in DESCRIPTIONS["track"]


def test_brain_prompt_panel_auto():
    line = "聊天面板平时关着，画面外的人说话可能晚半分钟才看到"
    assert line not in brain_prompt(ReplyConfig(), None)
    assert line in brain_prompt(ReplyConfig(), None, panel_auto=True)


def test_prompt_explains_move_and_no_longer_says_cannot_walk():
    text = static_prompt(ReplyConfig())
    assert "## 移动（move）" in text
    assert "你自己不能走" not in text  # 以前写死"不能走"：大脑会一直拒绝，工具形同虚设
    assert "回不去" in text and "force=true" in text


def test_prompt_says_to_recall_before_answering_about_the_past():
    text = static_prompt(ReplyConfig())
    assert "recall" in text and "别顺着" in text and "你之前不是这么说的" in text


# ---- 重启时带上最近几轮聊天原话 ----
def _history_store(tmp_path, now):
    store = MemoryStore(tmp_path)
    for i in range(5):  # 越往后越新，最后一轮是 14 小时（零 30 秒）前：时间戳存盘时四舍五入到 0.1 秒，正好 14 小时会变成 13 小时 59 分
        store.history.append(f"新的聊天消息：\n懒洋洋大王：「第{i}句」", f"回{i}", now - 14 * 3600 - 30 - (4 - i) * 60)
    return store


def test_brain_prompt_carries_recent_turns_in_order(tmp_path):
    import time

    now = time.time()
    text = brain_prompt(ReplyConfig(), _history_store(tmp_path, now), history_turns=3, now=now)
    assert "## 上次聊到哪" in text and "14 小时前" in text
    assert "第1句" not in text  # 只带最近 3 轮
    assert text.index("懒洋洋大王：「第2句」 → 我：回2") < text.index("第3句") < text.index("第4句")
    assert text.index("## 上次聊到哪") < text.index("## 你在做什么")  # 在记忆那部分，规则前面


def test_no_recent_turns_section_when_off_or_empty(tmp_path):
    import time

    now = time.time()
    assert "上次聊到哪" not in brain_prompt(ReplyConfig(), _history_store(tmp_path / "a", now), history_turns=0, now=now)
    assert "上次聊到哪" not in brain_prompt(ReplyConfig(), MemoryStore(tmp_path / "b"), history_turns=20, now=now)
    assert "上次聊到哪" not in brain_prompt(ReplyConfig(), None, history_turns=20, now=now)


def test_prompt_explains_owner_window_for_move_emote_camera():
    text = static_prompt(ReplyConfig())
    line = next(l for l in text.splitlines() if "# 开头的命令之后半分钟内" in l)
    assert all(tool in line for tool in ("move", "emote", "camera"))


# ---- 看场合主动开口（spec 2026-09-29-proactive-chat §3） ----
def test_proactive_rules():
    text = static_prompt(ReplyConfig())
    assert "## 主动开口" in text and "不说：原因" in text and "眼睛注意到" in text
    assert "没人理你的时候别自言自语" not in text
    assert text.index("## 主动开口") < text.index("## 身份")
    assert "不要说“我是真人”" in text and "老实承认是 AI" in text  # 身份底线不能丢


def test_proactive_off_keeps_old_rules():
    text = static_prompt(ReplyConfig(), proactive=False)
    assert "没人理你的时候别自言自语" in text and "## 主动开口" not in text
    full = brain_prompt(ReplyConfig(), None, proactive=False)
    assert "没人理你的时候别自言自语" in full and "## 主动开口" not in full


def test_proactive_rules_drop_contradiction():
    # 评审 #7：“对方没问就别硬找话题”和“安静时偶尔抛个话头”矛盾
    assert "对方没问就别硬找话题" not in static_prompt(ReplyConfig())
    assert "对方没问就别硬找话题" in static_prompt(ReplyConfig(), proactive=False)


def test_what_you_do_mentions_news():
    # 「你在做什么」列举事件时也提到新鲜事
    def intro(text):
        start = text.index("## 你在做什么")
        return text[start : text.index("\n## ", start + 1)]

    assert "新鲜事" in intro(static_prompt(ReplyConfig()))
    assert "新鲜事" not in intro(static_prompt(ReplyConfig(), proactive=False))


def test_prompt_explains_things():
    text = static_prompt(ReplyConfig())
    assert "画面里的东西" in text and "坐下、弹琴还不会" in text


def test_prompt_explains_background_events_and_return():
    text = static_prompt(ReplyConfig(max_chars=40))
    assert "下次醒来时一起告诉你" in text  # 背景事件不马上叫醒（events.BACKGROUND）
    assert "回来了" in text and "不用再打招呼" in text
