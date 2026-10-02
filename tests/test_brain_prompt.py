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


def test_emote_does_not_release_hands_in_prompt():
    text = static_prompt(ReplyConfig())
    assert "做动作会松开手" not in text and "不会松开手" in text  # 用户实测：做动作不松手，走动才会


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


BUBBLE = "有人跟你说话时，身体已经替你冒了输入气泡"


def test_bubble_note_only_when_enabled():
    assert BUBBLE in brain_prompt(ReplyConfig(), None, bubble=True)
    assert BUBBLE not in brain_prompt(ReplyConfig(), None)


def test_brain_prompt_inner_adds_days_and_rules():
    from skydango.brain.prompt import INNER_RULES

    text = brain_prompt(ReplyConfig(), None, days="## 日子\n今天……", inner=True)
    assert "## 日子\n今天……" in text and INNER_RULES in text
    assert text.index(INNER_RULES) < text.index("- 记住聊过的内容和对方的名字")
    assert "别报数字" in INNER_RULES


def test_brain_prompt_inner_off_unchanged():
    assert brain_prompt(ReplyConfig(), None) == brain_prompt(ReplyConfig(), None, days="", inner=False)
    assert "## 日子" not in brain_prompt(ReplyConfig(), None)


def test_days_before_recent_turns(tmp_path):
    import time

    now = time.time()
    text = brain_prompt(ReplyConfig(), _history_store(tmp_path, now), history_turns=3, now=now, days="## 日子\n今天……", inner=True)
    assert text.index("## 日子") < text.index("## 上次聊到哪")


def test_mind_rules_after_inner_rules():
    from skydango.brain.prompt import INNER_RULES, MIND_RULES

    text = brain_prompt(ReplyConfig(), None, inner=True, mind=True)
    assert text.index(INNER_RULES) < text.index(MIND_RULES) < text.index("- 记住聊过的内容和对方的名字")
    assert "别扭立刻作废" in MIND_RULES and "牵手" in MIND_RULES
    assert brain_prompt(ReplyConfig(), None, inner=True) == brain_prompt(ReplyConfig(), None, inner=True, mind=False)
    assert MIND_RULES not in brain_prompt(ReplyConfig(), None, inner=True)


# ---- 内心层第 3 期：脾气、你攒下的性格 ----
def test_temper_rules_and_loosened_line():
    from skydango.brain.prompt import GO_ON_NEW, GO_ON_OLD, MIND_RULES, TEMPER_RULES

    text = brain_prompt(ReplyConfig(), None, inner=True, mind=True, temper=True)
    assert text.index(MIND_RULES) < text.index(TEMPER_RULES) < text.index("- 记住聊过的内容和对方的名字")
    assert GO_ON_NEW in text and GO_ON_OLD not in text
    for s in ("有立场", "损事不损人", "不要，懒", "收着点", "# 命令"):
        assert s in TEMPER_RULES


def test_temper_loosened_line_without_proactive():
    from skydango.brain.prompt import GO_ON_NEW, NO_NEW_TOPIC

    text = brain_prompt(ReplyConfig(), None, proactive=False, temper=True)
    assert GO_ON_NEW + NO_NEW_TOPIC in text


def test_temper_off_identical_to_phase2():  # Review Focus 5
    assert brain_prompt(ReplyConfig(), None, inner=True, mind=True) == brain_prompt(
        ReplyConfig(), None, inner=True, mind=True, temper=False, persona_text="")


def test_bottom_lines_untouched():
    on = brain_prompt(ReplyConfig(), None, inner=True, mind=True, temper=True)
    off = brain_prompt(ReplyConfig(), None, inner=True, mind=True)
    for head in ("## 身份", "## 底线"):
        seg = lambda t: t[t.index(head): t.index("\n## ", t.index(head) + 1)]  # noqa: E731
        assert seg(on) == seg(off)


def test_persona_section_before_days(tmp_path):
    text = brain_prompt(ReplyConfig(), None, days="## 日子\n今天……", persona_text="## 你攒下的性格（…）\n口头禅：害", temper=True)
    assert text.index("## 你攒下的性格") < text.index("## 日子")
    store = MemoryStore(tmp_path)
    (tmp_path / "profile.md").write_text("我是团子", encoding="utf-8")
    text = memory_prompt(ReplyConfig(), store, days="## 日子\n今天……", persona_text="## 你攒下的性格（…）\n口头禅：害")
    assert text.index("我是团子") < text.index("## 你攒下的性格") < text.index("## 日子")


def test_go_on_line_does_not_contradict_proactive():  # 终审：别和“别硬转话题”打架
    from skydango.brain.prompt import GO_ON_NEW

    assert "换个话头" not in GO_ON_NEW
    text = brain_prompt(ReplyConfig(), None, inner=True, mind=True, temper=True, proactive=True)
    assert GO_ON_NEW in text


def test_prompt_mentions_idle_looking():
    text = static_prompt(ReplyConfig())
    assert "东张西望" in text and "attention" in text


# ---- 认装扮（spec 2026-10-01-appearance §6） ----
def test_prompt_appearance_rules_only_when_enabled():
    from skydango.brain.prompt import APPEARANCE_RULES, TEMPER_RULES

    assert APPEARANCE_RULES not in brain_prompt(ReplyConfig(), None)
    assert APPEARANCE_RULES in brain_prompt(ReplyConfig(), None, appearance=True)
    text = brain_prompt(ReplyConfig(), None, inner=True, mind=True, temper=True, appearance=True)
    assert text.index(TEMPER_RULES) < text.index(APPEARANCE_RULES) < text.index("- 记住聊过的内容和对方的名字")
    for s in ("看图猜的", "像小明", "随口提", "“你自己”那行"):
        assert s in APPEARANCE_RULES
    assert brain_prompt(ReplyConfig(), None, appearance=False) == brain_prompt(ReplyConfig(), None)


def test_look_person_description_mentions_stranger_id():
    from skydango.brain.tools import DESCRIPTIONS

    assert "陌生人A" in DESCRIPTIONS["look_person"]


# ---- 冷场时的心理活动（spec 2026-10-01-lull-musing §2 §4） ----
def test_lull_rules_added():
    from skydango.brain.prompt import LULL_POINTER

    p = brain_prompt(ReplyConfig(), None, lull=True)
    assert "## 冷场的时候" in p and LULL_POINTER in p and "心里：……" in p
    assert p.index("## 主动开口") < p.index("## 冷场的时候") < p.index("## 身份")


def test_lull_rules_mention_earlier_musings():  # 叫醒时附的“之前冷场时你想过…”要有说明，不然照样想“他在忙，我等等”
    p = brain_prompt(ReplyConfig(), None, lull=True)
    assert "“之前冷场时你想过”" in p


def test_lull_off_no_rules():
    p = brain_prompt(ReplyConfig(), None)
    assert "冷场" not in p
    assert "你接了别人的话、对方没再说，不算主动开口没人接。" in p


def test_lull_without_proactive():
    p = brain_prompt(ReplyConfig(), None, proactive=False, lull=True)
    assert "## 冷场的时候" in p and p.index("## 冷场的时候") < p.index("## 身份")


def test_lull_with_backstage():  # 幕后换「身份」整节时冷场那节还在
    from skydango.brain.backstage import section

    p = brain_prompt(ReplyConfig(), None, lull=True, backstage=section("卡洛", "sonnet", "haiku", "sonnet", []))
    assert "## 冷场的时候" in p and "## 身份" not in p
    assert p.index("## 主动开口") < p.index("## 冷场的时候") < p.index("## 幕后")


def test_brain_prompt_backstage_replaces_identity():
    from skydango.brain.backstage import section

    off = brain_prompt(ReplyConfig(), None)
    on = brain_prompt(ReplyConfig(), None, backstage=section("卡洛", "sonnet", "haiku", "sonnet", []))
    assert "## 身份" in off and "## 身份" not in on and "## 幕后" in on

    def floor(p):
        return p.split("## 底线")[1].split("\n## ")[0]

    assert floor(on) == floor(off)
    assert on.index("## 主动开口") < on.index("## 幕后") < on.index("## 底线")


def test_brain_prompt_without_backstage_unchanged():
    assert brain_prompt(ReplyConfig(), None, backstage="") == brain_prompt(ReplyConfig(), None)
