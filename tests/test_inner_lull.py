"""冷场时的心理活动 §1~§3：冷场追踪（spec 2026-10-01-lull-musing）。"""

from skydango.chat.tracker import similar
from skydango.config import LullConfig
from skydango.inner.lull import Cue, LullTracker, parse_musing

T0 = 1_790_000_000.0
FRIENDS = ["懒洋洋大王", "阿花"]
NEAR = ["懒洋洋大王"]


def is_friend(who):
    return bool(who) and who != "我" and any(similar(who, n, 0.75) for n in FRIENDS)


def tracker():
    return LullTracker(LullConfig(), is_friend)


def chatting():
    return [(T0, "懒洋洋大王", "在吗"), (T0 + 5, "我", "在呢")]


def started():
    t, chat = tracker(), chatting()
    assert len(t.tick(T0 + 65, NEAR, chat)) == 1
    return t, chat


def test_starts_after_first_stage():
    t, chat = tracker(), chatting()
    assert t.tick(T0 + 64, NEAR, chat) == []
    cues = t.tick(T0 + 65, NEAR, chat)
    assert cues == [Cue("冷场  懒洋洋大王 1 分钟没说话了。最后是你说的「在呢」，他没接。", False)]


def test_each_stage_once():
    t, chat = started()
    assert t.tick(T0 + 100, NEAR, chat) == []
    cues = t.tick(T0 + 185, NEAR, chat)
    assert len(cues) == 1 and "3 分钟" in cues[0].text and not cues[0].final
    cues = t.tick(T0 + 365, NEAR, chat)
    assert len(cues) == 1 and "6 分钟" in cues[0].text and cues[0].final
    assert t.tick(T0 + 900, NEAR, chat) == []


def test_fast_forward_only_latest():
    t, chat = tracker(), chatting()
    cues = t.tick(T0 + 400, NEAR, chat)
    assert len(cues) == 1 and "6 分钟" in cues[0].text and cues[0].final


def test_friend_spoke_last():
    t = tracker()
    cues = t.tick(T0 + 60, NEAR, [(T0, "懒洋洋大王", "我去倒水")])
    assert cues[0].text == "冷场  懒洋洋大王 1 分钟没说话了。最后是懒洋洋大王说的「我去倒水」，你没接。"


def test_group():
    t = tracker()
    chat = [(T0, "阿花", "好看"), (T0 + 10, "懒洋洋大王", "是吧"), (T0 + 20, "我", "嗯")]
    cues = t.tick(T0 + 90, ["懒洋洋大王", "阿花"], chat)
    assert cues[0].text.startswith("冷场  大家（阿花、懒洋洋大王） 1 分钟没说话了。最后是你说的「嗯」，没人接。")


def test_no_recent_friend_talk():
    t = tracker()
    chat = [(T0, "懒洋洋大王", "在吗"), (T0 + 301, "我", "在呢")]
    assert t.tick(T0 + 1000, NEAR, chat) == []
    assert t.tick(T0 + 1000, NEAR, [(T0, "", "hi"), (T0 + 5, "我", "嗯")]) == []


def test_ocr_typo():
    t = tracker()
    cues = t.tick(T0 + 65, NEAR, [(T0, "懒洋洋大玉", "在吗"), (T0 + 5, "我", "在呢")])
    assert cues[0].text.startswith("冷场  懒洋洋大王 1 分钟")


def test_said_during_lull_no_restart():
    t, chat = started()
    chat.append((T0 + 100, "我", "人呢"))
    cues = t.tick(T0 + 185, NEAR, chat)
    assert cues[0].text.startswith("冷场  懒洋洋大王 3 分钟没说话了。")
    assert cues[0].text.endswith("这之间你又说了「人呢」。")


def test_heard_ends_with_note():
    t, chat = started()
    assert t.muse("他是不是去忙了", T0 + 70)
    assert t.heard(T0 + 130, "懒洋洋大王", "刚去倒水") == "（冷场了 2 分钟，你刚才在想：他是不是去忙了）"
    assert t.active() == []
    done = t.pop_finished()
    assert len(done) == 1
    assert "你心里想过：他是不是去忙了。后来懒洋洋大王说「刚去倒水」" in t.summary(done[0], T0 + 130)
    assert t.heard(T0 + 131, "懒洋洋大王", "嗯") == ""


def test_heard_without_musing():
    t, chat = started()
    assert t.heard(T0 + 130, "懒洋洋大王", "刚去倒水") == "（冷场了 2 分钟）"


def test_heard_stranger_does_not_end():
    t, chat = started()
    assert t.heard(T0 + 100, "", "hi") == ""
    assert len(t.active()) == 1


def test_same_last_line_not_restarted():
    t, chat = started()
    t.heard(T0 + 130, "懒洋洋大王", "嗯")  # 结束了（这句不在 chat 里：最后一句还是“在呢”）
    assert t.tick(T0 + 500, NEAR, chat) == []
    chat.append((T0 + 510, "懒洋洋大王", "嗯"))
    assert len(t.tick(T0 + 575, NEAR, chat)) == 1


def test_paused_does_not_advance():
    t, chat = started()  # T0+65 开始
    assert t.tick(T0 + 100, NEAR, chat) == []
    assert t.tick(T0 + 300, NEAR, chat, paused=True) == []  # 上一圈到这一圈的 200 秒算黑屏
    assert t.tick(T0 + 310, NEAR, chat) == []  # 暂停的 200 秒不算：才冷了 105 秒
    cues = t.tick(T0 + 385, NEAR, chat)
    assert len(cues) == 1 and "3 分钟" in cues[0].text


def test_objects_left_ends_with_summary():  # 评审 #1：对象走了也要留下总结
    t, chat = started()
    t.muse("他是不是去忙了", T0 + 66)
    assert t.tick(T0 + 70, [], chat) == []
    assert t.active() == []
    done = t.pop_finished()
    assert len(done) == 1 and t.summary(done[0], T0 + 70).endswith("你心里想过：他是不是去忙了。后来他走开了")


def test_long_lull_friend_walks_off():  # 评审 #1 路径 A：冷了很久才走，不算“聊着聊着走了”，冷场照样留下总结
    t, chat = started()
    t.muse("他是不是去忙了", T0 + 70)
    t.tick(T0 + 200, NEAR, chat)
    assert t.left("懒洋洋大王", T0 + 400, chat) is False
    t.tick(T0 + 401, [], chat)
    done = t.pop_finished()
    assert len(done) == 1 and done[0].musings[0][2] == "他是不是去忙了"


def test_expire_after_last_stage():  # 评审 #2：最后一个节点之后再过 stages[0] 秒就结束
    t, chat = started()
    t.tick(T0 + 365, NEAR, chat)
    assert t.tick(T0 + 424, NEAR, chat) == [] and len(t.active()) == 1
    assert t.tick(T0 + 425, NEAR, chat) == []
    assert t.active() == [] and t.status(T0 + 425) == ""
    assert t.pop_finished()[0].ending == "后来就一直安静着"
    assert t.tick(T0 + 900, NEAR, chat) == []  # 同一句最后一句不重新开始


def test_musing_only_for_lulls_before_turn():  # 评审 #3：这一轮开始之后才有的冷场不挂
    t, chat = started()  # T0+65 开始
    assert t.muse("嗯", T0 + 100, since=T0 + 60) is False
    assert t.active()[0].musings == []
    assert t.muse("嗯", T0 + 100, since=T0 + 70) is True


def test_muse_without_lull():
    assert tracker().muse("随便", T0) is False


def test_status():
    t, chat = tracker(), chatting()
    assert t.status(T0 + 10) == ""
    t.tick(T0 + 65, NEAR, chat)
    assert t.status(T0 + 65) == "冷场：懒洋洋大王 1 分钟没说话（最后是你说的「在呢」）· 在想：（还没想过）"
    t.muse("他是不是去忙了", T0 + 70)
    assert t.status(T0 + 80).endswith("· 在想：他是不是去忙了（1 分钟时）")


def test_parse_musing():
    assert parse_musing("想了想\n心里：他忙去了吧\n不说：等", 60) == "他忙去了吧"
    assert parse_musing("心里: 嗯 ", 60) == "嗯"
    assert parse_musing("心里：一\n心里：二", 60) == "二"
    assert parse_musing("心里：" + "长" * 80, 60) == "长" * 60
    assert parse_musing("不说：等", 60) == ""


def test_flush():
    t, chat = started()
    done = t.flush(T0 + 200)
    assert len(done) == 1 and done[0].ending == "到下线还没结束"
    assert t.active() == []


def test_snapshot():
    t, chat = started()
    t.muse("他是不是去忙了", T0 + 70)
    snap = t.snapshot(T0 + 80)
    assert snap[0]["kind"] == "silent" and snap[0]["who"] == ["懒洋洋大王"]
    assert snap[0]["last"] == ["我", "在呢"] and snap[0]["musings"][0]["text"] == "他是不是去忙了"


# ---- 情况②：聊着聊着走了（spec §1；走开先等 leave_grace 秒再叫醒，评审 #4） ----
LEFT_AT = T0 + 100
GRACE = LEFT_AT + 15


def opening(t, chat_left):
    assert t.left("懒洋洋大王", LEFT_AT, chat_left)
    assert t.tick(LEFT_AT + 14, [], []) == []
    cues = t.tick(GRACE, [], [])
    assert len(cues) == 1 and not cues[0].final
    return cues[0].text


def test_left_after_he_spoke():
    t = tracker()
    text = opening(t, [(T0, "懒洋洋大王", "我去拿个东西")])
    assert text == "冷场  懒洋洋大王 聊着聊着走开了。走之前最后是他说的「我去拿个东西」。"


def test_left_after_i_spoke():
    t = tracker()
    assert "走之前最后是你说的「你要牵好我哦」" in opening(t, [(T0 - 200, "懒洋洋大王", "哈哈"), (T0 + 50, "我", "你要牵好我哦")])


def test_left_not_chatting():
    t = tracker()
    chat = [(LEFT_AT - 121, "懒洋洋大王", "哈哈"), (LEFT_AT - 61, "我", "嗯")]
    assert t.left("懒洋洋大王", LEFT_AT, chat) is False
    assert t.left("懒洋洋大王", LEFT_AT, [(LEFT_AT - 30, "懒洋洋大玉", "拜")]) is True  # OCR 错字照样算


def test_flicker_within_grace_is_nothing():  # 评审 #4：名字标签闪一下
    t = tracker()
    assert t.left("懒洋洋大王", LEFT_AT, [(T0, "懒洋洋大王", "我去拿个东西")])
    assert t.status(LEFT_AT + 5) == ""
    assert t.returned("懒洋洋大王", LEFT_AT + 5) is None
    assert t.active() == [] and t.pop_finished() == []
    assert t.tick(GRACE, [], []) == []


def left_one(musing=None):
    t = tracker()
    opening(t, [(T0, "懒洋洋大王", "我去拿个东西")])
    if musing:
        assert t.muse(musing, GRACE + 10)
    return t


def test_left_stages():
    t = left_one()
    assert t.tick(LEFT_AT + 179, [], []) == []
    assert t.tick(LEFT_AT + 180, [], []) == [Cue("冷场  懒洋洋大王 走开 3 分钟了，还没回来。", False)]
    cues = t.tick(LEFT_AT + 360, [], [])
    assert len(cues) == 1 and cues[0].final
    t.tick(LEFT_AT + 420, [], [])
    assert t.active() == [] and t.pop_finished()[0].ending == "一直没回来"  # 评审 #2
    assert t.returned("懒洋洋大王", LEFT_AT + 3600) is None


def test_returned():
    t = left_one("是不是我说错话了")
    assert t.returned("阿花", LEFT_AT + 200) is None
    assert t.returned("懒洋洋大王", LEFT_AT + 200) == "（走开了 3 分钟，你刚才在想：是不是我说错话了）"
    assert t.active() == []
    summary = t.summary(t.pop_finished()[0], LEFT_AT + 200)
    assert summary.startswith("懒洋洋大王聊着聊着走开了 3 分钟。") and summary.endswith("后来他回来了")


def test_left_heard_ends():
    t = left_one()
    assert t.heard(LEFT_AT + 70, "懒洋洋大玉", "我回来了") == "（走开了 1 分钟）"
    assert t.active() == []


def test_left_takes_over_silent():
    t, chat = started()
    t.muse("他是不是去忙了", T0 + 70)
    assert t.left("懒洋洋大王", T0 + 80, chat) is True
    lulls = t.active()
    assert len(lulls) == 1 and lulls[0].kind == "left"
    assert lulls[0].musings[0][2] == "他是不是去忙了"
    assert t.tick(T0 + 81, [], chat) == []  # 情况① 没有再冒出来


def test_both_kinds_status():
    t = tracker()
    chat = [(T0, "阿花", "好看"), (T0 + 5, "懒洋洋大王", "我去拿个东西")]
    t.left("懒洋洋大王", T0 + 10, chat)
    t.tick(T0 + 70, ["阿花"], chat)
    status = t.status(T0 + 70)
    assert status.startswith("冷场：") and "；" in status and "阿花" in status and "懒洋洋大王 走开" in status
