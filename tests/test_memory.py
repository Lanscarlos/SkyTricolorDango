import json
import logging
from datetime import datetime
from pathlib import Path

from skydango.chat.memory import (
    MEMO_SYSTEM,
    MEMO_SYSTEM_ENTRIES,
    NOTES_SYSTEM,
    NOTES_SYSTEM_ENTRIES,
    ChatMemory,
    MemoryStore,
    NotesKeeper,
    Turn,
    format_gap,
    render_transcript,
)
from skydango.chat.reader import Message
from skydango.chat.responder import Responder, build_system_prompt
from skydango.config import ReplyConfig
from skydango.vision.bubbles import Rect


class ScriptedLlm:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages, max_tokens=None):
        self.calls.append((system, [dict(m) for m in messages]))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def msg(text, speaker="懒洋洋大王"):
    return Message(text, Rect(0, 0, 1, 1), 0.0, speaker)


# ---- 提示词 ----
def test_friends_go_into_system_prompt():
    cfg = ReplyConfig(friends={"懒洋洋大王": "本名卡洛", "番茄炒蛋盖饭": "团子平时喊她“老登”"})
    prompt = build_system_prompt(cfg)
    assert "懒洋洋大王：本名卡洛" in prompt and "番茄炒蛋盖饭：团子平时喊她“老登”" in prompt
    assert "## 认识的人" not in build_system_prompt(ReplyConfig())


def test_profile_friends_and_notes_files_go_into_prompt():
    cfg = ReplyConfig(persona="配置里的人设")
    prompt = build_system_prompt(cfg, profile="文件里的人设：三彩团子", friends="## 懒洋洋大王\n- 本名卡洛", notes="## 自己\n- 不会粤语")
    assert "文件里的人设" in prompt and "配置里的人设" not in prompt  # 人设文件优先
    assert "本名卡洛" in prompt and "不会粤语" in prompt
    assert "配置里的人设" in build_system_prompt(cfg)  # 没有人设文件时用配置


def test_responder_rereads_files_so_edits_apply_without_restart(tmp_path):
    store = MemoryStore(tmp_path)
    (tmp_path / "profile.md").write_text("我是旧人设", encoding="utf-8")
    llm = ScriptedLlm(["嗯", "嗯"])
    r = Responder(llm, ReplyConfig(), store=store)
    r.reply([msg("在吗")])
    (tmp_path / "profile.md").write_text("我是新人设", encoding="utf-8")
    r.reply([msg("在吗")])
    assert "旧人设" in llm.calls[0][0] and "新人设" in llm.calls[1][0]


# ---- 聊天记录 ----
def test_history_survives_restart(tmp_path):
    t = [1000.0]
    r = Responder(ScriptedLlm(["在呢"]), ReplyConfig(), store=MemoryStore(tmp_path), clock=lambda: t[0])
    assert r.reply([msg("醒醒")]).text == "在呢"
    assert len((tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    llm = ScriptedLlm(["记得呀"])
    t[0] = 1060.0
    r2 = Responder(llm, ReplyConfig(), store=MemoryStore(tmp_path), clock=lambda: t[0])
    assert r2.reply([msg("你还记得我吗")]).text == "记得呀"
    sent = llm.calls[0][1]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert "醒醒" in sent[0]["content"] and sent[1]["content"] == "在呢"


def test_history_keeps_only_recent_turns(tmp_path):
    mem = ChatMemory(tmp_path / "h.jsonl")
    for i in range(10):
        mem.append(f"第{i}句", f"回{i}", 100.0 + i)
    assert [t.user for t in ChatMemory(tmp_path / "h.jsonl").load(3)] == ["第7句", "第8句", "第9句"]


def test_history_tolerates_broken_lines(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text('{"t": 1, "user": "a", "reply": "b"}\n{坏掉的行\n', encoding="utf-8")
    assert [t.user for t in ChatMemory(path).load(10)] == ["a"]


def test_long_gap_is_mentioned(tmp_path):
    t = [1000.0]
    llm = ScriptedLlm(["早", "下午好"])
    r = Responder(llm, ReplyConfig(), store=MemoryStore(tmp_path), clock=lambda: t[0])
    r.reply([msg("早上好")])
    t[0] += 3 * 3600
    r.reply([msg("下午好")])
    last = llm.calls[-1][1][-1]["content"]
    assert "距离上次聊天过了 3 小时" in last
    assert json.loads((tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()[-1])["user"] == last


def test_format_gap():
    assert format_gap(20 * 60) == "20 分钟"
    assert format_gap(3 * 3600 + 100) == "3 小时"
    assert format_gap(2 * 86400) == "2 天"


# ---- 长期记忆（笔记） ----
def test_render_transcript_marks_own_replies_and_skips():
    turns = [
        Turn(1.0, "新的聊天消息：\n懒洋洋大王：「粤语会吗」", "真不会"),
        Turn(2.0, "新的聊天消息：\n番茄炒蛋盖饭：「哎呀」", "<skip>"),
    ]
    text = render_transcript(turns)
    assert "懒洋洋大王：「粤语会吗」" in text and "我：真不会" in text
    assert "<skip>" not in text and "新的聊天消息" not in text


def test_notes_keeper_consolidates_pending_turns(tmp_path):
    store = MemoryStore(tmp_path)
    (tmp_path / "friends.md").write_text("## 懒洋洋大王\n- 本名卡洛", encoding="utf-8")
    for i, (u, r) in enumerate([("懒洋洋大王：「来句粤语」", "真不会"), ("懒洋洋大王：「我今天加班」", "辛苦啦")]):
        store.history.append(u, r, 100.0 + i)
    llm = ScriptedLlm(["```markdown\n## 自己\n- 不会粤语\n## 懒洋洋大王\n- 今天加班\n```"])
    keeper = NotesKeeper(llm, store, persona="三彩团子", every=2, background=False)
    assert keeper.pending == 2
    keeper.maybe_update()
    assert store.notes() == "## 自己\n- 不会粤语\n## 懒洋洋大王\n- 今天加班"  # 去掉了代码块标记
    prompt = llm.calls[0][1][0]["content"]
    assert "来句粤语" in prompt and "本名卡洛" in prompt and "三彩团子" in prompt
    assert keeper.pending == 0
    # 重启后不会重复整理已经整理过的
    assert NotesKeeper(llm, MemoryStore(tmp_path), persona="", every=2, background=False).pending == 0


def test_notes_keeper_waits_for_enough_turns_and_keeps_old_notes_on_failure(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_notes("## 自己\n- 旧笔记")
    store.history.append("懒洋洋大王：「在吗」", "在", 1.0)
    llm = ScriptedLlm(["无", RuntimeError("network"), ""])
    keeper = NotesKeeper(llm, store, persona="", every=2, background=False)
    keeper.maybe_update()
    assert llm.calls == []  # 只有 1 轮，不够 2 轮
    store.history.append("懒洋洋大王：「嗯」", "嗯", 2.0)
    keeper.turn_added(Turn(2.0, "懒洋洋大王：「嗯」", "嗯"))  # 随手记（无）→ 够 2 轮了，整理：调用失败
    assert store.notes() == "## 自己\n- 旧笔记" and keeper.pending == 2
    keeper.maybe_update()  # 模型返回空
    assert store.notes() == "## 自己\n- 旧笔记"


def test_responder_triggers_notes_update(tmp_path):
    store = MemoryStore(tmp_path)
    llm = ScriptedLlm(["嗯", "无", "好", "无", "## 自己\n- 喜欢樱花发型"])  # 回复、随手记交替，第 2 轮后整理
    keeper = NotesKeeper(llm, store, persona="", every=2, background=False)
    r = Responder(llm, ReplyConfig(), store=store, notes=keeper)
    r.reply([msg("你喜欢啥发型")])
    r.reply([msg("好看")])
    assert store.notes() == "## 自己\n- 喜欢樱花发型"


# ---- 随手记：每轮回复后单独调一次模型，只挑值得记的新信息 ----
def test_parse_memos():
    from skydango.chat.memory import parse_memos

    assert parse_memos("- 卡洛答应教团子粤语\n- 老登今天加班\n") == ["卡洛答应教团子粤语", "老登今天加班"]
    assert parse_memos("无") == [] and parse_memos("") == [] and parse_memos("- 无") == []


def test_memo_is_taken_after_reply_and_used_right_away(tmp_path):
    store = MemoryStore(tmp_path)
    llm = ScriptedLlm(["辛苦啦", "- 卡洛今天加班到九点", "那早点休息", "无"])
    keeper = NotesKeeper(llm, store, persona="", every=100, background=False)
    r = Responder(llm, ReplyConfig(), store=store, notes=keeper)
    assert r.reply([msg("我今天加班到九点")]).text == "辛苦啦"
    memo_prompt = llm.calls[1][1][0]["content"]
    assert "我今天加班到九点" in memo_prompt and "我：辛苦啦" in memo_prompt
    assert store.inbox() == "- 卡洛今天加班到九点"
    r.reply([msg("好累")])
    assert "卡洛今天加班到九点" in llm.calls[2][0]  # 下一句回复的提示词里已经有了
    assert store.inbox() == "- 卡洛今天加班到九点"  # 第二轮没什么可记的


def test_memo_is_taken_even_when_not_replying(tmp_path):
    store = MemoryStore(tmp_path)
    llm = ScriptedLlm(["<skip>", "- 老登最爱番茄炒蛋"])
    keeper = NotesKeeper(llm, store, persona="", every=100, background=False)
    r = Responder(llm, ReplyConfig(), store=store, notes=keeper)
    assert r.reply([msg("我最爱番茄炒蛋", "番茄炒蛋盖饭")]) is None
    assert store.inbox() == "- 老登最爱番茄炒蛋"


def test_memo_failure_does_not_break_reply(tmp_path):
    store = MemoryStore(tmp_path)
    llm = ScriptedLlm(["在呢", RuntimeError("network")])
    keeper = NotesKeeper(llm, store, persona="", every=100, background=False)
    assert Responder(llm, ReplyConfig(), store=store, notes=keeper).reply([msg("在吗")]).text == "在呢"
    assert store.inbox() == ""


def test_consolidation_folds_inbox_into_notes_and_keeps_new_memos(tmp_path):
    store = MemoryStore(tmp_path)
    store.history.append("懒洋洋大王：「我加班」", "辛苦啦", 1.0)
    store.add_memos(["卡洛今天加班"])

    class SlowLlm(ScriptedLlm):
        def complete(self, system, messages, max_tokens=None):
            store.add_memos(["整理期间新记的"])  # 整理进行中，聊天那边又记了一条
            return super().complete(system, messages, max_tokens)

    llm = SlowLlm(["## 懒洋洋大王\n- 最近常加班"])
    NotesKeeper(llm, store, persona="", every=1, background=False).maybe_update()
    assert "卡洛今天加班" in llm.calls[0][1][0]["content"]  # 随手记交给了整理
    assert store.notes() == "## 懒洋洋大王\n- 最近常加班"
    assert store.inbox() == "- 整理期间新记的"  # 整理过的清掉，新记的留着


def test_memo_system_records_opinions():
    from skydango.chat.memory import MEMO_SYSTEM

    assert "喜好和评价" in MEMO_SYSTEM


def test_notes_update_failure_waits_before_retry(tmp_path):
    from skydango.chat.memory import NOTES_RETRY

    store = MemoryStore(tmp_path)
    store.history.append("懒洋洋大王：「在吗」", "在", 1.0)
    llm = ScriptedLlm([RuntimeError("quota"), "## 自己\n- 新笔记"])
    now = [1000.0]
    keeper = NotesKeeper(llm, store, persona="", every=1, background=False, wall=lambda: now[0])
    keeper.maybe_update()  # 第一次整理：失败
    assert len(llm.calls) == 1
    keeper.pending += 1
    keeper.maybe_update()  # 冷却中，不再试
    assert len(llm.calls) == 1
    now[0] += NOTES_RETRY + 1
    keeper.maybe_update()  # 冷却过了，再试一次，成功
    assert len(llm.calls) == 2 and store.notes() == "## 自己\n- 新笔记"
    assert keeper._retry_after == float("-inf")


def test_update_now_ignores_retry_after(tmp_path):
    store = MemoryStore(tmp_path)
    store.history.append("懒洋洋大王：「在吗」", "在", 1.0)
    llm = ScriptedLlm([RuntimeError("quota"), "## 自己\n- 手动整理"])
    keeper = NotesKeeper(llm, store, persona="", every=1, background=False, wall=lambda: 1000.0)
    keeper.maybe_update()
    assert keeper.update_now() is True  # memory update 命令直接调，不看冷却


# ---- 记忆条目化（spec 2026-10-31-memory-entries） ----
BASELINE = Path(__file__).parent / "data" / "memory_prompts_baseline.json"


def at(text):
    t = datetime.strptime(text, "%Y-%m-%d %H:%M").timestamp()
    return lambda: t


def entries_keeper(store, llm, wall, cfg=None):
    return NotesKeeper(llm, store, persona="", every=100, background=False, wall=wall, entries=cfg or ReplyConfig())


def test_entries_off_prompts_unchanged():
    assert [MEMO_SYSTEM, NOTES_SYSTEM] == json.loads(BASELINE.read_text("utf-8"))["prompts"]


def test_entries_off_keeper_uses_old_prompts(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_notes("## 自己\n- 旧")
    store.add_memos(["小明：新事"])
    for cfg in (None, ReplyConfig(memory_entries=False)):
        llm = ScriptedLlm(["## 自己\n- 旧\n- 新"])
        NotesKeeper(llm, store, persona="", background=False, entries=cfg).update_now()
        assert llm.calls[0][0] == NOTES_SYSTEM
        store.add_memos(["小明：新事"])
    llm = ScriptedLlm(["无"])
    NotesKeeper(llm, store, persona="", background=False, entries=ReplyConfig(memory_entries=False)).jot(Turn(1.0, "a", "b"))
    assert llm.calls[0][0] == MEMO_SYSTEM


def test_update_supersedes_and_archives(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_notes("## 番茄\n- 早饭吃榴莲包子 〔2026-10-24｜早饭〕\n")
    llm = ScriptedLlm(["## 番茄\n- 早饭吃榴莲包子 〔2026-10-24｜早饭〕\n- 早饭吃油条 〔2026-10-31｜早饭〕\n"])
    k = entries_keeper(store, llm, at("2026-10-31 22:00"))
    store.add_memos(["番茄：早饭吃油条 〔2026-10-31｜早饭〕"])
    assert k.update_now()
    assert llm.calls[0][0] == NOTES_SYSTEM_ENTRIES
    assert "榴莲" not in store.notes() and "油条" in store.notes()
    assert "被取代：早饭吃油条" in store.archive_path.read_text("utf-8")


def test_update_rejects_bad_rewrite(tmp_path, caplog):
    old = "## 番茄\n" + "".join(f"- 第{i}条 〔2026-10-31〕\n" for i in range(10))
    store = MemoryStore(tmp_path)
    store.write_notes(old)
    store.add_memos(["番茄：x"])
    k = entries_keeper(store, ScriptedLlm(["## 番茄\n- 第1条\n"]), at("2026-10-31 22:00"))
    with caplog.at_level(logging.WARNING):
        assert not k.update_now()
    assert store.notes() == old.strip() and store.inbox()  # 拒绝时 inbox 不消化
    assert "拒绝" in caplog.text and not store.archive_path.exists()


def test_first_entries_update_backs_up_once(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_notes("## 番茄\n- 旧笔记")
    store.history.append("a", "b", 100.0)
    llm = ScriptedLlm(["## 番茄\n- 旧笔记 〔2026-10-31｜长期〕", "## 番茄\n- 旧笔记 〔2026-10-31｜长期〕\n- 新 〔2026-10-31〕"])
    k = entries_keeper(store, llm, at("2026-10-31 22:00"))
    assert k.update_now()
    store.add_memos(["番茄：新"])
    assert k.update_now()
    backups = list((tmp_path / "archive").glob("notes-*.md"))
    assert len(backups) == 1 and backups[0].read_text("utf-8").strip() == "## 番茄\n- 旧笔记"
    state = json.loads((tmp_path / "notes_state.json").read_text("utf-8"))
    assert state["until"] == 100.0 and "entries_since" in state  # 合并写，until 没被盖掉


def test_update_accepts_fenced_output_with_preamble(tmp_path):
    out = "好的，以下是更新后的笔记：\n```markdown\n## 番茄\n- 养仓鼠 〔2026-10-31｜长期〕\n" + "".join(
        f"- 第{i}条 〔2026-10-31〕\n" for i in range(6)) + "```"
    store = MemoryStore(tmp_path)
    store.add_memos(["番茄：养仓鼠"])
    k = entries_keeper(store, ScriptedLlm([out]), at("2026-10-31 22:00"))
    assert k.update_now()
    assert "养仓鼠" in store.notes() and "好的" not in store.notes() and "```" not in store.notes()


def test_entries_prompts_mention_rules():
    assert "不同的两天" in NOTES_SYSTEM_ENTRIES and "〔" in NOTES_SYSTEM_ENTRIES and "键" in NOTES_SYSTEM_ENTRIES
    assert "〔" in MEMO_SYSTEM_ENTRIES and "长期" in MEMO_SYSTEM_ENTRIES


def test_jot_entries_uses_entries_prompt(tmp_path):
    store = MemoryStore(tmp_path)
    llm = ScriptedLlm(["- 番茄：煮了酸辣粉 〔2026-10-31〕"])
    entries_keeper(store, llm, at("2026-10-31 22:00")).jot(Turn(1.0, "a", "b"))
    assert llm.calls[0][0] == MEMO_SYSTEM_ENTRIES and "酸辣粉" in store.inbox()


def test_responder_renders_entries(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_notes("## 番茄\n- 养仓鼠 〔2026-10-31｜长期〕\n- 去青岛 〔2026-10-31｜出差〕\n")
    store.add_memos(["番茄：煮了酸辣粉 〔2026-10-30〕"])
    t = datetime.strptime("2026-10-31 22:00", "%Y-%m-%d %H:%M").timestamp()
    llm = ScriptedLlm(["嗯", "嗯"])
    Responder(llm, ReplyConfig(), store=store, clock=lambda: t).reply([msg("在吗")])
    on = llm.calls[0][0]
    assert "养仓鼠（今天）" in on and "酸辣粉（昨天）" in on and "〔" not in on
    Responder(llm, ReplyConfig(memory_entries=False), store=store, clock=lambda: t).reply([msg("在吗")])
    off = llm.calls[1][0]
    assert "- 养仓鼠 〔2026-10-31｜长期〕" in off and "刚记下的：\n- 番茄：煮了酸辣粉" in off
