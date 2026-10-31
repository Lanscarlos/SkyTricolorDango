# 记忆条目化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** notes.md / inbox.md 的每条记忆带日期、长期标记和可选的键；同键新的取代旧的、近况过期进 archive；系统提示词只常驻长期条目（有上限），近况按这一轮的话检索、放进唤醒消息的「想起来的：」。

**Architecture:** 纯函数模块 `chat/entries.py` 管解析、渲染、取代 / 淡出、常驻切分；`NotesKeeper` 换新提示词并在写盘前过 `settle` 和把关；`brain/recollect.py` 的 `Recollector` 在身体线程里按事件打分，经 `Brain.recollect` 钩子插进唤醒消息（和「回法」同一套路）。开关 `[reply] memory_entries` 关掉时逐字照旧。

**Tech Stack:** Python 3.13 标准库（`dataclasses`、`datetime`、`difflib`、`re`），pytest。不加依赖。

**Spec:** `docs/superpowers/specs/2026-10-31-memory-entries-design.md`

## Global Constraints

- 不加新依赖；不改 `history.jsonl`、`friends.md`、`profile.md` 的格式。
- 系统提示词在一次上线内不变（前缀缓存）；「想起来的」只进唤醒消息。
- `memory_entries = false`：系统提示词、唤醒消息、整理 / 随手记的提示词、反思材料、`recall`、`--no-brain` 回复与现在逐字一样。
- 条目行格式：`- 正文 〔YYYY-MM-DD｜长期｜键〕`，方括号用全角〔〕，分隔用全角｜，字段顺序固定，长期和键可省，`长期` 不能当键。
- 时间一律用调用方给的钟（身体的 `wall`、`NotesKeeper.wall`），不直接 `time.time()` / `date.today()`：沙盒是模拟时钟。
- 默认值：`recent_days = 30`、`pinned_max = 1500`、`recall_per_turn = 5`、`recall_chars = 400`、`recall_half_days = 14`、`recall_min_score = 2.0`。
- 出错只记日志、退回照旧的样子，不打断大脑 / 整理。
- 中文注释、中文日志，照周围代码的密度。

## Review Focus

- 用户手改 notes 时写了别的括号（`[2026-10-31]`、`【长期】`、半角 `|`）：整行当正文、照常常驻，不丢、不报错。→ Task 1 `test_other_brackets_are_text`
- 模型写了未来的日期（笔误 `2026-11-30`）：`show` 写成绝对日期，不出现「-29 天前」；近况淡出不把它当过期。→ Task 1 `test_show_future_date`、Task 2 `test_future_date_not_expired`
- 沙盒的模拟时钟比真实时间晚 / 早：相对日期按世界的钟算。→ Task 5 `test_recollect_uses_given_clock`
- 整理输出包了 ```` ``` ```` 或开头多一句「好的，以下是…」：照样解析，那一句当解析不了的行计入把关比例（少量不触发拒绝）。→ Task 3 `test_update_accepts_fenced_output_with_preamble`
- 说话人是 OCR 错字或叫法（「番茄」「番茄炒蛋盖坂」）：能归到 `## 番茄炒蛋盖饭` 那一节。→ Task 5 `test_speaker_alias_and_typo_match_section`

---

## 文件

| 文件 | 职责 |
|---|---|
| `src/skydango/chat/entries.py`（新） | `Entry`、解析、渲染、`show` 相对日期、`settle`、`check_rewrite`、`split_pinned`、`render_shown` |
| `src/skydango/chat/memory.py` | 新提示词 `NOTES_SYSTEM_ENTRIES` / `MEMO_SYSTEM_ENTRIES`；`MemoryStore` 的 archive / 备份 / `entries_since`；`NotesKeeper.update_now` 走 settle + 把关 |
| `src/skydango/config.py` | `ReplyConfig` 七个新字段 |
| `src/skydango/console/settings.py` | 设置页四项 |
| `src/skydango/brain/prompt.py` | `memory_prompt` 只常驻长期条目 |
| `src/skydango/brain/recollect.py`（新） | `Recollector`：候选池、打分、一次上线只带一次 |
| `src/skydango/brain/loop.py` | `Brain.recollect` 钩子 |
| `src/skydango/brain/body.py` | `recollect_line`；反思材料按节取 |
| `src/skydango/cli.py` | 组装 Recollector、挂钩子、`memory show` |
| `src/skydango/chat/recall.py`、`chat/responder.py` | 渲染 + archive |
| `CLAUDE.md` | 记忆一节 |

---

### Task 1: 条目解析、渲染、相对日期（`chat/entries.py`）

**Files:**
- Create: `src/skydango/chat/entries.py`
- Test: `tests/test_memory_entries.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Entry: section: str; text: str; date: datetime.date | None; pinned: bool; key: str; raw: str`（`pinned` = 长期；`raw` = 原始行去掉首尾空白）
  - `parse_notes(text: str) -> list[Entry]`：`## 标题` 定节，节外的条目归 `""`；非 `- ` 开头的非空行跳过
  - `parse_inbox(text: str) -> list[Entry]`：`- 谁：内容〔…〕`，「我」归 `自己`；没有「谁：」的归 `""`
  - `render_notes(entries: list[Entry]) -> str`：按节第一次出现的顺序输出 `## 节` + 各行，节之间空一行；`section == ""` 的放最前、无标题
  - `entry_line(e: Entry) -> str`：`- 正文 〔日期｜长期｜键〕`（没有日期、长期、键时不写方括号）
  - `show(e: Entry, today: datetime.date) -> str`：正文 +（相对日期），日期不详只给正文

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date
from skydango.chat.entries import Entry, parse_notes, parse_inbox, render_notes, show

NOTES = """## 番茄炒蛋盖饭
- 养仓鼠，叫芝麻球 〔2026-10-31｜长期｜仓鼠.名字〕
- 下周去青岛出差三天 〔2026-10-31｜出差〕
- 新耳机是粉色猫耳造型 〔2026-10-31〕

## 自己
- 无翼，跑图容易出事
"""

def test_parse_fields():
    es = parse_notes(NOTES)
    assert [(e.section, e.text, e.date, e.pinned, e.key) for e in es] == [
        ("番茄炒蛋盖饭", "养仓鼠，叫芝麻球", date(2026, 10, 31), True, "仓鼠.名字"),
        ("番茄炒蛋盖饭", "下周去青岛出差三天", date(2026, 10, 31), False, "出差"),
        ("番茄炒蛋盖饭", "新耳机是粉色猫耳造型", date(2026, 10, 31), False, ""),
        ("自己", "无翼，跑图容易出事", None, True, ""),   # 没方括号 = 长期、日期不详
    ]

def test_render_roundtrip():
    assert render_notes(parse_notes(NOTES)) == NOTES

def test_bad_bracket_is_text():
    (e,) = parse_notes("## 小明\n- 去了雨林 〔昨天｜长期〕\n")
    assert e.text == "去了雨林 〔昨天｜长期〕" and e.date is None and e.pinned

def test_other_brackets_are_text():
    es = parse_notes("## 小明\n- 去了雨林 [2026-10-31]\n- 爱吃辣【长期】\n- 换发型 〔2026-10-30|发型〕\n")
    assert [e.text for e in es] == ["去了雨林 [2026-10-31]", "爱吃辣【长期】", "换发型 〔2026-10-30|发型〕"]
    assert all(e.pinned and e.date is None for e in es)

def test_parse_inbox_speaker():
    es = parse_inbox("- 番茄炒蛋盖饭：煮了酸辣粉 〔2026-10-31〕\n- 我：不会吹萨克斯 〔2026-10-31｜长期〕\n- 随手一句\n")
    assert [(e.section, e.text, e.pinned) for e in es] == [
        ("番茄炒蛋盖饭", "煮了酸辣粉", False), ("自己", "不会吹萨克斯", True), ("", "随手一句", True)]

@pytest.mark.parametrize("d,want", [
    (date(2026, 10, 31), "今天"), (date(2026, 10, 30), "昨天"), (date(2026, 10, 21), "10 天前"),
    (date(2026, 10, 2), "10 月 2 日"), (date(2026, 9, 26), "上个月 09-26"), (date(2026, 3, 1), "3 月 1 日"),
    (date(2025, 9, 17), "去年 09-17"), (date(2024, 9, 17), "2024 年 9 月 17 日"),
])
def test_show_relative(d, want):
    e = Entry("小明", "去了雨林", d, False, "", "")
    assert show(e, date(2026, 10, 31)) == f"去了雨林（{want}）"

def test_show_future_date():
    e = Entry("小明", "约好跑图", date(2026, 11, 30), False, "", "")
    assert show(e, date(2026, 10, 31)) == "约好跑图（11 月 30 日）"

def test_show_unknown_date():
    assert show(Entry("自己", "无翼", None, True, "", ""), date(2026, 10, 31)) == "无翼"
```

相对日期规则（按顺序取第一个命中的）：0 天「今天」；1「昨天」；2~13「N 天前」；未来的日期、或同一年同一个月 ≥ 14 天 → 走「同一年」；上一个日历月「上个月 MM-DD」；同一年「M 月 D 日」；去年「去年 MM-DD」；更早「YYYY 年 M 月 D 日」。

- [ ] **Step 2: Run** `python -m pytest -q tests/test_memory_entries.py` — Expected: FAIL（`ModuleNotFoundError: skydango.chat.entries`）
- [ ] **Step 3: Implement** `entries.py`：行尾方括号用正则 `\s*〔([^〔〕]*)〕\s*$` 取；字段按 `｜` 拆，第一项必须是合法日期（`date.fromisoformat`），否则整行当正文（日期不详、长期）；第二项起 `长期` 设 pinned、其余第一个当键。
- [ ] **Step 4: Run** `python -m pytest -q tests/test_memory_entries.py` — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(memory): 记忆条目的解析、渲染和相对日期"`

---

### Task 2: 取代、补日期、淡出、把关、常驻切分（`chat/entries.py`）

**Files:**
- Modify: `src/skydango/chat/entries.py`
- Test: `tests/test_memory_entries.py`

**Interfaces:**
- Consumes: Task 1 的 `Entry`、`parse_notes`
- Produces:
  - `settle(old: list[Entry], new: list[Entry], today: date, recent_days: int) -> tuple[list[Entry], list[tuple[Entry, str]]]`：返回（留下的、[(挪走的, 原因)]）
  - `check_rewrite(old_text: str, new_text: str) -> str | None`：该拒绝就返回原因
  - `split_pinned(entries: list[Entry], today: date, pinned_max: int) -> tuple[list[Entry], list[Entry]]`：（常驻的、挪出常驻的长期条目），只看 `pinned` 的条目
  - `archive_line(e: Entry, today: date, reason: str) -> str`：`- 〔挪走 2026-10-31：原因〕<e.raw>`

- [ ] **Step 1: Write the failing tests**

```python
T = date(2026, 10, 31)
def E(sec, text, d=None, pinned=False, key=""):
    return Entry(sec, text, d, pinned, key, entry_line(Entry(sec, text, d, pinned, key, ""))[2:])

def test_same_key_keeps_newest():
    old = [E("番茄", "早饭吃榴莲包子", date(2026, 10, 24), key="早饭")]
    new = [E("番茄", "早饭吃榴莲包子", date(2026, 10, 24), key="早饭"), E("番茄", "早饭吃油条", T, key="早饭")]
    keep, gone = settle(old, new, T, 30)
    assert [e.text for e in keep] == ["早饭吃油条"]
    assert gone == [(new[0], "被取代：早饭吃油条")]

def test_same_key_other_section_untouched():
    keep, gone = settle([], [E("番茄", "a", T, key="早饭"), E("小明", "b", T, key="早饭")], T, 30)
    assert len(keep) == 2 and gone == []

def test_fill_missing_date_from_similar_old():
    old = [E("番茄", "养仓鼠，叫芝麻球", date(2026, 10, 1), True)]
    keep, _ = settle(old, [E("番茄", "养仓鼠叫芝麻球", None, True)], T, 30)
    assert keep[0].date == date(2026, 10, 1)

def test_no_fill_when_not_similar():
    keep, _ = settle([E("番茄", "养仓鼠", date(2026, 10, 1), True)], [E("番茄", "要去青岛出差", None)], T, 30)
    assert keep[0].date is None

def test_recent_expires_pinned_stays():
    new = [E("番茄", "在追长风渡", date(2026, 9, 30)), E("番茄", "喜欢云野", date(2026, 1, 1), True)]
    keep, gone = settle([], new, T, 30)
    assert [e.text for e in keep] == ["喜欢云野"] and gone[0][1] == "近况过期"

def test_future_date_not_expired():
    keep, gone = settle([], [E("番茄", "约好跑图", date(2026, 11, 30))], T, 30)
    assert len(keep) == 1 and gone == []

def test_split_pinned_overflow_oldest_first_self_last():
    es = [E("自己", "x" * 600, None, True), E("番茄", "y" * 600, date(2026, 1, 1), True),
          E("番茄", "z" * 600, date(2026, 10, 1), True), E("番茄", "近况", T)]
    pinned, moved = split_pinned(es, T, 1500)
    assert [e.text[0] for e in moved] == ["y"] and {e.text[0] for e in pinned} == {"x", "z"}

def test_check_rewrite_rejects_shrink_and_garbage():
    old = "## 番茄\n" + "".join(f"- 第{i}条 〔2026-10-31〕\n" for i in range(10))
    assert check_rewrite(old, "## 番茄\n- 第1条 〔2026-10-31〕\n") is not None
    assert check_rewrite(old, "乱七八糟\n" * 20 + "## 番茄\n- 一条\n") is not None
    assert check_rewrite(old, old) is None
    assert check_rewrite("## 番茄\n- 一条\n", "## 番茄\n") is None      # 旧的不到 6 条不管骤减

def test_archive_line():
    e = E("番茄", "在追长风渡", date(2026, 9, 30))
    assert archive_line(e, T, "近况过期") == "- 〔挪走 2026-10-31：近况过期〕在追长风渡 〔2026-09-30〕"
```

`settle` 规则：取代按（节, 键）分组、留日期最新的（`None` 当最旧；一样新留 `new` 里靠后的）；补日期在取代之前做，相似度 `difflib.SequenceMatcher(None, a, b).ratio() >= 0.8`、只在同一节里找；近况淡出：非 pinned、日期不为空、`0 <= (today - date).days > recent_days` 才挪。`check_rewrite` 的「解析不了」= 新文本里既不是 `## ` 标题、也不是 `- ` 开头的非空行；超过非空行的一半拒绝；旧条目 ≥ 6 且新条目 < 旧的一半拒绝。`split_pinned` 按渲染后的 `entry_line` 长度累加，超了从「日期最旧（None 最旧）、非 `自己`」开始挪，`自己` 最后挪。

- [ ] **Step 2: Run** `python -m pytest -q tests/test_memory_entries.py` — Expected: FAIL（`ImportError: settle`）
- [ ] **Step 3: Implement** 上面四个函数
- [ ] **Step 4: Run** — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(memory): 同键取代、补日期、近况淡出、整理把关、常驻切分"`

---

### Task 3: 配置 + 整理 / 随手记接上条目（`config.py`、`chat/memory.py`）

**Files:**
- Modify: `src/skydango/config.py`（`ReplyConfig`）、`src/skydango/console/settings.py`、`src/skydango/chat/memory.py`、`src/skydango/cli.py`（建 `NotesKeeper` 的几处传 `entries`）
- Test: `tests/test_memory.py`、`tests/data/memory_prompts_baseline.json`（新，开关关着时的提示词基准）

**Interfaces:**
- Consumes: Task 1、2 全部
- Produces:
  - `ReplyConfig.memory_entries: bool = True`、`recent_days: int = 30`、`pinned_max: int = 1500`、`recall_per_turn: int = 5`、`recall_chars: int = 400`、`recall_half_days: float = 14.0`、`recall_min_score: float = 2.0`
  - 设置页 `Field`：`reply.memory_entries`（bool）、`reply.recent_days`、`reply.pinned_max`、`reply.recall_per_turn`，分组同 `reply.notes_every`
  - `MemoryStore.archive_path -> Path`（`notes_archive.md`）、`append_archive(lines: list[str])`、`backup_notes(stamp: str) -> Path | None`（复制到 `archive/notes-<stamp>.md`，没有 notes 返回 None）、`entries_since() -> float | None`、`set_entries_since(t: float)`；`set_notes_until` 改成合并写 `notes_state.json`（不覆盖别的键）
  - `NotesKeeper(..., entries: ReplyConfig | None = None)`：`None` = 关（逐字照旧）
  - `MEMO_SYSTEM_ENTRIES`、`NOTES_SYSTEM_ENTRIES` 两段新提示词

- [ ] **Step 1: Write the failing tests**（`tests/test_memory.py`，沿用该文件的假 LLM / `tmp_path` 写法）

```python
def test_entries_off_prompts_unchanged():          # 基准：先在改动前跑一次生成 baseline，再加断言
    assert (MEMO_SYSTEM, NOTES_SYSTEM) == tuple(json.loads(BASELINE.read_text("utf-8"))["prompts"])

def test_update_supersedes_and_archives(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_notes("## 番茄\n- 早饭吃榴莲包子 〔2026-10-24｜早饭〕\n")
    llm = FakeLLM("## 番茄\n- 早饭吃榴莲包子 〔2026-10-24｜早饭〕\n- 早饭吃油条 〔2026-10-31｜早饭〕\n")
    k = keeper(store, llm, entries=ReplyConfig(), wall=at("2026-10-31 22:00"))
    store.add_memos(["番茄：早饭吃油条 〔2026-10-31｜早饭〕"])
    assert k.update_now()
    assert "榴莲" not in store.notes() and "被取代：早饭吃油条" in store.archive_path.read_text("utf-8")

def test_update_rejects_bad_rewrite(tmp_path, caplog):
    old = "## 番茄\n" + "".join(f"- 第{i}条 〔2026-10-31〕\n" for i in range(10))
    store = MemoryStore(tmp_path); store.write_notes(old); store.add_memos(["番茄：x"])
    k = keeper(store, FakeLLM("## 番茄\n- 第1条\n"), entries=ReplyConfig(), wall=at("2026-10-31 22:00"))
    assert not k.update_now() and store.notes() == old and store.inbox()   # 拒绝时 inbox 不消化
    assert "拒绝" in caplog.text

def test_first_entries_update_backs_up_once(tmp_path):
    ...  # 两次 update_now：archive/notes-*.md 只有 1 个；notes_state.json 里 until 和 entries_since 都在

def test_update_accepts_fenced_output_with_preamble(tmp_path):
    out = "好的，以下是更新后的笔记：\n```markdown\n## 番茄\n- 养仓鼠 〔2026-10-31｜长期〕\n" + "".join(
        f"- 第{i}条 〔2026-10-31〕\n" for i in range(6)) + "```"
    ...  # update_now() 为真，notes 里有「养仓鼠」、没有「好的」

def test_entries_prompts_mention_rules():
    assert "不同的两天" in NOTES_SYSTEM_ENTRIES and "〔" in NOTES_SYSTEM_ENTRIES and "键" in NOTES_SYSTEM_ENTRIES
    assert "〔" in MEMO_SYSTEM_ENTRIES and "长期" in MEMO_SYSTEM_ENTRIES
```

- [ ] **Step 2: Run** `python -m pytest -q tests/test_memory.py` — Expected: 新测试 FAIL
- [ ] **Step 3: Implement**
  - 两段新提示词：在旧的基础上改（旧的原样留着给开关关时用）。`NOTES_SYSTEM_ENTRIES` 必须写进：条目格式和三个字段的含义（照 spec §1 原文）、什么算长期（「本人说成习惯、喜好、设定的，或者在不同的两天都提起过的；只说过一次的具体事件一律是近况，不许把几次单独的事合并成喜好」）、再次提起改日期、会变的事给键、新旧同键只写新的、长期部分不超过 1500 字。`MEMO_SYSTEM_ENTRIES`：每条 `- 谁：内容 〔日期｜长期｜键〕`。
  - `update_now` 在 `entries` 打开时：剥代码块后丢掉第一个 `## ` 之前的非条目行 → `check_rewrite(旧, 新)` 不过就 WARNING「整理结果不像样，拒绝写盘（原因）」、设 `_retry_after`、返回 False（inbox 不动）→ 第一次（`entries_since() is None`）先 `backup_notes` + `set_entries_since` → `settle(parse_notes(旧), parse_notes(新), today, recent_days)` → `write_notes(render_notes(keep))` + `append_archive` → 其余同旧。`today` 用 `datetime.fromtimestamp(self.wall()).date()`。
  - `jot` 用哪段提示词看 `entries`。
- [ ] **Step 4: Run** `python -m pytest -q tests/test_memory.py tests/test_config.py tests/test_console_settings.py` — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(memory): 整理和随手记按条目写，写盘前取代 / 淡出 / 把关"`

---

### Task 4: 系统提示词只常驻长期条目（`brain/prompt.py`、`chat/entries.py`）

**Files:**
- Modify: `src/skydango/chat/entries.py`、`src/skydango/brain/prompt.py:236-255`、调 `memory_prompt` 的地方（`cli.py`、沙盒）传 `entries` 和 `today`
- Test: `tests/test_memory_entries.py`、`tests/test_brain_prompt.py`

**Interfaces:**
- Consumes: Task 1、2
- Produces:
  - `pinned_view(notes: str, inbox: str, today: date, pinned_max: int) -> tuple[str, list[Entry]]`：（「记得的事」正文、候选池）。正文按节 `## 节` + `- show(e)`；候选池 = 非 pinned 的 notes 和 inbox 条目 + `split_pinned` 挪出的。inbox 的 pinned 条目并进正文对应的节。
  - `render_shown(notes: str, inbox: str, today: date) -> str`：全部条目都用 `show` 渲染（给 `--no-brain` 和反思用）
  - `memory_prompt(..., entries: ReplyConfig | None = None, today: date | None = None)`：`entries` 为 None 逐字照旧

- [ ] **Step 1: Write the failing tests**

```python
def test_pinned_view_splits():
    text, pool = pinned_view(NOTES, "- 番茄炒蛋盖饭：煮了酸辣粉 〔2026-10-31〕\n", date(2026, 10, 31), 1500)
    assert "养仓鼠，叫芝麻球（今天）" in text and "无翼" in text
    assert "青岛" not in text and "酸辣粉" not in text
    assert {e.text for e in pool} == {"下周去青岛出差三天", "新耳机是粉色猫耳造型", "煮了酸辣粉"}

def test_memory_prompt_entries_caps_length(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_notes("## 番茄\n" + "".join(f"- 喜欢第{i}样东西{'很' * 20} 〔2026-10-{i + 1:02d}｜长期〕\n" for i in range(28)))
    p = memory_prompt(ReplyConfig(), store, entries=ReplyConfig(pinned_max=300), today=date(2026, 10, 31))
    section = p.split("记得的事")[1]
    assert len(section) < 600 and "第27样" in section and "第0样" not in section

def test_memory_prompt_entries_off_unchanged(tmp_path):
    ...  # 同一个 store，entries=None 时输出和改动前一样（含「刚记下的：」）
```

- [ ] **Step 2: Run** — Expected: FAIL
- [ ] **Step 3: Implement**：`memory_prompt` 在 `entries` 打开时用 `pinned_view` 的正文代替 notes + inbox（标题照 `identity_sections` 里现在给 notes 的那个，不改措辞）；出异常 WARNING 并退回旧拼法。
- [ ] **Step 4: Run** `python -m pytest -q tests/test_memory_entries.py tests/test_brain_prompt.py` — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(memory): 系统提示词只常驻长期条目，超上限挪进候选池"`

---

### Task 5: 想起来的（`brain/recollect.py`）

**Files:**
- Create: `src/skydango/brain/recollect.py`
- Test: `tests/test_brain_recollect.py`

**Interfaces:**
- Consumes: Task 4 `pinned_view`；`brain/events.Event`（`kind`、`who`、`speech`）
- Produces:
  - `class Recollector(load: Callable[[], tuple[str, str]], match: Callable[[str], str | None], cfg: ReplyConfig, wall: Callable[[], float])`
    - 建的时候读一次 `load()`，记下当时 inbox 的原始行（只有这些 inbox 行进候选池）
    - `line(events: list[Event], epoch: int) -> str | None`：唤醒消息那一段；`epoch` 变了清空「带过的」
  - `bigrams(text: str) -> set[str]`

- [ ] **Step 1: Write the failing tests**

```python
NOTES = "## 番茄炒蛋盖饭\n- 下周去青岛出差三天 〔2026-10-21｜出差〕\n- 养仓鼠，叫芝麻球 〔2026-10-21〕\n- 在追长风渡 〔2026-10-30〕\n- 买了新耳机 〔2026-10-29〕\n## 小明\n- 腿摔了 〔2026-10-30〕\n## 自己\n- 不会吹萨克斯 〔2026-10-20〕\n"
def chat(who, text): return Event("chat", f"{who}：{text}", 0.0, speech=(who, text))

def test_topic_and_speaker():
    r = make(NOTES)
    out = r.line([chat("番茄炒蛋盖饭", "我下周出差要去青岛")], 0)
    assert out.startswith("想起来的：\n- 番茄炒蛋盖饭：下周去青岛出差三天（10 天前）")
    assert "小明" not in out

def test_threshold_needs_person_or_words():
    assert make(NOTES).line([Event("notice", "天黑了", 0.0)], 0) is None

def test_once_per_session_until_epoch_changes():
    r = make(NOTES); ev = [chat("番茄炒蛋盖饭", "青岛出差")]
    assert "青岛" in r.line(ev, 0) and (r.line(ev, 0) or "").find("青岛") < 0 and "青岛" in r.line(ev, 1)

def test_arrive_brings_latest_three():
    out = make(NOTES).line([Event("arrive", "番茄炒蛋盖饭 来到身边", 0.0, who="番茄炒蛋盖饭")], 0)
    assert [l.split("：", 1)[1].split("（")[0] for l in out.splitlines()[1:4]] == ["在追长风渡", "买了新耳机", "下周去青岛出差三天"]

def test_limits():
    out = make(NOTES, ReplyConfig(recall_per_turn=2)).line([Event("arrive", "", 0.0, who="番茄炒蛋盖饭")], 0)
    assert len(out.splitlines()) == 3

def test_self_section_when_talked_to():
    assert "萨克斯" in make(NOTES).line([chat("小明", "你会吹萨克斯吗")], 0)

def test_speaker_alias_and_typo_match_section():
    r = make(NOTES, match=lambda w: "番茄炒蛋盖饭" if w in ("番茄", "番茄炒蛋盖坂") else None)
    assert "仓鼠" in r.line([chat("番茄炒蛋盖坂", "我家仓鼠又跑了")], 0)

def test_mid_session_inbox_not_in_pool():
    ...  # load 第二次返回多一行 inbox「番茄炒蛋盖饭：去了霞谷 〔…〕」，line() 不带它

def test_recollect_uses_given_clock():
    ...  # wall 设成 2026-11-10：同一条显示「20 天前」

def test_notes_rewritten_rebuilds_pool():
    ...  # load 第二次返回的 notes 多了一条近况：带得出来
```

打分（每条候选）：人 = 节是这批 `chat` / `aside` 的说话人（经 `match`）或 `arrive` / `return` 的 `who` → +2；节是 `自己` 且这批有 `chat` 事件 → +0.5。话 = `bigrams(这批 chat / aside 原文)` 与 `bigrams(条目正文)` 交集大小，最多 +3。新 = `0.5 ** (天数 / recall_half_days)`，日期不详 0.3，未来日期按 0 天。人和话都是 0 不带；总分 < `recall_min_score` 不带。`arrive` 的人：先带他那一节日期最新的 3 条近况（一样新按文件里的顺序），不受门槛限制，但**算在上限里**。其余按分数降序（一样高按文件顺序）补，到 `recall_per_turn` 条或累计 `recall_chars` 字为止。每条带出的写一行 DEBUG「想起来：<节>：<正文> 分数 人 x 话 y 新 z」。格式 `- 节：show(e)`，`自己` 那一节写「你自己：」。`bigrams`：只留汉字和字母数字，按连续片段取相邻两字，去掉 `STOP = {"一个","这个","那个","什么","不是","没有","我们","你们","他们","自己","就是","还是","可以","知道","今天","明天"}`。每次 `line()` 重新 `load()`：notes 文本变了就重建池子（inbox 只认建时记下的行）。出错 `log.warning` 返回 None。

- [ ] **Step 2: Run** `python -m pytest -q tests/test_brain_recollect.py` — Expected: FAIL（模块不存在）
- [ ] **Step 3: Implement** `recollect.py`
- [ ] **Step 4: Run** — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(brain): 想起来的——按这一轮的人和话从近况里挑几条"`

---

### Task 6: 接进唤醒消息（`loop.py`、`body.py`、`cli.py`）

**Files:**
- Modify: `src/skydango/brain/loop.py:140-150`（`Brain.message` + 新属性）、`src/skydango/brain/body.py`（`recollector` 属性 + `recollect_line`）、`src/skydango/cli.py`（`_run_brain` 里组装，挨着 `brain.manner = …` 第 3239 行）
- Test: `tests/test_brain_loop.py`、`tests/test_brain_body.py`

**Interfaces:**
- Consumes: Task 5 `Recollector`
- Produces:
  - `Brain.recollect: Callable[[list[Event], int], str | None] | None = None`；`message()` 在「回法」之后、「状态：」之前插它的返回值，`epoch = getattr(self.session, "compactions", 0)`
  - `Body.recollector: Recollector | None = None`、`Body.recollect_line(events: list[Event], epoch: int) -> str | None`
  - cli：`[reply] memory_entries` 且有 store 时 `body.recollector = Recollector(lambda: (store.notes(), store.inbox()), match, cfg.reply, body.wall)`，`match` 用 body 现有的 `match_friend` + `parse_aliases(friends.md)`；`brain.recollect = lambda ev, ep: body.call(lambda: body.recollect_line(ev, ep), timeout=3)`

- [ ] **Step 1: Write the failing tests**

```python
def test_message_inserts_recollect_before_status():      # test_brain_loop.py，沿用该文件的 make_brain
    brain = make_brain(); brain.recollect = lambda ev, ep: "想起来的：\n- 小明：腿摔了（昨天）"
    msg = brain.message(0.0, [Event("chat", "小明：疼死了", 0.0, speech=("小明", "疼死了"))])
    assert msg.index("想起来的：") < msg.index("状态：")

def test_message_recollect_error_skipped():
    brain = make_brain(); brain.recollect = lambda ev, ep: (_ for _ in ()).throw(RuntimeError("坏了"))
    assert "想起来的" not in brain.message(0.0, [])

def test_message_passes_compactions_as_epoch():
    seen = []; brain = make_brain(); brain.session.compactions = 3
    brain.recollect = lambda ev, ep: seen.append(ep); brain.message(0.0, [])
    assert seen == [3]
```

- [ ] **Step 2: Run** — Expected: FAIL
- [ ] **Step 3: Implement**（`Body.recollect_line` 在 `recollector` 为 None 时返回 None；cli 组装出错 WARNING、不挂钩子）
- [ ] **Step 4: Run** `python -m pytest -q tests/test_brain_loop.py tests/test_brain_body.py tests/test_cli*.py` — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(brain): 唤醒消息带上「想起来的」"`

---

### Task 7: 其他用到笔记的地方 + 文档

**Files:**
- Modify: `src/skydango/brain/body.py:3088-3089`（反思材料）、`src/skydango/chat/recall.py:58-62`、`src/skydango/chat/responder.py:217-220`、`src/skydango/cli.py`（`cmd_memory` 的 show、`body.memory_notes`）、`CLAUDE.md`（「记忆」一节和代码结构表）
- Test: `tests/test_brain_body.py`、`tests/test_recall.py`、`tests/test_responder.py`、`tests/test_cli_memory.py`（没有就加进现有的 memory 命令测试）

**Interfaces:**
- Consumes: Task 1 `parse_notes`/`show`、Task 4 `render_shown`
- Produces:
  - `entries.section_lines(notes: str, inbox: str, names: list[str], today: date, limit: int = 10) -> list[str]`：这些名字那几节的条目，`f"{节}：{show(e)}"`
  - `recall(..., archive: str = "")`：附带笔记行用 `show` 渲染（`entries` 开时），archive 匹配的行前缀「（以前的）」

- [ ] **Step 1: Write the failing tests**

```python
def test_reflect_notes_by_section():                     # 现有 bug 的回归：条目行里不带名字
    lines = section_lines("## 番茄\n- 养仓鼠 〔2026-10-31〕\n## 小明\n- 腿摔了\n", "", ["番茄"], date(2026, 10, 31))
    assert lines == ["番茄：养仓鼠（今天）"]

def test_recall_renders_and_reads_archive():
    ...  # notes 里「- 养仓鼠 〔2026-10-21〕」、archive 里「- 〔挪走 …：近况过期〕在追长风渡 〔2026-09-30〕」
         # recall(query="仓鼠 长风渡") 的结果里有「养仓鼠（10 天前）」和「（以前的）在追长风渡」，没有「〔」

def test_responder_renders_entries():
    ...  # entries 开：system_prompt 里笔记没有「〔」、有「（今天）」；关：和现在一样

def test_memory_show_counts():
    ...  # 输出里有「常驻 N 条 / M 字」「候选池 K 条」
```

- [ ] **Step 2: Run** — Expected: FAIL
- [ ] **Step 3: Implement**：`Body` 反思那两行在 `self.recollector is not None` 时改用 `section_lines(...)`，否则照旧；`recall` 的 archive 由调用方读 `store.archive_path` 传进来；CLAUDE.md「记忆」表格里 notes.md / inbox.md 两行写上条目格式、加 `notes_archive.md` 一行，「记忆」段落加一段「条目化（`[reply] memory_entries`）」说明常驻 / 想起来的 / 取代 / 淡出 / 把关，代码结构表加 `chat/entries.py`、`brain/recollect.py`
- [ ] **Step 4: Run** 全量 `python -m pytest -q`（后台跑，约 7 分钟）— Expected: 只有基线里原本就挂的 5 个（`test_cli_camera` ×2、`test_console_jobs::test_dango_start_with_stop_job_stops_first`、`test_console_server` ×2）
- [ ] **Step 5: Commit** `git commit -m "feat(memory): 反思、recall、普通回复都按条目渲染；文档"`

---

### Task 8: 沙盒验收（真 DeepSeek，不写代码）

按 spec §5 三步，用 worktree 的代码起沙盒（`PYTHONPATH=src`，worktree 里放 `config.toml` / `console.toml` / `secrets.toml` 副本和 `private` 联接；面板用 `console --port 19393`，主目录 19390 的面板别点「让它退出」）。

- [ ] **Step 1:** 上线 → 番茄说「早饭吃了榴莲包子」→ 下线；再上线 → 说「早饭吃了油条」→ `python -m skydango memory update`（`-c` 指沙盒配置）或攒够轮数。Expected: `private/sandbox/memory/notes.md` 里没有「爱吃 / 喜欢吃」之类的早饭喜好；要么只剩油条（另一条在 `notes_archive.md`、原因「被取代」），要么两条各带日期的近况。
- [ ] **Step 2:** 手写一条 `- 下周去青岛出差三天 〔<10 天前>〕` 到番茄那一节 → 番茄说「出差好累」。Expected: 大脑控制台里这一轮唤醒消息有「想起来的：」带这一条、写「10 天前」。
- [ ] **Step 3:** 把 notes 塞到 3000 字长期条目 → 重启沙盒 → `sandbox-ctl brain --prompt`。Expected: 「记得的事」不超过约 1500 字；日志 INFO 有挪出常驻的提醒；被挪出的那条话题被提起时出现在「想起来的」。
- [ ] **Step 4:** 结果写进 `docs/progress/2026-10-31-memory-entries.md`（通过 / 没通过、截取的原文），提交。
