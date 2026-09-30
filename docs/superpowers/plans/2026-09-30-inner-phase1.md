# 内心层第 1 期（关系卡 + 日子）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 身体在主循环里给好友记关系卡、给每次上线记一行，大脑在 `arrive` 事件、`status`、系统提示词「日子」里看到交情和日子。

**Architecture:** 新包 `src/skydango/inner/`：`ledger.py`（纯数据：卡、这次上线、拼文字）、`store.py`（`memory/inner/` 的读写、原子写、坏文件改名）、`days.py`（「日子」一节和 `memory show` 的文字）、`backfill.py`（从 history.jsonl 回填）、`__init__.py`（`open_ledger`：启动时补意外结束 → 回填 → 建 Ledger）。身体只多几处 `_ledger_call(...)`；cli 建 Ledger、退出时 `close`。不调模型、不发游戏输入。

**Tech Stack:** Python 3.11+ 标准库（json、dataclasses、difflib 经 `chat.tracker.similar`）；pytest。

**Spec:** `docs/superpowers/specs/2026-09-30-inner-phase1-design.md`

## Global Constraints

- 只给好友建卡：名字是 friends.md 的 `## 标题`（`MemoryStore.friend_names()` + `reply.friends`，即 cli 的 `_friend_names(cfg)`）；陌生人、`"我"`、看不出是谁的不建卡
- 账本在 `<memory_dir>/inner/`：`people.json`、`days.jsonl`、`current.json`；只在 live 写盘，dry-run 内存里照记、磁盘上什么都不动（包括回填、补意外结束）
- 时间一律墙上时间 `time.time()`（身体用 `self.wall()`），日期按本地时区 `"YYYY-MM-DD"`
- `[inner]` 默认值：`enabled = true`、`visit_gap = 1800`、`session_gap = 7200`、`long_gap = 7`（天）、`save_every = 60`、`max_step = 5`
- 记账、写盘、拼文字出错只 `log.exception`，身体、大脑照常；拼不出就退回原来的文字
- `[inner] enabled = false` 或没有 `memory_dir`：不记账、不回填、提示词和 status 和现在逐字一样
- 原子写：先写 `<name>.tmp` 再 `replace`
- 中文注释、中文日志，风格同 `brain/occasion.py`、`brain/reflex.py`

## Review Focus

1. **dry-run 碰了磁盘**：dry-run 启动时 `people.json` 不存在要回填、`current.json` 残留要补 —— 都只在内存里做，`memory/inner/` 下不能多出任何文件（Task 5 测）
2. **两个好友名字很像**（`懒洋洋大王` / `懒洋洋大王子`）：OCR 出 `懒洋洋大王子` 必须记到 `懒洋洋大王子`，不能记到先出现的 `懒洋洋大王`（两个都过 `similar` 的子串规则）（Task 1 测）
3. **跨午夜挂机**：23:50 到 00:20 一直在场，`today_minutes` 在 0 点清零，读 `status_line` 不显示昨天的时长（Task 1 测）
4. **意外结束后连着重启两次**：`crash` 那行只补一次（`current.json` 补完就删）（Task 2 测）
5. **`people.json` 坏了**：改名放一边、空卡开始、**不回填**，这次运行照常记账（Task 5 测）

---

### Task 1: `[inner]` 配置 + `Ledger` 记账和拼文字（纯内存）

**Files:**
- Modify: `src/skydango/config.py`（`ReflexConfig` 后面加 `InnerConfig`，`Config` 加 `inner` 字段）
- Create: `src/skydango/inner/__init__.py`（这一步只放模块说明）
- Create: `src/skydango/inner/ledger.py`
- Test: `tests/test_inner_ledger.py`

**Interfaces:**
- Produces:
  - `InnerConfig`（字段同 Global Constraints，`long_gap` 单位是天）；`Config.inner`
  - `day_of(t: float) -> str`：本地日期 `"2026-09-30"`
  - `match_friend(name: str, friends: Sequence[str]) -> str | None`
  - `@dataclass Card`：`first_met: float`、`last_seen: float | None = None`、`visits: int = 0`、`days: list[str]`、`minutes: float = 0.0`、`today: str = ""`、`today_minutes: float = 0.0`、`today_visits: int = 0`、`lines: int = 0`、`to_me: int = 0`、`last_line: dict | None = None`（`{"t": float, "text": str}`）；`to_dict() -> dict`、`Card.from_dict(d) -> Card`
  - `@dataclass Session`：`start: float`、`end: float | None = None`、`live: bool = True`、`ended: str = ""`、`friends: list[str]`、`heard: int = 0`、`said: int = 0`、`summary: str = ""`、`saved: float | None = None`；`to_dict()` / `Session.from_dict(d)`
  - `class Ledger(cfg: InnerConfig, friends: Callable[[], list[str]], now: float, cards: dict[str, Card] | None = None, history: list[Session] | None = None, store=None, persist: bool = False)`：
    `present(names, now) -> dict[str, str]`、`heard(speaker, text, to_me, now) -> None`、`said(now) -> None`、`status_line(names, now) -> str`、`card(name) -> Card | None`（拷贝）、`session: Session`、`cards: dict[str, Card]`、`history: list[Session]`（以前的上线，旧到新）
  - `card_line(name: str, card: Card, now: float) -> str`（`memory show` 用）

- [ ] **Step 1: 写失败的测试** `tests/test_inner_ledger.py`（`T0 = 时间戳(2026-09-30 20:00 本地)`，用 `time.mktime` 造；`FRIENDS = ["小明", "阿花", "懒洋洋大王", "懒洋洋大王子"]`；注意两个字的名字错一个字 `similar` 只有 0.5，过不了 0.75，错字测试用长名字）

```python
def test_match_friend_prefers_closest():                 # Review Focus 2
    assert match_friend("懒洋洋大王子", FRIENDS) == "懒洋洋大王子"
    assert match_friend("懒洋羊大王", FRIENDS) == "懒洋洋大王"    # 错一个字
    assert match_friend("小明", FRIENDS) == "小明"
    assert match_friend("", FRIENDS) is None and match_friend("我", FRIENDS) is None
    assert match_friend("路人甲", FRIENDS) is None

def test_first_meeting_note_and_card():
    led = ledger()
    assert led.present(["小明"], T0) == {"小明": "（第一次在身边见到）"}
    c = led.card("小明")
    assert (c.first_met, c.visits, c.today_visits, c.days) == (T0, 1, 1, ["2026-09-30"])

def test_same_visit_within_gap_counts_once():
    led = ledger()
    led.present(["小明"], T0)
    led.present([], T0 + 60)
    assert led.present(["小明"], T0 + 1799) == {}           # 30 分钟内再出现：同一次
    assert led.card("小明").visits == 1

def test_new_visit_note_after_gap():
    led = ledger()
    led.present(["小明"], T0)
    note = led.present(["小明"], T0 + 3 * 3600)["小明"]
    assert note == "（第 2 次见；上次 3 小时前；今天第 2 次）"

def test_long_gap_and_last_line_before_today():
    led = ledger()
    led.heard("小明", "明天要考试", False, T0 - 12 * 86400)
    led.present(["小明"], T0 - 12 * 86400)
    note = led.present(["小明"], T0)["小明"]
    assert note == "（第 2 次见；上次 12 天前，好久没见了；今天第一次）。他上次最后说的（12 天前）：「明天要考试」"

def test_last_line_today_not_attached():
    led = ledger()
    led.present(["小明"], T0 - 3 * 3600)
    led.heard("小明", "晚点见", False, T0 - 3 * 3600)
    assert "上次最后说的" not in led.present(["小明"], T0)["小明"]

def test_minutes_step_capped_and_clock_back():
    led = ledger()
    led.present(["小明"], T0)
    led.present(["小明"], T0 + 2)          # +2 s
    led.present(["小明"], T0 + 600)        # 卡了 598 s：只算 5
    led.present(["小明"], T0 + 590)        # 时间往回跳：0
    assert led.card("小明").minutes == pytest.approx(7 / 60)
    assert led.card("小明").last_seen == T0 + 600

def test_midnight_resets_today():                       # Review Focus 3
    led = ledger()
    night = time.mktime((2026, 9, 30, 23, 59, 58, 0, 0, -1))
    led.present(["小明"], night - 600)
    for t in range(int(night - 600), int(night + 4), 2):
        led.present(["小明"], t)
    c = led.card("小明")
    assert c.today == "2026-10-01" and c.today_minutes < 0.1 and c.days == ["2026-09-30", "2026-10-01"]
    assert "今天刚来" in led.status_line(["小明"], night + 4)

def test_heard_counts_friend_lines_only():
    led = ledger()
    led.heard("懒洋羊大王", "x" * 50, True, T0)     # 错字也记到懒洋洋大王
    led.heard("路人", "你好", False, T0)
    led.heard("", "？", False, T0)
    c = led.card("懒洋洋大王")
    assert (c.lines, c.to_me, c.last_line) == (1, 1, {"t": T0, "text": "x" * 40})
    assert led.card("路人") is None and led.session.heard == 3 and led.session.friends == ["懒洋洋大王"]
    assert c.last_seen is None                  # 只说话没见过：不算见过

def test_status_line():
    led = ledger()
    led.present(["小明", "阿花"], T0 - 21 * 86400)
    led.present(["小明"], T0)
    for t in range(int(T0), int(T0 + 2405), 5):     # 到 T0+2400：480 × 5 s = 40 分钟
        led.present(["小明"], t)
    led.present(["阿花"], T0 + 2400)
    assert led.status_line(["小明", "阿花", "路人"], T0 + 2400) == (
        "小明（今天一起 40 分钟·认识 21 天·一起玩过 2 天）、阿花（今天刚来·认识 21 天·一起玩过 2 天）、路人"
    )

def test_status_line_new_friend():
    led = ledger()
    led.present(["阿花"], T0)
    assert led.status_line(["阿花"], T0 + 10) == "阿花（今天刚来·今天刚认识）"

def test_said_counts():
    led = ledger(); led.said(T0); led.said(T0)
    assert led.session.said == 2

def test_card_roundtrip():
    c = Card(first_met=T0, days=["2026-09-30"], last_line={"t": T0, "text": "hi"})
    assert Card.from_dict(c.to_dict()) == c
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_inner_ledger.py -q`
Expected: FAIL（`ModuleNotFoundError: skydango.inner.ledger`）

- [ ] **Step 3: 实现** `InnerConfig`（docstring 指向 spec）和 `inner/ledger.py`

- `match_friend`：先用 `similar(name, f, 0.75)` 过滤，再取 `SequenceMatcher(None, normalize(name), normalize(f)).ratio()` 最高的（`normalize` 同在 `chat/tracker.py`）；`name` 去空白后为空或是 `occasion.ME`（`"我"`）返回 `None`
- `present`：对 `names` 里每个 `match_friend` 到的好友：没卡就建（`first_met = now`）；`last_seen is None or now - last_seen > visit_gap` 算新见面 → 先按**更新前**的卡写说明，再 `visits += 1`；跨日期先清 `today_minutes` / `today_visits`；`dt = min(max(0, now - prev), max_step)`，`prev` 是这张卡上一次 `present` 的时刻（新见面时 `dt = 0`）；`last_seen = max(last_seen, now)`；追加日期、这次上线的 `friends`
- 说明文字（`format_gap` 用 `chat.memory.format_gap`，不到 1 分钟写"不到 1 分钟"）：
  - `last_seen is None` → `"（第一次在身边见到）"`
  - 否则 `"（第 {visits} 次见；上次 {gap}前{，好久没见了}；{今天第一次 | 今天第 N 次}）"`（`visits`、`today_visits` 都是加上这一次之后；`gap > long_gap * 86400` 才加"，好久没见了"）
  - `last_line` 的日期早于今天时再接 `"。他上次最后说的（{gap}前）：「{text}」"`
- `heard`：`session.heard += 1`；匹配到好友 → 没卡就建（`last_seen` 留 `None`）、`lines += 1`、`to_me` 为真再加、`last_line = {"t": now, "text": text[:40]}`、追加日期和 `friends`
- `status_line`：每个名字：没卡 → 名字；有卡 → `名字（a·b·c）`：a = 今天待了不到 1 分钟（或 `today` 不是今天）写"今天刚来"，否则"今天一起 N 分钟"（取整）；b = `first_met` 日期到今天 0 天写"今天刚认识"，否则"认识 N 天"；c = `len(days) > 1` 才写"一起玩过 N 天"。用"、"连
- `card_line(name, card, now)`：`status_line` 那一段 + `"；见过 {visits} 次，上次 {gap}前；说过 {lines} 句，跟你说过 {to_me} 句"`（`last_seen is None` 写"还没在身边见过"）
- `Ledger.__init__` 里 `session = Session(start=now, live=persist)`；`save` / `close` 在 Task 2 加

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_inner_ledger.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py src/skydango/inner tests/test_inner_ledger.py
git commit -m "feat(inner): [inner] 配置和账本：好友关系卡、这次上线、arrive 说明和 status 文字"
```

---

### Task 2: `InnerStore` 读写 + `Ledger.save` / `close`

**Files:**
- Create: `src/skydango/inner/store.py`
- Modify: `src/skydango/inner/ledger.py`（加 `save`、`close`）
- Test: `tests/test_inner_store.py`

**Interfaces:**
- Consumes: Task 1 的 `Card`、`Session`、`Ledger`
- Produces:
  - `class InnerStore(directory: str | Path)`：
    - `people_exists() -> bool`
    - `load_people() -> tuple[dict[str, Card], float | None, bool]`：`(cards, backfilled, ok)`；文件不存在 → `({}, None, True)`；读不了（坏 JSON / `version != 1`）→ 改名 `people.json.bad-<YYYYmmdd-HHMMSS>`、`({}, None, False)`、WARNING
    - `write_people(cards, backfilled: float | None) -> None`（`{"version": 1, "backfilled": …, "people": {名字: card.to_dict()}}`）
    - `days() -> list[Session]`（读不了的行跳过、WARNING）
    - `append_day(s: Session) -> None`、`append_days(ss: list[Session]) -> None`
    - `write_current(s: Session) -> None`、`peek_current() -> Session | None`（只读：不在 / 读不了都是 `None`，不删不改名，dry-run 用）、`take_current() -> Session | None`（读出来就删；读不了 → 改名 `.bad-<时间>`、返回 `None`）
  - `Ledger.save(now) -> bool`：`persist` 且距上次写 ≥ `save_every` → `session.saved = now`、写 `people.json` + `current.json`，返回是否写了；第一次调用就写（上次写的时间初始为 `-inf`）
  - `Ledger.close(summary: str, now) -> None`：`persist` 时 `session.end = now`、`ended = "normal"`、`summary`、写 `people.json`、`append_day`、删 `current.json`；重复调用只生效一次
  - `Ledger` 构造多一个参数 `backfilled: float | None = None`（写 `people.json` 时带上）

- [ ] **Step 1: 写失败的测试** `tests/test_inner_store.py`（`tmp_path`）

```python
def test_people_roundtrip_and_atomic(tmp_path):
    st = InnerStore(tmp_path)
    st.write_people({"小明": Card(first_met=T0, days=["2026-09-30"])}, backfilled=T0)
    cards, backfilled, ok = st.load_people()
    assert ok and backfilled == T0 and cards["小明"].days == ["2026-09-30"]
    assert not list(tmp_path.glob("*.tmp"))

def test_bad_people_renamed(tmp_path):
    (tmp_path / "people.json").write_text("{坏", encoding="utf-8")
    cards, backfilled, ok = InnerStore(tmp_path).load_people()
    assert (cards, backfilled, ok) == ({}, None, False)
    assert not (tmp_path / "people.json").exists() and list(tmp_path.glob("people.json.bad-*"))

def test_days_skip_bad_lines(tmp_path):
    st = InnerStore(tmp_path)
    st.append_day(Session(start=T0, end=T0 + 60, ended="normal"))
    with (tmp_path / "days.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("不是 json\n")
    st.append_day(Session(start=T0 + 100, end=T0 + 200, ended="normal"))
    assert [s.start for s in st.days()] == [T0, T0 + 100]

def test_take_current_once(tmp_path):                       # Review Focus 4
    st = InnerStore(tmp_path)
    st.write_current(Session(start=T0, saved=T0 + 60))
    assert st.peek_current().saved == T0 + 60 and (tmp_path / "current.json").exists()
    assert st.take_current().saved == T0 + 60
    assert st.take_current() is None and st.peek_current() is None

def test_ledger_save_throttled_and_close(tmp_path):
    st = InnerStore(tmp_path)
    led = Ledger(CFG, lambda: FRIENDS, T0, store=st, persist=True)
    led.present(["小明"], T0)
    assert led.save(T0) is True and (tmp_path / "current.json").exists()
    assert led.save(T0 + 30) is False                        # save_every = 60
    led.said(T0 + 40)
    led.close("一起看了日落", T0 + 100)
    led.close("又一次", T0 + 200)                            # 只生效一次
    (day,) = st.days()
    assert (day.end, day.ended, day.summary, day.said, day.friends) == (T0 + 100, "normal", "一起看了日落", 1, ["小明"])
    assert not (tmp_path / "current.json").exists() and st.load_people()[0]["小明"].visits == 1

def test_ledger_not_persisting_writes_nothing(tmp_path):
    led = Ledger(CFG, lambda: FRIENDS, T0, store=InnerStore(tmp_path), persist=False)
    led.present(["小明"], T0); led.save(T0); led.close("x", T0 + 1)
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_inner_store.py -q`
Expected: FAIL（`ModuleNotFoundError: skydango.inner.store`）

- [ ] **Step 3: 实现** `inner/store.py`（目录不存在时写之前 `mkdir(parents=True, exist_ok=True)`，读的时候不建目录；`days.jsonl` 追加用 `open("a")`，每行 `json.dumps(..., ensure_ascii=False)`）和 `Ledger.save` / `close`

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_inner_store.py tests/test_inner_ledger.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/inner tests/test_inner_store.py
git commit -m "feat(inner): memory/inner/ 读写：people.json、days.jsonl、current.json，原子写、坏文件改名"
```

---

### Task 3: 回填 + `open_ledger`

**Files:**
- Create: `src/skydango/inner/backfill.py`
- Modify: `src/skydango/inner/__init__.py`（`open_ledger`）
- Test: `tests/test_inner_backfill.py`

**Interfaces:**
- Consumes: `chat.memory.Turn`（`t`、`user`、`reply`；`SKIP`）、Task 1 的 `Card` / `Session` / `match_friend` / `day_of`、Task 2 的 `InnerStore`
- Produces:
  - `backfill(turns: list[Turn], friends: Sequence[str], session_gap: float) -> tuple[dict[str, Card], list[Session]]`
  - `open_ledger(cfg: InnerConfig, directory: str | Path, friends: Callable[[], list[str]], turns: Callable[[], list[Turn]], persist: bool, now: float) -> Ledger`

- [ ] **Step 1: 写失败的测试** `tests/test_inner_backfill.py`

history 的 `user` 格式同 `responder.format_incoming`：`"新的聊天消息：\n小明：「在吗」\n「看不出是谁」"`；`reply` 是团子说的或 `SKIP`。

```python
def turn(t, *lines, reply="好"):
    return Turn(t, "新的聊天消息：\n" + "\n".join(lines), reply)

TURNS = [
    turn(D1 + 0, "小明：「在吗」", "路人：「hi」"),
    turn(D1 + 600, "小明：「今天跑图吗」"),
    turn(D1 + 4 * 3600, "阿花：「晚上好」", reply=SKIP),     # 隔 4 小时：第二次上线
    turn(D2 + 0, "小明：「明天要考试」"),                    # 第二天
]

def test_backfill_cards():
    cards, sessions = backfill(TURNS, FRIENDS, 7200)
    m = cards["小明"]
    assert (m.first_met, m.lines, m.last_seen) == (D1, 3, D2)
    assert m.last_line == {"t": D2, "text": "明天要考试"}
    assert m.days == [day_of(D1), day_of(D2)] and m.minutes == 0 and m.to_me == 0
    assert m.visits == 2                                      # 出现在 2 次回填上线里
    assert "路人" not in cards

def test_backfill_sessions():
    _, sessions = backfill(TURNS, FRIENDS, 7200)
    assert [(s.start, s.end, s.ended, s.friends, s.heard, s.said) for s in sessions] == [
        (D1, D1 + 600, "backfill", ["小明"], 3, 2),
        (D1 + 4 * 3600, D1 + 4 * 3600, "backfill", ["阿花"], 1, 0),
        (D2, D2, "backfill", ["小明"], 1, 1),
    ]

def test_open_ledger_backfills_once(tmp_path):
    led = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: TURNS, persist=True, now=NOW)
    assert led.card("小明").lines == 3 and len(led.history) == 3
    assert (tmp_path / "people.json").exists() and len(InnerStore(tmp_path).days()) == 3
    led2 = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: TURNS * 2, persist=True, now=NOW + 10)
    assert led2.card("小明").lines == 3                       # 已有 people.json：不再回填

def test_open_ledger_days_exist_only_cards(tmp_path):
    InnerStore(tmp_path).append_day(Session(start=D1 - 86400, end=D1 - 80000, ended="normal"))
    led = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: TURNS, persist=True, now=NOW)
    assert len(led.history) == 1 and led.card("小明") is not None

def test_open_ledger_recovers_crash(tmp_path):
    st = InnerStore(tmp_path)
    st.write_people({}, None)
    st.write_current(Session(start=NOW - 900, saved=NOW - 300, friends=["小明"]))
    led = open_ledger(CFG, tmp_path, lambda: FRIENDS, lambda: [], persist=True, now=NOW)
    assert (led.history[-1].end, led.history[-1].ended) == (NOW - 300, "crash")
    assert not (tmp_path / "current.json").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_inner_backfill.py -q`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现**

- `backfill`：按 `t` 排序；相邻两轮隔 `> session_gap` 分段；每轮 `user` 去掉第一行 `"新的聊天消息："`，每行用 `^(.+?)：「(.*)」$` 拆说话人（拆不出的行算 `heard` 不算人）；`said` = 这段里 `reply` 不是空、不是 `SKIP` 的轮数；好友卡：`first_met` 最早、`last_seen` / `last_line` 最后、`lines`、`days`；`visits` = 出现过这个好友的段数；`today` / `today_*` 留空 / 0
- `open_ledger`（每一步都 try/except，出错就 `log.exception` 并用空的继续）：
  1. `store = InnerStore(directory)`
  2. `crash = store.take_current()` 只在 `persist` 时调（dry-run 用 `store.days()` 之外什么都不动）；拿到了 → `end = saved or start`、`ended = "crash"`、`append_day`
  3. `people_exists()` → `load_people()`；`ok = False` 时**不回填**（Task 5 测）；不存在 → `backfill(turns(), friends(), session_gap)`：`persist` 时 `write_people(cards, now)`，`store.days()` 为空才 `append_days(sessions)`；dry-run 只放进内存
  4. `history = store.days()`（dry-run 且回填了、磁盘上又没有 days 时用回填出来的 sessions；dry-run 下用 `peek_current()` 把残留的那次当 crash 行追加进内存 history，不删文件）
  5. `Ledger(cfg, friends, now, cards, history, store, persist, backfilled)`；日志 `"内心账本：N 个好友、以前上线 M 次（回填了 …）"`

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_inner_backfill.py tests/test_inner_store.py tests/test_inner_ledger.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/inner tests/test_inner_backfill.py
git commit -m "feat(inner): 从 history.jsonl 回填关系卡和上线记录，启动时补上意外结束的那次"
```

---

### Task 4: 「日子」一节 + 大脑提示词

**Files:**
- Create: `src/skydango/inner/days.py`
- Modify: `src/skydango/inner/ledger.py`（`Ledger.days_prompt(now) -> str` 调 `days.days_prompt`）
- Modify: `src/skydango/brain/prompt.py`（`memory_prompt` / `brain_prompt` 多 `days: str = ""` 和 `inner: bool = False`）
- Test: `tests/test_inner_days.py`、`tests/test_brain_prompt.py`

**Interfaces:**
- Consumes: Task 1 的 `Card` / `Session` / `day_of`；`chat.memory.format_date`、`format_gap`
- Produces:
  - `days_prompt(history: list[Session], cards: dict[str, Card], friends: Sequence[str], now: float, long_gap_days: float) -> str`
  - `recent_days(history: list[Session], n: int = 10) -> list[str]`（`memory show` 用）
  - `Ledger.days_prompt(now) -> str`
  - `brain_prompt(..., days: str = "", inner: bool = False)`；`INNER_RULES`

- [ ] **Step 1: 写失败的测试**

`tests/test_inner_days.py`（`NOW` = 2026-09-30 20:00 本地，周三）：

```python
def test_first_time():
    assert days_prompt([], {}, FRIENDS, NOW, 7) == (
        "## 日子\n今天 2026年9月30日（周三）。这是你第一次上线。\n这些是给你心里有数的，别跟人报数字。"
    )

def test_nth_time_crash_and_long_gap():
    history = [
        Session(start=NOW - 5 * 86400, end=NOW - 5 * 86400 + 3600, ended="normal"),
        Session(start=NOW - 4 * 3600, end=NOW - 3 * 3600, ended="crash", friends=["小明", "阿花"], summary="一起看了日落"),
    ]
    cards = {"阿花": Card(first_met=0, last_seen=NOW - 12 * 86400), "小红": Card(first_met=0, last_seen=NOW - 20 * 86400),
             "阿蓝": Card(first_met=0, last_seen=NOW - 30 * 86400), "小绿": Card(first_met=0, last_seen=NOW - 40 * 86400),
             "小明": Card(first_met=0, last_seen=NOW - 3600), "旧名字": Card(first_met=0, last_seen=NOW - 9 * 86400)}
    text = days_prompt(history, cards, ["小明", "阿花", "小红", "阿蓝", "小绿"], NOW, 7)
    assert text.splitlines() == [
        "## 日子",
        "今天 2026年9月30日（周三）。这是你第 3 次上线，今天第 2 次；上次下线是 3 小时前（意外断了）。最近 7 天上线了 2 天。",
        "上次见到了：小明、阿花。上次的经过：一起看了日落",
        "很久没见的好友：阿花（12 天前）、小红（20 天前）、阿蓝（30 天前）",
        "这些是给你心里有数的，别跟人报数字。",
    ]

def test_no_long_gap_line_and_empty_last():
    history = [Session(start=NOW - 7200, end=NOW - 60 * 30, ended="normal")]
    lines = days_prompt(history, {}, FRIENDS, NOW, 7).splitlines()
    assert lines[1].endswith("上次下线是 30 分钟前。最近 7 天上线了 1 天。") and len(lines) == 3

def test_recent_days():
    history = [Session(start=NOW - 3600, end=NOW - 600, ended="crash", friends=["小明"], summary="x" * 60)]
    assert recent_days(history) == ["9月30日 19:00  50 分钟  小明  意外断了  " + "x" * 40 + "…"]
```

- "今天第 N 次"：`history` 里开始日期是今天的行数 + 1；"最近 7 天上线了 N 天"：`start ≥ now − 7×86400` 的不同日期，**加上今天**
- "上次见到了" / "上次的经过"：取 `history[-1]`，两样都空就不写这一行；只有一样就只写那一样
- "很久没见"：只看 `friends` 里的名字，`last_seen` 不是 `None` 且超过 `long_gap_days` 天，按 `last_seen` 从近到远最多 3 个
- `recent_days`：最后 `n` 行、新到旧；`"{月}月{日}日 {HH:MM}  {时长}  {好友或"没见到好友"}  {正常下线|意外断了|回填}  {经过前 40 字，超了加…}"`；时长用 `format_gap`，没有 `end` 写"没有结束时间"

`tests/test_brain_prompt.py` 加：

```python
def test_brain_prompt_inner_adds_days_and_rules():
    text = brain_prompt(ReplyConfig(), None, days="## 日子\n今天……", inner=True)
    assert "## 日子\n今天……" in text and INNER_RULES in text
    assert text.index(INNER_RULES) < text.index("- 记住聊过的内容和对方的名字")

def test_brain_prompt_inner_off_unchanged():
    assert brain_prompt(ReplyConfig(), None) == brain_prompt(ReplyConfig(), None, days="", inner=False)

def test_days_before_recent_turns(tmp_path):   # 有 store 时「日子」在「上次聊到哪」之前
    ...  # 用现有测试造 history 的办法造一轮，断言 text.index("## 日子") < text.index("## 上次聊到哪")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_inner_days.py tests/test_brain_prompt.py -q`
Expected: FAIL（`ImportError: days_prompt`、`brain_prompt() got an unexpected keyword argument 'days'`）

- [ ] **Step 3: 实现** `inner/days.py`、`Ledger.days_prompt`，以及 `brain/prompt.py`：

`INNER_RULES` 原文（插在「说话」一节 `"- 记住聊过的内容和对方的名字"` 那一条前面，`inner=True` 才插）：

```text
- 状态和事件里写了你和每个好友的交情（见过几次、上次什么时候、他上次说了什么）。按交情说话：熟的人随便点，刚认识的客气点；隔了很久见面可以表现出来（“好久不见”“你去哪了”）。
- 别报数字（不说“我们见过 13 次”“你今天待了 40 分钟”）。
- 他上次提过的事可以接着问（“考试怎么样了”），拿不准先 recall。
```

`memory_prompt(reply, store, history_turns, now, days="")`：`days` 非空时放在 `recent` 之前；`store is None` 时也追加在人设之后。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_inner_days.py tests/test_brain_prompt.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/inner src/skydango/brain/prompt.py tests/test_inner_days.py tests/test_brain_prompt.py
git commit -m "feat(inner): 系统提示词的「日子」一节和交情规矩"
```

---

### Task 5: 身体接线

**Files:**
- Modify: `src/skydango/brain/body.py`（`__init__` 加 `ledger=None`；`step`、`_watch_people`、`_watch_comings`、`_heard`、`say`、`status`）
- Test: `tests/test_brain_inner_body.py`（用 `tests/test_brain_body.py` 的 `body()` / `FakeEnv` / `msg()`：`from test_brain_body import body, FakeEnv, msg`，同 `test_brain_reflex_body.py` 的做法）

**Interfaces:**
- Consumes: Task 1~3 的 `Ledger`（`present`、`heard`、`said`、`save`、`status_line`）、`open_ledger`
- Produces: `Body(..., ledger=None)`、`Body.ledger`；`Body._ledger_call(method: str, *args, default=None)`

- [ ] **Step 1: 写失败的测试** `tests/test_brain_inner_body.py`

```python
def make(clock, tmp_path, live=False, **kw):
    env = FakeEnv()
    led = Ledger(Config().inner, lambda: ["懒洋洋大王", "阿花"], WALL0, store=InnerStore(tmp_path), persist=live)
    b, dev, reader, events = body(clock, live=live, env=env, ledger=led, wall=lambda: WALL[0], **kw)
    b.friend_names = lambda: ["懒洋洋大王", "阿花"]
    return b, env, reader, events, led

def test_arrive_carries_note_return_does_not(clock, tmp_path):
    b, env, _, events, _ = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]; b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == ["懒洋洋大王 来到身边（第一次在身边见到）"]
    env.near = []; clock.advance(10); b.step()
    env.near = ["懒洋洋大王"]; clock.advance(10); b.step()
    assert [e.text for e in events.drain() if e.kind in ("arrive", "return")] == ["懒洋洋大王 回来了"]

def test_status_uses_status_line(clock, tmp_path):
    b, env, *_ = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]; b.step()
    assert "身边的好友：懒洋洋大王（今天刚来·今天刚认识）" in b.status()

def test_heard_and_said_counted(clock, tmp_path):
    b, env, reader, _, led = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]
    reader.batches = [[msg("团子在吗")]]; b.step()
    b.say("在")
    c = led.card("懒洋洋大王")
    assert (c.lines, c.to_me, led.session.said) == (1, 1, 1)

def test_present_during_track(clock, tmp_path):     # 跟踪中不发人来人走，但账照记
    b, env, _, _, led = make(clock, tmp_path)
    b.skills.active = SimpleNamespace(quiet_people=True, goal="盯着")
    env.near = ["懒洋洋大王"]; b._watch_people(clock())       # 直接调：不经过 skills.tick
    assert led.card("懒洋洋大王").visits == 1

def test_live_saves_each_loop_throttled(clock, tmp_path):
    b, env, *_ = make(clock, tmp_path, live=True)
    env.near = ["懒洋洋大王"]; b.step()
    assert (tmp_path / "current.json").exists()

def test_dry_run_disk_untouched(clock, tmp_path):     # Review Focus 1
    (tmp_path / "current.json").write_text(json.dumps(Session(start=WALL0 - 99, saved=WALL0 - 9).to_dict()), encoding="utf-8")
    led = open_ledger(Config().inner, tmp_path, lambda: ["懒洋洋大王"], lambda: [Turn(WALL0 - 999, "新的聊天消息：\n懒洋洋大王：「嗨」", "嗨")], persist=False, now=WALL0)
    led.present(["懒洋洋大王"], WALL0); led.save(WALL0); led.close("", WALL0 + 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["current.json"]
    assert led.card("懒洋洋大王").lines == 1 and led.history[-1].ended == "crash"

def test_bad_people_json_no_backfill_keeps_running(clock, tmp_path):   # Review Focus 5
    (tmp_path / "people.json").write_text("{坏", encoding="utf-8")
    led = open_ledger(Config().inner, tmp_path, lambda: ["懒洋洋大王"], lambda: [Turn(WALL0 - 999, "新的聊天消息：\n懒洋洋大王：「嗨」", "嗨")], persist=True, now=WALL0)
    assert led.card("懒洋洋大王") is None
    led.present(["懒洋洋大王"], WALL0)
    assert led.card("懒洋洋大王").visits == 1

def test_ledger_errors_do_not_break_body(clock, tmp_path, caplog):
    b, env, _, events, led = make(clock, tmp_path)
    led.present = lambda *a: 1 / 0
    env.near = ["懒洋洋大王"]; b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == ["懒洋洋大王 来到身边"]
    assert "身边的好友：懒洋洋大王" in b.status()

def test_no_ledger_unchanged(clock):
    b, *_ = body(clock, env=FakeEnv())
    assert b.ledger is None and "身边的好友：没看到" in b.status()
```

（`WALL0` = 固定墙上时间，`WALL = [WALL0]`；`clock` 是 conftest 的 `Clock`，`clock.advance(dt)`；`msg(text)` 的说话人默认是 `懒洋洋大王`。）

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_inner_body.py -q`
Expected: FAIL（`Body.__init__() got an unexpected keyword argument 'ledger'`）

- [ ] **Step 3: 实现**

- `_ledger_call(method, *args, default=None)`：`self.ledger is None` → `default`；调 `getattr(self.ledger, method)(*args)`，异常 `log.exception("内心账本出错（%s）", method)` 返回 `default`
- `_watch_people` 开头（跟踪分支之前）：`self._arrive_notes = self._ledger_call("present", self.env.nearby(now), self.wall(), default={})`
- `_watch_comings` 放 `arrive` 那行：`f"{name} 来到身边{self._arrive_notes.get(name, '')}"`；`return` 不变
- `_heard`：写完 `self.chat` 之后、判断大脑离线之前，每句 `self._ledger_call("heard", m.speaker, m.text, self._addressed(m, now), self.wall())`
- `say`：`self.said.append(full)` 之后 `self._ledger_call("said", self.wall())`
- `step` 末尾：`self._ledger_call("save", self.wall())`
- `status`："身边的好友"：`near` 非空时 `self._ledger_call("status_line", near, self.wall(), default="、".join(near))`，结果为空也退回名字列表

- [ ] **Step 4: 跑测试确认通过（含身体原有测试）**

Run: `python -m pytest tests/test_brain_inner_body.py tests/test_brain_body.py tests/test_brain_reflex_body.py tests/test_brain_bubble.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py tests/test_brain_inner_body.py
git commit -m "feat(body): 身体记内心账本：在场、听到、说话；arrive 带交情、status 的身边好友带交情"
```

---

### Task 6: cli 接线、`farewell` 返回经过、`memory show`、配置模板、管理面板、文档

**Files:**
- Modify: `src/skydango/brain/loop.py`（`farewell() -> str`）
- Modify: `src/skydango/cli.py`（大脑模式建 Ledger、传给 `Body` 和 `brain_prompt`、退出时 `close`；`cmd_memory show`）
- Modify: `src/skydango/inner/__init__.py`（`show_lines(cfg, directory, friends, now) -> list[str]`：只读）
- Modify: `config.example.toml`（`[inner]` 一节，在 `[reflex]` 后面）、`src/skydango/console/settings.py`（`Field("inner.enabled", "内心账本", "给好友记关系卡（见过几次、上次什么时候）和每次上线的日子；关掉就回到老样子", "bool", "brain")`，放在 `reflex.bubble` 后面）
- Modify: `CLAUDE.md`（「记忆」表加 `memory/inner/` 一行；代码结构表加 `src/skydango/inner/`；新一节「内心层（`[inner]`，大脑模式）」写第 1 期做了什么、**还没在真机上跑过**）、spec 状态改成"代码已完成，待真机验证"
- Test: `tests/test_brain_loop.py`、`tests/test_console_settings.py`、`tests/test_inner_show.py`、`tests/test_cli_brain.py`

**Interfaces:**
- Consumes: `open_ledger`、`Ledger.days_prompt` / `close`、`card_line`、`recent_days`、`InnerStore`
- Produces: `Brain.farewell() -> str`（写出的经过，没写出来 `""`）；`show_lines(...) -> list[str]`

- [ ] **Step 1: 写失败的测试**

- `tests/test_brain_loop.py`：`test_farewell_writes_summary_to_inbox` 改成 `assert brain.farewell() == "<假 session 返回的那段文字>"`、正在失败时 `== ""`；`test_farewell_reports_to_trace` 同样改
- `tests/test_console_settings.py`：清单里有 `inner.enabled`（照第 31 行那组键的写法加进去）
- `tests/test_inner_show.py`：

```python
def test_show_lines(tmp_path):
    st = InnerStore(tmp_path)
    st.write_people({"小明": Card(first_met=NOW - 86400, last_seen=NOW - 3600, visits=2, days=["2026-09-29", "2026-09-30"], lines=5, to_me=2)}, None)
    st.append_day(Session(start=NOW - 7200, end=NOW - 3600, ended="normal", friends=["小明"]))
    lines = show_lines(Config().inner, tmp_path, ["小明"], NOW)
    assert lines[0] == "===== 关系卡 ====="
    assert lines[1].startswith("小明（今天刚来·认识 1 天·一起玩过 2 天）；见过 2 次，上次 1 小时前；说过 5 句，跟你说过 2 句")
    assert "===== 最近 10 次上线 =====" in lines
    assert sorted(p.name for p in tmp_path.iterdir()) == ["days.jsonl", "people.json"]   # 只读，没多出文件

def test_show_lines_empty(tmp_path):
    assert show_lines(Config().inner, tmp_path / "none", [], NOW) == ["===== 关系卡 =====", "（空）", "===== 最近 10 次上线 =====", "（空）"]
```

- `tests/test_cli_brain.py`：照这个文件现有造大脑 run 的办法（先读它的 fixture），加一条 live 跑完后 `memory/inner/days.jsonl` 有一行、`summary` 是假大脑写的经过；`[inner] enabled = false` 时 `memory/inner/` 不存在

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_loop.py tests/test_console_settings.py tests/test_inner_show.py tests/test_cli_brain.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

- `farewell`：原来返回 `False` 的地方返回 `""`，成功返回 `text`
- cli 大脑模式（`store` 建好之后）：`cfg.inner.enabled and store is not None` 时 `ledger = open_ledger(cfg.inner, store.dir / "inner", _friend_names(cfg), store.history.all, persist=not cfg.reply.dry_run, now=time.time())`（try/except：失败就 `ledger = None` 并 `log.exception`）；`Body(..., ledger=ledger)`；`brain_prompt(..., days=ledger.days_prompt(time.time()) if ledger else "", inner=ledger is not None)`
- 退出：`summary = brain.farewell()`（原来的条件不变，没跑就是 `""`）；之后 `if ledger: ledger.close(summary, time.time())`（try/except）
- `show_lines`：只用 `InnerStore.load_people()` 的读取部分 —— **坏文件时 `show` 不能改名**：给 `load_people` 加参数 `quarantine: bool = True`，`show_lines` 传 `False`（坏了就显示"people.json 读不了"）；`days()` 本来就只读
- `cmd_memory show` 末尾：`cfg.inner.enabled` 时打印 `show_lines(cfg.inner, store.dir / "inner", _friend_names(cfg)(), time.time())`
- `config.example.toml`：

```toml
[inner]                      # 内心层第 1 期：给好友记关系卡、每次上线记一行（memory/inner/，只在 --live 时写）
enabled = true
visit_gap = 1800             # 离上次在场超过这么多秒再出现，算新的一次见面
session_gap = 7200           # 回填 history.jsonl 时，相邻两轮隔这么多秒算两次上线
long_gap = 7                 # 超过这么多天没见算“好久没见”
save_every = 60              # live 时每隔多少秒存一次
max_step = 5                 # 算在一起待了多久时，单圈最多算几秒
```

- [ ] **Step 4: 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部 PASS（基线 1392 passed + 新增）

- [ ] **Step 5: 提交**

```bash
git add -A src/skydango config.example.toml CLAUDE.md docs/superpowers/specs/2026-09-30-inner-phase1-design.md tests
git commit -m "feat(inner): 大脑模式接上内心账本（启动回填、退出记经过）、memory show、配置和文档"
```
