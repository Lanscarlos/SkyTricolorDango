# 内心层第 2 期（反思：心情、精力、别扭、心愿、日记）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 团子有心情、精力、别扭、心愿和日记：反思（一次性 `claude -p`）定时写心情 / 别扭 / 心愿，下线写日记；精力按规则现算；状态接到主动开口额度、反射、心跳和大脑看到的"心里"。

**Architecture:** `inner/` 新增 `energy.py`（纯计算）、`mind.py`（心情 / 别扭 / 心愿的数据和规矩 + 解析反思结果）、`effects.py`（倍数）、`reflect.py`（什么时候反思、拼材料、后台线程）；`store.py` 加 `mind.json` / `diary.md` 读写。身体每圈算精力、`expire`、给 Reflector 报动静、取回结果套进 `Mind`，并把倍数交给 `occasion` / `Reflexes`；`Brain` 困时心跳慢一档。下线时同步跑一次"最终反思"，写日记和要点，代替 `brain.farewell()`。

**Tech Stack:** Python 3.11+ 标准库；pytest。

**Spec:** `docs/superpowers/specs/2026-09-30-inner-phase2-design.md`（第 1 期：`2026-09-30-inner-phase1-design.md`）

## Global Constraints

- `[inner]` 新键和默认值：`reflect = true`、`reflect_model = "sonnet"`、`reflect_every = 1200`、`reflect_after_quiet = 180`、`reflect_min_lines = 6`、`reflect_timeout = 90`、`rest_gap = 3600`、`grudge_min_days = 3`、`grudge_max = 7200`、`wants_max = 3`、`want_days = 7`、`diary_prompt = 1`
- 心情档位只有 `开心` / `平常` / `低落` / `烦`；精力档位只有 `精神` / `还行` / `有点累` / `困`；心愿 kind 只有 `惦记` / `想做` / `小心思`
- 只在 live 写 `mind.json` / `diary.md` / inbox；dry-run 照样反思、结果只在内存里；坏 `mind.json` 改名 `.bad-<时间>`（dry-run 不改名）
- 接话永远照常：倍数只管**主动**开口（`occasion` 额度）、反射、心跳；主动额度 `max(1, int(基数 × 倍数))`，倍数下限 0.25
- 牵手等互动请求照常接；别扭只冲 `len(card.days) ≥ grudge_min_days` 的好友，最长 `grudge_max`，同时最多 1 条
- `[inner] reflect = false`（或 `enabled = false`）：和第 1 期逐字一样（提示词、status、退出走 `brain.farewell()`）
- 出错只 `log.exception` / WARNING，身体、大脑照常；反思失败沿用上一份
- 中文注释、中文日志

## Review Focus

1. **反思给别扭写了个陌生人或 OCR 错字名**：陌生人丢，错一个字的好友名用 `match_friend` 对上正名（Task 2 测）
2. **模型输出带 ```json 围栏、前后有一句话、或者根本不是 JSON**：前两种能解析，第三种返回 `None` 沿用旧状态（Task 2 测）
3. **下线时后台反思还在跑**：最终反思照跑，后台那次的结果回来后丢掉，不会写两篇日记 / 覆盖最终状态（Task 4 测）
4. **别扭对象和别的好友在同一批里说话**：照常冒输入气泡（只有整批都是别扭对象才不冒）（Task 5 测）
5. **dry-run 下线**：最终反思照跑，但 `mind.json` / `diary.md` / inbox 都不写（Task 7 测）

---

### Task 1: 配置 + 精力

**Files:**
- Modify: `src/skydango/config.py`（`InnerConfig` 追加 Global Constraints 里的 12 个键）
- Create: `src/skydango/inner/energy.py`
- Test: `tests/test_inner_energy.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Energy(level: str, score: int, note: str)`
  - `energy(hour: float, awake_min: float, cheered: bool, busy_min: float) -> Energy`
  - `awake_minutes(now: float, start: float, history: list[Session], rest_gap: float) -> float`：`(now - start)/60`，再加上一次（`history[-1]`，`end` 为空用 `saved or start`）的时长如果 `start - 上次结束 < rest_gap`

- [ ] **Step 1: 写失败的测试**

```python
def test_day_fresh():
    assert energy(15.0, 10, False, 0) == Energy("精神", 100, "精神")

def test_late_and_long():
    e = energy(0.8, 130, False, 0)            # 半夜 60，挂了 2 个多小时 −20
    assert (e.level, e.score) == ("有点累", 40) and e.note == "有点累（半夜了，连着挂了 2 个多小时）"

def test_small_hours_sleepy():
    assert energy(3.0, 200, False, 0).level == "困"   # 40 − 30

def test_cheered_and_busy():
    assert energy(23.0, 0, True, 0).score == 90       # 80 + 10
    assert energy(15.0, 0, False, 60).score == 85     # (60−30)/10 × 5 = 15
    assert energy(15.0, 0, False, 200).score == 85    # 封顶 −15
    assert energy(15.0, 0, True, 0).score == 100      # 夹到 100
    assert "有人陪着聊" in energy(23.0, 0, True, 0).note and "闹了好一阵" in energy(15.0, 0, False, 60).note

def test_awake_minutes_continues_without_sleep():
    prev = [Session(start=T0 - 7200, end=T0 - 1800)]            # 上次挂了 90 分钟，半小时前下线
    assert awake_minutes(T0 + 600, T0, prev, 3600) == 100
    assert awake_minutes(T0 + 600, T0, [Session(start=T0 - 20000, end=T0 - 10000)], 3600) == 10
    assert awake_minutes(T0 + 600, T0, [], 3600) == 10
```

- 时段分：`[9,22)` 100、`[22,24)` 80、`[7,9)` 80、`[0,2)` 60、`[2,7)` 40；原因：`[22,24)`"挺晚了"、`[0,2)`"半夜了"、`[2,7)`"凌晨了"、`[7,9)`"刚起"
- 挂机：`min(40, int(awake_min // 60) × 10)`，`awake_min ≥ 60` 时原因"连着挂了 {int(awake_min // 60)} 个多小时"
- 被逗 +10、原因"有人陪着聊"；闹累 `min(15, int(max(0, busy_min − 30) // 10) × 5)`，>0 时原因"闹了好一阵"
- 档位：≥ 70 精神、≥ 50 还行、≥ 30 有点累、其余困；`note` = 档位 + （原因用"，"连，括号包；没有原因就只写档位）

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_inner_energy.py -q` → Expected: FAIL（`ModuleNotFoundError`）
- [ ] **Step 3: 实现 `InnerConfig` 新键和 `inner/energy.py`**
- [ ] **Step 4: 跑测试确认通过** — Run: `python -m pytest tests/test_inner_energy.py tests/test_config.py -q` → Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 第 2 期配置和精力（现实时间、连续挂了多久、被逗、闹累）"`

---

### Task 2: `Mind`（心情 / 别扭 / 心愿）+ 解析 + 存储

**Files:**
- Create: `src/skydango/inner/mind.py`
- Modify: `src/skydango/inner/store.py`（`load_mind` / `write_mind` / `append_diary` / `last_diaries`）
- Test: `tests/test_inner_mind.py`

**Interfaces:**
- Consumes: `ledger.Card`、`match_friend`、`ago`；`energy.Energy`；`chat.tracker.similar`
- Produces:
  - `MOODS = ("开心", "平常", "低落", "烦")`、`KINDS = ("惦记", "想做", "小心思")`
  - `@dataclass Mood(level="平常", text="", since=0.0)`、`Grudge(who, why, since, until)`、`Want(kind, text, who="", since=0.0, until=0.0)`、`Mind(mood=Mood(), grudge: Grudge | None = None, wants: list[Want], updated: float | None = None)`；`Mind.to_dict()` / `Mind.from_dict(d)`
  - `Mind.apply(result: dict, cards: dict[str, Card], friends: list[str], now: float, cfg: InnerConfig) -> list[str]`（返回丢掉的原因）
  - `Mind.expire(now) -> None`、`grudge_on(name, now) -> bool`、`line(now, energy: Energy | None) -> str`（不带"心里："前缀）、`want_note(name) -> str`
  - `parse_reflection(text: str) -> dict | None`
  - `InnerStore.load_mind(quarantine: bool = True) -> Mind`、`write_mind(mind)`、`append_diary(text: str, now: float)`、`last_diaries(n: int) -> list[str]`

- [ ] **Step 1: 写失败的测试**（`CFG = InnerConfig()`；`CARDS = {"小明": Card(first_met=0, days=["d1","d2","d3"]), "懒洋洋大王": Card(first_met=0, days=["d1","d2","d3"]), "阿花": Card(first_met=0, days=["d1"])}`；`FRIENDS = ["小明", "懒洋洋大王", "阿花"]`；两个字的名字错一个字 `similar` 只有 0.5，错字测试用长名字）

```python
def test_apply_mood_and_since():
    m = Mind()
    m.apply({"mood": {"level": "低落", "text": "有点闷，小明说好来又没来" + "啊" * 30}}, CARDS, FRIENDS, T0, CFG)
    assert (m.mood.level, len(m.mood.text), m.mood.since) == ("低落", 40, T0)
    m.apply({"mood": {"level": "低落", "text": "还是闷"}}, CARDS, FRIENDS, T0 + 60, CFG)
    assert m.mood.since == T0                               # 档位没变：since 不动
    m.apply({"mood": {"level": "狂喜", "text": "x"}}, CARDS, FRIENDS, T0 + 99, CFG)
    assert m.mood.level == "平常"

def test_grudge_rules():                                    # Review Focus 1
    m = Mind()
    assert m.apply({"grudge": {"who": "路人", "why": "x"}}, CARDS, FRIENDS, T0, CFG) and m.grudge is None
    m.apply({"grudge": {"who": "阿花", "why": "x"}}, CARDS, FRIENDS, T0, CFG)
    assert m.grudge is None                                 # 不够熟（1 天）
    m.apply({"grudge": {"who": "懒洋羊大王", "why": "放鸽子"}}, CARDS, FRIENDS, T0, CFG)   # 错字对上正名
    assert (m.grudge.who, m.grudge.until) == ("懒洋洋大王", T0 + 7200)
    m.apply({"grudge": {"who": "小明", "why": "又放鸽子"}}, CARDS, FRIENDS, T0, CFG)       # 新的替换旧的
    assert m.grudge.who == "小明"
    m.apply({"grudge": "keep"}, CARDS, FRIENDS, T0 + 1, CFG)
    assert m.grudge is not None
    m.apply({"grudge": None}, CARDS, FRIENDS, T0 + 2, CFG)
    assert m.grudge is None

def test_wants_rules():
    m = Mind()
    m.apply({"wants_add": [
        {"kind": "惦记", "text": "阿花考试考得怎么样", "who": "阿花"},
        {"kind": "惦记", "text": "路人怎么样", "who": "路人"},       # 惦记非好友：丢
        {"kind": "想做", "text": "想看日落", "who": "阿花"},          # who 清空
        {"kind": "想去", "text": "霞谷"},                           # kind 不认识：丢
        {"kind": "小心思", "text": "想换个发型"},
        {"kind": "小心思", "text": "想换个发型呀"},                  # 和已有的几乎一样：丢
        {"kind": "想做", "text": "想听小明弹琴"},                    # 第 4 条：挤掉最旧的
    ]}, CARDS, FRIENDS, T0, CFG)
    assert [w.text for w in m.wants] == ["想看日落", "想换个发型", "想听小明弹琴"]
    assert m.wants[0].who == "" and m.wants[0].until == T0 + 7 * 86400
    m.apply({"wants_done": ["想看日落了"]}, CARDS, FRIENDS, T0, CFG)
    assert [w.text for w in m.wants] == ["想换个发型", "想听小明弹琴"]

def test_expire():
    m = Mind(grudge=Grudge("小明", "x", T0, T0 + 10), wants=[Want("想做", "a", "", T0, T0 + 5)])
    m.expire(T0 + 11)
    assert m.grudge is None and m.wants == []

def test_line_and_want_note():
    m = Mind(Mood("低落", "有点闷，小明说好来又没来", T0), Grudge("小明", "说好来又没来", T0, T0 + 2400),
             [Want("惦记", "阿花考试考得怎么样", "阿花", T0, T0 + 99999)])
    e = Energy("困", 20, "困（半夜了）")
    assert m.line(T0, e) == "有点闷，小明说好来又没来 · 困（半夜了） · 跟小明闹别扭（说好来又没来，还有 40 分钟消气） · 惦记：阿花考试考得怎么样"
    assert Mind().line(T0, Energy("精神", 100, "精神")) == "平常 · 精神"
    assert m.want_note("阿花") == "。你惦记着：阿花考试考得怎么样" and m.want_note("小明") == ""
    assert m.grudge_on("小明", T0) and not m.grudge_on("小明", T0 + 2400) and not m.grudge_on("阿花", T0)

def test_parse_reflection():                                # Review Focus 2
    assert parse_reflection('```json\n{"mood": {"level": "开心"}}\n```') == {"mood": {"level": "开心"}}
    assert parse_reflection('好的，结果如下：{"grudge": null} 就这样') == {"grudge": None}
    assert parse_reflection("我今天挺开心的") is None
    assert parse_reflection('{"坏": ') is None
    assert parse_reflection('[1, 2]') is None               # 不是对象

def test_mind_roundtrip_and_bad_file(tmp_path):
    st = InnerStore(tmp_path)
    m = Mind(Mood("开心", "x", T0), None, [Want("想做", "a", "", T0, T0 + 1)], T0)
    st.write_mind(m)
    assert st.load_mind() == m
    (tmp_path / "mind.json").write_text("坏", encoding="utf-8")
    assert st.load_mind(quarantine=False) == Mind() and (tmp_path / "mind.json").exists()
    assert st.load_mind() == Mind() and list(tmp_path.glob("mind.json.bad-*"))

def test_diary(tmp_path):
    st = InnerStore(tmp_path)
    st.append_diary("今天和小明看了日落。", T0)
    st.append_diary("晚上又上来挂了会儿。", T0 + 3600)
    st.append_diary("第二天。", T0 + 86400)
    assert st.last_diaries(1) == ["第二天。"]
    assert st.last_diaries(2) == ["今天和小明看了日落。\n\n晚上又上来挂了会儿。", "第二天。"]
    assert (tmp_path / "diary.md").read_text(encoding="utf-8").count("## ") == 2
    assert InnerStore(tmp_path / "none").last_diaries(1) == []
```

- `line`：心情部分 = `mood.text or mood.level`；精力部分 = `energy.note`（`None` 时不写）；别扭"跟{who}闹别扭（{why}，还有 {ago(until − now)}消气）"；心愿按 kind：`惦记：`/`想：`/`小心思：` + text；各部分用 `" · "` 连
- `want_note`：只 `惦记` 且 `who == name`，多条用"；"连
- `parse_reflection`：去掉 ``` 围栏后取第一个 `{` 到最后一个 `}` `json.loads`；不是 dict 返回 `None`
- `diary.md`：每天一节 `## {format_date(now)}`；同一天追加时在这一节末尾空一行接着写；`last_diaries(n)` 返回最后 n 天每天的正文（段落之间空一行，去掉标题），旧到新

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_inner_mind.py -q` → Expected: FAIL（`ModuleNotFoundError`）
- [ ] **Step 3: 实现 `inner/mind.py` 和 `InnerStore` 的四个方法**（`load_mind` 的坏文件处理照 `load_people`）
- [ ] **Step 4: 跑测试确认通过** — Run: `python -m pytest tests/test_inner_mind.py tests/test_inner_store.py -q` → Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 心情、别扭、心愿的数据和规矩，反思结果解析，mind.json / diary.md 读写"`

---

### Task 3: 后果倍数，接进 occasion / Reflexes / Brain

**Files:**
- Create: `src/skydango/inner/effects.py`
- Modify: `src/skydango/brain/occasion.py`（`assess(..., quota_scale: float = 1.0)`）、`src/skydango/brain/reflex.py`（`stir(now, scale=1.0)`、`pick_addressed(now, on_wheel, scale=1.0)`）、`src/skydango/brain/loop.py`（`Brain(..., slow: Callable[[], bool] | None = None)`）
- Test: `tests/test_inner_effects.py`、`tests/test_brain_occasion.py`、`tests/test_brain_reflex.py`、`tests/test_brain_loop.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) Effects(quota: float = 1.0, addressed: float = 1.0, idle: float = 1.0, slow: bool = False)`；`effects(mood_level: str, energy_level: str) -> Effects`；`NEUTRAL = Effects()`

- [ ] **Step 1: 写失败的测试**

```python
def test_effects_table():
    assert effects("平常", "精神") == Effects()
    assert effects("开心", "还行") == Effects(quota=1.5, addressed=1.5)
    assert effects("烦", "精神") == Effects(quota=0.5, addressed=0.0)
    assert effects("低落", "困") == Effects(quota=0.25, idle=0.6, slow=True)   # 0.5 × 0.5
    assert effects("平常", "有点累").quota == 0.75
    assert effects("烦", "困").quota == 0.25                                    # 下限
    assert effects("不认识", "不认识") == Effects()
```

`tests/test_brain_occasion.py` 加：`quota_scale=0.5` 时安静档 `limit = max(1, int(2 × 0.5)) = 1`（主动说过 1 句后 `left == 0`、`blocked` 非空）；`quota_scale=0.25` 仍至少 1；`quota_scale=1.5` 热闹档能说 6 句。
`tests/test_brain_reflex.py` 加：`pick_addressed(now, wheel, scale=0)` 永远不做；`scale=2` 时 `addressed_chance = 0.3` 按 0.6 抽（注入 `Rng` 返回 0.5 → 做）；`stir(now, scale=0.5)` 后 `_idle_due` 落在 `now + idle_min×0.5 … now + idle_max×0.5`。
`tests/test_brain_loop.py` 加：`slow=lambda: True` 时身边有好友的心跳取第二档（90）、闲着一档后取第三档（180）且不越界。

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_inner_effects.py tests/test_brain_occasion.py tests/test_brain_reflex.py tests/test_brain_loop.py -q` → Expected: FAIL
- [ ] **Step 3: 实现**：`effects` 按 spec §5 表格，心情 × 精力相乘、`quota` 下限 0.25；`assess` 只改 `limit` 那一行；`pick_addressed` 概率 `min(1, addressed_chance × scale)`；`stir` 两端乘 `scale`；`heartbeat` 下标 `start + idle + (1 if slow() else 0)`
- [ ] **Step 4: 跑测试确认通过**（同上命令）→ Expected: PASS，原有测试不变
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 心情和精力的倍数，接进主动额度、反射和心跳"`

---

### Task 4: `Reflector`（什么时候反思、材料、后台线程、最终反思）

**Files:**
- Create: `src/skydango/inner/reflect.py`
- Test: `tests/test_inner_reflect.py`

**Interfaces:**
- Consumes: `brain.claude.ClaudeError`（`.limit`）、`mind.parse_reflection`
- Produces:
  - `REFLECT_SYSTEM: str`（spec §2 的要点，要求只输出 JSON，字段 `mood` / `grudge` / `wants_add` / `wants_done` / `diary` / `memos`；`final` 才写 `diary` / `memos`）
  - `materials(now: float, energy_note: str, mind_line: str, chat: list[tuple[float, str, str]], comings: list[str], cards: list[str], notes: list[str], persona: str, final: bool) -> str`
  - `class Reflector(cfg: InnerConfig, llm, clock=time.monotonic, threaded: bool = True)`：
    - `heard(friend: bool, now)`（有人说话；好友的算进 `reflect_min_lines`）、`stirred(now)`（好友来去、团子说话）
    - `due(now) -> bool`、`start(content: str, now) -> None`（`threaded=False` 时同步跑，测试用）
    - `poll() -> dict | None`（后台结果；最终反思开始后永远 `None`）
    - `final(content: str) -> dict | None`（同步，最多 `reflect_timeout`；额度暂停中也跑一次）
    - `running: bool`

- [ ] **Step 1: 写失败的测试**（假 llm：`complete(system, messages)` 记下调用、返回预设文本或抛 `ClaudeError`；`clock` 用 conftest 的 `Clock`）

```python
def test_due_needs_activity_and_interval(clock):
    r = Reflector(CFG, FakeLlm('{"mood": {"level": "开心"}}'), clock, threaded=False)
    clock.advance(1300)
    assert not r.due(clock())                     # 没动静
    r.stirred(clock())
    assert r.due(clock())
    r.start("材料", clock())
    assert r.poll() == {"mood": {"level": "开心"}} and r.poll() is None and not r.due(clock())

def test_due_after_conversation_quiet(clock):
    r = Reflector(CFG, FakeLlm("{}"), clock, threaded=False)
    for _ in range(6):
        r.heard(True, clock()); clock.advance(5)
    assert not r.due(clock())
    clock.advance(180)
    assert r.due(clock())
    r2 = Reflector(CFG, FakeLlm("{}"), clock, threaded=False)
    for _ in range(6):
        r2.heard(False, clock())                  # 陌生人说的不算
    clock.advance(200)
    assert not r2.due(clock())

def test_failure_keeps_old_and_limit_pauses(clock):
    llm = FakeLlm(ClaudeError("额度", limit=True))
    r = Reflector(CFG, llm, clock, threaded=False)
    r.stirred(clock()); clock.advance(1300)
    r.start("材料", clock())
    assert r.poll() is None
    r.stirred(clock()); clock.advance(1300)
    assert not r.due(clock())                     # 600 秒内（limit_retry）不反思……
    clock.advance(10_000); r.stirred(clock())
    assert r.due(clock())

def test_bad_json_is_none(clock):
    r = Reflector(CFG, FakeLlm("今天挺好"), clock, threaded=False)
    r.stirred(clock()); clock.advance(1300); r.start("x", clock())
    assert r.poll() is None

def test_final_discards_background(clock):         # Review Focus 3
    gate = threading.Event()
    llm = FakeLlm('{"mood": {"level": "烦"}}', wait=gate)
    r = Reflector(CFG, llm, clock)
    r.stirred(clock()); clock.advance(1300); r.start("x", clock())
    assert r.running
    llm.reply = '{"mood": {"level": "开心"}, "diary": "今天不错", "memos": ["小明考试过了"]}'
    gate.set()
    assert r.final("最终材料")["diary"] == "今天不错"
    time.sleep(0.2)
    assert r.poll() is None                       # 后台那次回来了也丢掉

def test_materials_mentions_everything():
    text = materials(T0, "有点困（半夜了）", "平常 · 有点困", [(T0, "小明", "在吗"), (T0 + 1, "我", "在")],
                     ["小明 来了"], ["小明（…）"], ["- 小明在准备考试"], "人设……", final=True)
    for s in ("有点困（半夜了）", "小明：在吗", "我：在", "小明 来了", "- 小明在准备考试", "人设……", "日记"):
        assert s in text
    assert "日记" not in materials(T0, "", "", [], [], [], [], "", final=False)
```

- `materials` 里聊天最多 80 行（多了留最新的）；`final` 时末尾写"这是今天下线前的最后一次：再写一段第一人称日记（≤ 200 字）和给以后的自己的要点（≤ 5 条）"
- 最终反思开始时置一个标记；后台线程结果回来时看到标记就丢

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_inner_reflect.py -q` → Expected: FAIL（`ModuleNotFoundError`）
- [ ] **Step 3: 实现 `inner/reflect.py`**：`llm.complete(REFLECT_SYSTEM, [{"role": "user", "content": content}])`；后台线程 `daemon=True, name="reflect"`；`limit_retry` 取 600 秒常量（同 `[brain] limit_retry` 默认）
- [ ] **Step 4: 跑测试确认通过** — Run: `python -m pytest tests/test_inner_reflect.py -q` → Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 反思——什么时候跑、拼材料、后台线程、下线前的最终反思"`

---

### Task 5: 身体接线

**Files:**
- Modify: `src/skydango/brain/body.py`
- Test: `tests/test_brain_mind_body.py`（夹具同 `tests/test_brain_inner_body.py`）

**Interfaces:**
- Consumes: Task 1~4 全部；第 1 期的 `Ledger`
- Produces: `Body(..., mind=None, reflector=None)`；`Body.effects() -> Effects`；`Body.energy_now() -> Energy | None`；`Body.reflect_materials(final: bool) -> str`；`Body.apply_reflection(result: dict) -> None`

- [ ] **Step 1: 写失败的测试**（`make()` 同第 1 期测试，外加 `mind=Mind()`、`reflector=Reflector(CFG, FakeLlm(reflect_reply), clock, threaded=False)`；墙上时间固定在**本地下午 3 点**（`time.mktime(...)`），免得精力把额度也压下去；懒洋洋大王的关系卡 `days` 够 3 天；`FakeLlm` 放 `tests/conftest.py` 给 Task 4 / 5 共用）

```python
def test_status_has_heart_line(clock, tmp_path):
    b, env, *_ = make(clock, tmp_path)
    assert "心里：平常 · " in b.status()

def test_arrive_carries_want_note(clock, tmp_path):
    b, env, _, events, _ = make(clock, tmp_path)
    b.mind.wants = [Want("惦记", "考试考得怎么样", "懒洋洋大王", 0, 9e12)]
    env.near = ["懒洋洋大王"]; b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == ["懒洋洋大王 来到身边（第一次在身边见到）。你惦记着：考试考得怎么样"]

def test_mood_scales_proactive_quota(clock, tmp_path):
    b, env, *_ = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]
    b.mind.mood = Mood("低落", "闷", 0)
    assert b.occasion().left == 1                  # 安静 2 × 0.5

def test_grudge_target_no_bubble_others_yes(clock, tmp_path):   # Review Focus 4
    b, env, reader, *_ = make(clock, tmp_path, live=True)
    b.mind.grudge = Grudge("懒洋洋大王", "放鸽子", 0, 9e12)
    env.near = ["懒洋洋大王"]
    reader.batches = [[msg("团子在吗")]]; b.step()
    assert b._bubble_at is None
    reader.batches = [[msg("团子在吗"), msg("团子！", speaker="阿花")]]; clock.advance(40); b.step()
    assert b._bubble_at is not None

def test_reflection_result_applied_on_body_thread(clock, tmp_path):
    b, env, reader, *_ = make(clock, tmp_path, reflect_reply='{"mood": {"level": "开心", "text": "小明来了"}}')
    reader.batches = [[msg("嗨")]]; b.step()
    clock.advance(1300); b.step(); b.step()
    assert b.mind.mood.level == "开心"

def test_show_info_has_mood_and_energy(clock, tmp_path):
    ...  # 用现有 viewer 夹具（test_brain_body 里 FakeViewer 的做法），断言 info 里有 "心情" 和 "精力"

def test_errors_fall_back_to_neutral(clock, tmp_path):
    b, *_ = make(clock, tmp_path)
    b.mind.line = lambda *a: 1 / 0
    assert "心里：" not in b.status() and b.effects() == Effects()   # 拼不出不写这一项
    b.mind = None
    assert b.effects() == Effects()

def test_no_mind_unchanged(clock):
    b, *_ = body(clock, env=FakeEnv())
    assert "心里" not in b.status() and b.effects() == Effects()
```

- 精力：`energy(本地小时, awake_minutes(wall, ledger.session.start, ledger.history, cfg.inner.rest_gap), wall − _cheered_at < 600, 最近 60 分钟热闹分钟数)`；每圈算一次缓存在 `self._energy`
- `_cheered_at`：`_heard` 里 `to_me` 为真时记墙上时间；热闹分钟：每圈 `occasion().level == "busy"` 时把这一圈的时长（≤ `max_step`）记进 `deque[(wall, 秒)]`，只留 3600 秒内
- 反思材料攒在 `self._reflect_chat`（听到 + 说出的每一句，`(wall, 说话人, 内容)`，"我" 表示团子）和 `self._reflect_comings`（arrive / leave 的文字），`start` 时清空；`cards` 用在场 + 说过话的好友的 `card_line`，`notes` 取 notes.md / inbox.md 里含这些名字的行（最多 10 行），人设 `store.profile() or cfg.reply.persona`（`store` 为空时只用 persona）
- 每圈（`step` 末尾，`_ledger_call("save")` 之前）：`mind.expire`、算精力、`poll()` 有结果就 `apply_reflection`、`reflector.due()` 就 `start(reflect_materials(False))`；`apply_reflection` 用 `ledger.cards` 和 `friend_names()` 调 `mind.apply`，live 时 `store.write_mind`（经 `ledger.store`）
- `occasion()` 传 `quota_scale=self.effects().quota`；`_on_heard` 里：`pick_addressed(..., scale=effects.addressed)`、整批说话人都是别扭对象（`mind.grudge_on`）时 return（不做小动作、不开框）；`reflexes.stir(now)` 的四处都传 `scale=effects.idle`
- `status`：`场合` 后面加 `"心里：" + mind.line(wall, energy)`；`_show` 的 `info` 加 `心情`（`mood.text or level`）和 `精力`（`energy.note`）；arrive 文字再接 `mind.want_note(name)`
- 全部经 `_ledger_call` 同样的 try 包法（可以新增 `_inner_call`）；`mind is None` 时 `effects()` 返回 `NEUTRAL`

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_brain_mind_body.py -q` → Expected: FAIL（`unexpected keyword argument 'mind'`）
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过（含身体原有测试）** — Run: `python -m pytest tests/test_brain_mind_body.py tests/test_brain_inner_body.py tests/test_brain_body.py tests/test_brain_reflex_body.py tests/test_brain_bubble.py -q` → Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(body): 身体接上心情、精力和反思：主动额度、反射、别扭不冒气泡、status 的心里、arrive 带惦记"`

---

### Task 6: 提示词 + 「日子」带日记

**Files:**
- Modify: `src/skydango/brain/prompt.py`（`brain_prompt(..., mind: bool = False)`，`MIND_RULES`）、`src/skydango/inner/days.py`（`days_prompt(..., diaries: list[str] | None = None)`）、`src/skydango/inner/ledger.py`（`Ledger.days_prompt(now, diaries=None)`）
- Test: `tests/test_brain_prompt.py`、`tests/test_inner_days.py`

- [ ] **Step 1: 写失败的测试**

```python
def test_mind_rules_after_inner_rules():
    from skydango.brain.prompt import INNER_RULES, MIND_RULES
    text = brain_prompt(ReplyConfig(), None, inner=True, mind=True)
    assert text.index(INNER_RULES) < text.index(MIND_RULES) < text.index("- 记住聊过的内容和对方的名字")
    assert "别扭立刻作废" in MIND_RULES and "牵手" in MIND_RULES
    assert brain_prompt(ReplyConfig(), None, inner=True) == brain_prompt(ReplyConfig(), None, inner=True, mind=False)

def test_days_prompt_with_diary():
    history = [Session(start=NOW - 7200, end=NOW - 3600, ended="normal", friends=["小明"], summary="要点")]
    lines = days_prompt(history, {}, FRIENDS, NOW, 7, diaries=["今天和小明看了日落。" + "啊" * 300]).splitlines()
    assert lines[2] == "上次见到了：小明。"
    assert lines[3].startswith("上次的日记：今天和小明看了日落。") and len(lines[3]) == len("上次的日记：") + 200
    assert days_prompt(history, {}, FRIENDS, NOW, 7) == days_prompt(history, {}, FRIENDS, NOW, 7, diaries=[])
```

`MIND_RULES` 原文（spec §6 的四条）：

```text
- 状态里的“心里”是你现在的心情、精力、别扭和惦记的事。照着它说话：开心可以皮一点，低落、困了话短、兴致不高，困了可以说困；别每句都提自己的心情。
- 跟谁闹别扭：可以嘲两句、慢点接、装作不在乎；他认真问、说难过、说不舒服、真的生气了，别扭立刻作废，照常关心他。不骂人、不翻旧账、不拿牵手拥抱这些互动冷落人。
- 惦记的事：见到那个人、场合合适时自然地问一句，别一见面就查户口。
- 想做的事、小心思：场合合适时提，别硬塞；做不到的（坐下、弹琴、自己去远处）只能说想，请别人配合。
```

- 「日子」：`diaries` 非空时，"上次的经过"不写，另起一行"上次的日记：{最后一篇截 200 字}"（`diary_prompt > 1` 时多篇用"上上次…"不做，只取最后 `diary_prompt` 篇用空格连起来再截 200 字）

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_brain_prompt.py tests/test_inner_days.py -q` → Expected: FAIL
- [ ] **Step 3: 实现**（`MIND_RULES` 插在 `REMEMBER_ANCHOR` 前、`INNER_RULES` 之后：先插 `MIND_RULES` 再插 `INNER_RULES` 会反过来，注意顺序）
- [ ] **Step 4: 跑测试确认通过**（同上命令）→ Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 提示词加心情 / 别扭 / 惦记的规矩，「日子」带上一篇日记"`

---

### Task 7: cli 接线、最终反思代替 farewell、配置模板、管理面板、文档

**Files:**
- Modify: `src/skydango/inner/__init__.py`（`finish_reflection`）、`src/skydango/cli.py`、`config.example.toml`、`src/skydango/console/settings.py`（`Field("inner.reflect", "反思", "定时想想刚才发生的事：心情、精力、闹别扭、惦记的事，下线写日记；关掉就只记账", "bool", "brain")`，放在 `inner.enabled` 后面）、`CLAUDE.md`（「内心层」一节加第 2 期、「记忆」表 `inner/` 那行加 `mind.json` / `diary.md`）、spec 状态
- Test: `tests/test_inner_finish.py`、`tests/test_cli_brain.py`、`tests/test_console_settings.py`

**Interfaces:**
- Produces: `finish_reflection(result: dict | None, mind: Mind, store: InnerStore, memory, cards, friends, persist: bool, now: float, cfg) -> str`：`result` 为空返回 `""`；否则 `mind.apply`、`persist` 时 `write_mind` + `append_diary(diary)`（有才写）+ `memory.add_memos([f"{format_date(now)} 的要点：{m}" for m in memos])`（`memory` 不为空时）；返回 `"；".join(memos)`（没有要点就用日记前 100 字）

- [ ] **Step 1: 写失败的测试**

```python
def test_finish_writes_diary_memos_and_mind(tmp_path):
    st, mem = InnerStore(tmp_path / "inner"), MemoryStore(tmp_path)
    m = Mind()
    s = finish_reflection({"mood": {"level": "开心", "text": "好"}, "diary": "今天不错。", "memos": ["小明考试过了", "约了周六"]},
                          m, st, mem, {}, [], True, T0, InnerConfig())
    assert s == "小明考试过了；约了周六" and st.last_diaries(1) == ["今天不错。"]
    assert "小明考试过了" in mem.inbox() and st.load_mind().mood.level == "开心"

def test_finish_dry_run_writes_nothing(tmp_path):            # Review Focus 5
    st, mem = InnerStore(tmp_path / "inner"), MemoryStore(tmp_path)
    s = finish_reflection({"diary": "今天不错。", "memos": []}, Mind(), st, mem, {}, [], False, T0, InnerConfig())
    assert s == "今天不错。" and not (tmp_path / "inner").exists() and mem.inbox() == ""

def test_finish_none():
    assert finish_reflection(None, Mind(), None, None, {}, [], True, T0, InnerConfig()) == ""
```

`tests/test_cli_brain.py`（照第 1 期那几条的写法，假 claude 把收到的内容以"收到："前缀原样回）：
- live 跑 3 秒：反思的最终那次**被调用了**（假 claude 日志里有一条 `--system-prompt` 是 `REFLECT_SYSTEM`），大脑**没收到** `SUMMARY_REQUEST`，`days.jsonl` 那一行 `ended == "normal"`（假模型回的不是 JSON → summary 为空也照样 close）
- `inner.reflect = false`：大脑收到 `SUMMARY_REQUEST`（第 1 期原样）、`prompt.md` 里没有 `MIND_RULES`
- dry-run：`memory/inner/` 不存在、`prompt.md` 有 `MIND_RULES`
`tests/test_console_settings.py`：清单里 `inner.enabled` 后面是 `inner.reflect`

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_inner_finish.py tests/test_cli_brain.py tests/test_console_settings.py -q` → Expected: FAIL
- [ ] **Step 3: 实现**
  - cli：`ledger` 不为空且 `cfg.inner.reflect` 时 `mind = ledger.store.load_mind(quarantine=not dry_run)`、`reflector = Reflector(cfg.inner, ClaudeLlm(base, claude_vars, cfg.inner.reflect_model, run.path / "brain" / "reflect", cfg.inner.reflect_timeout))`，交给 `Body(..., mind=mind, reflector=reflector)`；`Brain(..., slow=lambda: body.effects().slow)`；`brain_prompt(..., days=ledger.days_prompt(now, diaries=ledger.store.last_diaries(cfg.inner.diary_prompt)), mind=reflector is not None)`（都包 try，同第 1 期）
  - 退出：`checkpoint` 之后，`reflector` 不为空 → `summary = finish_reflection(reflector.final(body.reflect_materials(True)), …, persist=not dry_run, …)`，**不调** `brain.farewell()`；否则照旧
  - `config.example.toml` 的 `[inner]` 追加 12 个键（带注释，同第 1 期风格，注明"数字都是估的"）
- [ ] **Step 4: 跑全部测试** — Run: `python -m pytest -q` → Expected: 全部 PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 大脑模式接上反思（下线写日记代替经过）、配置、管理面板和文档"`
