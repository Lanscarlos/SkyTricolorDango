# 对每个人的感觉 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 关系卡上加「印象 + 亲近度 + 走向」，下线那次的反思在旧的基础上改写，好友来时（arrive）和 status 里带给大脑。

**Architecture:** 新的纯函数模块 `inner/feelings.py` 管数据、把关、拼文字；`Card` 多一个 `feeling` dict。最终反思的系统提示词临时接上 `FEELINGS_SYSTEM`、材料多一节，结果在 `finish_reflection` 里经 `apply_feelings` 写进卡，`ledger.close` 照旧写 `people.json`。身体拼 arrive / status 时调 feelings 的文字函数。

**Tech Stack:** Python 3.13，pytest；不加依赖。

**Spec:** `docs/superpowers/specs/2026-11-01-friend-feelings-design.md`

## Global Constraints

- 亲近度三档 `("刚认识", "熟了", "很铁")`（有序）；走向写盘三档 `("在变近", "平稳", "有点生疏")`；显示时现算的一档 `好久没见了`。
- `FEEL_MAX = 40`（印象字数，超了整条不收、不截）；`FEEL_PAST = 3`（旧版本最多几版）。
- 配置 `[inner]`：`feelings: bool = True`（要 `reflect` 开着才生效）、`feel_close_days: int = 7`（很铁要一起玩过几天）、`feel_stale_days: int = 14`（多少天没在身边算好久没见）；熟了的门槛沿用 `grudge_min_days`（3）。
- 只有最终反思（下线那次）改感觉；在线反思的系统提示词、材料逐字不变。
- `feelings = false` 或 `reflect = false`：反思提示词和材料、arrive、status、提示词「说话」一节**逐字照旧**；卡上已有的 `feeling` 留着。
- 只在 live 写盘（`ledger.persist`），dry-run 只改内存。
- 中文注释、照周围代码的密度和写法；用户可见文字全角标点。

## Review Focus

1. 旧 `people.json` 里的卡没有 `feeling` 键 → 载入不报错、`feeling == {}`，arrive / status 照旧（Task 1 测 `Card.from_dict` 缺字段）。
2. 反思 JSON 的 `feelings` 是字符串 / dict / 项缺字段 / `who` 是 null → 整项跳过、不炸、心情日记照常写（Task 2 测）。
3. 好友名 OCR 错一个字（「懒洋洋大玉」）→ 归到最像的好友；完全对不上的丢（Task 1 测）。
4. 同一个人在 `feelings` 里出现两次 → 只收第一条（Task 1 测）。
5. 印象和旧的几乎一样但档位变了 → 只改档位、推一版 past（Task 1 测）。

---

## File Structure

| 文件 | 改什么 |
|---|---|
| `src/skydango/inner/feelings.py`（新） | 常量、`apply_feelings`、`distressed`、`feeling_note`（arrive）、`feeling_bit`（status）、`feelings_lines`（反思材料）、`introspect_text` |
| `src/skydango/inner/ledger.py` | `Card.feeling`；`describe(…, feel=False)` 带档位 |
| `src/skydango/config.py` | `InnerConfig` 三个字段 |
| `src/skydango/inner/reflect.py` | `FEELINGS_SYSTEM`；`materials(…, feelings=None)`；`Reflector.final(content, extra="")` |
| `src/skydango/inner/__init__.py` | `finish_reflection(…, feelings=None)` 套感觉 |
| `src/skydango/inner/log.py` | `diff` 多「对谁的感觉」 |
| `src/skydango/brain/body.py` | `reflect_materials` 带感觉一节；arrive 带 `feeling_note`；status 带档位；`introspect("关系")` |
| `src/skydango/cli.py` | `_final_reflection` 传 `extra` / `feelings`；`brain_prompt(feelings=)` |
| `src/skydango/brain/prompt.py` | `FEEL_RULE` 接进 `INNER_RULES` 后面 |
| `src/skydango/brain/tools.py` | `INTROSPECT_TOPICS` 加「关系」、`INTROSPECT_DESCRIPTION` 写上 |
| `src/skydango/console/inner_view.py` `static/inner.js` `settings.py` | 内心页显示、设置项 |
| `CLAUDE.md` | 「内心层」加一节 |

---

### Task 1: feelings 模块和关系卡字段

**Files:**
- Create: `src/skydango/inner/feelings.py`
- Modify: `src/skydango/inner/ledger.py`（`Card`）、`src/skydango/config.py`（`InnerConfig`）
- Test: `tests/test_inner_feelings.py`

**Interfaces:**
- Produces:
  - `CLOSE_LEVELS: tuple[str, ...]`、`TRENDS: tuple[str, ...]`、`STALE = "好久没见了"`、`FEEL_MAX = 40`、`FEEL_PAST = 3`、`CLINGY: tuple[str, ...]`、`GRUMBLE: tuple[str, ...]`
  - `Card.feeling: dict`（默认 `{}`）
  - `apply_feelings(cards: dict[str, Card], proposed, *, met: set[str], soft: set[str], today: str, friends: Sequence[str], cfg: InnerConfig) -> tuple[list[str], list[str]]`：返回（改了什么的说明「懒洋洋大王：熟了→很铁」，丢掉的原因），原地改 `cards[name].feeling`
  - `distressed(chat: Iterable[tuple[float, str, str]], friends: Sequence[str]) -> set[str]`：聊天里说过 `mind.DISTRESS` 词的好友（`match_friend` 归名）
  - `trend_shown(card: Card, now: float, cfg: InnerConfig) -> str`：`feeling["trend"]`，或离 `last_seen` ≥ `feel_stale_days` 天时 `STALE`
  - `feeling_note(card: Card, now: float, cfg: InnerConfig, soft: bool) -> str`：「。你对他：<text>（<close>，最近<trend>）」；soft 时「。你对他：（<close>）」；没印象 `""`
  - `feeling_bit(card: Card, now: float, cfg: InnerConfig) -> str`：status 括号里的一段：档位，走向是「有点生疏」/ `STALE` 时「熟了·有点生疏」；没印象 `""`

- [ ] **Step 1: 写失败的测试**（`tests/test_inner_feelings.py`；卡用 `Card(first_met=…, days=[…], last_seen=…)` 造，`cfg = InnerConfig()`）

```python
def test_old_card_without_feeling_loads():
    assert Card.from_dict({"first_met": 0.0}).feeling == {}

def test_apply_new_feeling():
    cards = {"小明": card(days=5)}
    changes, dropped = apply_feelings(cards, [{"who": "小明", "text": "嘴贫但靠谱", "close": "熟了", "trend": "在变近"}],
                                      met={"小明"}, soft=set(), today="2026-11-01", friends=["小明"], cfg=cfg)
    assert cards["小明"].feeling == {"text": "嘴贫但靠谱", "close": "熟了", "trend": "在变近", "since": "2026-11-01", "past": []}
    assert changes and dropped == []

def test_not_met_or_unknown_dropped(): ...          # who 不在 met → 丢，原因含「这次没见到」；对不上名单 → 丢
def test_ocr_typo_matches_friend(): ...             # who="小眀" 归到「小明」
def test_duplicate_who_keeps_first(): ...
def test_close_moves_one_step(): ...                # 刚认识 → 给很铁 → 存熟了（days=10）
def test_close_needs_days(): ...                    # days=1 给熟了 → 刚认识；days=5 给很铁且旧档熟了 → 熟了（<7 天），dropped 有原因
def test_bad_level_dropped(): ...                   # close="好朋友" 或 trend="变远" → 整条丢
def test_too_long_dropped_not_cut(): ...            # 41 字 → 不收，feeling 不变
def test_sensitive_clingy_distress_dropped(): ...   # 含「胖」/「离不开」/「难过」 → 不收
def test_soft_no_cooling_no_grumble(): ...          # soft={"小明"}：trend 有点生疏 → 平稳；text 含「放鸽子」→ 只收档位走向，text 留旧的
def test_same_text_new_close_pushes_past(): ...     # 文字 similar≥0.9、close 变了 → past 多一版
def test_trend_only_no_past(): ...                  # 只改走向 → past 不变
def test_past_max_three(): ...                      # 连改 5 次 → len(past)==3，最旧的被挤掉
def test_non_list_or_bad_items_ignored(): ...       # proposed="x" / [{"who": None}] / [{"text": "a"}] → ([], [...]) 不抛
def test_distressed(): ...                          # [(0,"小明","我今天好难过"),(1,"我","难过")] → {"小明"}
def test_notes_and_bits():
    c = card(days=5, feeling={"text": "嘴贫但靠谱", "close": "熟了", "trend": "平稳", "since": "2026-11-01", "past": []})
    assert feeling_note(c, NOW, cfg, soft=False) == "。你对他：嘴贫但靠谱（熟了，最近平稳）"
    assert feeling_note(c, NOW, cfg, soft=True) == "。你对他：（熟了）"
    assert feeling_bit(c, NOW, cfg) == "熟了"
def test_stale_shown(): ...                         # last_seen 15 天前 → trend_shown == "好久没见了"，feeling_bit == "熟了·好久没见了"
```

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_inner_feelings.py -q` → ImportError / FAIL

- [ ] **Step 3: 实现**
  - `Card` 末尾加 `feeling: dict = field(default_factory=dict)`；`InnerConfig` 加三个字段（注释同 Global Constraints）
  - `feelings.py`：敏感词用 `persona.SENSITIVE`，真人用 `chat.responder.claims_human`，相似用 `chat.tracker.similar`，归名用 `ledger.match_friend`；`CLINGY = ("离不开", "只喜欢", "只要他", "只要她", "吃醋", "嫉妒", "独占", "最爱")`、`GRUMBLE = ("烦", "讨厌", "放鸽子", "迟到", "没良心", "气死", "不理我")`；规矩按 spec §3 的顺序 1→5，每条丢弃写一句原因（「小明：一起玩过 1 天，熟了 压回 刚认识」）

- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_inner_feelings.py tests/test_inner_ledger.py -q` → 全过

- [ ] **Step 5: 提交** `feat(inner): 对每个人的感觉——数据和把关`

---

### Task 2: 下线反思写感觉

**Files:**
- Modify: `src/skydango/inner/reflect.py`、`src/skydango/inner/__init__.py`、`src/skydango/inner/log.py`、`src/skydango/brain/body.py`（`reflect_materials`）、`src/skydango/cli.py`（`_final_reflection`）
- Test: `tests/test_inner_reflect.py`、`tests/test_inner_finish.py`、`tests/test_inner_log.py`

**Interfaces:**
- Consumes: Task 1 的 `apply_feelings`、`distressed`
- Produces:
  - `FEELINGS_SYSTEM: str`（spec §2 的提示词要点，最后一句写 JSON 多一项 `"feelings": [{"who": "...", "text": "...", "close": "...", "trend": "..."}]`）
  - `feelings_lines(cards: dict[str, Card], names: Sequence[str], now: float, cfg) -> list[str]`（在 `feelings.py`）：每人一行「小明：嘴贫但靠谱（熟了，平稳，11-01 写的；一起玩过 5 天）」或「小明：还没有印象（一起玩过 1 天）」
  - `materials(…, feelings: list[str] | None = None)`：非 None 时在「相关的好友」后面加 `_block("你对他们的感觉（下线时可以改，见 feelings）：", feelings)`
  - `Reflector.final(content: str, extra: str = "") -> dict | None`：`extra` 非空时这一次的 system = `self.system + "\n\n" + extra`
  - `finish_reflection(…, feelings: dict | None = None)`：`feelings = {"met": set, "soft": set, "today": str}`；非 None 时调 `apply_feelings(cards, result.get("feelings"), …)`，丢弃原因并进 `dropped`，改了什么交给 `mind_log`
  - `log.diff(…, feelings: list[str] = ())`：把 `apply_feelings` 返回的改动说明原样追加在最后

- [ ] **Step 1: 写失败的测试**
  - `test_materials_feelings_block`：`materials(..., feelings=["小明：还没有印象（一起玩过 1 天）"])` 含「你对他们的感觉」和这一行；`feelings=None` 时输出和不传参数**完全相等**
  - `test_final_extra_system`：假 llm 记下 system；`final(c, extra="X")` 的 system 以 `REFLECT_SYSTEM` 开头、以 `X` 结尾；之后的 `_ask`（在线反思）system 仍 == `REFLECT_SYSTEM`
  - `test_finish_applies_feelings`：result 带合法 `feelings`，`feelings={"met": {"小明"}, "soft": set(), "today": "2026-11-01"}` → `cards["小明"].feeling["text"]` 有值；`feelings=None` → 卡不变
  - `test_finish_bad_feelings_still_writes_diary`：`result["feelings"] = "坏的"` → 不抛、日记照写（Review Focus 2）
  - `test_diff_lists_feelings`：`diff(..., feelings=["小明：熟了→很铁"])` 最后一项是它

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_inner_reflect.py tests/test_inner_finish.py tests/test_inner_log.py -q`

- [ ] **Step 3: 实现**
  - `body.reflect_materials(final)`：`final and self._feelings_on()` 时（`_feelings_on` = `cfg.inner.feelings and cfg.inner.reflect and self.ledger is not None`）传 `feelings=feelings_lines(self.ledger.cards, names, wall, self.cfg.inner)`，否则 `None`
  - `body` 加 `feeling_context(self) -> dict | None`：开着时 `{"met": 这次上线 ledger 当前 Session.friends ∪ names（同 reflect_materials 的算法）, "soft": self.soft_names_this_session() | distressed(self._session_chat, friends), "today": day_of(self.wall())}`，关着 `None`
  - `cli._final_reflection`：`ctx = body.feeling_context()`；`reflector.final(materials, extra=FEELINGS_SYSTEM if ctx else "")`；`finish_reflection(..., feelings=ctx)`

- [ ] **Step 4: 跑测试确认通过**：同 Step 2 的命令，再加 `tests/test_brain_inner_body.py tests/test_brain_inner_log_body.py -q`

- [ ] **Step 5: 提交** `feat(inner): 下线反思在旧印象的基础上改写对每个人的感觉`

---

### Task 3: 大脑看到感觉（arrive、status、提示词、introspect）

**Files:**
- Modify: `src/skydango/inner/ledger.py`（`describe`、`Ledger.status_line`）、`src/skydango/brain/body.py`（arrive、status、introspect）、`src/skydango/brain/prompt.py`、`src/skydango/brain/tools.py`、`src/skydango/cli.py`（`brain_prompt(... feelings=)`）
- Test: `tests/test_inner_ledger.py`、`tests/test_brain_inner_body.py`、`tests/test_brain_prompt*.py`（找现有的 brain_prompt 测试文件）

**Interfaces:**
- Consumes: Task 1 的 `feeling_note`、`feeling_bit`、`trend_shown`
- Produces:
  - `describe(name, card, now, look="", feel: str = "")`：`feel` 非空时接在「一起玩过 N 天」后面（`·熟了`）；`Ledger.status_line(names, now, looks=None, feel=None)`：`feel` 是 `Callable[[Card], str]`，None = 照旧
  - `FEEL_RULE: str`（spec §4 那一句，以「- 」开头）；`brain_prompt(..., feelings: bool = False)`：True 时 `FEEL_RULE` 接在 `INNER_RULES` 后面
  - `INTROSPECT_TOPICS` 末尾加 `"关系"`；`feelings.introspect_text(cards, friends, now, cfg) -> str`

- [ ] **Step 1: 写失败的测试**
  - `test_describe_with_feel`：`describe(..., feel="熟了")` == 「小明（今天一起 40 分钟·认识 21 天·一起玩过 9 天·熟了）」；不传 == 原来的输出
  - `test_arrive_carries_feeling`（body 测试，照 `test_brain_inner_body.py` 里现有 arrive 测试的搭法）：卡上有 feeling、`feelings` 开 → arrive 文字含「你对他：嘴贫但靠谱（熟了，最近平稳）」且在「你惦记着」之前；他在 `_soft_until` 里 → 只有「（熟了）」；`return` 不带
  - `test_arrive_unchanged_when_off`：`cfg.inner.feelings = False` → arrive 文字和改动前逐字一样（不含「你对他」）
  - `test_status_feel_bit`：开着时 status 的「身边的好友」括号末尾有「·熟了」；关着没有
  - `test_prompt_feel_rule`：`brain_prompt(..., inner=True, feelings=True)` 含 `FEEL_RULE`；`feelings=False` 和不传参数输出完全相等
  - `test_introspect_relations`：开着 → 列出印象、档位、走向、`since`、旧版本；`feelings` 关着 → 「没开」

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现**：arrive 在 `body.py` 拼 `text` 那一行把 `feel` 插在 `self._arrive_notes.pop(name, '')` 后、`want` 前；status 调 `status_line(..., feel=lambda c: feeling_bit(c, wall, cfg.inner) if on else None)`；`INTROSPECT_DESCRIPTION` 的话题说明加「关系 = 你对每个好友的感觉和它怎么变过来的」；`cli` 组 `brain_prompt` 时传 `feelings=ledger is not None and cfg.inner.reflect and cfg.inner.feelings`

- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_inner_ledger.py tests/test_brain_inner_body.py tests/test_brain_tools*.py -q`，再全量（放后台）`python -m pytest -q`

- [ ] **Step 5: 提交** `feat(brain): 好友来时和 status 带上对他的感觉`

---

### Task 4: 内心页、设置项、文档

**Files:**
- Modify: `src/skydango/console/inner_view.py`（`_cards`）、`src/skydango/console/static/inner.js`（关系卡）、`src/skydango/console/settings.py`（`FIELDS`）、`CLAUDE.md`
- Test: `tests/test_console_inner.py`、`tests/test_console_settings*.py`（找现有的 FIELDS 测试）

**Interfaces:**
- Consumes: `Card.feeling`
- Produces: `/api/inner` 每张卡多 `"feeling": {"text", "close", "trend", "since", "past": [...]}`（没有 = `{}`）

- [ ] **Step 1: 写失败的测试**：`test_inner_cards_feeling`：store 里一张带 feeling 的卡 → `_cards` 输出里 `feeling` 原样；没有的卡 `{}`。`Field("inner.feelings", …)` 在 `FIELDS` 里、类型 bool
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：`inner.js` 关系卡在装扮那块前面加一行「印象：…（熟了 · 平稳 · 11-01）」，`past` 折叠成「以前：…」（照 outfits 的循环写法，颜色只用 `var(--…)`）；`settings.py` 加 `Field("inner.feelings", "对每个人的感觉", "下线反思时改写对每个好友的印象、亲近度和走向，他来时带给大脑；要开着「反思」", "bool", "brain")`；CLAUDE.md「内心层」第 3 期后面加一节「对每个人的感觉」（设计 / 计划路径、要点、「还没在沙盒 / 真机验证，数字是估的」）
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_console_inner.py -q` + 全量（后台）
- [ ] **Step 5: 提交** `feat(console): 内心页显示对每个人的感觉；文档`

---

### Task 5: 沙盒验收（真 DeepSeek，不写代码）

按 spec §6 沙盒四步：起管理面板 → `sandbox-ctl start` → 每一步后看 `private/sandbox/memory/inner/people.json` 里那张卡的 `feeling` 和流水账；第 2 步用 `private/sandbox/scenarios/放鸽子-懒洋洋大王.toml`；第 3 步挑 `people.json` 里 `days` 只有 1 天的好友（没有就沙盒里手动 `come` 一个只在 friends.md 里的好友）。
结果写进 CLAUDE.md 那一节（过了几步、看到的原文），停沙盒、停面板，私有仓库提交推送。
