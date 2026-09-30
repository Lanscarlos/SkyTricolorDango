# 内心层第 3 期（性格：口头禅、老梗、看法，放开太乖的规则）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 反思顺带沉淀团子的口头禅 / 老梗 / 看法（`persona.json`，有上限、会淡出、代码守敏感词和熟人门槛），启动时进系统提示词；提示词加「脾气」放开四样太乖；好友说难过身体挂"收着点"。

**Architecture:** 新文件 `inner/persona.py`（`Persona` 数据 + 规矩 + 拼文字），`store.py` 加读写；`reflect.py` 的系统提示词可传、材料可带性格段；`prompt.py` 加 `TEMPER_RULES` 和「你攒下的性格」；身体在 `apply_reflection` 顺带套性格、arrive 接老梗、记"收着点"；cli 接线、`finish_reflection` 也套性格。沿用第 1 / 2 期的模式：纯数据模块 + 身体 `_inner_call` 包出错 + 只在 live 写盘。

**Tech Stack:** Python 3.11+ 标准库；pytest。

**Spec:** `docs/superpowers/specs/2026-09-30-inner-phase3-design.md`（前两期：`2026-09-30-inner-phase1-design.md`、`2026-09-30-inner-phase2-design.md`）

## Global Constraints

- `[inner]` 新键和默认值：`persona = true`、`fade_days = 14`、`catchphrases_max = 5`、`jokes_per_friend = 3`、`jokes_max = 20`、`opinions_max = 10`、`soft_minutes = 30`
- `SENSITIVE = ("胖", "瘦", "丑", "矮", "长相", "身材", "脸", "爸", "妈", "家里", "成绩", "考试", "分数", "几岁", "年纪", "学校", "班")`
- 条目 `text` / `stance` 截 30 字，`topic` 截 10 字；老梗的 `who` 是好友且 `len(card.days) ≥ grudge_min_days`；收着点的人不记新老梗
- 超上限淘汰：`hits` 最少的先删，同样多删 `last_used or since` 最旧的；淡出：`now − (last_used or since) > fade_days × 86400`
- 只在 live 写 `persona.json`；dry-run 只在内存里；坏文件改名 `.bad-<时间>`（dry-run 不改名）
- `persona = false`：大脑提示词、反思提示词和材料与第 2 期**逐字一样**，不沉淀、不记"收着点"
- 底线各节（身份、隐私、不约线下、未成年人）一字不动
- 出错只记日志（身体里走 `_inner_call`），中文注释 / 日志

## Review Focus

1. **反思要沉淀一条带敏感词的老梗**（"小明又胖了"）：丢掉（Task 1 测）
2. **老梗挂在陌生人、不熟的好友、或 OCR 错一个字的好友名下**：前两种丢，错字对上正名（Task 1 测）
3. **同一个话题来了新看法 / 超上限**：换立场不重复；超上限按 hits、再按最旧淘汰（Task 1 测）
4. **dry-run 下 `persona.json` 坏了**：不改名、从空的开始（Task 1 测）
5. **`persona = false`**：`brain_prompt`、`REFLECT_SYSTEM`、`materials` 与第 2 期一模一样（Task 2 / 3 测）

---

### Task 1: 配置 + `Persona` + 存储

**Files:**
- Modify: `src/skydango/config.py`（`InnerConfig` 追加 7 个键）、`src/skydango/inner/store.py`（`load_persona` / `write_persona`，文件 `persona.json`）
- Create: `src/skydango/inner/persona.py`
- Test: `tests/test_inner_persona.py`

**Interfaces:**
- Consumes: `ledger.Card` / `match_friend` / `ago`、`chat.tracker.similar`、`InnerConfig`
- Produces:
  - `SENSITIVE`（见 Global Constraints）、`@dataclass Trait(text: str, who: str = "", topic: str = "", since: float = 0.0, last_used: float = 0.0, hits: int = 0)`
  - `@dataclass Persona(catchphrases: list[Trait], jokes: list[Trait], opinions: list[Trait])`（都 `default_factory=list`）；`to_dict()` 里看法写 `{"topic", "stance", "since", "last_used", "hits"}`（`stance` = `Trait.text`），口头禅 `{"text", …}`，老梗 `{"who", "text", …}`；`Persona.from_dict(d)` 反过来
  - `Persona.apply(result: dict, cards: dict[str, Card], friends: list[str], now: float, cfg: InnerConfig, soft: set[str] = frozenset()) -> list[str]`（丢掉的原因）；只看 `result["persona_add"]` / `result["persona_used"]`
  - `Persona.fade(now: float, cfg: InnerConfig) -> None`、`section() -> str`、`joke_note(name: str) -> str`、`show_lines() -> list[str]`
  - `InnerStore.load_persona(quarantine: bool = True) -> Persona`、`write_persona(p: Persona) -> None`

- [ ] **Step 1: 写失败的测试**（`CARDS = {"小明": Card(first_met=0, days=["d1","d2","d3"]), "懒洋洋大王": Card(…3 天…), "阿花": Card(first_met=0, days=["d1"])}`，`FRIENDS` 同键；`CFG = InnerConfig()`）

```python
def add(**kw): return {"persona_add": kw}

def test_add_three_kinds_and_rules():
    p = Persona()
    dropped = p.apply(add(
        catchphrases=["害，懒得动", "妈呀好累"],                                    # 第二条带“妈”：丢
        jokes=[{"who": "小明", "text": "上次把团子带进冥龙嘴里"},
               {"who": "阿花", "text": "x"},                                        # 不熟：丢
               {"who": "路人", "text": "y"},                                        # 不是好友：丢
               {"who": "懒洋羊大王", "text": "路痴带路"},                             # 错字对上正名
               {"who": "小明", "text": "小明又胖了"}],                                # 敏感词：丢  (Review Focus 1/2)
        opinions=[{"topic": "雨林啊啊啊啊啊啊啊啊啊", "stance": "湿漉漉的，谁爱去谁去" + "呀" * 30}]),
        CARDS, FRIENDS, T0, CFG)
    assert [t.text for t in p.catchphrases] == ["害，懒得动"]
    assert [(t.who, t.text) for t in p.jokes] == [("小明", "上次把团子带进冥龙嘴里"), ("懒洋洋大王", "路痴带路")]
    assert (len(p.opinions[0].topic), len(p.opinions[0].text)) == (10, 30)
    assert len(dropped) == 4

def test_soft_friend_gets_no_new_jokes():
    p = Persona()
    p.apply(add(jokes=[{"who": "小明", "text": "a"}]), CARDS, FRIENDS, T0, CFG, soft={"小明"})
    assert p.jokes == []

def test_same_topic_replaces_and_duplicates_skip():             # Review Focus 3
    p = Persona()
    p.apply(add(opinions=[{"topic": "雨林", "stance": "丑"}], catchphrases=["害，懒得动"]), CARDS, FRIENDS, T0, CFG)
    p.apply(add(opinions=[{"topic": "雨林", "stance": "其实还行"}], catchphrases=["害，懒得动啊"]), CARDS, FRIENDS, T0 + 9, CFG)
    assert [(t.topic, t.text, t.since) for t in p.opinions] == [("雨林", "其实还行", T0 + 9)]
    assert len(p.catchphrases) == 1

def test_used_bumps_hits():
    p = Persona(catchphrases=[Trait("害，懒得动", since=T0)], opinions=[Trait("丑", topic="雨林", since=T0)])
    p.apply({"persona_used": ["害，懒得动", "雨林"]}, CARDS, FRIENDS, T0 + 5, CFG)
    assert (p.catchphrases[0].hits, p.catchphrases[0].last_used, p.opinions[0].hits) == (1, T0 + 5, 1)

def test_caps_evict_fewest_hits_then_oldest():                    # Review Focus 3
    p = Persona(catchphrases=[Trait(f"口头禅{i}", since=T0 + i, hits=1 if i == 0 else 0) for i in range(5)])
    p.apply(add(catchphrases=["新的"]), CARDS, FRIENDS, T0 + 99, CFG)
    assert [t.text for t in p.catchphrases] == ["口头禅0", "口头禅2", "口头禅3", "口头禅4", "新的"]
    jokes = [{"who": "小明", "text": f"梗{i}"} for i in range(4)]
    p.apply(add(jokes=jokes), CARDS, FRIENDS, T0, CFG)
    assert [t.text for t in p.jokes] == ["梗1", "梗2", "梗3"]      # 每人 3 条

def test_fade():
    p = Persona(catchphrases=[Trait("旧", since=T0), Trait("用过", since=T0, last_used=T0 + 10 * 86400)])
    p.fade(T0 + 15 * 86400, CFG)
    assert [t.text for t in p.catchphrases] == ["用过"]

def test_junk_shapes():
    p = Persona()
    assert p.apply({"persona_add": "不是对象", "persona_used": 5}, CARDS, FRIENDS, T0, CFG) and p == Persona()
    assert p.apply({"persona_add": {"jokes": "x", "opinions": [5]}}, CARDS, FRIENDS, T0, CFG)

def test_section_and_joke_note():
    p = Persona(catchphrases=[Trait("害，懒得动"), Trait("樱花发型天下第一")],
                jokes=[Trait("上次把团子带进冥龙嘴里", who="小明", hits=2), Trait("路痴带路", who="小明"), Trait("第三个", who="小明")],
                opinions=[Trait("湿漉漉的，谁爱去谁去", topic="雨林")])
    assert p.section().splitlines() == [
        "## 你攒下的性格（慢慢和大家玩出来的；人设里写的优先）",
        "口头禅：害，懒得动；樱花发型天下第一",
        "看法：雨林——湿漉漉的，谁爱去谁去",
        "和小明的老梗：上次把团子带进冥龙嘴里；路痴带路；第三个",
        "用得自然，别每句都用；同一个梗一次上线最多用一两回。",
    ]
    assert p.joke_note("小明") == "。你们的老梗：上次把团子带进冥龙嘴里；路痴带路"   # 最多 2 条，hits 多的先
    assert p.joke_note("阿花") == "" and Persona().section() == ""

def test_roundtrip_and_bad_file(tmp_path):                         # Review Focus 4
    st = InnerStore(tmp_path)
    p = Persona([Trait("a", since=1)], [Trait("b", who="小明")], [Trait("c", topic="雨林", hits=2)])
    st.write_persona(p)
    assert st.load_persona() == p
    assert json.loads((tmp_path / "persona.json").read_text("utf-8"))["opinions"][0]["stance"] == "c"
    (tmp_path / "persona.json").write_text("坏", encoding="utf-8")
    assert st.load_persona(quarantine=False) == Persona() and (tmp_path / "persona.json").exists()
    assert st.load_persona() == Persona() and list(tmp_path.glob("persona.json.bad-*"))
```

- 重复判定：口头禅比 `text`、老梗比**同一个人**的 `text`、看法比 `topic`（`similar ≥ 0.75`）；`persona_used` 的每一项按同样的相似度匹配口头禅 / 老梗的 `text` 或看法的 `topic`
- 淘汰：先按类上限（老梗先按每人再按总数）；`apply` 结束时调一次 `fade`
- `section()` 各行只在有内容时写；老梗按人分行，人名按第一次出现的顺序；`show_lines()` 给 `memory show`：`["===== 性格档案 =====", "口头禅：…（用过 N 次）", …]`，空的写"（空）"

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_inner_persona.py -q` → Expected: FAIL（`ModuleNotFoundError`）
- [ ] **Step 3: 实现 `InnerConfig` 新键、`inner/persona.py`、`InnerStore` 两个方法**（坏文件处理照 `load_mind`）
- [ ] **Step 4: 跑测试确认通过** — Run: `python -m pytest tests/test_inner_persona.py tests/test_inner_store.py tests/test_config.py -q` → Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 性格档案（口头禅、老梗、看法）的数据、规矩、淡出和 persona.json 读写"`

---

### Task 2: 反思带上性格

**Files:**
- Modify: `src/skydango/inner/reflect.py`
- Test: `tests/test_inner_reflect.py`

**Interfaces:**
- Produces: `PERSONA_SYSTEM: str`；`Reflector(cfg, llm, clock=time.monotonic, threaded=True, system: str = REFLECT_SYSTEM)`（`_ask` 用 `self.system`）；`materials(..., final: bool, traits: str | None = None)`（新参数加在最后；`None` 不写这一段，`""` 写"（还没有）"）

- [ ] **Step 1: 写失败的测试**

```python
def test_reflector_uses_given_system(clock):
    llm = FakeLlm("{}")
    r = Reflector(CFG, llm, clock, threaded=False, system="系统X")
    r.stirred(clock()); clock.advance(1300); r.start("m", clock())
    assert llm.calls[0][0] == "系统X"

def test_persona_system_mentions_keys():
    for s in ("persona_add", "persona_used", "catchphrases", "jokes", "opinions", "外貌"):
        assert s in PERSONA_SYSTEM

def test_materials_traits_section():
    base = dict(now=T0, energy_note="", mind_line="", chat=[], comings=[], cards=[], notes=[], persona="", final=False)
    assert "你攒下的性格" not in materials(**base)                             # Review Focus 5：不传和第 2 期一样
    assert materials(**base) == materials(**base, traits=None)
    assert "你攒下的性格：\n（还没有）" in materials(**base, traits="")
    assert "口头禅：害" in materials(**base, traits="口头禅：害")
```

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_inner_reflect.py -q` → Expected: FAIL
- [ ] **Step 3: 实现**：`PERSONA_SYSTEM` 照 spec §3 的要点写（顺带想团子的性格；口头禅 / 老梗 / 看法各是什么；用了已有的且有人接写进 `persona_used`；损人的、拿外貌 / 家里 / 成绩 / 年龄开玩笑的不记；没有就空的；输出在原 JSON 里多 `persona_add` / `persona_used`，附一行格式示例）；材料里性格段放在"相关的好友"前，标题"你攒下的性格："
- [ ] **Step 4: 跑测试确认通过** — Run: `python -m pytest tests/test_inner_reflect.py -q` → Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 反思可带性格：PERSONA_SYSTEM、系统提示词可传、材料带已有的性格"`

---

### Task 3: 大脑提示词「脾气」「你攒下的性格」+ 人设模板

**Files:**
- Modify: `src/skydango/brain/prompt.py`、`src/skydango/cli.py`（`PROFILE_TEMPLATE`）
- Test: `tests/test_brain_prompt.py`、`tests/test_cli_brain.py`（`test_profile_template_has_likes` 旁边）

**Interfaces:**
- Produces: `TEMPER_RULES: str`；`GO_ON_OLD = "接对方的话往下聊；"`、`GO_ON_NEW = "接得住就接，不想接也可以换个话头或者吐槽一句；"`；`brain_prompt(..., persona_text: str = "", temper: bool = False)`；`memory_prompt(..., persona_text: str = "")`（「你攒下的性格」放在「日子」之前）

- [ ] **Step 1: 写失败的测试**

```python
def test_temper_rules_and_loosened_line():
    from skydango.brain.prompt import GO_ON_NEW, GO_ON_OLD, MIND_RULES, TEMPER_RULES
    text = brain_prompt(ReplyConfig(), None, inner=True, mind=True, temper=True)
    assert text.index(MIND_RULES) < text.index(TEMPER_RULES) < text.index("- 记住聊过的内容和对方的名字")
    assert GO_ON_NEW in text and GO_ON_OLD not in text
    for s in ("有立场", "损事不损人", "不要，懒", "收着点", "# 命令"):
        assert s in TEMPER_RULES

def test_temper_off_identical_to_phase2():                           # Review Focus 5
    assert brain_prompt(ReplyConfig(), None, inner=True, mind=True) == brain_prompt(
        ReplyConfig(), None, inner=True, mind=True, temper=False, persona_text="")

def test_bottom_lines_untouched():
    on = brain_prompt(ReplyConfig(), None, inner=True, mind=True, temper=True)
    off = brain_prompt(ReplyConfig(), None, inner=True, mind=True)
    for head in ("## 身份", "## 底线"):
        seg = lambda t: t[t.index(head): t.index("\n## ", t.index(head) + 1)]
        assert seg(on) == seg(off)

def test_persona_section_before_days(tmp_path):
    text = brain_prompt(ReplyConfig(), None, days="## 日子\n今天……", persona_text="## 你攒下的性格（…）\n口头禅：害", temper=True)
    assert text.index("## 你攒下的性格") < text.index("## 日子")
```

`tests/test_cli_brain.py`：`PROFILE_TEMPLATE` 含 `"## 脾气"`、`"毛病"`、`"执念"`、`"雷点"`。

`TEMPER_RULES` 原文（spec §2 的五条）：

```text
- 有立场：不同意就直说（“我觉得雨林就是丑”），别动不动“你说得对”；可以争两句、嘴硬，但不吵架、不人身攻击。
- 熟人之间可以互损（状态里写了交情、一起玩过好几天的）：损事不损人——路痴、手残、放鸽子可以；外貌、身材、家里、成绩、年龄不碰。对方不接梗、语气不对就收；刚认识的、陌生人不损。
- 会拒绝、会偷懒：不想做就说不（“不要，懒”），不想聊可以敷衍两句；卡洛的 # 命令、有人真着急或真难过时别偷懒。
- 有执念和怪癖：人设和「你攒下的性格」里的执念可以反复提、口头禅会用，偶尔说点出人意料的话；别每句都用。
- 状态里写着“收着点”的人：不损、不唱反调、不拒绝，好好说话。
```

`PROFILE_TEMPLATE` 在「喜好和看法」后面加 spec §2 的「## 脾气」三行。

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_brain_prompt.py tests/test_cli_brain.py -q -k "temper or bottom or persona_section or profile_template"` → Expected: FAIL
- [ ] **Step 3: 实现**（`TEMPER_RULES` 插在 `REMEMBER_ANCHOR` 前、`MIND_RULES` 之后：按 INNER → MIND → TEMPER 的顺序依次插；`GO_ON_OLD` → `GO_ON_NEW` 只在 `temper` 时换，要在主动开口替换 `NO_NEW_TOPIC` 之前或之后都成立）
- [ ] **Step 4: 跑测试确认通过** — Run: `python -m pytest tests/test_brain_prompt.py tests/test_cli_brain.py -q` → Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 提示词加「脾气」「你攒下的性格」，放开“接对方的话往下聊”；人设模板加脾气"`

---

### Task 4: 身体：收着点、套性格、arrive 带老梗、材料带性格

**Files:**
- Modify: `src/skydango/brain/body.py`
- Test: `tests/test_brain_persona_body.py`（夹具同 `tests/test_brain_mind_body.py` 的 `make()`，外加 `persona=Persona()`，`cfg.inner.persona = True`）

**Interfaces:**
- Consumes: Task 1 的 `Persona`；Task 2 的 `materials(traits=)`
- Produces: `Body(..., persona=None)`；`Body.soft_names(wall: float) -> set[str]`

- [ ] **Step 1: 写失败的测试**

```python
def test_upset_friend_gets_soft_in_status(clock, tmp_path):
    b, env, reader, _ = make(clock, tmp_path)
    env.near = ["阿花"]
    said = "今天真的很难过" + "呜" * 20
    reader.batches = [[msg(said, speaker="阿花")]]
    b.step()
    assert f"对阿花收着点（他刚说「{said[:20]}」）" in b.status()   # 原话截 20 字
    b.wall = lambda: WALL + 31 * 60
    assert "收着点" not in b.status() and b.soft_names(WALL + 31 * 60) == set()

def test_arrive_carries_jokes(clock, tmp_path):
    b, env, _, events = make(clock, tmp_path)
    b.persona.jokes = [Trait("路痴带路", who="懒洋洋大王")]
    env.near = ["懒洋洋大王"]; b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == ["懒洋洋大王 来到身边（第一次在身边见到）。你们的老梗：路痴带路"]

def test_reflection_applies_persona_live_only(clock, tmp_path):
    reply = '{"persona_add": {"catchphrases": ["害，懒得动"]}}'
    for live in (False, True):
        d = tmp_path / str(live)
        b, env, reader, _ = make(clock, d, live=live, reflect_reply=reply)
        reader.batches = [[msg("嗨")]]; b.step(); clock.advance(1300); b.step(); b.step()
        assert [t.text for t in b.persona.catchphrases] == ["害，懒得动"]
        assert (d / "persona.json").exists() is live

def test_soft_friend_passed_to_persona(clock, tmp_path):
    reply = '{"persona_add": {"jokes": [{"who": "懒洋洋大王", "text": "路痴"}]}}'
    b, env, reader, _ = make(clock, tmp_path, reflect_reply=reply)
    reader.batches = [[msg("我好难过")]]; b.step(); clock.advance(1300); b.step(); b.step()
    assert b.persona.jokes == []

def test_materials_have_traits(clock, tmp_path):
    b, *_ = make(clock, tmp_path)
    b.persona.catchphrases = [Trait("害，懒得动")]
    assert "口头禅：害，懒得动" in b.reflect_materials(False)

def test_persona_off_no_soft(clock, tmp_path):
    b, env, reader, _ = make(clock, tmp_path)
    b.persona = None
    reader.batches = [[msg("我好难过")]]; b.step()
    assert "收着点" not in b.status()
```

- 收着点：`_heard` 里 `mind` 那段遍历每句时，`who is not None and sounds_upset(m.text)` 且 `self.persona is not None` → `self._soft_until[who] = (wall + cfg.inner.soft_minutes * 60, m.text[:20])`（这一段在 `forgive` 之前 / 之后都行，两者独立）
- status：`心里：…` 后面加 `"对" + "、".join(名字) + "收着点（他刚说「原话」）"`——一个人时带原话；多人时"对阿花、小明收着点（他们刚说难过）"
- `apply_reflection`：`mind.apply` 之后 `persona.apply(result, ledger.cards, friends, wall, cfg.inner, soft=self.soft_names(wall))`，live 时 `ledger.store.write_persona`
- arrive：`want` 之后接 `persona.joke_note(name)`
- `reflect_materials`：`traits = self.persona.section() if self.persona is not None else None`（`section()` 为空时传 `""`）

- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_brain_persona_body.py -q` → Expected: FAIL（`unexpected keyword argument 'persona'`）
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过（含身体原有测试）** — Run: `python -m pytest tests/test_brain_persona_body.py tests/test_brain_mind_body.py tests/test_brain_inner_body.py tests/test_brain_body.py tests/test_brain_reflex_body.py tests/test_brain_bubble.py -q` → Expected: PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(body): 身体接上性格：好友说难过挂收着点、反思套性格档案、arrive 带老梗"`

---

### Task 5: cli 接线、最终反思、`memory show`、配置模板、管理面板、文档

**Files:**
- Modify: `src/skydango/inner/__init__.py`（`finish_reflection(..., persona=None, soft: set[str] = frozenset())`）、`src/skydango/cli.py`、`config.example.toml`、`src/skydango/console/settings.py`（`Field("inner.persona", "性格", "攒口头禅、老梗和看法，敢唱反调、会拒绝、熟人之间互损；关掉就回到第 2 期", "bool", "brain")`，放在 `inner.reflect` 后面）、`CLAUDE.md`（「内心层」加第 3 期一段、「记忆」表 `inner/` 那行加 `persona.json`、代码结构表加 `persona.py`）、spec 状态
- Test: `tests/test_inner_finish.py`、`tests/test_cli_brain.py`、`tests/test_console_settings.py`

- [ ] **Step 1: 写失败的测试**
  - `test_inner_finish.py`：`finish_reflection` 带 `persona=Persona()` 时 `persona_add` 被套进来、`persist=True` 写 `persona.json`、`persist=False` 不写
  - `test_cli_brain.py`（照第 2 期那几条的写法；假模型回的不是 JSON，`persona.json` 只在有结果时写，所以不断言文件）：live 跑完 `prompt.md` 里有 `TEMPER_RULES`、假 claude 日志里反思那次的 `--system-prompt` 以 `REFLECT_SYSTEM` 开头且含 `PERSONA_SYSTEM`；`inner.persona = false` 时 `prompt.md` 没有 `TEMPER_RULES`、反思系统提示词就是 `REFLECT_SYSTEM`；`memory show` 输出有"===== 性格档案 ====="
  - `test_console_settings.py`：`inner.reflect` 后面是 `inner.persona`
- [ ] **Step 2: 跑测试确认失败** — Run: `python -m pytest tests/test_inner_finish.py tests/test_cli_brain.py tests/test_console_settings.py -q` → Expected: FAIL
- [ ] **Step 3: 实现**
  - cli：新函数 `_inner_persona(cfg, ledger) -> Persona | None`（`ledger` 不为空且 `cfg.inner.persona`：`load_persona(quarantine=ledger.persist)` → `fade(time.time(), cfg.inner)`，出错返回 `None`）；`Body(..., persona=persona)`；`_inner_mind` 建 `Reflector` 时 `system = REFLECT_SYSTEM + "\n\n" + PERSONA_SYSTEM if cfg.inner.persona else REFLECT_SYSTEM`；
    `brain_prompt(..., persona_text=persona.section() if persona else "", temper=ledger is not None and cfg.inner.persona)`；`_final_reflection` 把 `persona=body.persona, soft=body.soft_names(time.time())` 传给 `finish_reflection`；`memory show` 在 `show_lines` 之后打印 `store.load_persona(quarantine=False).show_lines()`（`cfg.inner.persona` 时）
  - `config.example.toml` `[inner]` 追加 7 个键（带注释，注明数字是估的）
- [ ] **Step 4: 跑全部测试** — Run: `python -m pytest -q` → Expected: 全部 PASS
- [ ] **Step 5: 提交** — `git commit -m "feat(inner): 大脑模式接上性格（反思沉淀、提示词、收着点）、memory show、配置、管理面板和文档"`
