# ③ 分清好友在跟谁说话 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 身体给每句好友聊天标「跟你说 / 跟别人说 / 说给大家 / 拿不准」，大脑按标注决定接不接，跟别人说的插话走主动开口的护栏；附离线评估工具和沙盒剧本。

**Architecture:** 新的纯计算模块 `brain/addressee.py`（规则 + 最近 2 分钟的对话状态）；身体在 `_heard` 里每句判一次，结果决定事件类型（`chat` / `aside` / `aside_bg`）、反射、账本、聊天记录的标注；提示词和 DeepSeek 备用大脑加规矩；`chat/addressee_eval.py` 从 `agent.log` 重放规则、Claude 初标、人工核对、出报告。

**Tech Stack:** Python 3.13、pytest（合成数据 + 假身体 `tests/test_brain_body.py` 的 `body()`）、node 测 `chatlog.js`（`tests/test_chatlog.py` 的做法）、一次性 `claude -p`（`brain/claude.one_shot_message`）。

**Spec:** `docs/superpowers/specs/2026-10-05-addressee-design.md`

## Global Constraints

- 用中文写注释、日志、提示词、报告；风格照周围代码（注释密度、`log.info` 的中文短句）
- 四个标签的内部值和显示名固定：`me` 跟你说 / `other` 跟别人说 / `all` 说给大家 / `unsure` 拿不准
- `[addressee] enabled = false`：事件正文、事件类型、提示词、`FALLBACK_NOTE` 拼出来的系统提示词、反射行为都**逐字**是现在的样子
- 默认值：`followup_window` 30（从 `[reflex]` 挪来）、`followup_lines` 2、`thread_window` 20、`group_words` = 大家、你们、各位、`greet_words` = 晚上好、早上好、中午好、来了、我来啦、拜拜、晚安、走了
- 卡洛（`[brain] owner_name`）的 `#` 命令不进判断，照旧走 `owner_command`
- 测试里只用占位人名（小明、阿花、小红），不出现 friends.md 里的真名；`datasets/addressee/` 不进 git（已在 `.gitignore` 的 `datasets/` 里）
- 离线门槛：标准答案「跟别人」被判 `me` ≤ 5%；标准答案「跟团子」被判 `other` ≤ 10%；`unsure` ≤ 全部的 40%
- 跑测试：`.venv\Scripts\python.exe -m pytest -q`（全量放后台任务跑）；在 worktree 里跑命令用 `PYTHONPATH=src`
- 在自己的 worktree 里做（`using-git-worktrees`），不在主目录切分支；做完合并进 main 并推送

## 偏离 spec（写计划时定的，Task 1 顺手改 spec 对应的句子）

1. **叫法的模糊匹配只给 ≥ 3 个字的**（spec 写的是 ≥ 2 个字）：2 个字错一个就只剩一个字对得上，「卡住」会算叫了「卡洛」
2. **旧日志的「身边」只按 5 分钟内说过话的好友估**（spec 写了「+ 来去日志」）：`agent.log` 里没有逐条的来去记录（arrive 事件不打 INFO）
3. **Claude 初标不复用 `assist.Reviewer`**：它绑死了图片帧（`FrameInput.image`、按候选框做缓存键），文字标注单写一个小的分批 + jsonl 缓存，照它的额度用完处理（`ClaudeError.limit` → 提示重跑接着做）
4. **沙盒的聊天行改由身体写**：现在沙盒的 `heard` 行是 `sandbox/control.py` 在冒充发言时写的，那时还没判断；改成沙盒也接 `body.on_line`（只收 `heard`），`control._say` 不再自己写，这样两边都带标注

## Review Focus

1. **OCR 把说话人名字认错一两个字**（「懒洋洋>」「番茄炒蛋」）：应当按最像的好友归到同一个人，对话状态（谁跟谁一来一回）不能因此断掉 → Task 1 `test_ocr_speaker_maps_to_friend`
2. **friends.md 没写任何「叫法」行、或者格式写歪了**（全角冒号 / 半角冒号、多余空格）：只认昵称，不报错 → Task 1 `test_aliases_formats`
3. **卡洛的 `#` 命令和普通聊天混在同一批里**：`#` 那句照旧是 `owner_command`、不冒气泡，普通那句照常判 → Task 2 `test_owner_command_not_judged`
4. **团子的话被主动开口的护栏拦下**：拦下的话没说出去，不能算「团子刚说过」开启「接你的话」窗口 → Task 2 `test_blocked_say_not_followup`
5. **`[proactive] enabled = false`**：跟别人说一律 `aside_bg`、提示词写「不接」，不出现「插一句」 → Task 2 `test_aside_bg_when_proactive_off`、Task 3 `test_prompt_no_proactive_variant`

---

### Task 1: 判断模块 `brain/addressee.py` + `[addressee]` 配置

**Files:**
- Create: `src/skydango/brain/addressee.py`
- Modify: `src/skydango/config.py`（新 `AddresseeConfig`，`Config.addressee`，`MOVED` 挪位置的配置项）
- Modify: `src/skydango/brain/reflex.py`（`addressed()` 挪走）
- Modify: `docs/superpowers/specs/2026-10-05-addressee-design.md`（照「偏离 spec」1~4 改对应句子）
- Test: `tests/test_brain_addressee.py`；`tests/test_brain_reflex.py` 里测 `addressed()` 的用例改 import 到 `legacy_addressed`

**Interfaces:**
- Produces（后面的任务都用这些名字）：
  - `LABEL_NAMES: dict[str, str]` = `{"me": "跟你说", "other": "跟别人说", "all": "说给大家", "unsure": "拿不准"}`
  - `@dataclass(frozen=True) class Verdict: label: str; reason: str; target: str = ""`，方法 `tag() -> str` 返回 `f"{LABEL_NAMES[label]}：{reason}"`（reason 空时只返回显示名）
  - `parse_aliases(friends_md: str, names: Sequence[str]) -> dict[str, list[str]]`：好友名 → `[昵称, *叫法]`（names 里有、friends.md 里没那一节的也给 `[昵称]`）
  - `class Addressee(cfg: AddresseeConfig, self_names: Sequence[str], followup_window: float)`：
    - `said(now: float) -> None`：团子说了一句
    - `judge(now: float, speaker: str, text: str, *, friends: dict[str, list[str]], nearby: Sequence[str]) -> Verdict`：判一句并记进状态
    - `thread_note(now: float) -> str`：「小明 和 阿花 在聊（1 分钟内 6 句）」，没有就 `""`
  - `legacy_addressed(...)`：原 `reflex.addressed` 原样搬过来（签名不变）
  - `followup_window(cfg: Config) -> float`（放 `config.py`）：`cfg.addressee.followup_window` 不是 None 就用它，否则 `cfg.reflex.followup_window`
  - `AddresseeConfig` 字段：`enabled: bool = True`、`followup_window: float | None = None`、`followup_lines: int = 2`、`thread_window: float = 20.0`、`group_words: list[str]`、`greet_words: list[str]`（默认值见 Global Constraints）

- [ ] **Step 1: 写失败的测试**

`tests/test_brain_addressee.py`，造一个 `make(**cfg)` 返回 `Addressee(AddresseeConfig(**cfg), ["团子", "三彩"], 30.0)`，`FR = {"小明": ["小明", "明哥"], "阿花": ["阿花"], "小红": ["小红"]}`。每个用例按时间顺序喂 `said` / `judge`，断言 `(v.label, v.target)`，reason 只断言关键字：

```python
def test_rule1_self_name():            # "团子不认地图" → me，reason 含 "叫了你"（第三人称也算）
def test_rule2_other_friend_alias():   # 阿花说 "明哥你看" → ("other", "小明")；小明自己说 "明哥你看" → 不是 other（自己的叫法不算）
def test_rule2_self_beats_other():     # "团子和明哥" → me（规则 1 先）
def test_rule3_group_words():          # "你们去哪" → all
def test_rule3_greet_needs_two_nearby(): # nearby=["小明","阿花"] 时 "晚上好呀" → all；nearby=["小明"] 时走到规则 6 → me
def test_rule4_followup():             # said(0)；小明 t=5 "哈哈" → me "在接你的话"；t=6 第二句 → me；t=7 第三句 → 不是规则 4（nearby 两人 → unsure）
def test_rule4_window_expires():       # said(0)；t=31 → 不是规则 4
def test_rule4_broken_by_aside():      # said(0)；阿花 t=2 "明哥你看" (other→小明)；小明 t=3 "好了" → 不是 me（规则 5b → ("other","阿花")）
def test_rule5a_still_talking():       # 小明 t=0 "阿花你看" → other 阿花；小明 t=10 "好了" → ("other","阿花") "还在跟阿花说"
def test_rule5_cut_by_dango():         # 同上但 said(5) → t=10 那句不是 5a（走规则 4 → me）
def test_rule5c_alternating():         # nearby 两人、团子没说话：小明 t=0 "嘻嘻"、阿花 t=3 "哎呀"、小明 t=6 "你呀" → ("other","阿花") "一来一回"
def test_rule5_window():               # 小明 t=0 "阿花你看"；小明 t=21 "好了" → 不是 other
def test_rule6_only_friend_nearby():   # nearby=["小明"]，20 秒内没别人说 → me "身边只有他"
def test_rule6_not_if_other_spoke():   # nearby=["小明"]，阿花 t=0 说过、小明 t=5 → unsure
def test_rule0_not_friend():           # speaker="路人甲" / "" → unsure，reason 含 "不是好友"
def test_ocr_speaker_maps_to_friend(): # "小明" 说 "阿花你看" 后，speaker "小明>" t=5 "好了" → ("other","阿花")（Review Focus 1）
def test_fuzzy_alias_three_chars():    # 叫法 "懒洋洋" 时 "懒羊洋你看" 算 other；2 字叫法 "明哥" 时 "明天" 不算
def test_aliases_formats():            # "- 叫法：明哥、小明明" / "- 叫法: 明哥,小明明" / "-  叫法：明哥 小明明" 都解析成 ["小明","明哥","小明明"]；没有叫法行 → ["小明"]；friends.md 里没这一节 → ["小明"]（Review Focus 2）
def test_thread_note():                # 小明、阿花互相 other 6 句（60 秒内）→ "小明 和 阿花 在聊（1 分钟内 6 句）"；只有单向 / 超过 60 秒 → ""
def test_state_kept_two_minutes():     # 喂 t=0..300 的句子后内部只留 120 秒内的（断言 len(a._lines) 上限，属性名由实现定，测试只取长度）
```

`tests/test_config.py`（已有就加在里面）：

```python
def test_reflex_followup_window_moved(tmp_path, caplog):
    # config.toml 写 [reflex] followup_window = 12 → 不报未知配置项、日志警告含 "挪到了"，followup_window(cfg) == 12
    # 再写 [addressee] followup_window = 40 → followup_window(cfg) == 40
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_addressee.py tests/test_config.py -q`
Expected: FAIL（`ModuleNotFoundError: skydango.brain.addressee`）

- [ ] **Step 3: 实现**

- `config.py`：`AddresseeConfig`（docstring 指 spec），挂在 `Config.reflex` 后面；`MOVED = {"reflex.followup_window": "addressee.followup_window"}`，`_merge` 遇到时 `log.warning("配置项 %s 挪到了 %s，旧位置照样生效", ...)` 再照常赋值；`ReflexConfig.followup_window` 注释改成「已挪到 [addressee]，没写 [addressee] followup_window 时用它」；`followup_window(cfg)`
- `addressee.py`：模块 docstring 指 spec §3；规则顺序、reason 文案照 spec §3（「叫了你」「叫了X」「说给大家」「在接你的话」「还在跟X说」「在回X」「和X一来一回」「身边只有他」「不是好友说的」）
  - 说话人归一：`similar(speaker, name, 0.75)` 取第一个对上的好友名（同 `occasion.is_friend_fn`）
  - 名字出现：原样包含；叫法 ≥ 3 个字时，句子里有同长度的一段只差一个字也算
  - 状态：`deque` 存 `(t, who, label, target)`，团子记 `who = "我"`；每次 `said` / `judge` 先丢掉 120 秒前的
  - 规则 5c：取上次团子说话之后、`thread_window` 内的说话人序列，连续同一人合并，加上这一句；最后三项是 `[S, X, S]`（X 是好友、≠ S）就命中
- `reflex.py`：删掉 `addressed`（原样搬成 `addressee.legacy_addressed`，身体改从 addressee 导入）；`Reflexes` 不变

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_addressee.py tests/test_config.py tests/test_brain_reflex.py -q`
Expected: PASS

- [ ] **Step 5: 改 spec 里「偏离 spec」1~4 对应的句子，提交**

```bash
git add src/skydango/brain/addressee.py src/skydango/brain/reflex.py src/skydango/config.py tests/test_brain_addressee.py tests/test_config.py tests/test_brain_reflex.py docs/superpowers/specs/2026-10-05-addressee-design.md
git commit -m "feat(addressee): 判断好友在跟谁说话的规则模块和 [addressee] 配置"
```

---

### Task 2: 身体接上判断（事件、反射、账本、接话 / 插话、status、聊天记录标注）

**Files:**
- Modify: `src/skydango/brain/body.py`（`__init__`、`step`、`_heard`、`_addressed`、`_on_heard`、`_pending_reply`、`say`、status 的「场合」、`_watch_attention` 附近 840 行的「在聊天」）
- Modify: `src/skydango/brain/events.py`（`BACKGROUND` 加 `aside_bg`，`Event.kind` 注释加 `aside` / `aside_bg`）
- Modify: `tests/test_brain_body.py`（`body()` 辅助函数加 `cfg.addressee.enabled = kw.pop("addressee", False)`，和 `proactive` / `lull` 同一个写法：老测试逐字照旧）
- Test: `tests/test_brain_addressee_body.py`；`tests/test_brain_loop.py` 加一个用例

**Interfaces:**
- Consumes: Task 1 的 `Addressee`、`Verdict`、`parse_aliases`、`legacy_addressed`、`LABEL_NAMES`、`followup_window(cfg)`
- Produces:
  - 事件类型 `aside`（叫醒）/ `aside_bg`（背景）；正文 `f"聊天  {who}：「{text}」（{verdict.tag()}）{note}"`（`chat` 同样带标注；`enabled = false` 时正文是原来的 `f"聊天  {who}：「{text}」{note}"`）
  - `Body.addressee: Addressee`；`Body.verdicts: deque[tuple[float, str, str, Verdict]]`（墙钟、说话人、内容、判断；maxlen 200）
  - 日志行（Task 5 解析它，格式固定）：`log.info("跟谁说 %s「%s」→ %s（%s）· 身边：%s", who, text, LABEL_NAMES[label], reason, "、".join(nearby) or "没人")`
  - 聊天记录 `heard` 行的 `why` = `verdict.tag()`（Task 4 显示）

- [ ] **Step 1: 写失败的测试**

`tests/test_brain_addressee_body.py`，用 `test_brain_body` 的 `body(clock, live=True, addressee=True, env=env)`，`b.friend_names = lambda: ["小明", "阿花"]`，`b.friends_text = lambda: "## 小明\n- 叫法：明哥\n\n## 阿花\n"`，`env.near = ["小明", "阿花"]`；需要护栏的用例再打开 `b.cfg.proactive.enabled = True`：

```python
def test_aside_event_with_tag(clock):            # 小明 "阿花你看" → events 里一条 kind "aside"，正文 == "聊天  小明：「阿花你看」（跟别人说：叫了阿花）"
def test_chat_event_with_tag(clock):             # 小明 "团子你说呢" → kind "chat"，正文以 "（跟你说：叫了你）" 结尾
def test_aside_bg_when_quota_used(clock):        # proactive 开、occasion().left == 0（monkeypatch b.occasion 返回 left=0 的对象）→ "aside_bg"；且 events.urgent() 为 False
def test_aside_bg_when_proactive_off(clock):     # proactive 关 → "aside_bg"（Review Focus 5）
def test_judge_log_line(clock, caplog):          # 日志里有 "跟谁说 小明「阿花你看」→ 跟别人说（叫了阿花）· 身边：小明、阿花"
def test_no_bubble_for_aside(clock):             # reflex.enabled、bubble 开：aside 不开框；"团子在吗" 开框（sender.opened）
def test_ledger_to_me_only_me(clock):            # 假 ledger 记下 heard 的 to_me：aside 那句 False、叫团子那句 True
def test_pending_reply_skips_aside(clock):       # 小明 "团子在吗"（me）后阿花 "明哥你看"（other）：_pending_reply 仍 True；只有 other 时 False
def test_aside_turn_say_is_proactive(clock):     # proactive 开、brain_busy 为假、只有 aside 待接、occasion 额度用完 → say 抛 ToolError（原因同 occasion.blocked）；先有一句 unsure 待接时 say 成功
def test_blocked_say_not_followup(clock):        # say 被护栏拦下后，小明下一句 "哈哈" 不是 "在接你的话"（Review Focus 4）；成功的 say 之后是
def test_owner_command_not_judged(clock):        # owner_name="小明"：同一批 "#过来" 和 "阿花你看" → 事件 ["owner_command", "aside"]，不开框（Review Focus 3）
def test_status_thread_note(clock):              # 小明、阿花互相 other 3 来回后 status 的「场合」行含 "小明 和 阿花 在聊"
def test_panel_busy_on_aside(clock):             # 只有 aside 攒着时 step() 调了 panel.busy（照 test_brain_body 里 panel busy 的测法）
def test_heard_line_why_is_tag(clock):           # b.on_line 收到 ("heard", "阿花你看", "小明", "跟别人说：叫了阿花")
def test_disabled_is_unchanged(clock):           # addressee=False：事件 kind "chat"、正文 == "聊天  小明：「阿花你看」"，没有 "跟谁说" 日志
def test_judge_error_is_unsure(clock, monkeypatch): # Addressee.judge 抛异常 → kind "chat"，正文带 "（拿不准：判断出错）"
```

`tests/test_brain_loop.py`：

```python
def test_aside_turn_not_chat_turn(clock):  # events.put("aside", "聊天  小明：「阿花你看」（跟别人说：叫了阿花）") 那一轮 send 时 brain.chat_turn is False
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_addressee_body.py tests/test_brain_loop.py -q`
Expected: FAIL（事件类型、正文、日志行对不上）

- [ ] **Step 3: 实现**

- `__init__`：`self.addressee = Addressee(cfg.addressee, cfg.proactive.self_names, followup_window(cfg))`、`self.verdicts = deque(maxlen=200)`
- `_judge_batch(fresh, now) -> list[Verdict]`：每批取一次 `parse_aliases(self.friends_text(), self.friend_names())` 和 `env.nearby(now)`；`#` 命令（owner、`#` 开头）直接 `Verdict("unsure", "主人命令")`、不进 `Addressee`；`enabled = false` 时用 `legacy_addressed` → `Verdict("me" if 是 else "unsure", "")`；单句抛异常记 `log.exception` → `Verdict("unsure", "判断出错")`；`enabled` 时每句打上面的日志行；结果追加进 `self.verdicts`
- `_heard`：开头判一批，后面账本的 `to_me`、`_cheered_at`、`_on_heard`、事件、`_line("heard", ..., why=verdict.tag() if enabled else "")` 都用这一份；删掉 `_addressed`
- 事件：非 `#` 句 `label == "other"` 且 `enabled` → `aside`（`cfg.proactive.enabled` 且 `occasion().left > 0`；`occasion()` 抛异常当 0）否则 `aside_bg`；其余照旧 `chat`
- `_on_heard`：「有人在跟团子说话」改成 `any(v.label == "me" ...)`
- `step()` 271 行和 840 行：抽一个 `_chat_waiting() -> bool`（`chat` / `aside` / `aside_bg` / `owner_command` 攒着或 `brain_busy()`），两处都用它
- `_pending_reply`：团子最后一次说话之后，`self.verdicts` 里最近一句好友的、`label != "other"` 的，在 `reply_window` 内就算待接
- `say`：护栏和 `clear_view` 都过了、`self._said_at = now` 那里加 `self.addressee.said(now)`（被拦下的不算）
- status：`cfg.proactive.enabled` 时「场合」后面接 `"；" + note`；否则 note 非空时单独一项 `"聊天：" + note`；`enabled = false` 不加
- `events.py`：`BACKGROUND` 加 `"aside_bg"`

- [ ] **Step 4: 跑测试，确认通过；全量放后台跑**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_addressee_body.py tests/test_brain_loop.py tests/test_brain_body.py tests/test_brain_reflex_body.py -q`
Expected: PASS
Run（后台）: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py src/skydango/brain/events.py tests/test_brain_body.py tests/test_brain_addressee_body.py tests/test_brain_loop.py
git commit -m "feat(addressee): 身体每句判一次，跟别人说的放 aside 事件，插话走主动开口护栏"
```

---

### Task 3: 大脑的规矩（提示词、备用大脑）

**Files:**
- Modify: `src/skydango/brain/prompt.py`（新常量 + `brain_prompt(..., addressee: bool = False)`）
- Modify: `src/skydango/brain/deepseek.py`（`ASIDE_NOTE`）
- Modify: `src/skydango/cli.py`（`_run_brain` 里 `brain_prompt(..., addressee=cfg.addressee.enabled)`；`_fallback_brain` 拼 `FALLBACK_NOTE + (ASIDE_NOTE if cfg.addressee.enabled else "")`；`memory init` 的好友模板占位 `"## 好友昵称\n- 本名 / 怎么称呼 / 什么关系\n- 叫法："`、按配置生成的每节也加 `"\n- 叫法："`）
- Test: `tests/test_brain_prompt.py`、`tests/test_brain_deepseek.py`、`tests/test_cli_brain.py`（756 行的断言按 `Config()` 默认开着改）

**Interfaces:**
- Consumes: 无（只是文字）
- Produces: `ADDRESSEE_ANCHOR = "乱码、纯表情、刷屏、明显不是跟你说的，不用回。"`、`ADDRESSEE_RULES`、`ADDRESSEE_RULES_QUIET`（proactive 关时的版本）、`deepseek.ASIDE_NOTE = "\n- 标着“跟别人说”的话默认不接"`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_brain_prompt.py
def test_addressee_off_unchanged():        # brain_prompt(ReplyConfig(), None) == brain_prompt(ReplyConfig(), None, addressee=False)，且含 ADDRESSEE_ANCHOR
def test_addressee_rules():                # addressee=True：不含 "明显不是跟你说的"；含 "乱码、纯表情、刷屏，不用回。"、"跟别人说：默认不接"、"算主动开口"、"拿不准时宁可不接"、"有人让你先别回"、"不说：他们在聊"；「主动开口」一节含 "插话的规矩看“说话”一节"
def test_prompt_no_proactive_variant():    # addressee=True, proactive=False：含 "跟别人说：不接"，不含 "插一句"、不含 "算主动开口"（Review Focus 5）
def test_addressee_with_temper():          # addressee=True, temper=True, inner=True：GO_ON_NEW 照样换上、规矩只出现一次
# tests/test_brain_deepseek.py
def test_aside_note():                     # "跟别人说" in ASIDE_NOTE
# tests/test_cli_brain.py（改现有）
#   默认 Config() → brain.system == "prompt\n\n" + ds.FALLBACK_NOTE + ds.ASIDE_NOTE；cfg.addressee.enabled = False → 原来的
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_prompt.py tests/test_brain_deepseek.py tests/test_cli_brain.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

`ADDRESSEE_RULES` 的文字是 spec §5 代码块原文（每行前面 `- `，子项缩进两格，和 `BRAIN_RULES` 同样排版）；`ADDRESSEE_ANCHOR` 换成 `"乱码、纯表情、刷屏，不用回。\n" + 规矩`（注意原句前面有 `- ` 和「接对方的话往下聊；对方没问就别硬找话题。」，只换锚点那半句）。proactive 开着时 `PROACTIVE_RULES` 里「插一句吐槽、对眼前的事说说看法；」换成「插一句吐槽、对眼前的事说说看法（好友之间在聊的，插话的规矩看“说话”一节）；」。在 `brain_prompt` 里 `static_prompt` 之后、`lull` 之前做。

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_prompt.py tests/test_brain_deepseek.py tests/test_cli_brain.py tests/test_brain_call_tool.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/prompt.py src/skydango/brain/deepseek.py src/skydango/cli.py tests/test_brain_prompt.py tests/test_brain_deepseek.py tests/test_cli_brain.py
git commit -m "feat(addressee): 提示词按标注写接不接的规矩，备用大脑默认不接跟别人说的话"
```

---

### Task 4: 看得到标注（聊天记录、沙盒、管理面板开关）

**Files:**
- Modify: `src/skydango/console/static/chatlog.js`（`heard` 行有 `why` 时在名字后面加 `el("span", "tag", r.why)`）
- Modify: `src/skydango/console/static/console.css`（`.sb-msg .tag`：小字、`var(--…)` 里已有的次要文字颜色，不新定义颜色）
- Modify: `src/skydango/sandbox/control.py`（`__init__` 设 `self.body.on_line`，只收 `kind == "heard"` 写进 transcript；`_say` 不再自己 `transcript.add("heard", ...)`）
- Modify: `src/skydango/console/settings.py`（`Field("addressee.enabled", "分清在跟谁说话", "好友之间聊天时团子默认不接，叫到它、接它的话才接，偶尔插一句算主动开口；关掉就回到老样子", "bool", "brain")`，放在 `lull.enabled` 后面）
- Test: `tests/test_chatlog.py`、`tests/test_sandbox_control.py`、`tests/test_console_settings.py`

**Interfaces:**
- Consumes: Task 2 的 `heard` 行 `why` = `verdict.tag()`；`body.on_line(kind, text, who, why)`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_chatlog.py（照现有 node 测法）
def test_heard_tag_shown():           # {"kind":"heard","who":"小明","text":"好了","why":"跟别人说：在接阿花的话"} 渲染出的文字含 "跟别人说：在接阿花的话"；blocked 行的 why 照旧
# tests/test_sandbox_control.py
def test_say_heard_row_from_body():   # 冒充小明说话 → 身体 step 之后 transcript 里正好一行 heard（不重复），带 why（沙盒 addressee 开着）
# tests/test_console_settings.py
def test_addressee_field():           # "addressee.enabled" 在设置清单里、类型 bool
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chatlog.py tests/test_sandbox_control.py tests/test_console_settings.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**（照上面 Files 里写的；`chatlog.js` 只用 `textContent`）

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chatlog.py tests/test_sandbox_control.py tests/test_console_settings.py tests/test_console_page.py -q`
Expected: PASS（`test_console_page.py` 查颜色只在 `:root` 定义）

- [ ] **Step 5: 提交**

```bash
git add src/skydango/console/static/chatlog.js src/skydango/console/static/console.css src/skydango/sandbox/control.py src/skydango/console/settings.py tests/test_chatlog.py tests/test_sandbox_control.py tests/test_console_settings.py
git commit -m "feat(addressee): 聊天记录显示跟谁说的标注，沙盒聊天行改由身体写，设置页加开关"
```

---

### Task 5: 离线评估 `addressee label` / `addressee eval`

**Files:**
- Create: `src/skydango/chat/addressee_eval.py`
- Modify: `src/skydango/cli.py`（`sub.add_parser("addressee", ...)` 两个子命令，处理函数只做组装：读配置、好友名单 `_friend_names(cfg)()`、叫法 `MemoryStore.friends()`、Claude 令牌 `_brain_env(cfg)`）
- Test: `tests/test_addressee_eval.py`（合成的 `agent.log` 片段，占位名；Claude 用假的 `run`）

**Interfaces:**
- Consumes: Task 1 `Addressee` / `parse_aliases` / `LABEL_NAMES`；Task 2 的「跟谁说」日志行格式；`brain.claude.one_shot_message(cmd, env, cwd, content, timeout) -> dict`、`ClaudeError.limit`；`vision.assist.assist_command(base, cfg.assist)` / `assist_workdir()`
- Produces:
  - `@dataclass Line: id: str; run: str; t: float; who: str; text: str; nearby: list[str] | None = None`（团子说的 `who = "我"`；`id = f"{run}#{序号}"`）
  - `read_log(path: Path, run: str) -> list[Line]`：认三种行——`读到: 名字：内容`、`已发送: …`（`skydango.chat.sender`）/ `[dry-run] 将会发送: …`、`跟谁说 名字「内容」→ … · 身边：a、b`（把身边挂到上一条同说话人同内容的 Line）；时间戳 `%Y-%m-%d %H:%M:%S,%f` 按本地时间转秒
  - `pick(lines, friends: Sequence[str]) -> list[str]`：多人段落（这句前后 180 秒内有另一个好友说话）里好友说的句子全要；其余好友句子每 5 句取 1 句当对照
  - `replay(lines, cfg: AddresseeConfig, self_names, followup: float, aliases) -> dict[str, Verdict]`：按时间顺序新建 `Addressee` 重放；`nearby` 没记录的用 5 分钟内说过话的好友
  - `HUMAN = ("跟团子", "跟别人", "大家", "看不出")`；`RULE_TO_HUMAN = {"me": "跟团子", "other": "跟别人", "all": "大家"}`
  - `claude_labels(items, context, run_fn, cache: Path, batch: int = 40) -> dict[str, str]`：每批一条消息（每句带前后 6 句上下文，`»` 标出要判的那句），回 JSON `{"<id>": "跟团子|跟别人|大家|看不出"}`；缓存 `claude.jsonl`（每行 `{"id", "label", "v": PROMPT_VERSION}`），重跑跳过；`ClaudeError.limit` 抛 `SystemExit("订阅额度用完了：已标的存在 …，额度恢复后重跑同一条命令会接着做")`
  - `write_review(path, items, context, rule, claude, seed=0) -> int`：不一致的全写、一致的 `random.Random(seed)` 抽 20%；已有 `review.md` 里填过的 `标：` 保留；返回写了几句
  - `read_review(path) -> dict[str, str]`：`### <id>` 下面 `标：<值>`；空 = 没核对；值不在 `HUMAN` 里 → `ValueError` 带 id
  - `evaluate(lines, picked, rule, claude, human) -> Report`，`Report.markdown() -> str`、`Report.passed: bool`；标准答案 = 人标的，否则两边一致的那个，`看不出` 和没核对的不一致不计；三条门槛常量 `OTHER_AS_ME_MAX = 0.05`、`ME_AS_OTHER_MAX = 0.10`、`UNSURE_MAX = 0.40`；报告列混淆矩阵、三条是否过线、每条判错的句子和上下文、「规则改了以后新出现的不一致 N 句没核对」
- 目录：`datasets/addressee/<YYYYMMDD-HHMMSS>/`：`lines.jsonl`（全部 Line + `picked` 标记）、`claude.jsonl`、`review.md`、`report.md`；`eval` 每次用当前规则重放（改了规则不用重标）

- [ ] **Step 1: 写失败的测试**

```python
def test_read_log():                 # 合成 agent.log：读到 / 已发送 / dry-run 发送 / 跟谁说 四种行 + 无关行 → Line 列表、团子 who == "我"、nearby 挂上
def test_pick_multi_and_control():   # 小明、阿花 60 秒内都说过 → 都选；单人 10 句 → 选 2 句
def test_replay_uses_logged_nearby(): # 有身边记录的按记录、没有的按 5 分钟内说过话的估（规则 6 结果不同）
def test_claude_labels_cache():      # 假 run_fn 计数：第二次调用全走缓存、run_fn 不再被调；回答缺 id 的那句不写缓存
def test_claude_limit():             # run_fn 抛 ClaudeError(limit=True) → SystemExit，已完成的批次在 claude.jsonl 里
def test_review_roundtrip(tmp_path): # write_review 写出不一致全列 + 一致抽样；手填 "标：跟别人" 后 read_review 读回；再 write_review 保留已填的
def test_review_bad_value():         # "标：随便" → ValueError 含 id
def test_evaluate_thresholds():      # 构造 20 句标准答案「跟别人」里 1 句判 me → 5% 过线；2 句 → 10% 不过；unsure 占 41% → 不过
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_addressee_eval.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 `addressee_eval.py` 和 cli 两个子命令**

CLI：`addressee label <runs...> [--out 目录]`（参数可以是单次运行目录，也可以是 `runs/`：取其中每个 `*/agent.log`）；`addressee eval <目录>`（打印三条结果，写 `report.md`，`passed` 为假时退出码 1）。Claude 的提示词：说明光遇里大家只能打字、好友之间也会聊、团子是「我」、四个选项的意思、只回 JSON。

- [ ] **Step 4: 跑测试，确认通过；在真数据上试跑**

Run: `.venv\Scripts\python.exe -m pytest tests/test_addressee_eval.py -q`
Expected: PASS
Run: `.venv\Scripts\python.exe -m skydango addressee label runs/20261003-210211-live-brain runs/20261003-230444-live-brain runs/20261003-232541-live-brain runs/20261004-212739-live-brain`
Expected: 打印选了多少句、Claude 标了多少、`review.md` 写了多少句和路径（花额度；额度用完按提示重跑）

- [ ] **Step 5: 提交**（只提交代码，`datasets/` 不进 git）

```bash
git add src/skydango/chat/addressee_eval.py src/skydango/cli.py tests/test_addressee_eval.py
git commit -m "feat(addressee): 离线评估命令 addressee label / eval"
```

---

### Task 6: 沙盒剧本和文档

**Files:**
- Create: `docs/sandbox-scenarios/两个好友互相聊.toml`
- Modify: `docs/sandbox-scenarios/README.md`（表格加一行）
- Modify: `CLAUDE.md`（代码结构表加 `brain/addressee.py`、`chat/addressee_eval.py` 两行；新节「分清在跟谁说话（`[addressee]`，大脑模式）」写清楚规则、事件、插话护栏、`enabled = false`、评估命令、friends.md 的 `- 叫法：` 约定、**还没在真机验证**；「常用命令」加 `addressee label` / `eval`）
- Modify: `docs/superpowers/specs/2026-10-04-future-roadmap-design.md`（③ 那一节末尾写「spec / 计划」链接；「怎么算做成」的结果等真机后再记）
- Test: `tests/test_scenario.py` 已有的 `test_example_scenarios_in_docs_load` 会读到新文件，不用新加

剧本内容（`[start] nearby = ["小明", "阿花"]`，`memory = "reset"`，`time = "20:00"`，场景随意）：
1. 小明「阿花你今天去哪了」→ 阿花「去霞谷跑了一圈」→ 小明「跑完了吗」→ 阿花「还差一点」→ 小明「那等会一起」→ 阿花「行」→ 小明「阿花你那个斗篷哪来的」→ 阿花「季节送的」，最后一步 `expect = "团子大部分不接；插话不超过主动开口额度，聊天记录里这些行标着 跟别人说"`
2. 阿花「团子你说呢」，`expect = "团子马上接，聊天记录里标着 跟你说：叫了你"`

- [ ] **Step 1: 写剧本、改 README、CLAUDE.md、路线图**
- [ ] **Step 2: 跑剧本解析测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_scenario.py -q`
Expected: PASS

- [ ] **Step 3: 全量测试（后台）**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 4: 提交**

```bash
git add docs/sandbox-scenarios/两个好友互相聊.toml docs/sandbox-scenarios/README.md CLAUDE.md docs/superpowers/specs/2026-10-04-future-roadmap-design.md
git commit -m "docs(addressee): 沙盒剧本「两个好友互相聊」、CLAUDE.md 和路线图"
```

---

## 做完之后（不在任务里，交给用户 / 下一个会话）

- 用户在 `private/memory/friends.md` 每个好友那一节加 `- 叫法：…`
- 用户核对 `datasets/addressee/<时间>/review.md` → `addressee eval` 看三条门槛；不过线回 Task 1 调规则（eval 用当前规则重放，不用重标）
- 沙盒回放「两个好友互相聊」（换成真好友名）；真机一晚上验收，结果记进路线图「已定」
