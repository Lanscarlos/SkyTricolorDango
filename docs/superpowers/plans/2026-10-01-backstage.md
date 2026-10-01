# 幕后 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打开 `[backstage] enabled` 后，团子知道自己是 AI、卡洛做了她，能用 `introspect` 查自己的内心细节，启动时知道卡洛上次以来改了她什么。

**Architecture:** 新模块 `brain/backstage.py` 拼「幕后」一节、取更新记录、读写标记；`brain_prompt` 多一个 `backstage` 参数，非空时整节替换「## 身份」。`introspect` 是只读工具：眼睛一项在 `ToolBox` 里读 `Eyes.latest`，其余经 `Body.introspect(topic)` 在身体线程读内心层。开关关掉时提示词、工具列表和现在逐字一样。

**Tech Stack:** Python 3.13、pytest、mcp（FastMCP）、git 命令行（subprocess）。

**Spec:** `docs/superpowers/specs/2026-10-01-backstage-design.md`

## Global Constraints

- `[backstage] enabled` 默认 `false`；`changelog_max = 10`、`changelog_days = 7`。
- `enabled = false`：`brain_prompt` 输出逐字不变；MCP 工具列表 == 现在的 `TOOL_NAMES`（没有 `introspect`）。
- 「## 底线」一节、`clean_reply` 的 `_CLAIMS_HUMAN` 不动。
- 主人名一律取 `cfg.brain.owner_name`；为空时提示词没有主人那一档、也没有"做出你的人是…"。
- 更新记录：`--no-merges`；只要 `feat|fix|perf`；scope 为 `console` / `viewer` 的不要；git 超时 2 秒；失败返回空并记 WARNING。
- 标记 `<记忆目录>/inner/backstage.json` = `{"seen": "<完整哈希>"}`，只在 live 写（原子写：先写 `.tmp` 再 `replace`）。
- 测试命令：`.venv\Scripts\python.exe -m pytest -q`（系统 Python 没装 pytest 时）。
- 用中文写注释和提示词，照周围代码的风格。

## Review Focus

1. 中文提交标题里有冒号、括号、`!`（`feat(brain)!: …`）——正则要认对 scope，不能把标题截坏。→ Task 2 `test_changelog_parses_bang_and_chinese`
2. `seen` 是 rebase 后不存在的哈希 / 文件是坏 JSON —— 退回按天，不能抛异常。→ Task 2 `test_changelog_unknown_seen_falls_back_to_days`、`test_read_seen_bad_json`
3. `owner_name` 为空 —— 不能出现空的"做出你的人是。"。→ Task 3 `test_section_without_owner`
4. 内心层关着（`ledger is None`）时调 `introspect` —— 回"没开"，不能 `AttributeError`。→ Task 4 `test_introspect_inner_off`
5. 更新记录里提交标题含 `{` `}` —— 拼提示词不能用 `str.format` 二次格式化。→ Task 3 `test_section_keeps_braces_in_changelog`

---

### Task 1: 精力逐项明细 `energy_parts`

**Files:**
- Modify: `src/skydango/inner/energy.py`
- Test: `tests/test_inner_energy.py`

**Interfaces:**
- Produces: `energy_parts(hour, awake_min, cheered, busy_min) -> list[tuple[int, str]]`：第一项是 `(基础分, 原因或"白天")`，之后每项是 `(加减分, 原因)`（扣分为负数，只列非 0 项）；`energy()` 用它求和、夹到 0~100，结果和改前一样。`format_parts(parts, e: Energy) -> str` 返回 "基础 100（白天）− 20（连着挂了 2 个多小时）+ 10（有人陪着聊）= 90，精神"。

- [ ] **Step 1: 写失败的测试**

```python
from skydango.inner.energy import energy, energy_parts, format_parts

def test_energy_parts_sum_matches_energy():
    for args in [(14, 0, False, 0), (1, 130, True, 0), (23, 200, False, 90), (5, 600, True, 80)]:
        parts = energy_parts(*args)
        assert max(0, min(100, sum(p for p, _ in parts))) == energy(*args).score

def test_format_parts():
    parts = energy_parts(14, 130, True, 0)
    text = format_parts(parts, energy(14, 130, True, 0))
    assert text == "基础 100（白天）− 20（连着挂了 2 个多小时）+ 10（有人陪着聊）= 90，精神"
```

- [ ] **Step 2: 跑测试确认失败**（`ImportError: energy_parts`）
- [ ] **Step 3: 实现** `energy_parts` / `format_parts`，`energy()` 改成用 `energy_parts` 求和；原因文字沿用 `_BANDS` 和现有的"连着挂了 N 个多小时""有人陪着聊""闹了好一阵"。分数被夹过（>100 或 <0）时 `format_parts` 末尾照 `e.score` 写。
- [ ] **Step 4: 跑 `tests/test_inner_energy.py` 全部通过**（原有测试也不能挂）
- [ ] **Step 5: 提交** `feat(inner): 精力逐项明细（给 introspect 用）`

---

### Task 2: 更新记录和标记（`brain/backstage.py` 前半）

**Files:**
- Create: `src/skydango/brain/backstage.py`
- Test: `tests/test_brain_backstage.py`

**Interfaces:**
- Produces:
  - `Runner = Callable[[list[str]], str]`；`git_runner(repo: Path) -> Runner`：`subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, encoding="utf-8", timeout=2, check=True).stdout`
  - `repo_root() -> Path | None`：从 `skydango` 包目录往上找含 `.git` 的目录（`.git` 是文件也算：worktree）
  - `read_seen(path: Path) -> str | None`（不存在 / 坏 JSON / 没有 `seen` → `None`）；`write_seen(path: Path, sha: str) -> None`（原子写，建父目录）
  - `changelog(run: Runner, seen: str | None, now: float, max_n: int = 10, days: int = 7) -> tuple[list[str], str | None]`：返回 (给提示词的行, HEAD 哈希)；任何异常 → `([], None)` 并 `log.warning`
- 取法：`git rev-parse HEAD`；`seen` 非空时 `git merge-base --is-ancestor <seen> HEAD`（非 0 退出 → `CalledProcessError` → 退回按天）；
  `git log --no-merges --format=%H%x1f%ct%x1f%s <seen>..HEAD` 或 `--since=<days>.days`。标题正则 `^(feat|fix|perf)(\(([^)]*)\))?!?:\s*`，scope 在 `{"console", "viewer"}` 的丢掉。
- 行格式：`f"- {when} {原标题}"`，`when` = 今天 / 昨天 / N 天前（按 `now` 和 `%ct` 的本地日期差）；超出 `max_n` 时最后加 `f"- 还有 {n} 条小改动"`。

- [ ] **Step 1: 写失败的测试**（`tmp_path` 里真 `git init`，设 `user.name/email`，用 `GIT_COMMITTER_DATE` / `GIT_AUTHOR_DATE` 造时间）

```python
def test_changelog_filters(tmp_path): # 造 feat(brain)、fix、docs、style、feat(console)、fix(viewer)、perf、合并提交
    lines, head = changelog(git_runner(repo), None, now)
    assert [l.split(" ", 2)[2] for l in lines] == ["perf: 快一点", "fix: 修好了", "feat(brain): 新本事"]  # 新到旧
    assert head == rev_parse_head(repo)

def test_changelog_since_seen(tmp_path):        # seen = 第 2 个提交 → 只有它之后的
def test_changelog_unknown_seen_falls_back_to_days(tmp_path):  # seen = "0"*40 → 按 7 天；8 天前的提交不在
def test_changelog_limit(tmp_path):             # 13 个 feat，max_n=10 → 10 行 + "- 还有 3 条小改动"
def test_changelog_parses_bang_and_chinese(tmp_path):  # "feat(inner)!: 心情：更别扭（试试）" 原样保留
def test_changelog_when_labels(tmp_path):       # 今天 / 昨天 / 3 天前
def test_changelog_runner_failure():
    def boom(args): raise subprocess.TimeoutExpired("git", 2)
    assert changelog(boom, None, 0.0) == ([], None)
def test_read_seen_bad_json(tmp_path):          # 写 "{坏" → None；不存在 → None
def test_write_then_read_seen(tmp_path):        # 父目录不存在也能写；没有残留 .tmp
```

- [ ] **Step 2: 跑测试确认失败**（模块不存在）
- [ ] **Step 3: 实现上面的接口**
- [ ] **Step 4: 跑 `tests/test_brain_backstage.py` 通过**
- [ ] **Step 5: 提交** `feat(brain): 幕后的更新记录（git 提交）和标记文件`

---

### Task 3: 「幕后」一节 + 配置 + 替换「身份」

**Files:**
- Modify: `src/skydango/config.py`（`BackstageConfig`、`Config.backstage`）、`config.example.toml`（`[backstage]` 一节，放在 `[inner]` 之后）
- Modify: `src/skydango/brain/backstage.py`（加 `section`）
- Modify: `src/skydango/brain/prompt.py`
- Test: `tests/test_brain_backstage.py`、`tests/test_brain_prompt.py`

**Interfaces:**
- Consumes: Task 2 的行列表。
- Produces:
  - `@dataclass class BackstageConfig: enabled: bool = False; changelog_max: int = 10; changelog_days: int = 7`
  - `section(owner: str, brain_model: str, eyes_model: str, reflect_model: str, changelog: list[str]) -> str`：返回以 `## 幕后` 开头的整节（文字逐字按 spec §1 代码块，`{owner}` 等用字符串拼接 / `replace` 代入，**不对整段用 `str.format`**）；`owner == ""` 时删掉"做出你的人是…"那半句和整条主人档、知情好友档里"这个你问{owner}"换成"这个你别问了"、"{owner}能看、能改、能清空"换成"能被看、被改、被清空"、沙盒那条换成"沙盒是假世界：里面的人是冒充的，你在里面看不见东西"、最后一条"跟你聊幕后"那句整条删掉；`changelog` 非空时末尾接 spec §3 的 `### …改了你什么（…）` 小标题和各行（owner 为空时小标题用"上次以来你被改了什么"）。
  - `brain_prompt(..., backstage: str = "")`：非空时在 `static_prompt` 之后，用正则 `r"## 身份\n.*?(?=\n\n## )"`（DOTALL）把「## 身份」整节换成 `backstage`（用函数作替换值，避免反斜杠转义）。

- [ ] **Step 1: 写失败的测试**

```python
def test_section_with_owner():
    s = section("卡洛", "sonnet", "haiku", "sonnet", [])
    assert s.startswith("## 幕后") and "做出你的人是卡洛" in s and "Claude（sonnet）" in s and "（haiku）" in s
    assert "知道团子是 AI" in s and "introspect" in s and "改了你什么" not in s

def test_section_without_owner():
    s = section("", "sonnet", "haiku", "sonnet", ["- 昨天 feat: x"])
    assert "做出你的人是" not in s and "（聊天里名字一字不差）" not in s and "{owner}" not in s and "上次以来你被改了什么" in s

def test_section_keeps_braces_in_changelog():
    assert "- 今天 feat: {x}" in section("卡洛", "s", "h", "s", ["- 今天 feat: {x}"])

def test_brain_prompt_backstage_replaces_identity():   # tests/test_brain_prompt.py
    off = brain_prompt(ReplyConfig(), None)
    on = brain_prompt(ReplyConfig(), None, backstage=section("卡洛", "sonnet", "haiku", "sonnet", []))
    assert "## 身份" in off and "## 身份" not in on and "## 幕后" in on
    底线 = lambda p: p.split("## 底线")[1].split("\n## ")[0]
    assert 底线(on) == 底线(off)
    assert on.index("## 主动开口") < on.index("## 幕后") < on.index("## 底线")

def test_brain_prompt_without_backstage_unchanged():
    assert brain_prompt(ReplyConfig(), None, backstage="") == brain_prompt(ReplyConfig(), None)

def test_backstage_config_defaults(): # load_config 读 [backstage] enabled = true；默认 False / 10 / 7
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** `BackstageConfig`、`section`、`brain_prompt` 的替换；`config.example.toml` 加注释过的 `[backstage]`
- [ ] **Step 4: 跑 `tests/test_brain_backstage.py tests/test_brain_prompt.py tests/test_config*.py` 通过**
- [ ] **Step 5: 提交** `feat(brain): 「幕后」一节替换「身份」（[backstage] enabled）`

---

### Task 4: `introspect` 工具

**Files:**
- Modify: `src/skydango/brain/body.py`（`Body.introspect`）、`src/skydango/inner/reflect.py`（`Reflector.next_in`）、`src/skydango/brain/tools.py`、`src/skydango/brain/mcp_server.py`
- Test: `tests/test_brain_tools.py`、`tests/test_brain_mcp.py`、`tests/test_brain_body*.py`（找现有建 `Body` 的测试辅助函数）

**Interfaces:**
- Consumes: Task 1 `energy_parts` / `format_parts`。
- Produces:
  - `INTROSPECT_TOPICS = ("精力", "反思", "性格", "日记", "眼睛")`、`INTROSPECT_DESCRIPTION`（spec §2 的工具说明原文）放在 `tools.py`；**不进** `DESCRIPTIONS` / `TOOL_NAMES` / `ACTIONS`。
  - `ToolBox(..., backstage: bool = False)`；`_exec("introspect", {"topic": …})`：`backstage` 为假 → `ToolError("没有这个工具")`；topic 不在五个里 → `ToolError("topic 只能是 精力 / 反思 / 性格 / 日记 / 眼睛")`；`眼睛` 在 ToolBox 里读 `self.eyes.latest`（`None` → "眼睛还没看过"；否则 `f"眼睛（{eyes_model}）{秒数:.0f} 秒前写的：\n{text}"`）；其余 `b.call(lambda: b.introspect(topic))`。沙盒也给（不进 `SANDBOX_MISSING`）。
  - `Body.introspect(topic: str) -> str`（身体线程）：
    - `精力`：`format_parts(energy_parts(...同 energy_now 的参数...), self.energy_now())`——把 `energy_now` 里算参数的部分抽成 `_energy_args() -> tuple[float, float, bool, float]` 两处共用。
    - `反思`：`self.reflector is None` → "反思没开（卡洛没打开这个）"；`mind_log.recent()` 里 `kind == "reflect"` 的最后 3 条：`多久前 · 下线那次? · 心情 level（text）· 改了：changes · 没收下：dropped`；末尾 `self.reflector.next_in(wall)` 有值就 "下次大概 N 分钟后（有动静才反思）"。没有记录 → "这次上线还没反思过"。
    - `性格`：`self.persona is None` → "性格没开（卡洛没打开这个）"；三类各列 `text`（老梗带 `who`、看法带 `topic`）、`hits` 次、`last_used` 多久前、离淡出还有几天（`cfg.inner.fade_days` 减去距 `max(last_used, since)` 的天数，下限 0）；全空 → "还没攒下什么"。
    - `日记`：`self.ledger is None or not cfg.inner.reflect` → "日记没开（卡洛没打开这个）"；`self.ledger.store.last_diaries(1)`，截 600 字；空 → "还没写过日记"。
    - 内心层整个关着（`ledger is None`）时 `精力` 照样能算（`energy_now` 本来就有退路）。
    - 主人名用 `cfg.brain.owner_name or "卡洛"` 填"（…没打开这个）"。
  - `Reflector.next_in(now: float) -> float | None`：`_final` 或 `_running` → `None`；否则 `max(0, _last_run + cfg.reflect_every - now)`。
  - `build_server`：`if toolbox.backstage:` 在最后注册 `@srv.tool(name="introspect", description=INTROSPECT_DESCRIPTION) def introspect(topic: str)`。

- [ ] **Step 1: 写失败的测试**

```python
def test_introspect_hidden_when_off():         # test_brain_mcp：ToolBox(FakeBody(), FakeEyes()) → names == TOOL_NAMES
def test_introspect_listed_when_on():          # backstage=True → names == TOOL_NAMES + ["introspect"]；调 ("introspect", {"topic": "精力"}) 不出错
def test_introspect_bad_topic():               # ToolBox(..., backstage=True).run("introspect", {"topic": "代码"}) → 出错、文字含 "精力 / 反思"
def test_introspect_eyes():                    # FakeEyes.latest = ("一棵树", now-30) → 含 "30 秒前" 和 "一棵树"；latest=None → "还没看过"
def test_introspect_not_action():              # 调完 tb.acted is False
def test_introspect_inner_off():               # Body(ledger=None, mind=None, reflector=None, persona=None)：反思 / 性格 / 日记 → 各含 "没开"；精力 → 含 "= "
def test_introspect_reflect_rows():            # MindLog 里记两次 reflect（changes=["心情：平常→开心"]）→ 两条都在、新的在后、含 "心情：平常→开心"
def test_introspect_persona():                 # 一个口头禅 hits=3、一个老梗 who="小明" → 都列出、含 "小明" 和 "3 次"
def test_introspect_diary_truncated(tmp_path): # 写 1000 字日记 → 返回 ≤ 600 字正文
def test_reflector_next_in():                  # last_run=0, reflect_every=1200, now=200 → 1000；_final → None
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** 上面的接口
- [ ] **Step 4: 跑 `tests/test_brain_tools.py tests/test_brain_mcp.py` 和 body / reflect 相关测试通过**
- [ ] **Step 5: 提交** `feat(brain): introspect 工具（查精力 / 反思 / 性格 / 日记 / 眼睛）`

---

### Task 5: 接线、管理面板开关、文档

**Files:**
- Modify: `src/skydango/cli.py`（`_run_brain` 和新的 `_backstage_prompt`）
- Modify: `src/skydango/console/settings.py`（`Field("backstage.enabled", …)`）
- Modify: `CLAUDE.md`
- Test: `tests/test_cli_backstage.py`（新）、`tests/test_console_settings.py`

**Interfaces:**
- Consumes: Task 2 `repo_root` / `git_runner` / `changelog` / `read_seen` / `write_seen`，Task 3 `section` / `brain_prompt(backstage=)`，Task 4 `ToolBox(backstage=)`。
- Produces: `_backstage_prompt(cfg: Config, store, now: float, run: Runner | None = None, repo: Path | None = None) -> str`：
  `not cfg.backstage.enabled` → `""`；标记路径 = `store.dir / "inner" / "backstage.json"`（`store is None` 时不读不写、`seen=None`）；`changelog(...)` 之后 `not cfg.reply.dry_run and head and store is not None` 才 `write_seen`；返回 `section(cfg.brain.owner_name, cfg.brain.model, cfg.brain.eyes_model, cfg.inner.reflect_model, lines)`；拼出错 → `log.exception`、返回 `""`（和 `_persona_prompt` 同样的写法）。`repo_root()` 为 `None` → 更新记录为空。
- `_run_brain`：`ToolBox(..., backstage=cfg.backstage.enabled)`；`brain_prompt(..., backstage=_backstage_prompt(cfg, store, wall()))`。
- 设置项：`Field("backstage.enabled", "幕后", "团子知道自己是 AI、卡洛做了她，能跟卡洛聊她自己怎么运作；好友要在 friends.md 里写「知道团子是 AI」才会跟他承认", "bool", "brain")`，放在 `inner.persona` 后面。

- [ ] **Step 1: 写失败的测试**

```python
def test_backstage_prompt_off():        # enabled=False → ""，假 runner 没被调
def test_backstage_prompt_live_writes_seen(tmp_path):  # 假 runner 回 HEAD="abc…"、一个 feat；dry_run=False → 返回含 "## 幕后" 和那条；backstage.json 的 seen == HEAD
def test_backstage_prompt_dry_run_does_not_write(tmp_path)
def test_backstage_prompt_no_store():   # store=None → 照样返回「幕后」，不写文件
def test_settings_has_backstage_toggle():  # test_console_settings：清单里有 "backstage.enabled"，类型 bool
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** `_backstage_prompt`、两处接线、设置项
- [ ] **Step 4: 跑全部测试** `.venv\Scripts\python.exe -m pytest -q`，全部通过
- [ ] **Step 5: 更新 CLAUDE.md**：代码结构表加 `brain/backstage.py`；新加「## 幕后（`[backstage]`，大脑模式）」一节（spec 链接、三档、`introspect`、更新记录和标记文件、默认关、friends.md 写"知道团子是 AI"、**还没用真 Claude 验证**、spec §7 五步）；记忆表 `inner/` 一行加 `backstage.json`。
- [ ] **Step 6: 提交** `feat(brain): 幕后接进大脑和管理面板；CLAUDE.md 加「幕后」`
