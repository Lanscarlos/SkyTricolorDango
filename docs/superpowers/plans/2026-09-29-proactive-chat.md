# 看场合主动开口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 眼睛对比上次挑出新鲜事发 `notice` 事件、身体算"场合"给大脑看并对主动开口加护栏、提示词放开主动开口并带自己的喜好。

**Architecture:** 新增纯计算模块 `brain/occasion.py`（场合 + 护栏判定），身体（`brain/body.py`）在 `status` / `say` / 主循环里调它；
眼睛（`brain/eyes.py`）在自动看时多要一项"新鲜事"，经回调交给身体，身体在自己的线程里过滤后放 `notice` 事件；
提示词（`brain/prompt.py`）按 `[proactive] enabled` 换规则。全部由新配置节 `[proactive]` 控制，关掉 = 现在的行为。

**Tech Stack:** Python 3.11+、dataclasses、pytest（合成数据 + 假设备，不需要模拟器）

**Spec:** `docs/superpowers/specs/2026-09-29-proactive-chat-design.md`

## Global Constraints

- 回答、注释、日志、给大脑看的文字都用中文，风格照周围代码（短句、全角标点）
- `[proactive] enabled = false` 时行为和现在完全一样：旧提示词、眼睛不要新鲜事、不发 `notice`、status 没有场合、`say` 不拦
- 护栏只管**主动**开口：接话（大脑这一轮是被聊天 / 主人命令叫醒的，`brain_busy()` 为真）和手动控制（`live=True`）不拦、不计数
- 身份底线不动：`clean_reply` 的硬过滤和提示词"身份"一节原样保留
- 所有数字是估值，只做成配置，不在代码里写死
- 跑测试：`python -m pytest -q`（云端要先 `pip install -e ".[dev,brain,openai,anthropic]"`）
- 每个任务一个提交，提交信息末尾带 CLAUDE.md 要求的署名行；全部完成后合并进 main 并推送（CLAUDE.md「Agent 规则」）

## Review Focus

1. **聊天里好友名字被 OCR 认错一两个字**（"懒洋洋大玉"）→ 仍算好友说话（接话 / 热闹判定用 `similar(…, 0.75)` 模糊匹配）→ Task 2 测试
2. **眼睛的新鲜事写法不规整**（`**新鲜事**：无。`、分几行列条目、没有这一项）→ 无 / 空 / 解析不到都当没有，多行合成一条 → Task 4 测试
3. **环境没有 `place` / `strangers` 属性、或根本没开 env**（`FakeEnv`、`env=None`）→ 场合和换地图检测不报错，当成没好友 / 没地名 → Task 3、Task 5 测试
4. **冷场暂停期间主人用手动控制说话** → 不拦、不计入主动、不解除冷场 → Task 3 测试
5. **说话人看不出是谁（空字符串）** → 算"别人说话"（热闹计数），但不算"有好友接" → Task 2 测试

---

### Task 1: 配置 `[proactive]`

**Files:**
- Modify: `src/skydango/config.py`（新 `ProactiveConfig`，挂到 `Config.proactive`）
- Modify: `config.example.toml`（加 `[proactive]` 一节，带注释，值等于默认值）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ProactiveConfig` 字段（名字和默认值照抄，后面所有任务都用）：
  `enabled: bool = True`、`notice_min: float = 60.0`、`prev_max_age: float = 600.0`、`auto_look_busy: float = 60.0`、
  `busy_window: float = 180.0`、`busy_lines: int = 4`、`reply_window: float = 90.0`、`quota_window: float = 600.0`、
  `quota_busy: int = 4`、`quota_quiet: int = 2`、`min_gap: float = 60.0`、`cold_after: int = 3`、
  `self_names: list[str] = ["团子", "三彩"]`（聊天里出现这些字算"叫了你"；spec §2"话里带团子的名字"）
- Produces: `Config.proactive: ProactiveConfig`

- [ ] **Step 1: 写失败的测试** `test_proactive_section`：`load_config` 读一个只有 `[proactive]\nquota_busy = 6\n` 的文件 → `cfg.proactive.quota_busy == 6`、`cfg.proactive.enabled is True`、`cfg.proactive.min_gap == 60.0`；
  并在 `test_example_config_loads` 里加 `assert cfg.proactive.enabled is True and cfg.proactive.quota_quiet == 2`
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_config.py -q` → FAIL（未知配置项 proactive）
- [ ] **Step 3: 实现** `ProactiveConfig`（docstring 一句话说明 + 指向 spec），`Config` 加字段；`config.example.toml` 加这一节，每个键一行注释（照 spec 配置表的"作用"列）
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_config.py -q` → PASS
- [ ] **Step 5: 提交** `feat(config): [proactive] 看场合主动开口的配置`

---

### Task 2: 场合计算 `brain/occasion.py`

**Files:**
- Create: `src/skydango/brain/occasion.py`
- Test: `tests/test_brain_occasion.py`

**Interfaces:**
- Consumes: `ProactiveConfig`（Task 1）
- Produces:
  ```python
  @dataclass(frozen=True)
  class Spoken:
      t: float          # 墙钟时间（和 body.chat 同一个时钟）
      text: str
      proactive: bool

  @dataclass(frozen=True)
  class Occasion:
      level: str                 # "busy" 热闹 / "quiet" 安静 / "alone" 没熟人
      friends: tuple[str, ...]
      strangers: int
      others: int                # busy_window 内别人（说话人不是"我"）说了几句
      last: Spoken | None        # 上次主动开口
      last_reply: str            # "called" / "replied" / "waiting" / "none"；没有上次时 ""
      replier: str               # called / replied 时是谁
      recent: int                # quota_window 内主动说了几句
      recent_replied: int        # 其中几句是 called / replied
      left: int                  # 还能主动说几句（额度；没熟人 / 冷场时 0）
      blocked: str               # 现在不能主动开口的原因（给大脑看），"" = 可以
      def line(self, now: float) -> str

  def is_friend_fn(names: Sequence[str]) -> Callable[[str], bool]   # speaker 与任一名字 similar(…, 0.75)；空 speaker 永远 False
  def reply_state(cfg, spoken: Spoken, chat, now: float, is_friend) -> tuple[str, str]   # (状态, 谁)
  def assess(cfg: ProactiveConfig, now: float, friends: Sequence[str], strangers: int,
             chat: Sequence[tuple[float, str, str]], spoken: Sequence[Spoken],
             is_friend: Callable[[str], bool]) -> Occasion
  LEVEL_NAMES = {"busy": "热闹", "quiet": "安静", "alone": "没熟人"}
  ```
  `chat` 是 `body.chat` 的元素 `(墙钟时间, 说话人, 内容)`，团子自己是 `"我"`。

**判定规则（照 spec §2 / §4）：**
- `level`：没有好友 → alone；`others >= busy_lines` → busy；否则 quiet
- `reply_state`：`spoken.t < t <= spoken.t + reply_window` 内好友说的话；有一句含 `self_names` 任一 → `("called", 谁)`，否则有好友说话 → `("replied", 第一个人)`；
  都没有：`now < spoken.t + reply_window` → `("waiting", "")`，否则 `("none", "")`
- 冷场：最近 `cold_after` 句主动的话（不足这么多句就不算）都是 `"none"`，且最后一句主动之后没有任何好友说话
- `blocked` 按顺序取第一个成立的（文案照抄）：
  1. 没熟人：`身边没有好友，不主动开口`
  2. 冷场：`连着 {cold_after} 句主动的话都没人接，等有好友说话再主动开口`
  3. 额度（按当前档位 busy→`quota_busy`、quiet→`quota_quiet`，`recent >= 上限`）：`最近 {quota_window/60:g} 分钟已经主动说了 {recent} 句，{秒} 秒后才能再主动开口`（秒 = 窗口内最早那句 + `quota_window` − now，向上取整）
  4. 间隔（距上次主动 < `min_gap`）：`刚主动说过，{秒} 秒后才能再主动开口`
- `left`：alone 或冷场 → 0；否则 `max(0, 上限 − recent)`（间隔不影响 `left`）

**`line(now)` 的格式**（各段用 ` / ` 连，照抄）：
- 档位段：`热闹（身边 小明、阿花；最近 3 分钟别人说了 7 句）`、`安静（身边 小明；最近 3 分钟别人说了 1 句）`；没熟人：`没熟人（身边只有 2 个陌生人）` / `没熟人（身边没人）`（分钟数 = `busy_window/60:g`）
- 上次段：没有 → `还没主动开过口`；有 → `上次主动开口 {多久}前「{text}」` + `，{谁}叫了你` / `，{谁}接了话` / `，还在等人接` / `，没人接`（多久：< 60 秒写 `N 秒`，否则 `N 分钟`，向下取整）
- 统计段：`最近 {quota_window/60:g} 分钟主动说了 {recent} 句，{recent_replied} 句有人接`
- 末段：`blocked` 非空 → `这会儿不能主动开口：{blocked}`；否则 `这会儿还能主动说 {left} 句`

- [ ] **Step 1: 写失败的测试**（`cfg = ProactiveConfig()`，`now = 10_000.0`，`is_friend = is_friend_fn(["懒洋洋大王", "阿花"])`）：
  - `test_levels`：friends=() → `"alone"`；friends=("阿花",) 且 180 秒内别人 3 句 → `"quiet"`；4 句 → `"busy"`；181 秒前的那句不算
  - `test_fuzzy_friend_and_unknown_speaker`：说话人 `"懒洋洋大玉"` → `is_friend` 为真；`""` → 假；`""` 说的话计入 `others`，但不让上次主动变成 `"replied"`（Review Focus 1、5）
  - `test_reply_state`：主动那句后 30 秒阿花说"哈哈" → `("replied", "阿花")`；说"团子你好" → `("called", "阿花")`；60 秒内没人说且 now 在窗口内 → `("waiting", "")`；过了 90 秒 → `("none", "")`；窗口外（100 秒后）才说 → `"none"`
  - `test_alone_blocks`：friends=() → `left == 0`、`blocked == "身边没有好友，不主动开口"`
  - `test_quota_by_level`：quiet，600 秒内主动 2 句（最早在 now−500）→ `left == 0`、`blocked == "最近 10 分钟已经主动说了 2 句，100 秒后才能再主动开口"`；同样数据 busy → `left == 2`、`blocked == ""`（最后一句在 now−70）
  - `test_min_gap`：quiet，只有一句主动在 now−20 → `left == 1`、`blocked == "刚主动说过，40 秒后才能再主动开口"`
  - `test_cold_and_recovery`：3 句主动都 `"none"` → `blocked` 以"连着 3 句"开头、`left == 0`；最后一句之后阿花说了一句（窗口外）→ 不再冷场；只有 2 句 none → 不冷场；非主动的话不参与
  - `test_line`：构造 busy + 上次 replied → `line(now) == "热闹（身边 懒洋洋大王、阿花；最近 3 分钟别人说了 7 句）/ 上次主动开口 4 分钟前「这图好黑」，阿花接了话 / 最近 10 分钟主动说了 2 句，1 句有人接 / 这会儿还能主动说 2 句"`；alone + 陌生人 2 + 没开过口 → 以 `"没熟人（身边只有 2 个陌生人）/ 还没主动开过口"` 开头、以 `"这会儿不能主动开口：身边没有好友，不主动开口"` 结尾
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_occasion.py -q` → FAIL（ModuleNotFoundError）
- [ ] **Step 3: 实现** `occasion.py`（模块 docstring 说明"纯计算、不碰设备"；`is_friend_fn` 用 `chat.tracker.similar`）
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_brain_occasion.py -q` → PASS
- [ ] **Step 5: 提交** `feat(brain): 场合计算（热闹 / 安静 / 没熟人、有没有人接、主动开口额度）`

---

### Task 3: 身体接场合：状态、护栏、网页和面板卡片

**Files:**
- Modify: `src/skydango/brain/body.py`（`__init__`、`status`、`say`、`_show`，新方法 `occasion`）
- Modify: `src/skydango/console/static/console.html`（`CARDS` 在"刚说过"后加 `"场合"`）
- Modify: `src/skydango/console/settings.py`（`FIELDS` 在 `brain.memory_model` 后加 5 项，group `"brain"`）
- Test: `tests/test_brain_body.py`、`tests/test_console_settings.py`

**Interfaces:**
- Consumes: `assess`、`is_friend_fn`、`Spoken`、`Occasion`、`LEVEL_NAMES`（Task 2）；`cfg.proactive`（Task 1）
- Produces:
  - `Body.friend_names: Callable[[], list[str]]`（默认 `lambda: []`，cli 在 Task 5 里设）
  - `Body.spoken: deque[Spoken]`（maxlen 50；大脑 `say` 成功后记，含 dry-run；`live=True` 不记）
  - `Body.occasion() -> Occasion`：`friends = env.nearby(clock())`（env 为 None → 空），`strangers = env.strangers(clock())`（没有这个方法 → 0），时间用 `self.wall()`
- 设置项（照抄）：
  - `Field("proactive.enabled", "看场合主动开口", "关掉就回到只接话、不主动的老样子", "bool", "brain")`
  - `Field("proactive.quota_busy", "热闹时主动额度", "10 分钟里最多主动说几句（好友在身边、聊得热闹）", "int", "brain")`
  - `Field("proactive.quota_quiet", "安静时主动额度", "10 分钟里最多主动说几句（好友在身边、没怎么说话）", "int", "brain")`
  - `Field("proactive.min_gap", "主动开口间隔（秒）", "两句主动的话之间至少隔多久", "float", "brain")`
  - `Field("proactive.auto_look_busy", "好友在身边时多久看一次（秒）", "眼睛（Haiku）最久多久看一次画面；越短越费订阅额度", "float", "brain")`

**行为：**
- `say`：`clean_reply` 和 `limiter.allow` 之后、`clear_view` 之前：`proactive = cfg.proactive.enabled and not live and not self.brain_busy()`；
  主动且 `occasion().blocked` 非空 → `raise ToolError(blocked)`（不记 limiter、不进 said）。成功后（dry-run 和真发都算）`live=False` 时 `spoken.append(Spoken(self.wall(), full, proactive))`
- `status`：`enabled` 时在"刚说过"那段之后加 `"场合：" + occasion().line(self.wall())`
- `_show`：`enabled` 时 `info["场合"] = f"{LEVEL_NAMES[o.level]} · " + ("不主动" if o.blocked else f"还能主动说 {o.left} 句")`

- [ ] **Step 1: 写失败的测试**（`body()` 帮手要能换墙钟：把 `wall=lambda: 1_790_000_000.0` 改成 `kw.setdefault("wall", …)` 再传；下面的测试传 `wall=clock`，`env = FakeEnv(); env.near = ["阿花"]`，`b.friend_names = lambda: ["阿花"]`）：
  - `test_proactive_say_blocked_when_alone`：`env.near = []`，`b.say("好无聊")` → `ToolError`，消息 `"身边没有好友，不主动开口"`；`b.said == []`
  - `test_reply_turn_not_limited`：`b.brain_busy = lambda: True`，`env.near = []` → `say` 成功，`b.spoken[-1].proactive is False`
  - `test_min_gap_and_recovery`：好友在身边，`say` 一次成功；`clock.advance(10)` 再 `say` → `ToolError` 含 `"刚主动说过"`；`clock.advance(55)` → 成功（`cfg.reply.min_interval` 设 0 免得旧限速挡住）
  - `test_cold_pause_then_friend_speaks`：`cfg.proactive.min_gap = 0`、`quota_quiet = 10`；主动 3 句、每句后 `clock.advance(100)`（没人说话）→ 第 4 句 `ToolError` 含 `"连着 3 句"`；`reader.batches = [[msg("嗯？", speaker="阿花")]]; b.step()` → 第 4 句成功
  - `test_manual_say_ignores_guard`（Review Focus 4）：冷场状态下 `b.say("主人让说的", live=True)` 成功，`len(b.spoken)` 不变，之后大脑主动 `say` 仍被冷场拦
  - `test_dry_run_counts`：dry-run 下主动说 2 句（quiet，`min_gap = 0`，每句之间 `clock.advance(5)` 躲开 `reply.min_interval` 的 3 秒）→ 第 3 句被额度拦
  - `test_status_has_occasion`：`"场合：安静（身边 阿花" in b.status()`；`env=None` 的身体 `status()` 含 `"场合：没熟人（身边没人）"`（Review Focus 3）
  - `test_disabled_keeps_old_behavior`：`cfg.proactive.enabled = False`，`env.near = []` → `say` 成功；`"场合" not in b.status()`
  - `test_console_settings.py::test_every_spec_field_is_listed`：列表在 `"brain.memory_model"` 后插入 `"proactive.enabled", "proactive.quota_busy", "proactive.quota_quiet", "proactive.min_gap", "proactive.auto_look_busy"`
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_body.py tests/test_console_settings.py -q` → 新测试 FAIL
- [ ] **Step 3: 实现** body 的改动、`FIELDS`、`CARDS`
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_brain_body.py tests/test_console_settings.py tests/test_console_server.py -q` → PASS
- [ ] **Step 5: 提交** `feat(brain): 身体算场合写进状态，主动开口过护栏；网页和管理面板显示场合`

---

### Task 4: 眼睛挑新鲜事

**Files:**
- Modify: `src/skydango/brain/eyes.py`
- Test: `tests/test_brain_eyes.py`

**Interfaces:**
- Consumes: `ProactiveConfig`（Task 1）
- Produces:
  - `Eyes.__init__` 新增关键字参数：`proactive: ProactiveConfig | None = None`（None 或 `enabled=False` = 旧行为）、
    `busy: Callable[[float], bool] = lambda now: False`（好友在不在身边）、`on_news: Callable[[str], None] = lambda text: None`
  - `NEWS_ITEM: str`、`LOOK_REQUEST_NEWS: str`、`parse_news(text: str) -> str`（没有新鲜事返回 `""`）
  - `describe_frame(frame, now, news: bool = False) -> str`（签名只加一个参数，大脑的 `look` 照旧不传）

**文案（照抄）：**
```python
NEWS_ITEM = """新鲜事：和上次比，有没有值得跟朋友提一句的变化？最多两条，一条一句；没有或拿不准就写“无”。
  值得提：天黑了 / 下雨了、到了新地方；好友换了斗篷 / 发型；好友在做有意思的事（弹琴、坐下、睡着冒 Z、放烟花、跳舞）；出现显眼的东西（篝火、先祖、冥龙、彩虹）
  不值得：镜头角度变了、人挪了几步、陌生人来来去去"""
LOOK_REQUEST_NEWS = LOOK_REQUEST.replace("按这四项写", "按这五项写") + "\n" + NEWS_ITEM
```
上一份描述放在请求前面：`f"上次（{秒:.0f} 秒前）看到的：\n{上一份}\n\n"`。

**行为：**
- `due`：`proactive` 开着且 `busy(now)` → 用 `auto_look_busy` 代替 `cfg.auto_look_max`
- `tick` 调 `describe_frame(frame, now, news=True)`；只有**这次真的要了第五项**（开着、有上一份、上一份不超过 `prev_max_age`）才 `parse_news`，非空就 `on_news(结果)`；`on_news` 抛异常只记日志
- `parse_news`：找 `新鲜事`（允许前面有 `**`、`- `，后面 `：` 或 `:`，允许 `**` 包着）那一行，取冒号后的内容 + 后面紧跟的非空行（去掉行首 `-`、`•`、`1.` 这类编号），用 `；` 连起来；
  去掉首尾空白和句末 `。`；是 `无` / `没有` / 空 → `""`；最长 80 字

- [ ] **Step 1: 写失败的测试**（`eyes()` 帮手加 `proactive=None, busy=False` 参数，`Describer` 支持按次返回指定文字）：
  - `test_parse_news`：`"…\n新鲜事：天黑了"` → `"天黑了"`；`"**新鲜事**：无。"` → `""`；`"新鲜事：\n- 天黑了\n- 小明坐下弹琴了"` → `"天黑了；小明坐下弹琴了"`；没有这一项 → `""`；`"新鲜事: 没有"` → `""`（Review Focus 2）
  - `test_first_look_has_no_news_item`：开着 proactive，第一次 `tick` 的请求文字不含 `"新鲜事"`，`on_news` 没被调
  - `test_second_look_compares_and_reports`：第二次 `tick`（+200 秒）请求含 `"上次（200 秒前）看到的：\n描述1"` 和 `"按这五项写"`；Describer 返回 `"…\n新鲜事：下雨了"` → `on_news` 收到 `["下雨了"]`
  - `test_old_description_not_compared`：两次间隔 700 秒 → 请求不含 `"新鲜事"`，不调 `on_news`
  - `test_brain_look_never_reports`：`describe_frame(frame(), now)`（不传 news）即使返回含新鲜事也不调 `on_news`，请求不含 `"新鲜事"`
  - `test_busy_looks_more_often`：`busy=True` → `tick(now+61)` 为真；`busy=False` → 为假（181 才真）
  - `test_disabled_is_old_behavior`：`proactive=ProactiveConfig(enabled=False)` → 第二次请求不含 `"新鲜事"`，`busy=True` 也不提前看
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_eyes.py -q` → 新测试 FAIL
- [ ] **Step 3: 实现** eyes 的改动
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_brain_eyes.py -q` → PASS
- [ ] **Step 5: 提交** `feat(brain): 眼睛对比上次描述挑新鲜事；好友在身边时看得勤一点`

---

### Task 5: 新鲜事变 `notice` 事件 + cli 接线

**Files:**
- Modify: `src/skydango/brain/body.py`（`news`、`_watch_news`，在 `_sense` 末尾调）
- Modify: `src/skydango/brain/events.py`（`Event.kind` 注释里加 `notice`）
- Modify: `src/skydango/cli.py`（`cmd_run` 大脑分支：`body.friend_names`、`Eyes(...)` 新参数）
- Test: `tests/test_brain_body.py`、`tests/test_cli_brain.py`（若已有构造 Eyes / Body 的接线测试就加断言，否则只跑一遍）

**Interfaces:**
- Consumes: `Body.occasion()`（Task 3）、`Eyes(on_news=…, busy=…, proactive=…)`（Task 4）
- Produces: `Body.news(text: str) -> None`（线程安全，只入队，眼睛线程调）；事件 `notice`

**行为：**
- `_watch_news(now)`（身体线程，只在 `cfg.proactive.enabled` 时）：
  - 候选 1：队列里眼睛给的每条 → 文字 `"眼睛注意到：" + text`
  - 候选 2：`place = getattr(self.env, "place", "")`（env 为 None → `""`）；非空时和上次见到的非空地名比，不同且上次非空 → `"看起来到了" + place`；然后记下这个非空地名。空值不更新、不发
  - 每个候选：`occasion()` 是 alone 或 `left == 0` → 丢（`log.debug`）；距上个 `notice`（`self.clock()`）< `notice_min` → 丢；和上一条发出的 `similar(…, 0.9)` → 丢；否则 `events.put("notice", 文字)` 并记时间、内容
- cli：`body.friend_names = _friend_names(cfg)`；`Eyes(..., proactive=cfg.proactive, busy=(lambda now: bool(env.nearby(now))) if env else (lambda now: False), on_news=body.news)`

- [ ] **Step 1: 写失败的测试**（好友在身边：`env.near = ["阿花"]`、`b.friend_names = lambda: ["阿花"]`）：
  - `test_news_becomes_notice`：`b.news("天黑了"); b.step()` → 事件 `("notice", "眼睛注意到：天黑了")`
  - `test_notice_min_and_duplicate`：连着两条，第二条在 60 秒内 → 只有一个事件；`clock.advance(61)` 再给同样的"天黑了" → 仍不发（内容相同）；给"下雨了" → 发
  - `test_notice_dropped_when_alone_or_no_quota`：`env.near = []` → 不发；好友在但主动额度用完（quiet 下主动说满 2 句）→ 不发
  - `test_place_change`：`env.place` 依次 `""` → `"云野"` → `""` → `"雨林"`，每次 `step()` → 只在最后一次发 `("notice", "看起来到了雨林")`；FakeEnv 没有 `place` 属性时不报错（Review Focus 3）
  - `test_notice_off_when_disabled`：`enabled = False` → `news` + `step()` 不发事件
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_body.py -q -k "news or notice or place"` → FAIL
- [ ] **Step 3: 实现** body 的 `news` / `_watch_news`、events 注释、cli 接线
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_brain_body.py tests/test_cli_brain.py tests/test_brain_eyes.py -q` → PASS
- [ ] **Step 5: 提交** `feat(brain): 眼睛的新鲜事和换地图变成 notice 事件（没熟人 / 额度用完不发）`

---

### Task 6: 提示词、人设模板、随手记

**Files:**
- Modify: `src/skydango/brain/prompt.py`（`PROACTIVE_RULES`、`static_prompt`、`brain_prompt`）
- Modify: `src/skydango/cli.py`（`PROFILE_TEMPLATE`；`brain_prompt(..., proactive=cfg.proactive.enabled)`）
- Modify: `src/skydango/chat/memory.py`（`MEMO_SYSTEM` 一处措辞）
- Test: `tests/test_brain_prompt.py`、`tests/test_memory.py`、`tests/test_cli_brain.py`（或 `memory init` 所在的测试文件）

**Interfaces:**
- Produces: `static_prompt(reply: ReplyConfig, proactive: bool = True) -> str`、`brain_prompt(..., proactive: bool = True)`、`PROACTIVE_RULES: str`

**文案（照抄）：**
- `proactive=True` 时把 `BRAIN_RULES` 里的 `没人理你的时候别自言自语。` 换成 `要不要主动开口，看下面“主动开口”一节。`，并在 `## 身份` 前插入：
```python
PROACTIVE_RULES = """## 主动开口
- 状态里的“场合”告诉你现在是什么场合：
  - 热闹（好友在身边、大家在聊）：可以接梗、插一句吐槽、对眼前的事说说看法；别人聊得正起劲时别硬转话题。
  - 安静（好友在身边、没怎么说话）：偶尔抛个话头就够了 —— 眼前的新鲜事、好友的近况（记忆里有的）、以前聊过的事（先 recall 确认）。
  - 没熟人（只有陌生人或没人）：不主动开口。
- 主动说的话要带你自己的看法，别播报：不说“天黑了”，说“一到晚上这图就阴森森的”；不说“小明在弹琴”，说“小明这首弹得还挺像样”。喜好照你的人设来，前后一致。
- 身体会把眼睛注意到的新鲜事（“眼睛注意到：…”“看起来到了…”）告诉你。这不是任务：大多数不用说，只挑你真有想法的。
- 分寸：上次主动开口没人接，就收一收，过一阵再说，别连着抛话题；一次只主动说一句；别用问句硬找话。
- 不开口也可以用动作：好友弹琴可以鼓掌，被夸了可以害羞 —— 动作不占聊天。
- 想了但决定不说，就在心里写一句“不说：原因”。
- 主动开口被身体拦下（说得太勤、没人接、身边没好友），照它说的等，别换个说法再试。"""
```
- `PROFILE_TEMPLATE` 末尾加 spec 附录 A 的"## 喜好和看法"一节（四条原样）
- `MEMO_SYSTEM`：`以及“我”新说出口的关于自己的设定（免得以后说法不一致）` → `以及“我”新说出口的关于自己的设定、喜好和评价（比如“我说雨林太湿，不喜欢待”；免得以后说法不一致）`

- [ ] **Step 1: 写失败的测试**：
  - `test_proactive_rules`：`static_prompt(ReplyConfig())` 含 `"## 主动开口"`、`"不说：原因"`、`"眼睛注意到"`，不含 `"没人理你的时候别自言自语"`；`"## 主动开口"` 在 `"## 身份"` 之前；身份底线断言（`不要说“我是真人”`、`老实承认是 AI`）仍成立
  - `test_proactive_off_keeps_old_rules`：`static_prompt(ReplyConfig(), proactive=False)` 含 `"没人理你的时候别自言自语"`、不含 `"## 主动开口"`；`brain_prompt(ReplyConfig(), None, proactive=False)` 同样
  - `test_profile_template_has_likes`（放 `tests/test_cli_brain.py`）：`cli.PROFILE_TEMPLATE` 含 `"## 喜好和看法"` 和 `"樱花发型天下第一"`；`cfg.reply.memory_dir = str(tmp_path)` 后 `cli.cmd_memory(cfg, argparse.Namespace(action="init"))`，生成的 `profile.md` 里也有
  - `test_memo_system_records_opinions`：`"喜好和评价" in MEMO_SYSTEM`
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_prompt.py tests/test_memory.py -q` → 新测试 FAIL
- [ ] **Step 3: 实现** 以上改动（`brain_prompt` 把 `proactive` 传给 `static_prompt`；cli 传 `cfg.proactive.enabled`）
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_brain_prompt.py tests/test_memory.py tests/test_cli_brain.py -q` → PASS
- [ ] **Step 5: 提交** `feat(brain): 提示词放开看场合主动开口；人设模板加喜好和看法，随手记记下团子的评价`

---

### Task 7: 文档、全量测试、合并

**Files:**
- Modify: `CLAUDE.md`（代码结构表加 `brain/occasion.py`；新增一节「看场合主动开口（`[proactive]`）」，要点：眼睛挑新鲜事 → `notice`、场合三档、护栏四条、总开关、`profile.md` 加"喜好和看法"要用户自己复制、**还没在真机上跑过**）
- Modify: `docs/superpowers/specs/2026-09-29-proactive-chat-design.md`（状态改成"代码已完成，待真机验证"，指向本计划）

- [ ] **Step 1: 写文档**
- [ ] **Step 2: 全量测试**：`python -m pytest -q` → 全部 PASS（基线 1062 passed + 6 skipped + 新增）
- [ ] **Step 3: 提交** `docs: 看场合主动开口`
- [ ] **Step 4: 推送功能分支**：`git push -u origin ccr-d1066432-4ftyad`
- [ ] **Step 5: 合并进 main 并推送**（CLAUDE.md 规矩）：`git fetch origin main && git checkout main && git merge --no-ff ccr-d1066432-4ftyad && python -m pytest -q && git push origin main`，再切回功能分支
