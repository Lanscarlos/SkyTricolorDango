# 内心层可视化（管理面板「内心」页）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 加一份内心流水账（`mind_log.jsonl`：每次反思改了什么、精力曲线、删性格条目），管理面板加「内心」页看现在 / 曲线 / 反思记录 / 性格档案 / 关系卡 / 日子和日记，能删沉淀歪了的性格条目。

**Architecture:** 新纯数据模块 `inner/log.py`（`MindLog` + `diff`）；身体在反思前后拍快照记 `reflect`、每 300 秒记 `energy`，新增 `forget` / `inner_snapshot`；viewer 加 `/inner`、`/inner/forget`（处理逻辑放 `inner/api.py`，之后沙盒复用）；管理面板 `console/inner_view.py` 读 `memory/inner/` 文件、在跑时合并实时数据，页面加「内心」页。**所有读取都按"内心目录 + 实时取数函数"参数化**，下一份计划（大脑沙盒）直接换成 `sandbox/memory/inner/` 和 `/sandbox/inner`。

**Tech Stack:** Python 3.13 标准库；pytest；原生 JS + 手写 SVG（不引外部库）。

**Spec:** `docs/superpowers/specs/2026-09-30-inner-viewer-design.md`（依赖内心层第 3 期，已在 main）

## Global Constraints

- 流水账 `memory/inner/mind_log.jsonl`，一行一条；kind 只有 `reflect` / `energy` / `forget`；live 追加写盘，dry-run 只在内存里；内存最多 200 条
- `ENERGY_EVERY = 300`（秒，墙上时间）、`LOG_DAYS = 30`（启动时 `trim` 删 30 天前的行，live 时原子重写）
- `reflect` 字段：`t`、`final`、`mood {level, text}`、`energy {level, score}`、`grudge {who, why, until} | null`、`wants [text…]`、`changes [str…]`、`dropped [str…]`
- `forget` 字段：`t`、`kind`（`catchphrase` / `joke` / `opinion`）、`text`、`who` / `topic`
- `changes` 的文字格式照 spec §1 逐字：`心情 平常→开心（有人来聊天）`、`心情：…`、`新别扭：小明（放鸽子）`、`别扭撤了：小明`、`新心愿：…`、`心愿了结：…`、`新口头禅：…`、`新老梗：小明——…`、`新看法：雨林——…`、`看法换了：雨林 丑→其实还行`、`淡出：…`
- 流水账的时间一律用**身体的 `wall()`**（下一份计划的沙盒会换成模拟时钟）
- 只能改的是删性格条目；别扭 / 心愿 / 心情 / 关系卡只看
- POST 规矩：本机 Host（`is_local_host`）、`X-Skydango: 1`、JSON、≤ 64 KB（`post_guard`）；团子实时数据取不到时超时 3 秒
- 删除：团子 `running` → 转发；`idle` / `exited` / `crashed` → 直接改 `persona.json`（原子写）并记 `forget`；`starting` / `stopping` → `{"ok": false, "error": "团子正在启动 / 停止，稍等再删"}`
- 返回里的 `source`：`"live"` / `"files"` / `"files_fallback"`
- 出错只记日志，不影响团子；中文注释 / 日志；页面不引外部资源（现有测试查 `http://`）

## Review Focus

1. **反思结果和上一次完全一样**（模型只换了心情那句话）：`changes` 只有一行 `心情：…`，不刷出一堆"新心愿"（Task 1 测）
2. **`mind_log.jsonl` 里有半行坏数据**（上次被强杀写到一半）：`read` / `trim` 跳过坏行，页面照常（Task 1 测）
3. **在网页上删一条刚好在反思里被淡出的条目**：返回"找不到这条（可能已经淡出了）"、不抛异常（Task 2 测）
4. **团子刚停、runner 还是 `stopping`** 时点删除：拒绝并提示稍等，不同时改文件（Task 4 测）
5. **没开反思（`inner.reflect = false`）**：`/inner` 各字段 `null`、页面这几块写"（没开反思）"，关系卡 / 日记照常（Task 3、4 测）

---

### Task 1: 流水账 `MindLog` + `diff`

**Files:**
- Create: `src/skydango/inner/log.py`
- Test: `tests/test_inner_log.py`

**Interfaces:**
- Consumes: `inner.mind.Mind`（`mood.level/text`、`grudge`、`wants[].text`）、`inner.persona.Persona` / `Trait`
- Produces:
  - `ENERGY_EVERY = 300`、`LOG_DAYS = 30`、`MEMORY_MAX = 200`
  - `diff(before_mind: Mind, after_mind: Mind, before_persona: Persona | None, after_persona: Persona | None) -> list[str]`
  - `class MindLog(path: Path | None, persist: bool)`：
    - `reflect(t: float, final: bool, mind: Mind, energy, changes: list[str], dropped: list[str]) -> dict`（`energy` 是 `inner.energy.Energy | None`，记 `{level, score}`）
    - `energy(t: float, energy) -> dict`、`forget(t: float, kind: str, text: str, who: str = "", topic: str = "") -> dict`
    - `recent() -> list[dict]`（内存里的，旧到新）
    - `trim(now: float, days: int = LOG_DAYS) -> None`（live 时原子重写文件：写 `.tmp` 再 `replace`）
  - `read(path: Path, since: float) -> list[dict]`（模块函数；文件不在返回 `[]`，坏行跳过）

- [ ] **Step 1: 写失败的测试**

```python
from skydango.inner.log import MindLog, diff, read
from skydango.inner.mind import Grudge, Mind, Mood, Want
from skydango.inner.persona import Persona, Trait

def mind(level="平常", text="", grudge=None, wants=()):
    return Mind(mood=Mood(level, text), grudge=grudge, wants=[Want("想做", w) for w in wants])

def test_diff_mood_level_and_text_only():
    assert diff(mind("平常", "还行"), mind("开心", "有人来聊天"), None, None) == ["心情 平常→开心（有人来聊天）"]
    assert diff(mind("平常", "还行"), mind("平常", "有点无聊"), None, None) == ["心情：有点无聊"]
    assert diff(mind("平常", "还行"), mind("平常", "还行"), None, None) == []

def test_diff_grudge_and_wants():
    g = Grudge("小明", "放鸽子", 0, 100)
    assert diff(mind(), mind(grudge=g), None, None) == ["新别扭：小明（放鸽子）"]
    assert diff(mind(grudge=g), mind(), None, None) == ["别扭撤了：小明"]
    assert diff(mind(wants=["看日落"]), mind(wants=["听弹琴"]), None, None) == ["新心愿：听弹琴", "心愿了结：看日落"]

def test_diff_persona():
    before = Persona(catchphrases=[Trait("害")], opinions=[Trait("丑", topic="雨林")], jokes=[Trait("旧梗", who="阿花")])
    after = Persona(catchphrases=[Trait("害"), Trait("懒得动")], opinions=[Trait("其实还行", topic="雨林"), Trait("好看", topic="云野")],
                    jokes=[Trait("冥龙嘴里", who="小明")])
    assert diff(mind(), mind(), before, after) == [
        "新口头禅：懒得动", "新老梗：小明——冥龙嘴里", "新看法：云野——好看", "看法换了：雨林 丑→其实还行", "淡出：阿花——旧梗"]

def test_log_persist_and_dry_run(tmp_path):
    p = tmp_path / "mind_log.jsonl"
    live, dry = MindLog(p, persist=True), MindLog(tmp_path / "x.jsonl", persist=False)
    live.energy(10.0, None); dry.energy(10.0, None)
    assert len(read(p, 0)) == 1 and not (tmp_path / "x.jsonl").exists() and len(dry.recent()) == 1

def test_memory_cap_and_trim_and_bad_lines(tmp_path):
    p = tmp_path / "mind_log.jsonl"
    log = MindLog(p, persist=True)
    for i in range(210):
        log.forget(float(i), "catchphrase", "害")
    assert len(log.recent()) == 200
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"t": 5, "kind": "ene')             # 被强杀写到一半
    assert len(read(p, 0)) == 210                       # 坏行跳过
    log.trim(now=31 * 86400 + 100, days=30)            # 30 天前（t < 100）的删掉
    assert [r["t"] for r in read(p, 0)][0] == 100.0
```

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_inner_log.py -q` → ImportError

- [ ] **Step 3: 实现 `inner/log.py`**：`changes` 顺序固定为 心情 → 别扭 → 新心愿 → 心愿了结 → 新口头禅 → 新老梗 → 新看法 → 看法换了 → 淡出；心愿按 `text` 比；性格条目按（口头禅：`text`；老梗：`(who, text)`；看法：`topic`）比，看法同话题立场不同算"换了"；before 里有、after 里没有的都是"淡出"（老梗写 `who——text`，看法写 `topic——stance`）。`energy` 为 `None` 时记 `null`。写盘失败只 `log.warning`。

- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_inner_log.py -q` → 全过

- [ ] **Step 5: 提交**：`git add src/skydango/inner/log.py tests/test_inner_log.py && git commit -m "feat(inner): 内心流水账 mind_log 和反思前后的 diff"`

---

### Task 2: 身体接上流水账、`forget`、`inner_snapshot`；cli 接线

**Files:**
- Modify: `src/skydango/brain/body.py`（`__init__` 加 `mind_log=None`；`apply_reflection`、`_inner_tick`、新 `forget` / `inner_snapshot`）
- Modify: `src/skydango/inner/__init__.py`（`finish_reflection` 加参数 `mind_log=None, energy=None`）
- Modify: `src/skydango/cli.py`（`_run_brain` 建 `MindLog`、启动时 `trim`、传给 Body 和 `_final_reflection`）
- Modify: `src/skydango/inner/persona.py`（加 `remove`）
- Test: `tests/test_brain_inner_log_body.py`、`tests/test_inner_finish.py`（追加）、`tests/test_inner_persona.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `MindLog`、`diff`、`ENERGY_EVERY`
- Produces:
  - `Body(..., mind_log=None)`；`Body.mind_log`
  - `Body.forget(kind: str, text: str, who: str = "", topic: str = "") -> str`：空字符串 = 删成功；否则原因（`"找不到这条（可能已经淡出了）"`、`"性格档案没开"`、`"不认识的类别"`）。live（`ledger.persist`）时写 `persona.json`，都记 `forget`
  - `Body.inner_snapshot() -> dict`：`{"running": True, "at": wall, "mood": {level, text, since} | None, "energy": {level, score, note} | None, "grudge": {who, why, until} | None, "wants": [{kind, text, who, until}], "soft": [{who, text, until}], "persona": Persona.to_dict() | None, "log": mind_log.recent()}`；没开反思时 `mood` / `energy` / `grudge` / `persona` 为 `None`、`wants` / `soft` / `log` 为 `[]`
  - `Persona.remove(kind: str, text: str, who: str = "", topic: str = "") -> bool`（`kind` ∈ `catchphrase` / `joke` / `opinion`；看法按 `topic` 删）
  - `finish_reflection(..., mind_log=None, energy=None)`：套完之后记一条 `reflect(final=True)`

- [ ] **Step 1: 写失败的测试**

```python
# make(clock, tmp_path, live=False, reflect_reply="{}", **kw) 照 tests/test_brain_mind_body.py 复制过来，
# 多接 mind_log=MindLog(tmp_path / "mind_log.jsonl", persist=live)、persona=…、reflect=True（False 时 mind / reflector 为 None）
def test_reflection_logs_changes(clock, tmp_path):
    b = make(clock, tmp_path)
    b.apply_reflection({"mood": {"level": "开心", "text": "有人来聊天"}, "grudge": "keep"})
    r = b.mind_log.recent()[-1]
    assert r["kind"] == "reflect" and r["final"] is False and r["changes"] == ["心情 平常→开心（有人来聊天）"]

def test_energy_every_300s(clock, tmp_path):
    b = make(clock, tmp_path)
    for dt in (0, 299, 300):
        b.wall = lambda dt=dt: WALL + dt
        b._inner_tick()
    assert [r["kind"] for r in b.mind_log.recent()] == ["energy", "energy"]

def test_forget_live_writes_persona_and_logs(clock, tmp_path):
    b = make(clock, tmp_path, live=True, persona=Persona(catchphrases=[Trait("害")]))
    assert b.forget("catchphrase", "害") == ""
    assert b.persona.catchphrases == [] and json.loads((tmp_path / "persona.json").read_text("utf-8"))["catchphrases"] == []
    assert b.mind_log.recent()[-1]["kind"] == "forget"
    assert b.forget("catchphrase", "害") == "找不到这条（可能已经淡出了）"

def test_snapshot_without_reflect(clock, tmp_path):
    s = make(clock, tmp_path, reflect=False).inner_snapshot()
    assert s["mood"] is None and s["persona"] is None and s["log"] == [] and s["running"] is True
```

`tests/test_inner_finish.py` 追加：`finish_reflection(..., mind_log=log)` 之后 `log.recent()[-1]["final"] is True`。`tests/test_inner_persona.py` 追加：`remove` 三类各删一条、删不存在的返回 `False`。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_inner_log_body.py tests/test_inner_finish.py tests/test_inner_persona.py -q`

- [ ] **Step 3: 实现**
  - `apply_reflection`：套之前 `copy.deepcopy` 出 `Mind` 和 `Persona` 快照，`_apply_persona` 之后 `diff(...)`，`mind_log.reflect(self.wall(), False, self.mind, self._energy, changes, dropped)`；全程包在 `_inner_call` 里
  - `_inner_tick`：算完精力后，`self.mind_log` 不为空且 `wall - self._energy_logged >= ENERGY_EVERY`（初值 `-inf`）就记一条 `energy`
  - `forget` / `inner_snapshot`：只在身体线程里调（viewer 那边经 `body.call`）
  - cli：`reflector` 不为空时建 `MindLog(ledger.store.dir / "mind_log.jsonl", persist=ledger.persist)`，`trim(time.time())`，传给 `Body(mind_log=…)` 和 `_final_reflection`（`energy=body._energy`）

- [ ] **Step 4: 跑测试确认通过**：同 Step 2，再跑一次全量 `python -m pytest -q`

- [ ] **Step 5: 提交**：`git commit -m "feat(body): 反思 / 精力记进流水账，能删性格条目、取内心快照"`

---

### Task 3: viewer 的 `/inner`、`/inner/forget`（处理逻辑可复用）

**Files:**
- Create: `src/skydango/inner/api.py`
- Modify: `src/skydango/vision/viewer.py`（`Viewer.inner`、`Viewer.forget` 两个可调用属性 + 路由）、`src/skydango/cli.py`（`run --view` 时挂上）
- Test: `tests/test_inner_api.py`、`tests/test_viewer.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 `Body.inner_snapshot` / `Body.forget`、`Body.call(fn, timeout)`
- Produces:
  - `KINDS = ("catchphrase", "joke", "opinion")`
  - `parse_forget(req: dict) -> tuple[str, str, str, str]`（`kind, text, who, topic`；`kind` 不认识、`text` / `topic` 都空 → `ValueError("…")`）
  - `forget_result(reason: str) -> dict`：`""` → `{"ok": True}`，否则 `{"ok": False, "error": reason}`
  - `EMPTY_SNAPSHOT: dict`（没有身体时 `/inner` 返回的：`running` 为 `False`，其余同"没开反思"）
  - `Viewer.inner: Callable[[], dict] | None`、`Viewer.forget: Callable[[str, str, str, str], str] | None`（都是 `None` 时路由 404）
  - cli：`viewer.inner = lambda: body.call(body.inner_snapshot, timeout=3)`，`viewer.forget = lambda k, t, w, tp: body.call(lambda: body.forget(k, t, w, tp), timeout=3)`

- [ ] **Step 1: 写失败的测试**：`parse_forget` 的好 / 坏输入；viewer 起在随机端口（照 `tests/test_viewer.py` 现有写法），`GET /inner` 返回 `viewer.inner()` 的 JSON；`POST /inner/forget` 缺 `X-Skydango` → 403、非本机 Host → 403、坏 JSON → 400、`kind="xx"` → 400、成功 → `{"ok": true}`；`viewer.inner is None` → 404

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_inner_api.py tests/test_viewer.py -q`

- [ ] **Step 3: 实现**：路由套在现有 `do_GET` / `do_POST` 里，POST 走现有 `post_guard`；`body.call` 抛 `ToolError`（超时 / 身体已经停了）→ 503 `{"ok": false, "error": "团子正忙，稍后再试"}`

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: 提交**：`git commit -m "feat(viewer): /inner 取内心快照、/inner/forget 删性格条目"`

---

### Task 4: 管理面板读文件 + 合并实时（`console/inner_view.py`）

**Files:**
- Create: `src/skydango/console/inner_view.py`
- Modify: `src/skydango/console/server.py`（`GET /api/inner`、`POST /api/inner/forget`）
- Test: `tests/test_console_inner.py`

**Interfaces:**
- Consumes: `InnerStore`（`load_people`、`days`、`last_diaries`、`load_mind`、`load_persona`，都用 `quarantine=False`）、`recent_days`、Task 1 的 `read`、`MindLog`、Task 2 的 `Persona.remove`、Task 3 的 `parse_forget` / `forget_result` / `EMPTY_SNAPSHOT`
- Produces（**参数化，沙盒计划复用**）：
  - `inner_state(inner_dir: Path, friends: list[str], state: str, live: Callable[[], dict | None], now: float) -> dict`
    返回 `{"source", "saved_at", "now": {mood, energy, grudge, wants, soft}, "persona", "log", "cards": [{name, friend, first_met, days, visits, last_seen, said, to_me}], "days": [str], "diaries": [str]}`；
    `state` 是 runner 状态；`running` 时调 `live()`（拿不到 → `files_fallback`）；实时优先的字段：`now` 各项和 `persona`；`log` = 文件最近 7 天 + 实时的，按 `(t, kind)` 去重、按 `t` 排；`saved_at` = `mind.json` 的修改时间
  - `forget_offline(inner_dir: Path, req: dict, now: float) -> dict`：改 `persona.json`（原子写）+ 追加 `forget` 到 `mind_log.jsonl`
  - `ConsoleServer.inner(body_query) -> (code, dict)`、`ConsoleServer.forget(body) -> (code, dict)`：目录 = `Path(cfg.reply.memory_dir) / "inner"`（按面板读到的配置）；实时取数 = `proxy("GET", "inner", "")`，删除转发 `proxy("POST", "inner/forget", …)`

- [ ] **Step 1: 写失败的测试**（假 runner：`status()` 返回给定 state；假 proxy）
  - 没在跑：`source == "files"`，关系卡按"在好友名单里"在前、不在的在后；`persona.json` 不存在 → `persona is None`
  - 在跑：`now.mood` 取实时；实时和文件里有同一条 `(t, kind)` 的记录只出现一次
  - 在跑但 proxy 返回 503：`source == "files_fallback"`
  - 删除：`idle` 改文件 + 记 `forget`；`running` 转发；`starting` / `stopping` → `{"ok": False, "error": "团子正在启动 / 停止，稍等再删"}`
  - `mind_log.jsonl` 有坏行：照常返回

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_console_inner.py -q`

- [ ] **Step 3: 实现**：`/api/inner` 在 `do_GET` 的 Host 校验之后；`/api/inner/forget` 加进 `routes`

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: 提交**：`git commit -m "feat(console): /api/inner 读内心文件、在跑时合并实时，/api/inner/forget"`

---

### Task 5: 「内心」页

**Files:**
- Modify: `src/skydango/console/static/console.html`
- Test: `tests/test_console_server.py`（追加页面测试）

**Interfaces:**
- Consumes: Task 4 的 `GET api/inner`、`POST api/inner/forget`
- Produces（沙盒计划复用这些 JS 函数，放在页面脚本顶层、别的函数不依赖闭包）：
  - `renderInnerNow(container, now)`：第 1 块"现在"（心情色点、精力分数条、别扭、收着点、心愿标签）
  - `loadInner(source)`：`source` 先只有 `"dango"`；拼 `api/inner` 的地址；下一份计划加 `?source=sandbox`
  - 容器 id：`tab-inner`、`inner-source`、`inner-now`、`inner-curve`、`inner-log`、`inner-persona`、`inner-cards`、`inner-days`

- [ ] **Step 1: 写失败的测试**：导航里有 `data-tab="inner"`（在「实时画面」后面）；上面八个 id 都在；`test_page_script_parses` 照样过（脚本能被 node 解析，现有测试）

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_console_server.py -q`

- [ ] **Step 3: 用 `frontend-design` skill 做排版，照 spec §3 实现六块**：
  - 心情档位颜色：开心 `--sakura`、平常 `--rice`、低落 `#7aa2d6`、烦 `--warn`（色点和曲线背景色带共用）
  - 曲线：手写 SVG，精力分数折线（0~100）+ 心情档位背景色带；反思点小圆点，点了滚到时间线那一条；切换"最近 24 小时 / 7 天"
  - 反思记录：新的在上；`changes` 逐行、`dropped` 灰色折叠；`final` 标"下线"；"只看有改动的"开关
  - 性格档案：每条"用过 N 次 · 上次 X 前" + "删"（`confirm` 后 POST；`state` 是 `starting` / `stopping` 时按钮置灰）
  - 关系卡表格、日子和日记
  - 刷新：团子 `running` 时每 5 秒，其他时候打开时读一次 + "刷新"按钮；来源标签三种文字照 spec
  - 宽屏两栏、窄屏竖排；沿用现有 CSS 变量

- [ ] **Step 4: 跑测试确认通过**，再用内置浏览器开 `python -m skydango console --no-browser`（团子不在跑，读本机 `memory/inner/`），切到「内心」页截图核对六块都显示、没有控制台报错

- [ ] **Step 5: 提交**：`git commit -m "feat(console): 内心页（现在、曲线、反思记录、性格档案、关系卡、日记）"`

---

### Task 6: 文档

**Files:**
- Modify: `CLAUDE.md`（「管理面板」一节加「内心」页一行；「记忆」表 `inner/` 那行加 `mind_log.jsonl`；代码结构表 `inner/` 加 `log.py` `api.py`、`console/` 加 `inner_view.py`）
- Modify: `docs/superpowers/specs/2026-09-30-inner-viewer-design.md`（状态改成"代码已完成，待真机验证"）

- [ ] **Step 1: 改文档**
- [ ] **Step 2: 全量测试**：`python -m pytest -q` → 全过
- [ ] **Step 3: 提交并合进 main、推送**：`git commit -m "docs: 内心页"`，按 CLAUDE.md 合并进 main 并 `git push`
