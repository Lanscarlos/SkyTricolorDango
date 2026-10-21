# 聊天节奏像人 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全是短句的一批多等一会儿再交给大脑（`[pacing] short_quiet`），并加 `sandbox-ctl play` 按真实时间线回放来验收。

**Architecture:** `Pacer`（纯计算）放行时用「有效安静时长」：全是短句取 `max(原 quiet, short_quiet)`；身体建 `Pacer` 时把 `[proactive] self_names` 传进去认「叫团子」。`sandbox-ctl play` 解析回放文件、按偏移发 `/op say`、不等回完，最后复用 `wait_quiet` 等 idle。

**Tech Stack:** Python 3.13、pytest；沙盒 JSON 接口（`sandbox/server.py`）。

**Spec:** `docs/superpowers/specs/2026-10-21-chat-rhythm-design.md`

## Global Constraints

- `short_quiet: float = 8.0`，`0` = 关（逐字照旧）；短句长度和回法共用 `short_len`（6，去掉空白数）
- 短句 = 不超过 `short_len` 字 且 不是 `is_question` 且 不是 `sounds_upset` 且 不含 `self_names` 任何一个
- `max_wait`（15 秒）、urgent 马上放、打字从最后看到起重新算，这三条不变
- 回放文件一行 `<秒> <说话人>：<原文>`，全角 / 半角冒号都行，空行和 `#` 开头的行跳过；格式不对报行号、整个文件不发
- 回放文件有聊天原话：只放 `tmp/`，不进 git
- 项目约定：中文注释、注释密度和周围一样；测试用 `python -m pytest -q -p no:cacheprovider`；worktree 里跑 `python -m skydango` 要 `PYTHONPATH=src`

## Review Focus

1. 一批先是短句、后来进来一句长句 / 问句：回到原来的 quiet，不能还按 8 秒等（Task 1 测）
2. 短句里含 `self_names` 的另一个名字（「三彩」）或重复（「团子团子」）：不算短句，照原 quiet（Task 1 测）
3. 短句攒着时对方在打字：照旧从最后看到打字起重新算有效安静时长（Task 1 测）
4. 回放文件原文里还有冒号（「懒洋洋大王：我说：好」）、带 BOM / CRLF：只按第一个冒号拆、BOM 和 `\r` 不进说话人 / 原文（Task 2 测）
5. 回放中途沙盒拒绝某句（说话人不在身边）或偏移倒着走：前者报出那一句、停止、退出码 1；后者解析时就报行号（Task 2 测）

---

### Task 1: 短句多等一会儿（`Pacer` + 配置 + 身体接线 + 文档）

**Files:**
- Modify: `src/skydango/brain/pacing.py`（`Pacer.__init__`、`ready`、`describe`；加 `_short_only`、`_effective_quiet`）
- Modify: `src/skydango/config.py`（`PacingConfig` 加 `short_quiet`）
- Modify: `src/skydango/brain/body.py:198`（`Pacer(cfg.pacing, self.rng, cfg.proactive.self_names)`）
- Modify: `src/skydango/console/settings.py`（`pacing.manner` 后面加 `Field("pacing.short_quiet", …, "float", "brain")`，写法照同文件别的 float 字段）
- Modify: `config.example.toml`（`[pacing]` 一节加一行 `short_quiet` 说明）、`CLAUDE.md`（「聊天的节奏和分寸」攒话那条加一句）
- Test: `tests/test_pacing.py`、`tests/test_brain_pacing_body.py`

**Interfaces:**
- Produces: `Pacer(cfg: PacingConfig, rng: random.Random, self_names: Sequence[str] = ())`；`PacingConfig.short_quiet: float = 8.0`

- [ ] **Step 1: Write the failing tests**（`tests/test_pacing.py`，用现有 `pacer()` / `release_time()`；`pacer()` 加 `names=()` 透传给 `Pacer`）

```python
def test_short_only_waits_short_quiet():
    p = pacer(quiet=3, short_quiet=8)
    p.heard(0, [("小明", "嘻嘻")], False)
    assert release_time(p, 0) == 8.0

def test_long_or_question_line_goes_back_to_quiet():
    p = pacer(quiet=3, short_quiet=8)
    p.heard(0, [("小明", "哈哈")], False)
    p.heard(1, [("小明", "今天跑图累死我了真的")], False)
    assert release_time(p, 1) == 4.0
    q = pacer(quiet=3, short_quiet=8)
    q.heard(0, [("小明", "哈哈")], False)
    q.heard(1, [("小明", "去哪")], False)  # 问句：quiet_min = 3
    assert release_time(q, 1) == 4.0

def test_calling_dango_is_not_short():
    for text in ("团子", "三彩", "团子团子"):
        p = pacer(quiet=3, short_quiet=8, names=("团子", "三彩"))
        p.heard(0, [("小明", text)], False)
        assert release_time(p, 0) == 3.0, text

def test_short_quiet_typing_extends():
    p = pacer(quiet=3, short_quiet=8)
    p.heard(0, [("小明", "嗯")], False)
    assert not p.ready(5, {"小明"})
    assert release_time(p, 5) == 13.0

def test_short_quiet_capped_by_max_wait_and_urgent():
    p = pacer(quiet=3, short_quiet=20, max_wait=15)
    p.heard(0, [("小明", "嗯")], False)
    assert release_time(p, 0) == 15.0
    u = pacer(quiet=3, short_quiet=8)
    u.heard(0, [("小明", "难过")], True)
    assert u.ready(0, ())

def test_short_quiet_zero_is_old_behavior():
    p = pacer(quiet=3, short_quiet=0)
    p.heard(0, [("小明", "嗯")], False)
    assert release_time(p, 0) == 3.0

def test_describe_uses_effective_quiet():
    p = pacer(quiet=3, short_quiet=8)
    p.heard(0, [("小明", "嗯")], False)
    assert "再等 6 秒" in p.describe(2, ())
```

`tests/test_brain_pacing_body.py`：照该文件已有的身体构造方式，断言 `body.pacer` 认得 `cfg.proactive.self_names`——喂一句「团子」后按原 quiet 放（不是 `short_quiet`）。

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q -p no:cacheprovider tests/test_pacing.py tests/test_brain_pacing_body.py`
Expected: 新测试 FAIL（`PacingConfig` 没有 `short_quiet` / `Pacer` 不收 `self_names`）

- [ ] **Step 3: Implement**

`_short_only()`：批里每一句都满足 Global Constraints 的短句定义（长度用 `len("".join(text.split()))`，和 `manner.py` 一致）；`short_quiet <= 0` 时直接 False。`_effective_quiet()` = `max(self._quiet, cfg.short_quiet) if _short_only() else self._quiet`；`ready` 和 `describe` 里原来用 `self._quiet` 的地方换成它。`self._quiet` 本身的赋值（随机 / 问句取 `quiet_min`）不动。模块 docstring 补一句。

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest -q -p no:cacheprovider tests/test_pacing.py tests/test_brain_pacing_body.py tests/test_manner.py tests/test_console_settings.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/skydango/brain/pacing.py src/skydango/config.py src/skydango/brain/body.py src/skydango/console/settings.py config.example.toml CLAUDE.md tests/test_pacing.py tests/test_brain_pacing_body.py
git commit -m "feat(brain): 全是短句的一批多等一会儿再叫醒大脑（[pacing] short_quiet）"
```

---

### Task 2: `sandbox-ctl play` 按真实时间线回放

**Files:**
- Modify: `src/skydango/sandbox/ctl.py`（加 `parse_script`、`Ctl.play`；`configure` 加子命令；`run` 分派）
- Modify: `CLAUDE.md`（「大脑沙盒」命令行客户端那条、「常用命令」各加一行）
- Test: `tests/test_sandbox_ctl.py`

**Interfaces:**
- Produces: `parse_script(text: str) -> list[tuple[float, str, str]]`（(偏移秒, 说话人, 原文)，偏移从 0 起、不降；坏行抛 `Failed("第 N 行…")`）；`Ctl.play(entries: list[tuple[float, str, str]]) -> int`；子命令 `play <文件> [--timeout 秒]`（`--timeout` 默认 `OP_TIMEOUT`，从最后一句发出后算）

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_script_formats():
    text = "\ufeff# 10-07\n0 小明：嘻嘻\r\n\n2.5 小明:团子\n12 小明：我说：好\n"
    assert ctl.parse_script(text) == [(0.0, "小明", "嘻嘻"), (2.5, "小明", "团子"), (12.0, "小明", "我说：好")]

@pytest.mark.parametrize("bad, n", [("0 小明 嘻嘻", 1), ("x 小明：嘻嘻", 1), ("0 小明：嘻嘻\n3 ：空", 2),
                                    ("5 小明：a\n2 小明：b", 2), ("0 小明：", 1)])
def test_parse_script_bad_line(bad, n):
    with pytest.raises(ctl.Failed) as e:
        ctl.parse_script(bad)
    assert f"第 {n} 行" in e.value.text

def test_play_sends_on_time_without_waiting_then_waits_idle(run, tmp_path):
    f = tmp_path / "s.txt"; f.write_text("0 小明：嘻嘻\n0.4 小明：团子\n", encoding="utf-8")
    # 两句之间 /state 一直不 idle（团子在想）也照发；发完才等 idle
    # sandbox 路由：/op 两次都 ok；/state 先 not idle、后 idle；/brain 有一轮
    ...  # 照 test_say_posts_op_and_prints_reaction_until_idle 的 Fake 写法
    # 断言：两次 /op 依次是 say 小明 嘻嘻 / say 小明 团子；两次 /op 的间隔 ≥ 0.35 秒（Fake.calls 里记时间：给 Fake 加 times 列表）；
    #       输出里有「+0.0s 小明：嘻嘻」「+0.4s 小明：团子」、团子说的行、大脑的文字；退出码 0

def test_play_refused_line_stops(run, tmp_path):
    # 第二句 /op 回 {"ok": False, "text": "小明 不在身边"}：输出含「第 2 句」和「不在身边」，没发第三句，退出码 1

def test_play_bad_file_sends_nothing(run, tmp_path):
    # 文件有坏行：不调 /op，输出含「第 N 行」，退出码 1

def test_play_json(run, tmp_path):
    # --json：一个对象，键 sent（[{t, who, text}]）、lines、turns，ok 为 True
```

`...` 处照该文件现有 `run` / `Fake` / `state` / `row` / `brain` / `turn` 辅助函数写；给 `Fake.calls` 每条多记一个 `time.monotonic()`（加在元组末尾，改动现有解包的地方一起改）或另记 `fake.times`，二选一。

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q -p no:cacheprovider tests/test_sandbox_ctl.py -k "parse_script or play"`
Expected: FAIL（`ctl.parse_script` 不存在）

- [ ] **Step 3: Implement**

- `parse_script`：去 BOM、按行、`rstrip("\r")`；正则 `^\s*(\d+(?:\.\d+)?)\s+([^：:]+?)\s*[：:]\s*(.+?)\s*$`（只按第一个冒号拆）；偏移比上一句小、说话人或原文空都算坏行；文件一句都没有也报错。
- `Ctl.play`：先 `before = self.state(0, 0)`、`brain0 = self.brain(0)`；以 `time.monotonic()` 为起点，每句等到偏移（`time.sleep` 剩余时间）后 `POST /op {"op": "say", "who", "text"}`，打印 `+{偏移:.1f}s 说话人：原文`；某句 `ok` 为假 → 打印「第 N 句没发出去：<text>」、返回 1。发完调 `self.wait_quiet(before["version"], a.timeout, seen)`，之后打印大脑轮次和 `footer`，和 `op()` 的收尾一样（抽一个 `_finish(before, brain0, lines, quiet, extra: dict) -> int` 让 `op` 和 `play` 共用，`op` 的输出逐字不变）。
- `run`：`play` 读文件（UTF-8，读不到 → `Failed`），`parse_script`，再 `ctl.play`。

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest -q -p no:cacheprovider tests/test_sandbox_ctl.py`
Expected: PASS（含原有 `say` 等测试：`op` 输出没变）

- [ ] **Step 5: Commit**

```bash
git add src/skydango/sandbox/ctl.py tests/test_sandbox_ctl.py CLAUDE.md
git commit -m "feat(sandbox): sandbox-ctl play 按真实时间线回放，不等团子回完"
```

---

### Task 3: 按 10-07 时间线验收（不改产品代码）

**Files:**
- Create: `tmp/rhythm-1021/play-1007.txt`（不进 git）、`tmp/rhythm-1021/metrics.py`（一次性脚本，不进 git）
- Modify: `docs/progress/2026-10-05-tonight.md`（末尾加「10-21 聊天节奏回放」一节）

- [ ] **Step 1: 取时间线**：从 `D:/Lanscarlos/Develop/SkyTricolorDango/runs/20261007-220825-live-brain/agent.log` 的 INFO `跟谁说 懒洋洋大王「…」` 行取时间和原文，写成回放文件（第一句偏移 0）；同一句 3 秒内重复出现的只留一次。检查：行数约 40，`parse_script` 能读。
- [ ] **Step 2: 跑两遍**：主目录起管理面板（独立进程，见 10-21 会话做法）→ `sandbox-ctl start 20:30` → `come 懒洋洋大王` → `PYTHONPATH=src python -m skydango sandbox-ctl play tmp/rhythm-1021/play-1007.txt --timeout 300`；一遍 `console.toml` 里 `[pacing] short_quiet = 0`、一遍 `= 8`（改完重启沙盒，跑完恢复 console.toml 原样并核对）。注意沙盒子进程是主目录面板起的，要用 worktree 代码就在 worktree 里起面板（`PYTHONPATH=src`）。
- [ ] **Step 3: 算三个数**：`metrics.py` 读两次运行的 `agent.log`（「跟谁说」= 收到、`已发送` / 沙盒 transcript 的 said = 团子开口），输出：开口延迟中位数和 < 4 秒的条数（除问句 / 叫团子 / 难过）；间隔 ≤ 有效安静时长的连发被拆开回的次数；后面 8 秒内还有话的短句没并进下一批的次数。目标见 spec §4。
- [ ] **Step 4: 写进进度文档并提交**

```bash
git add docs/progress/2026-10-05-tonight.md
git commit -m "docs(progress): 10-21 聊天节奏按 10-07 时间线回放"
```

没达标：记下数字，回 spec 讨论（不在这个任务里调参数以外的东西）；只是 `short_quiet` 取值不合适可以在 `config.py` 改默认值、补测试、再跑一遍。
