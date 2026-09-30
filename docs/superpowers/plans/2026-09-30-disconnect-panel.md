# 掉线弹框只报告、按钮认全 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 游戏掉线弹出「连接错误」时，团子认得出这是掉线弹框、两个按钮都认全、一个都不按，只告诉大脑和日志"掉线了，等卡洛处理"；顺带把 09-30 晚按 Q 的实测写进 game-ops，把反思回答坏掉时的原文留在日志里。

**Architecture:** 按钮词表加「重试」让 OCR 细读把它认成按钮；新加一张只靠文字认的特征卡 `disconnect`（通用兜底 OCR 读到"连接错误 / 网络连接失败"就认成它），卡上 `never = ["重试", "取消"]`、没有关法、`auto = false`，于是身体不自动关、大脑的 `panel_press` / `panel_close` 都按不下去；身体在这张卡打开时多记一条 WARNING、给大脑的事件多一句话。

**Tech Stack:** Python 3.13、pytest（合成画面 + 假 OCR，`tests/panels_helpers.py`）、TOML 特征卡

**Spec:** 没有单独的 spec，需求来自 2026-09-30 晚真机（用户定的：「重试」只报告、不自动点）：
- 21:56 游戏弹出「连接错误：网络连接失败。（错误码：140）请确认您的网络连接状态」，按钮 取消（灰）/ 重试（蓝）
- 当时细读结果（`runs/20260930-214438-live-brain/agent.log`）：标题「连接错误」，正文「网络连接失败。 (错误码：140) 请确认您的网络连接状态 重试」，按钮只有 `[1] 取消（可以按）` —— **「重试」被当成了正文**（不在 `button_words` 里）
- 大脑按了三次「取消」都"画面没变"，身体也用撤退类按钮试关了一次；「取消」按下去会怎样不知道（可能回主界面），所以这张弹框上两个按钮都不按
- 那一帧没留下截图（`tmp/now.png` 后来被覆盖了），测试用上面日志里的 OCR 原文造假 OCR 行

## Global Constraints

- 跑测试：在 worktree 里 `python -m pytest -q`（`tests/conftest.py` 会把 worktree 的 `src` 放在最前面）；**跑 CLI 要加 `PYTHONPATH=src`**，否则用的是主目录的代码
- 注释、日志、给大脑的文字都用中文，照周围代码的密度和口气
- 提交信息格式同仓库：`fix(panels): …` / `docs(game-ops): …`，末尾带 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- 不改 `classify()` 的优先级（never > retreat > allow > other），不改别的卡
- 不往游戏里发输入；这个计划全部离线可测

## Review Focus

1. OCR 漏读灰色的「取消」（只剩 标题 + 正文 + 「重试」）→ 仍然认成面板、认成 `disconnect` 卡 —— Task 1、Task 2 各有测试
2. 别的弹框上的「重试」（不是掉线）→ 类别 `other`（要卡洛 `#允许`），不是 `never` —— Task 1 测试
3. 句子里带"重试"的长正文（「网络连接断开，请重试」）不能变成按钮 —— Task 1 测试
4. 大脑对掉线弹框调 `panel_close` → 一下都不点、返回关不掉 —— Task 2 测试
5. 掉线弹框消失（重连成功）→ 照常发"关了：…"事件（通用兜底已有逻辑，不另写测试）

---

### Task 1: 「重试」算按钮词

**Files:**
- Modify: `src/skydango/config.py:400-402`（`PanelsConfig.button_words` 末尾加 `"重试"`）
- Modify: `config.example.toml:336`（同一行同步加 `"重试"`）
- Test: `tests/test_panels_read.py`、`tests/test_panels_unknown.py`

**Interfaces:**
- Produces: `PanelsConfig().button_words` 里有 `"重试"`；`classify("重试", cfg) == "other"`

- [ ] **Step 1: 写失败的测试**

`tests/test_panels_read.py`（照文件里现有 `split_reading` / `classify` 的写法）：

```python
DISCONNECT_LINES = [  # 2026-09-30 真机掉线弹框当时的 OCR（整张图坐标，1920×1080）
    line("连接错误", 495, 447, 120, 36),
    line("网络连接失败。 (错误码：140)", 495, 518, 320, 30),
    line("请确认您的网络连接状态", 495, 546, 245, 30),
    line("取消", 1245, 609, 50, 30),
    line("重试", 1357, 609, 50, 30),
]

def test_retry_is_a_button():
    panel = Panel(UNKNOWN, "不认识的面板", Rect(460, 407, 1000, 265), False, 100)
    r = split_reading(DISCONNECT_LINES, panel, None, PanelsConfig(), 1920, 1080, 0.0)
    assert r.title == "连接错误"
    assert "重试" not in r.text
    assert [(b.text, b.kind) for b in r.buttons] == [("取消", "retreat"), ("重试", "other")]

def test_retry_inside_long_sentence_is_not_a_button():
    panel = Panel(UNKNOWN, "不认识的面板", Rect(0, 0, 1920, 1080), False, 100)
    r = split_reading([line("出错了", 300, 20), line("网络连接断开，请重试", 200, 90)], panel, None, PanelsConfig(), 1920, 1080, 0.0)
    assert r.buttons == ()
```

`tests/test_panels_unknown.py`：

```python
def test_looks_like_panel_with_only_retry():  # OCR 漏读灰色的「取消」
    lines = [line("连接错误", 300, 20), line("网络连接失败。 (错误码：140)", 200, 90), line("重试", 500, 300)]
    assert looks_like_panel(lines, PanelsConfig())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_panels_read.py tests/test_panels_unknown.py -q`
Expected: `test_retry_is_a_button`、`test_looks_like_panel_with_only_retry` FAIL（「重试」进了正文 / 不算面板）；`test_retry_inside_long_sentence_is_not_a_button` PASS（本来就对，留作防回归）

- [ ] **Step 3: 两处 `button_words` 末尾加 `"重试"`**

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_panels_read.py tests/test_panels_unknown.py tests/test_panels_cards.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py config.example.toml tests/test_panels_read.py tests/test_panels_unknown.py
git commit -m "fix(panels): 「重试」算按钮词——掉线弹框的重试被当成了正文"
```

---

### Task 2: 掉线弹框卡 `disconnect`：两个按钮都不按，只报告

**Files:**
- Create: `assets/panels/disconnect/card.toml`
- Modify: `src/skydango/vision/panels.py`（加常量 `DISCONNECT = "disconnect"`，放在 `UNKNOWN` / `CHAT` 旁边）
- Modify: `src/skydango/brain/body.py:750-766`（`_watch_panels`）
- Modify: `docs/game-ops.md` §7 面板表 + 录样本清单；`CLAUDE.md`（代码结构表"五张特征卡"→"六张"，「面板识别」一节同步，写明掉线弹框只报告）
- Test: `tests/test_panels_cards.py`、`tests/test_panels_unknown.py`、`tests/test_panels_ops.py`、`tests/test_brain_panels.py`

**Interfaces:**
- Consumes: Task 1 的 `"重试"` 按钮词
- Produces: `vision.panels.DISCONNECT == "disconnect"`；卡 `disconnect`：`label = "掉线弹框（连接错误）"`、`layer = 100`、`region = [0.15, 0.08, 0.85, 0.92]`、`[[features]] kind = "text"`、`any = ["连接错误", "网络连接失败"]`、`[buttons] never = ["重试", "取消"]`、`[close] ways = []`、`auto = false`、`verified = false`（文件头注释写：2026-09-30 真机见过、没截到图；取消按下去会怎样不知道，所以也不按）

- [ ] **Step 1: 写失败的测试**

`tests/test_panels_cards.py::test_real_cards_load`：期望集合加 `"disconnect"`，并断言

```python
assert cards["disconnect"].never == ("重试", "取消") and cards["disconnect"].close_ways == ()
assert not cards["disconnect"].close_auto and cards["disconnect"].layer == 100
```

`tests/test_panels_unknown.py`（照 `test_text_card_identified`）：

```python
DISCONNECT = [line("连接错误", 300, 20), line("网络连接失败。 (错误码：140)", 200, 90),
              line("请确认您的网络连接状态", 200, 130), line("取消", 400, 300), line("重试", 500, 300)]

def test_disconnect_card_identified(clock):
    w = watcher(ListOcr(DISCONNECT))
    w.observe(scene(), clock())
    [c] = w.pop_changes()
    assert c.panel.name == "disconnect" and c.panel.layer == 100
    assert [(b.text, b.kind) for b in c.reading.buttons] == [("取消", "never"), ("重试", "never")]

def test_disconnect_card_without_grey_cancel(clock):  # OCR 漏读灰色的「取消」
    w = watcher(ListOcr([ln for ln in DISCONNECT if ln.text != "取消"]))
    w.observe(scene(), clock())
    [c] = w.pop_changes()
    assert c.panel.name == "disconnect" and [(b.text, b.kind) for b in c.reading.buttons] == [("重试", "never")]
```

`tests/test_panels_ops.py`（照文件里现有的假设备 / watcher 写法）：`test_close_disconnect_taps_nothing` —— 对 `disconnect` 面板调 `PanelOps.close(panel, reading)`，返回 `False`，假设备一次 `tap` 都没收到。

`tests/test_brain_panels.py`（照文件里现有构造身体、喂面板变化的写法）：
- `test_disconnect_open_reports`：`_watch_panels` 收到 `disconnect` 打开 → 事件队列里一条 `panel` 事件，文字包含 `"游戏掉线了"` 和 `"等卡洛"`；`caplog` 里有一条 WARNING 包含 `"掉线"`
- `test_disconnect_buttons_not_pressed`：`panel_press("重试")` 和 `panel_press("取消")` 都抛 `ToolError`，文字包含 `"不能按"`；设备没收到 `tap`

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_panels_cards.py tests/test_panels_unknown.py tests/test_panels_ops.py tests/test_brain_panels.py -q`
Expected: 上面新加 / 改的测试 FAIL（没有 disconnect 卡、没有 WARNING）

- [ ] **Step 3: 写卡片 `assets/panels/disconnect/card.toml`（值见 Interfaces）**

- [ ] **Step 4: `vision/panels.py` 加 `DISCONNECT`；`Body._watch_panels` 在 `open` 且 `p.name == DISCONNECT` 时**
  - `log.warning("游戏掉线了（%s），要手动点「重试」", describe_reading(change.reading) if change.reading else "还没读")`
  - 事件文字 = 原来 `开了：…` 那句 + `"。游戏掉线了：弹框上的按钮都别按，等卡洛自己点重试"`

- [ ] **Step 5: 跑测试确认通过，再跑全量**

Run: `python -m pytest tests/test_panels_cards.py tests/test_panels_unknown.py tests/test_panels_ops.py tests/test_brain_panels.py -q`，然后 `python -m pytest -q`
Expected: 全 PASS

- [ ] **Step 6: 文档**
  - `docs/game-ops.md` §7 表加一行：`| 掉线弹框（连接错误） | 屏幕中间的白框 | 网络断了（2026-09-30 晚见过，错误码 140） | 只能玩家自己点「重试」 | disconnect（读到"连接错误 / 网络连接失败"） | 未核对；「重试」「取消」都不按，只报告 |`；录样本清单里"断线重连"那句改成已有卡、下次掉线截一张存进 `assets/panels/disconnect/` 旁边备查
  - `CLAUDE.md`：代码结构表"五张特征卡"→"六张"；「面板识别」一节"五张卡（…）"加上掉线弹框，并加一句"掉线弹框（`disconnect`）两个按钮都不按：身体记 WARNING、告诉大脑等卡洛处理"

- [ ] **Step 7: 提交**

```bash
git add assets/panels/disconnect src/skydango/vision/panels.py src/skydango/brain/body.py docs/game-ops.md CLAUDE.md tests/
git commit -m "feat(panels): 掉线弹框卡——认出连接错误，重试 / 取消都不按，只报告等卡洛处理"
```

---

### Task 3: game-ops 记下按 Q 的呼唤特效

**Files:**
- Modify: `docs/game-ops.md` §2（`| Q | …` 那一行之后加一小节）

- [ ] **Step 1: 加小节 `### 呼唤特效（Q，2026-09-30 晚实测，录像 tmp/record/q-call-20260930-b、-c）`**，要点：
  - 短按：头上先冒一个小亮圈（约 1.3 倍头宽）→ 0.1~0.2 s 长到约 2 倍、最亮 → 扩到约 3 倍变淡消失；整个 **0.25~0.45 s**，亮的阶段约 0.2 s。5 fps 截图会漏，要 30 fps（`record --fps 30` 实测能到 ~31 张 / 秒，但相邻截图常重复，游戏实际约 15~20 帧 / 秒）
  - 长按（大喊）：身体先蹲下再跳起张开双臂，配一个罩住半个画面的淡大球；太显眼
  - **别人按 Q 的特效一模一样** → 靠特效认团子只能对时间：团子按下后 ~1 s 内谁头上冒圈；两人同时按对不上，要隔随机时间再按一次
  - 聊天面板开 / 关时画面整体横移，比较前后帧要避开
- [ ] **Step 2: 提交**

```bash
git add docs/game-ops.md
git commit -m "docs(game-ops): 按 Q 呼唤特效实测——0.25~0.45 s、别人的一样、只能对按键时间"
```

---

### Task 4（可选，同晚另一个问题）：反思回答坏了时把原文留下

下线那次最终反思 `反思的回答不是 JSON`，日志只截了前 120 字（到 `"diary":` 为止），看不出坏在哪、当天日记也丢了。
猜是日记里用了英文双引号引别人的话，但没证据，这一步只做"留证据 + 提示词里说一句"，不做 JSON 修补。

**Files:**
- Modify: `src/skydango/inner/reflect.py:173-176`（`_ask`）和反思的系统提示词（同文件，写 JSON 格式说明那段）
- Test: `tests/` 里现有的反思测试文件（`grep -l "parse_reflection\|Reflector" tests/`）

- [ ] **Step 1: 写失败的测试**：假 llm 返回一段坏 JSON（`'{"mood": {"level": "开心", "text": "好"}, "diary": "他说"来找我"了"}'`）→ `_ask` 返回 `None`，`caplog` 里有一条 DEBUG 记录包含完整原文（含 `来找我`）；系统提示词包含 `「」`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：失败时 WARNING 照旧（前 120 字），另记一条 `log.debug("反思的原文：%s", raw)`；系统提示词的 JSON 格式说明后加一句"字符串里引用别人的话用「」，不要用英文双引号"
- [ ] **Step 4: 跑测试确认通过，再跑全量** `python -m pytest -q`
- [ ] **Step 5: 提交** `fix(inner): 反思回答不是 JSON 时把原文记进 DEBUG 日志；提示词要求引号用「」`
