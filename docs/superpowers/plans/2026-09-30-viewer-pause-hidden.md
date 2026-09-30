# 识别可视化：看不见时暂停取画面 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** viewer 网页在标签页后台、或在管理面板里不在「实时画面」页时停止拉 `/snapshot`（服务端因此不压 JPEG），重新看得见时立刻接上。

**Architecture:** 只改前端。viewer 页面脚本用纯函数 `pullWhy()` 把三个暂停原因（按钮 / `document.hidden` / 父页面消息）合成现有的 `paused`；管理面板在切页签和 iframe 加载完时 `postMessage` 告诉 iframe 自己是否显示。服务端不改。

**Tech Stack:** 页面内联 JS（`vision/viewer.py` 的 `PAGE`、`console/static/console.html`），pytest + node 跑脚本片段。

**Spec:** `docs/superpowers/specs/2026-09-30-viewer-pause-hidden-design.md`

## Global Constraints

- 状态栏文字逐字：`已暂停`、`页面在后台，暂停取画面`、`不在实时画面页，暂停取画面`
- 消息格式：`{type:"skydango-view", shown:<布尔>}`；发送 targetOrigin = `location.origin`；接收只认 `e.source===window.parent && e.origin===location.origin`
- viewer 默认拉（收不到消息时 `frameShown = true`）
- 大脑时间线（`/brain`）、手动控制不暂停；服务端 `viewer.py` 的 Python 部分不改
- 不引外部资源、不用 `innerHTML` 放数据（现有页面测试会查）
- 跑测试：`python -m pytest -q`（没有 pytest 的机器用 `.venv\Scripts\python.exe -m pytest -q`）

## Review Focus

1. 后台自动暂停时用户又点了「暂停」、再切回前台：应该仍然暂停，按钮显示「继续」（`userPaused` 独立于另外两个原因）→ 真值表测试覆盖
2. 暂停 → 恢复很快（长轮询还在等）：不能起两个 `loop()` 并发拉 → 靠现有 `running` 防重入，Task 3 浏览器核对里快速切两次看请求不翻倍
3. 团子在用户停在「总览」时启动：iframe 加载完就要收到 `shown:false` → Task 2 测试查 iframe `load` 调 `tellLive`
4. 别的来源的 `message`（浏览器扩展、别的 iframe）不能改暂停状态 → Task 1 测试查来源检查
5. 直接打开 viewer（不在 iframe 里）时 `window.parent === window`：自己给自己发消息能生效，这正好用于 Task 3 的离线核对；不发消息时照常拉

---

### Task 1: viewer 页面按三个原因暂停

**Files:**
- Modify: `src/skydango/vision/viewer.py`（`PAGE` 脚本：约 471 行的变量、473 行的暂停按钮、493 行的 `loop()` 附近）
- Modify: `CLAUDE.md`（「识别可视化」一节加一行）
- Test: `tests/test_viewer.py`

**Interfaces:**
- Produces: 页面脚本里的 `function pullWhy(userPaused, docHidden, frameShown)`（**单独一行、从行首开始**，测试按行抽出来给 node 跑）；接收消息 `{type:"skydango-view", shown}`

- [ ] **Step 1: 写失败的测试**

```python
def test_pull_why_truth_table():
    # spec §1：按钮 > 后台 > 不在实时画面页；都不成立返回 ""（该拉）
    [fn] = [line for line in PAGE.splitlines() if line.startswith("function pullWhy(")]
    out = _node(fn + """
console.log(JSON.stringify([pullWhy(false,false,true),pullWhy(true,false,true),pullWhy(false,true,true),
  pullWhy(false,false,false),pullWhy(true,true,false),pullWhy(false,true,false)]));""")
    assert json.loads(out) == ["", "已暂停", "页面在后台，暂停取画面", "不在实时画面页，暂停取画面", "已暂停", "页面在后台，暂停取画面"]


def test_page_pauses_when_hidden_or_told_by_parent():
    script = PAGE.replace(" ", "")
    assert 'addEventListener("visibilitychange"' in script
    assert 'addEventListener("message"' in script
    assert "e.source===window.parent" in script and "e.origin===location.origin" in script  # Review Focus 4
    assert '"skydango-view"' in script
```

（`_node` 在本文件里定义在后面，模块级函数调用时已定义，放在 `test_viewer_page_script_parses` 附近即可。）

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_viewer.py -q -k "pull_why or pauses_when_hidden"`
Expected: 2 failed（`ValueError: not enough values to unpack` / 断言失败）

- [ ] **Step 3: 实现**

在 `PAGE` 脚本里：
- 变量：保留 `paused`，新增 `userPaused=false, docHidden=document.hidden, frameShown=true`
- `function pullWhy(userPaused,docHidden,frameShown)` 一行写完，返回值见 Global Constraints
- `function sync()`：`const why=pullWhy(...)`；`paused=!!why`；有原因 `status(why,true)`，没有就 `loop()`
- 暂停按钮：只翻转 `userPaused`，按钮文字跟 `userPaused`（`继续` / `暂停`），然后 `sync()`
- `document.addEventListener("visibilitychange",()=>{docHidden=document.hidden;sync()})`
- `window.addEventListener("message",e=>{…})`：来源检查 + `e.data&&e.data.type==="skydango-view"` 后 `frameShown=!!e.data.shown;sync()`
- 页面末尾原来的 `loop();` 改成 `sync();`（开页时标签页就在后台的情况也不拉）

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_viewer.py -q`
Expected: 全部通过（含 `test_viewer_page_script_parses`、不用 innerHTML 的检查）

- [ ] **Step 5: 更新 CLAUDE.md**

「识别可视化」一节"页面上能暂停、隐藏框、存图"之后加：标签页在后台、或在管理面板里不在「实时画面」页时自动暂停取画面（不再压 JPEG），大脑时间线照常。

- [ ] **Step 6: 提交**

```bash
git add src/skydango/vision/viewer.py tests/test_viewer.py CLAUDE.md
git commit -m "feat(viewer): 标签页在后台或管理面板不在实时画面页时暂停取画面"
```

### Task 2: 管理面板告诉 iframe 在不在显示

**Files:**
- Modify: `src/skydango/console/static/console.html`（`showTab()` 约 515 行；`render()` 里加载 iframe 约 572 行；页签绑定附近注册 iframe `load`）
- Test: `tests/test_console_server.py`

**Interfaces:**
- Consumes: Task 1 的消息格式 `{type:"skydango-view", shown}`
- Produces: `function tellLive()`

- [ ] **Step 1: 写失败的测试**

```python
def test_page_tells_live_frame_when_shown():  # spec 2026-09-30-viewer-pause-hidden §2
    page = _console_page().replace(" ", "")
    assert "functiontellLive(" in page
    assert 'type:"skydango-view"' in page and "location.origin" in page
    show = page[page.index("functionshowTab("):]
    assert "tellLive()" in show[: show.index("\n}")]  # showTab 里调
    assert 'addEventListener("load",tellLive)' in page  # Review Focus 3：iframe 加载完马上告诉它
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_console_server.py -q -k tells_live`
Expected: 1 failed

- [ ] **Step 3: 实现**

- `function tellLive()`：`const f=$("live-frame")`；没有 `src` 属性或没有 `contentWindow` 就返回；否则 `f.contentWindow.postMessage({type:"skydango-view",shown:!$("tab-live").hidden},location.origin)`
- `showTab()` 末尾调 `tellLive()`
- 脚本初始化处 `$("live-frame").addEventListener("load",tellLive)`

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_console_server.py -q`
Expected: 全部通过（含 `test_page_script_parses`、不引外部资源的检查）

- [ ] **Step 5: 提交**

```bash
git add src/skydango/console/static/console.html tests/test_console_server.py
git commit -m "feat(console): 切页签和实时画面加载完时告诉 viewer 在不在显示"
```

### Task 3: 浏览器离线核对 + 全量测试

不需要模拟器。

- [ ] **Step 1: 起回放**

`tmp/` 里放几张 1920×1080 截图（本机 `tmp/*.png` 就有），用内置浏览器的 `preview_start`（`.claude/launch.json` 加一项：`python -m skydango view --images tmp --no-browser --port 19399`，port 19399）打开。

- [ ] **Step 2: 模拟后台**

`javascript_tool`：`Object.defineProperty(document,"hidden",{configurable:true,get:()=>true});document.dispatchEvent(new Event("visibilitychange"))`
Expected: 状态栏显示 `页面在后台，暂停取画面`；等 5 秒后 `read_network_requests` 过滤 `snapshot`，没有新请求（最多多出 1 个在途的）

- [ ] **Step 3: 恢复 + 父页面消息**

改回 `get:()=>false` 再派发事件 → 状态栏回到帧率、请求恢复；然后 `window.postMessage({type:"skydango-view",shown:false},location.origin)` → 显示 `不在实时画面页，暂停取画面`；`shown:true` 恢复。
快速连发 `shown:false` / `shown:true` 两次后看 5 秒内请求频率没有翻倍（Review Focus 2）。截图留证。

- [ ] **Step 4: 全量测试**

Run: `python -m pytest -q`
Expected: 全部通过；有失败的逐个写进汇报

- [ ] **Step 5: 合并推送**

按 CLAUDE.md：功能分支合进 main 并推送。真机验证两步（spec「真机验证」）留到晚上，写进汇报。
