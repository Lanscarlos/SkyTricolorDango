# 管理面板改版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `python -m skydango console` 的网页改成「左栏 + 三栏工作台 + 和果子纸感」，顺手补上预检漏查、出错原因、报告阅读和内嵌浏览器里失效的原生对话框。

**Architecture:** 1150 行的 `console/static/console.html` 拆成骨架 html + `console.css` + `common.js` + 每页一个 js（无构建，浏览器直接加载）；后端只加字段和两组只读路由（报告、面板静态文件），现有接口不改。页面逻辑是**搬家 + 改 DOM**：轮询、长轮询、内心曲线、设置保存这些算法原样搬，不重写。

**Tech Stack:** Python 3.13 标准库 HTTP 服务（`console/server.py`）、原生 JS（ES2020，无框架）、CSS 变量；测试 pytest，JS 语法检查和纯函数测试用本机 `node`（没有 node 时 skip）。

**Spec:** `docs/superpowers/specs/2026-10-01-console-redesign-design.md`（先读它；效果图 `.superpowers/brainstorm/*/content/sandbox-page.html` 在本机，不进 git，可以参考但不是必需）

**改版前的页面**：Task 4 会重写 `console.html`，之后各页任务搬旧逻辑时看改版前的版本：`git show 00bda3d:src/skydango/console/static/console.html`（下文「旧第 N 行」都指这个版本的行号）。

## Global Constraints

- 跑测试：`.venv\Scripts\python.exe -m pytest -q`（系统 Python 没装 pytest）；git worktree 里也用主目录的 `.venv`
- 页面**不引任何外部资源**：不出现 `http://` / `https://`，不加载外部字体、CDN（`test_page_*` 会查）
- 页面里的请求一律相对路径（不以 `/` 开头），POST 带 `X-Skydango: 1` 和 JSON
- 不改：子进程 / 沙盒 / viewer 的接口和行为；现有 `/api/*` 的地址和参数（只加字段）；`vision/static/brain_trace.*`；安全检查（Host、`post_guard`）
- 不用 `confirm()` / `prompt()` / `alert()`：一律 `ask()` / `toast()`（Task 4）
- 颜色只在 `console.css` 的 `:root` 定义（spec §3 的变量名和值），别处只用 `var(--…)`
- 字体：`"Microsoft YaHei UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif`；等宽 `"Cascadia Mono",Consolas,monospace`
- `localStorage` 读写都包 try/catch
- 文案用中文，沿用现有文案（重置记忆、回放、删性格条目的确认文字照搬）
- 每个任务结束提交一次；提交信息末尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. 报告名带路径穿越 / 编码过的 `..`（`%2e%2e`、`..%5c`）、空名、目录不存在 → 一律 404 或空列表，绝不读到 reports 目录外（Task 3 测试覆盖编码变体）
2. 报告和聊天原话里有 `<script>`、`<img onerror>` → markdown 渲染和聊天行都只出文本，不执行（Task 7 渲染器测试、Task 6 用 `textContent`）
3. 页面打开时子进程状态在变（starting→running→crashed）→ 左栏卡片、两个运行页的启动按钮跟着变，不残留「运行中」（Task 4 卡片按 `/api/state` 每秒重画；Task 10 浏览器核对）
4. 直接打开 `#settings/secret.llm` 或旧书签 `#overview` → 落到对的页面和设置项（Task 4 路由测试点在 node 里跑）
5. `secrets.toml` 坏了 / 预检里有字符串以外的东西 → 页面照样能开，问题列表照样显示（Task 1 断言所有 problems 都是 dict）

---

### Task 1: 预检问题改成 dict、大脑模式补查大模型 Key、沙盒预检提前给

**Files:**
- Modify: `src/skydango/console/preflight.py`
- Modify: `src/skydango/console/server.py`（`_run_problems`、`state`、`start_run`、`launch_sandbox`、`start_sandbox`、`sandbox_info`，以及别处往 `problems` 里塞字符串的地方，`grep -n "problems" server.py` 全部过一遍）
- Test: `tests/test_console_preflight.py`、`tests/test_console_server.py`、`tests/test_console_sandbox.py`、`tests/test_sandbox_fix_console.py`

**Interfaces:**
- Produces: `preflight.problem(text: str, setting: str | None = None) -> dict` 返回 `{"text": text, "setting": setting}`；`preflight(...) -> list[dict]`；`ConsoleServer.sandbox_problems() -> tuple[list[dict], bool]`（问题列表、沙盒端口上是否有孤儿）；`/api/state` 的 `problems`、`/api/run/start` / `/api/sandbox/start` / `/api/sandbox/replay` 拒绝回复里的 `problems`、`/api/sandbox/info` 新增的 `problems` 都是 `list[dict]`

- [ ] **Step 1: 写失败的测试**（`tests/test_console_preflight.py` 追加；用文件里现有的 `store_with(tmp_path, console=, secrets=)`、`HAS`，文件已有 autouse 的 `no_registry` 把 Windows 用户环境变量那层关掉了）

```python
TOKEN = 'SKYDANGO_CLAUDE_TOKEN = "t"\n'
KEY = 'DEEPSEEK_API_KEY = "k"\n'

def test_brain_needs_llm_key_for_fallback(tmp_path):
    s = store_with(tmp_path, secrets="[env]\n" + TOKEN)  # 有 Claude 令牌、没有大模型 Key
    assert preflight(s, LaunchOptions(brain=True), busy=False, find_spec=HAS) == [
        {"text": "大脑离线时的备用回复要大模型的 API Key：去设置页填", "setting": "secret.llm"}]

def test_brain_echo_provider_needs_no_llm_key(tmp_path):
    s = store_with(tmp_path, console='[llm]\nprovider = "echo"\n', secrets="[env]\n" + TOKEN)
    assert preflight(s, LaunchOptions(brain=True), busy=False, find_spec=HAS) == []

def test_missing_claude_token_points_to_setting(tmp_path):
    s = store_with(tmp_path, secrets="[env]\n" + KEY)
    assert {"text": "大脑模式要 Claude 令牌：去设置页填", "setting": "secret.claude"} in preflight(s, LaunchOptions(brain=True), busy=False, find_spec=HAS)

def test_all_problems_are_dicts(tmp_path):
    s = store_with(tmp_path, console="[device]\nserail = 1\n")  # 现有测试里的坏配置
    for p in preflight(s, LaunchOptions(), busy=True, find_spec=HAS):
        assert set(p) == {"text", "setting"}
```

  其它测试文件（`test_console_server.py` 等）如果没有关掉 `_user_env`，同样加上 `monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")`——**这台机器的 Windows 用户环境变量里真的有 `DEEPSEEK_API_KEY` 和 `SKYDANGO_CLAUDE_TOKEN`**，不关的话缺 Key 的断言在这里会失败、在别的机器上又会过。

  `tests/test_console_server.py` 追加：

```python
def test_sandbox_info_has_problems(srv):  # 没有 Claude 令牌
    info = request(srv.url + "api/sandbox/info")[1]
    assert any(p["setting"] == "secret.claude" for p in info["problems"])
    assert all(set(p) == {"text", "setting"} for p in info["problems"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_console_preflight.py tests/test_console_server.py -q`
Expected: 新测试 FAIL（`problems` 是字符串 / 没有 `problems` 键）

- [ ] **Step 3: 实现**
  - `preflight.py`：加 `problem()`；所有 `append` 改成 `problem(...)`。Claude 令牌 → `setting="secret.claude"`；普通模式缺 Key → `"secret.llm"`；新增：`opts.brain and cfg.llm.provider != "echo" and not store.secret("llm")` → `problem("大脑离线时的备用回复要大模型的 API Key：去设置页填", "secret.llm")`（放在 mcp 检查之后）；配置错误、`busy` 的问题 `setting=None`
  - `server.py`：服务端自己拼的问题（另一种在跑、端口孤儿、正在检测设备、secrets.toml 坏、`runner.start` 抛错、正在回放）都改成 `problem(...)`（`setting=None`）
  - 把 `launch_sandbox` 里起进程前的检查抽成 `sandbox_problems() -> tuple[list[dict], bool]`（不加锁，调用方需要时自己拿 `_device_lock`）：另一种在跑 / 沙盒已在跑、`preflight(..., LaunchOptions(brain=True), False, ...)`、沙盒端口孤儿（第二个返回值）、团子端口孤儿（顺手 `self.orphan = True`）、secrets.toml 读不了。`launch_sandbox` 改用它
  - `sandbox_info()` 加 `"problems"`：沙盒自己在 starting/running/stopping 时为 `[]`，否则 `sandbox_problems()[0]`
- [ ] **Step 4: 更新老测试**：大脑模式现在还要大模型 Key，原来只给了 `SKYDANGO_CLAUDE_TOKEN` 就期望预检通过 / 启动成功的测试（如 `test_preflight_ok_with_token`、`test_start_refused_while_checking_device` 最后那次 200、沙盒启动成功的测试），secrets 里补上 `DEEPSEEK_API_KEY = "k"`；`test_preflight_brain_needs_token_and_mcp` 的期望加上新问题。断言问题文字的：凡是 `"…" in p` / `problems[0].startswith(…)` / `problems == [字符串]` 的，改成看 `p["text"]`（`test_console_preflight.py:47`、`test_console_server.py:157,173,318,353`、`test_console_sandbox.py:57,64,65,67`、`test_sandbox_fix_console.py:26`，再 `grep -rn "problems" tests/` 确认没漏）
- [ ] **Step 5: 跑全部测试**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全部 PASS（页面相关的 `test_page_*` 此时还是旧页面，照样过）

- [ ] **Step 6: 提交** `fix(console): 预检问题带跳转目标、大脑模式补查大模型 Key、沙盒 info 带预检`

---

### Task 2: 子进程出错时给出原因

**Files:**
- Modify: `src/skydango/console/runner.py`（`status()`）
- Test: `tests/test_console_runner.py`

**Interfaces:**
- Produces: `Runner.status()["error"]: str | None`

- [ ] **Step 1: 写失败的测试**（沿用文件里现有的造 crashed 状态的方式——用 `tests/fake_child.py` 让子进程打印几行后非 0 退出；看现有 crashed 测试怎么写的照抄）

```python
def test_crashed_status_has_error_line(...):
    # 子进程输出: "INFO 启动", "错误: 没有找到 API Key，请设置环境变量 DEEPSEEK_API_KEY", "bye"，然后 exit 1
    assert st["state"] == "crashed" and st["error"] == "没有找到 API Key，请设置环境变量 DEEPSEEK_API_KEY"

def test_crashed_without_error_line_uses_last_line(...):
    # 输出: "a", "Traceback (most recent call last):", "ValueError: boom", ""，exit 1
    assert st["error"] == "ValueError: boom"

def test_running_has_no_error(...):
    assert st["state"] == "running" and st["error"] is None
```

- [ ] **Step 2: 跑测试确认失败**：`.venv\Scripts\python.exe -m pytest tests/test_console_runner.py -q` → KeyError `error`
- [ ] **Step 3: 实现**：`status()` 里 crashed 时倒着扫 `self._lines`（`(n, text)` 环形缓冲）：第一行 `text.strip()` 以 `错误:` 开头的，去掉 `错误:` 和前后空白；没有就取最后一行非空的；截 200 字。其它状态 `None`。日志行可能带时间前缀（`11:23:18 INFO …`），「错误:」可能在行中间——用 `text.find("错误:")` 而不是 `startswith`
- [ ] **Step 4: 跑测试**：`.venv\Scripts\python.exe -m pytest -q` → PASS
- [ ] **Step 5: 提交** `feat(console): 子进程出错停下时 status 带出错原因`

---

### Task 3: 报告接口和面板自己的静态文件

**Files:**
- Create: `src/skydango/console/reports.py`
- Modify: `src/skydango/console/server.py`（GET 路由）、`pyproject.toml`（package-data）
- Test: `tests/test_console_reports.py`（新）、`tests/test_console_server.py`

**Interfaces:**
- Produces（`reports.py`）：
  - `list_reports(report_dir: Path) -> list[dict]`，每项 `{"name": str, "scenario": str, "time": float, "size": int}`，按 mtime 倒序
  - `read_report(report_dir: Path, name: str) -> tuple[int, dict]`：`(200, {"name", "text"})` / `(404, {"ok": False, "text": "没有这份报告"})` / `(413, {"ok": False, "text": "报告太大（超过 2 MB）"})`
  - `MAX_REPORT = 2 * 1024 * 1024`
- Produces（路由）：`GET /api/sandbox/reports`、`GET /api/sandbox/reports/<name>`（校验 Host）；`GET /console/static/<文件>`（不校验 Host）
- Consumes：沙盒目录 = `ConsoleServer.sandbox_dir()`，报告目录 = `sandbox_dir() / "reports"`（核对 `replay.py` 的 `report_dir` 是不是这个，以它为准）

- [ ] **Step 1: 写失败的测试**（`tests/test_console_reports.py`）

```python
def test_list_newest_first_only_md(tmp_path):
    d = tmp_path / "reports"; d.mkdir()
    _write(d / "放鸽子-20261001-113000.md", "# a", mtime=100)
    _write(d / "a-b-剧本-20261001-120000-2.md", "# b", mtime=200)
    _write(d / "notes.txt", "x", mtime=300)
    rs = list_reports(d)
    assert [r["name"] for r in rs] == ["a-b-剧本-20261001-120000-2.md", "放鸽子-20261001-113000.md"]
    assert rs[0]["scenario"] == "a-b-剧本" and rs[1]["scenario"] == "放鸽子"

def test_scenario_falls_back_to_stem(tmp_path):
    d = tmp_path / "reports"; d.mkdir(); _write(d / "手写的.md", "x", mtime=1)
    assert list_reports(d)[0]["scenario"] == "手写的"

def test_missing_dir_is_empty(tmp_path):
    assert list_reports(tmp_path / "nope") == []

@pytest.mark.parametrize("name", ["../x.md", "..\\x.md", "sub/x.md", "", "x.txt", "不存在.md", "..", "%2e%2e%2fx.md"])
def test_read_refuses_bad_names(tmp_path, name):
    d = tmp_path / "reports"; d.mkdir(); (tmp_path / "x.md").write_text("secret", encoding="utf-8")
    assert read_report(d, name)[0] == 404

def test_read_ok_and_too_big(tmp_path):
    d = tmp_path / "reports"; d.mkdir()
    (d / "a-20261001-000000.md").write_text("# 标题\n正文", encoding="utf-8")
    assert read_report(d, "a-20261001-000000.md") == (200, {"name": "a-20261001-000000.md", "text": "# 标题\n正文"})
    (d / "big.md").write_bytes(b"x" * (MAX_REPORT + 1))
    assert read_report(d, "big.md")[0] == 413
```

  `tests/test_console_server.py` 追加：

```python
def test_reports_routes(srv):  # 报告目录在沙盒目录下；沙盒目录怎么指到 tmp 看现有 sandbox 测试的 make_server
    ...  # 造一份报告后：
    assert request(srv.url + "api/sandbox/reports")[1]["reports"][0]["name"] == "r-20261001-000000.md"
    assert request(srv.url + "api/sandbox/reports/r-20261001-000000.md")[1]["text"] == "hi"
    assert request(srv.url + "api/sandbox/reports/..%2fx.md")[0] == 404
    assert request(srv.url + "api/sandbox/reports", headers={"Host": "evil.com"})[0] == 403  # 用现有 Host 测试的写法

def test_serves_console_static(srv):
    with urllib.request.urlopen(srv.url + "console/static/console.css", timeout=5) as r:
        assert r.status == 200 and "text/css" in r.headers["Content-Type"]
    for bad in ("console/static/console.html", "console/static/../server.py", "console/static/%2e%2e/server.py", "console/static/nope.js"):
        assert request(srv.url + bad)[0] == 404
```

  这一步先建一个最小的 `src/skydango/console/static/console.css`（一行注释即可，Task 4 填内容），让静态测试有文件可取。

- [ ] **Step 2: 跑测试确认失败**：`.venv\Scripts\python.exe -m pytest tests/test_console_reports.py tests/test_console_server.py -q`
- [ ] **Step 3: 实现**
  - `reports.py`：只看 `report_dir.glob("*.md")` 里的普通文件；`scenario` 用 `re.fullmatch(r"(.*)-(\d{8}-\d{6})(?:-\d+)?\.md", name)` 第 1 组，对不上取 `name[:-3]`。`read_report` 先 `urllib.parse.unquote(name)`，再要求它**恰好等于** `list_reports` 里某个 `name`（这一条就挡住了所有穿越），再看大小
  - `server.py`：`/console/static/` 前缀在 Host 检查之前处理（同 `/static/`）：`name` 必须匹配 `[A-Za-z0-9_.-]+\.(css|js)`、不含 `..`，从 `importlib.resources.files("skydango.console") / "static" / name` 读，不存在 404；类型 css → `text/css; charset=utf-8`、js → `text/javascript; charset=utf-8`。`/api/sandbox/reports` 和 `/api/sandbox/reports/<name>` 放在 Host 检查之后
  - `pyproject.toml`：package-data 加 `"console/static/*.css", "console/static/*.js"`
- [ ] **Step 4: 跑测试**：`.venv\Scripts\python.exe -m pytest -q` → PASS
- [ ] **Step 5: 提交** `feat(console): 报告列出 / 读取接口、面板静态文件路由`

---

### Task 4: 新骨架——左栏、主题、common.js（对话框、提示条、路由、正在跑卡片）

**Files:**
- Rewrite: `src/skydango/console/static/console.html`（骨架，**不再有内联 `<script>`**）
- Write: `src/skydango/console/static/console.css`（主题 + 左栏 + 通用组件 + 窄屏；各页的样式由各页任务追加到同一文件末尾，每页一段 `/* ---- 沙盒 ---- */` 注释开头）
- Create: `src/skydango/console/static/common.js`
- Test: `tests/test_console_page.py`（新；把 `tests/test_console_server.py` 里 `# ---- 页面 ----` 那一组 `test_page_*` / `_console_page` / `test_cards_show_mood_and_energy` / `emotes_allowed` 页面断言**移过来**并按新结构改写，旧的在 Task 4 里删掉；各页任务再往这个文件加自己的页面测试）

**Interfaces:**
- Produces（`common.js`，全部挂在 `window` 上，后面各页直接用）：
  - `$(id)`、`el(tag, cls?, text?)`、`getJSON(url)`、`post(url, body?) -> Promise<{status, data}>`（照搬旧实现）
  - 时间文字：`pad2(n)`、`hhmm(t)`、`dayTime(t)`、`span(sec)`、`fmtUptime(s)`（照搬旧实现）
  - `ask(text: string, opts?: {ok?: string, cancel?: string, danger?: boolean, fields?: Array<{name, label, placeholder?, value?}>}) -> Promise<boolean | object | null>`：页面内模态框，Esc = 取消、回车 = 确定，打开时焦点在第一个输入框或确定按钮，关闭后焦点回到原处
  - `toast(text: string, kind?: "ok" | "warn" | "bad")`：右下角，ok/warn 4 秒消失，bad 不自动消失（有 ×）
  - `problemList(container: Element, problems: Array<{text, setting}>)`：清空后渲染 `<ul class="problems">`；`setting` 不为空的项后面加 `<a href="#settings/<setting>">去设置 →</a>`；空列表时隐藏容器
  - `Pages`：`{[name]: {init?(): void, show?(arg: string|null): void, hide?(): void}}`；各页文件 `Pages.sandbox = {...}` 注册
  - `go(page: string, arg?: string)`：改 hash；路由规则 `parseHash(hash) -> {page, arg}`：`#settings/secret.llm` → `{page:"settings", arg:"secret.llm"}`；`#overview` → `live`；不认识 / 空 → `sandbox`
  - `S.state`：最近一次 `/api/state`；`onState(fn)` 注册回调，每次拉到新状态都调（`refresh()` 每秒一次）
  - `BUSY = ["starting","running","stopping"]`
- 页面骨架 id（后面任务往里填）：`#side`（左栏）、`#runcard`、`nav#nav`（按钮 `data-page="sandbox|live|inner|scenarios|settings|device"`，每个带 `<small class="mark" id="mark-<page>">`）、`main#work` 里六个 `<section id="page-<name>" hidden>`、`#dialog`（模态框根）、`#toasts`
- `console.html` 的 `<head>` 依次：`static/brain_trace.css`、`console/static/console.css`；`<body class="console">` 末尾依次：`static/brain_trace.js`、`console/static/common.js`、`markdown.js`、`inner.js`、`sandbox.js`、`live.js`、`scenarios.js`、`settings.js`、`device.js`（都在 `console/static/` 下；还没写的文件在本任务先建成只有一行注释的空文件，保证页面能开），最后一个内联启动脚本都不要——`common.js` 在 `DOMContentLoaded` 时调各页 `init()`、路由一次、开始轮询

- [ ] **Step 1: 写失败的测试**（`tests/test_console_page.py`）

```python
STATIC = importlib.resources.files("skydango.console") / "static"
JS = ["common.js", "markdown.js", "inner.js", "sandbox.js", "live.js", "scenarios.js", "settings.js", "device.js"]

def bundle() -> str:  # 页面 + 样式 + 全部脚本，页面断言都对它做
    return "\n".join((STATIC / n).read_text(encoding="utf-8") for n in ["console.html", "console.css", *JS])

def test_skeleton():
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    for id_ in ("side", "runcard", "nav", "work", "dialog", "toasts", *(f"page-{p}" for p in ("sandbox", "live", "inner", "scenarios", "settings", "device"))):
        assert f'id="{id_}"' in page, id_
    assert "<script>" not in page  # 没有内联脚本
    for n in JS:
        assert f'src="console/static/{n}"' in page
    assert page.index('src="static/brain_trace.js"') < page.index('src="console/static/common.js"') < page.index('src="console/static/sandbox.js"')

def test_offline_relative_and_no_native_dialogs():
    b = bundle()
    assert "http://" not in b and "https://" not in b
    assert not re.search(r"""fetch\(\s*[`"']/""", b)
    assert '"X-Skydango":"1"' in b.replace(" ", "")
    assert not re.search(r"\b(confirm|prompt|alert)\(", b)

def test_scripts_parse():
    node = shutil.which("node") or pytest.skip("没有 node")
    for n in JS:
        assert subprocess.run([node, "--check", str(STATIC / n)]).returncode == 0, n

def test_routes():  # 在 node 里跑 common.js 的 parseHash（common.js 末尾：typeof module!=="undefined" 时 module.exports={parseHash}，且顶层不碰 document）
    node = shutil.which("node") or pytest.skip("没有 node")
    js = f"const {{parseHash}}=require({json.dumps(str(STATIC / 'common.js'))});console.log(JSON.stringify(['#settings/secret.llm','#overview','#nope','','#inner'].map(parseHash)))"
    out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out == [{"page": "settings", "arg": "secret.llm"}, {"page": "live", "arg": None}, {"page": "sandbox", "arg": None},
                   {"page": "sandbox", "arg": None}, {"page": "inner", "arg": None}]

def test_theme_variables():
    css = (STATIC / "console.css").read_text(encoding="utf-8")
    for v in ("--paper:#f6f1e7", "--card:#fffdf8", "--ink:#3b332b", "--sakura:#e58aab", "--matcha:#7fae7d", "--brick:#c4573f"):
        assert v in css.replace(" ", "")
```

  从 `test_console_server.py` 删掉旧的页面测试（`test_page_has_four_tabs_and_uses_relative_api`、`test_page_has_inner_tab`、`test_page_script_parses`、`test_cards_show_mood_and_energy`、`test_page_has_sandbox_tab`、`test_page_has_scenario_controls`、`test_state_says_when_emotes_are_off_in_config` 末尾那句 `_console_page()` 断言、`_console_page`），它们的检查点由 Task 5~9 的页面测试接手。`test_serves_shared_brain_trace_assets` 留着。

- [ ] **Step 2: 跑测试确认失败**：`.venv\Scripts\python.exe -m pytest tests/test_console_page.py -q`
- [ ] **Step 3: 写 `console.css` 的公共部分**：spec §3 的全部变量；`body` 纸底、墨色、`tabular-nums`；左栏（200px、`--side` 底、右边线）、`#runcard` 四种状态（`data-state="idle|busy|stopping|crashed"`：idle 灰点；busy 抹茶左边框 + 呼吸点；stopping 琥珀；crashed 砖红边框）、导航分组小标题、选中项（`--card` 底 + 左侧 3px 樱花内阴影 + `--sakura-ink` 字）；按钮 `.btn` / `.btn.go`（抹茶）/ `.btn.halt`（砖红字）/ `.btn.sm`、输入框、`.pane`（卡片：`--card`、1px `--line`、12px 圆角、淡阴影）、`.pane>h3` 小标题（12px、字距 .12em、`--muted`）、`.problems`、`.banner.warn/.bad`、`.chip`、`.tag`、模态框（遮罩 + 居中卡片）、提示条；团子三颗球随 `body[data-state]` 变（照搬旧样式的状态逻辑，颜色换成纸感）；`@media (max-width: 999px)`：左栏变顶部横栏、导航横排可滚、`#work` 内的多栏网格一律单列。`brain_trace.css` 的纸感覆盖也写在这里（`.console .turn{…}` 等，选择器以 `.console` 开头，盖过 `brain_trace.css` 的深色），先看 `vision/static/brain_trace.css` 有哪些类
- [ ] **Step 4: 写 `console.html` 骨架和 `common.js`**
  - 左栏 `#runcard` 由 `renderRunCard(st)` 按 `S.state.run` 画：`kind` 为 `sandbox` / 其它 → 「沙盒」/「真机团子」；starting「正在启动…」、running「运行中」+ 一行摘要（沙盒：沙盒页提供的 `window.sandboxSummary?.()` 返回的文字，例如「10月1日 11:28 · 开心 · 精神」；真机：`真的发送`/`只打印` + 已运行时长）、stopping「在收尾…」、crashed「出错停下了」+ `run.error`（Task 2）+ `<details><summary>看日志</summary><pre>` 最近 20 行（`api/logs?after=0` 取最后 20 行，只在展开时拉）；idle / exited「都没在跑」。按钮：running/starting 时「停止」（沙盒是「下线（写日记）」→ `api/sandbox/stop`，真机 → `api/run/stop`）；点卡片空白处 `go(kind==="sandbox"?"sandbox":"live")`
  - `body.dataset.state` = `run.state`（让团子球变色；旧代码沙盒在跑时设 idle，新代码不用区分）
  - 导航标记 `#mark-live`：真机在跑「在跑」；`#mark-settings`：`S.state.problems` 里有 `setting` 不为空的项时「!」；其余标记由各页设置
  - 路由：`hashchange` → `parseHash` → 隐藏其它 section、调旧页 `hide()`、新页 `show(arg)`、导航 `aria-current="page"`；`#overview` 用 `history.replaceState` 改成 `#live`
  - 轮询：`refresh()` 每 1 秒拉 `api/state` → `S.state` → `renderRunCard` → 所有 `onState` 回调；拉失败时卡片显示「连不上面板（程序停了？）」并 `body.dataset.state="crashed"`
  - `ask()` / `toast()` / `problemList()` 按 Interfaces 实现；`ask` 用 `#dialog` 里的一个 `role="dialog" aria-modal="true"` 卡片
- [ ] **Step 5: 跑测试**：`.venv\Scripts\python.exe -m pytest -q` → PASS
- [ ] **Step 6: 提交** `feat(console): 新骨架——左栏、和果子纸感主题、页面内对话框和提示条、路由`

---

### Task 5: 内心页

**Files:**
- Write: `src/skydango/console/static/inner.js`
- Modify: `console.html`（`#page-inner` 的内容）、`console.css`（追加 `/* ---- 内心 ---- */`）
- Test: `tests/test_console_page.py`

**Interfaces:**
- Consumes: Task 4 的 `$ el getJSON post ask toast span hhmm dayTime Pages onState S BUSY`
- Produces: `window.Inner = {renderNow(container: Element, now: object, d: object): void, setSource(src: "dango"|"sandbox"): void}`（沙盒页右栏「现在」用 `renderNow`；「完整内心 →」用 `setSource("sandbox")` 后 `go("inner")`）

- [ ] **Step 1: 写失败的测试**

```python
def test_inner_page():
    b = bundle()
    for id_ in ("inner-source-toggle", "inner-source", "inner-now", "inner-curve", "inner-log", "inner-persona", "inner-cards", "inner-days", "inner-refresh", "inner-range", "inner-changed"):
        assert f'id="{id_}"' in b, id_
    for s in ("api/inner", "api/inner/forget", "Inner.renderNow", "Pages.inner", "IN.lastState"):
        assert s in b
    for text in ("（没开反思）", "实时取不到，显示的是上次保存的", "团子正在启动 / 停止，稍等再删", "只看有改动的", "删了团子就不会再用它（不能撤销）"):
        assert text in b
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：把旧 `console.html` 第 717~939 行（`IN`、`MOOD_COLOR`… `renderInner*`、`forgetTrait`、`pullInner`、来源切换）搬进 `inner.js`，改动只有：
  - `forgetTrait` 的 `confirm` → `await ask(<原文案>, {ok: "删", danger: true})`；`alert` → `toast(…, "bad")`
  - 依赖沙盒状态的 `innerClock()` 改成读 `window.sandboxClock?.()`（Task 6 提供，返回沙盒在跑时的沙盒时间、停着时 `max(现在, 下限)`；没有就用 `Date.now()/1000`）
  - `innerMine` / `innerBusy` 读 `S.state.run`（不变）
  - 页面 `show()` 时 `loadInner(IN.source)`；`setInterval(pullInner, 5000)` 和「状态一变就重读」改成在 `onState` 里判断
  - 布局：标题行（`<h2 id="inner-title">` + 两段式 `#inner-source-toggle` + `#inner-source` 来源标签 + 刷新）；两列网格（左：现在 / 曲线 / 反思记录；右：性格档案 / 关系卡 / 日子和日记）
  - CSS：心情色变量 `--m-happy` 等在纸感下重新取色（开心樱花、平常米灰 `#cbbfa8`、低落 灰蓝 `#8ea3b8`、烦 砖红浅 `#d98c78`），曲线网格线 / 坐标字用 `--line` / `--muted`
- [ ] **Step 4: 跑测试**：`.venv\Scripts\python.exe -m pytest -q` → PASS
- [ ] **Step 5: 提交** `feat(console): 内心页搬进新骨架`

---

### Task 6: 沙盒页（三栏工作台）

**Files:**
- Write: `src/skydango/console/static/sandbox.js`
- Modify: `console.html`（`#page-sandbox`）、`console.css`（追加 `/* ---- 沙盒 ---- */`）
- Test: `tests/test_console_page.py`

**Interfaces:**
- Consumes: Task 4 全部公共函数；`Inner.renderNow`（Task 5）；`/api/sandbox/info` 的 `problems`（Task 1）
- Produces: `window.sandboxSummary(): string | null`（左栏卡片用，例：`"10月1日 11:28 · 开心 · 精神"`）；`window.sandboxClock(): number`（内心页用）；`#mark-scenarios` 不归它管

- [ ] **Step 1: 写失败的测试**

```python
def test_sandbox_page():
    b = bundle()
    for id_ in ("sb-head", "sb-start", "sb-problems", "sb-clock", "sb-replay-bar", "sb-chat", "sb-say", "sb-brain", "sb-now", "sb-nearby", "sb-scene", "sb-rec", "sb-save", "sb-reset"):
        assert f'id="{id_}"' in b, id_
    assert 'mountBrainTrace($("sb-brain"),"sandbox/brain")' in b.replace(" ", "")
    for api in ("api/sandbox/start", "api/sandbox/stop", "api/sandbox/reset", "api/sandbox/info", "sandbox/state", "sandbox/op", "api/sandbox/save", "api/sandbox/record/new", "api/sandbox/replay"):
        assert api in b
    assert "会用 memory/ 覆盖沙盒记忆" in b and "Pages.sandbox" in b and "sandboxSummary" in b
    assert "1.15fr 1.1fr .85fr" in (STATIC / "console.css").read_text(encoding="utf-8")
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**——搬旧第 941~1079、1080~1142 行里沙盒相关的部分（剧本列表 / 回放按钮 / 报告留给 Task 7），按 spec §4.1 摆：
  - `#sb-head` 顶栏：停着 → `#sb-start` 表单（起始时间三选一 + 自定义输入 + 「启动沙盒」+ 下限说明 `#sb-floor` + `#sb-problems`，用 `problemList`，有问题时「启动沙盒」置灰；`SB.info.problems` 来自 `api/sandbox/info`，在 `show()`、沙盒停下时、每 10 秒（页面可见且停着）重拉）；运行中 → `#sb-clock`（大号时间 + 日期、精力、快进按钮、拨到、`#sb-limit` 额度提示）；右侧总有 `#sb-rec`「录制中 · N 步」chip、`#sb-save`「另存为剧本…」、`#sb-reset`「重置记忆」（运行中 / 回放中置灰）
  - `#sb-replay-bar`：回放中显示「回放『名字』第 i/n 步」+「停止回放」（`api/sandbox/replay/stop`），并给三栏加 `.sb-replaying`（冒充和世界操作置灰）；数据仍来自 `api/sandbox/replay`（每 1.5 秒，页面可见或回放中）
  - 三栏 `.sb-cols{display:grid;grid-template-columns:1.15fr 1.1fr .85fr;height:calc(100vh - 顶栏高度)}`，每栏 `.pane` 内 `.scroll{overflow:auto}`、页面本身不滚。左：`#sb-chat` + 底部固定 `#sb-say`；中：标题行（模型 · 轮数 · 在线，从 `sandbox/state` 或 brain_trace 已有信息里取；拿不到就只写「大脑」）+ `#sb-brain`；右：`#sb-now`（`Inner.renderNow`，右上「完整内心 →」）、`#sb-nearby`、`#sb-scene`
  - 聊天行样式按 spec §4.1（团子靠右樱花底、别人靠左米色底、事件分隔线、反思虚线框、动作斜体、被拦删除线 + 砖红原因）；反思行判断：`r.kind==="event"` 且文字以「反思」开头 → 虚线框样式
  - 说话人选择存 `localStorage["sb-who"]`（try/catch），`renderSbInfo` 时恢复
  - `confirm`/`prompt` 全换 `ask`：重置记忆（原文案，`danger`）、新录制（原文案）、另存为（一个 `ask` 带两个 `fields`：剧本名、说明；409 已存在再 `ask(r.data.text)` 覆盖）
  - 操作结果 `sbMsg` → `toast`
  - 启动被拒：`problemList(#sb-problems, r.data.problems)`；`r.data.orphan` 时在 `#sb-problems` 下加「让它退出」按钮（`api/orphan/stop {kind:"sandbox"}`）
  - 提供 `sandboxSummary()`（沙盒在跑时：`sbClockText(sbNow())` + 心情 + 精力，来自最近一次 `Inner.renderNow` 用的数据和 `SB.state.energy`）、`sandboxClock()`
- [ ] **Step 4: 跑测试**：`.venv\Scripts\python.exe -m pytest -q` → PASS
- [ ] **Step 5: 提交** `feat(console): 沙盒页改成三栏工作台`

---

### Task 7: 剧本和报告页 + markdown 渲染器

**Files:**
- Write: `src/skydango/console/static/markdown.js`、`src/skydango/console/static/scenarios.js`
- Modify: `console.html`（`#page-scenarios`）、`console.css`（追加 `/* ---- 剧本 ---- */`）
- Test: `tests/test_console_page.py`、`tests/test_console_markdown.py`（新）

**Interfaces:**
- Produces: `renderMarkdown(text: string) -> string`（HTML 字符串，挂 `window`；`typeof module!=="undefined"` 时 `module.exports={renderMarkdown}`）
- Consumes: `/api/sandbox/scenarios`、`/api/sandbox/replay`（POST 开始 / GET 进度）、`/api/sandbox/replay/stop`、`/api/sandbox/reports`、`/api/sandbox/reports/<name>`（Task 3）；`ask toast Pages go`

- [ ] **Step 1: 写失败的测试**（`tests/test_console_markdown.py`，node 跑，没有 node 就 skip）

```python
CASES = [
    ("# 标题\n正文", "<h1>标题</h1>\n<p>正文</p>"),
    ("## 二\n### 三", "<h2>二</h2>\n<h3>三</h3>"),
    ("- a\n- b", "<ul><li>a</li><li>b</li></ul>"),
    ("1. a\n2. b", "<ol><li>a</li><li>b</li></ol>"),
    ("> 引用", "<blockquote>引用</blockquote>"),
    ("```\nx < y\n```", "<pre><code>x &lt; y</code></pre>"),
    ("有 `code` 和 **粗**", "<p>有 <code>code</code> 和 <strong>粗</strong></p>"),
    ("<script>alert(1)</script>", "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>"),
    ("**<img src=x onerror=1>**", "<p><strong>&lt;img src=x onerror=1&gt;</strong></p>"),
    ("a\nb\n\nc", "<p>a<br>b</p>\n<p>c</p>"),
]
@pytest.mark.parametrize("src,html", CASES)
def test_render(src, html): assert _run(src) == html
```

  `tests/test_console_page.py` 追加：

```python
def test_scenarios_page():
    b = bundle()
    for id_ in ("sc-list", "sc-progress", "sc-reports", "sc-reader"):
        assert f'id="{id_}"' in b, id_
    for api in ("api/sandbox/scenarios", "api/sandbox/replay", "api/sandbox/replay/stop", "api/sandbox/reports"):
        assert api in b
    assert "回放时每一步会真的调 Claude，花额度" in b and "renderMarkdown(" in b and "Pages.scenarios" in b
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
  - `markdown.js`：逐行扫；先整体 HTML 转义（`& < > "`）再识别语法；代码块内不做行内替换；行内只认 `` `x` `` 和 `**x**`；段落内单换行变 `<br>`；块之间用 `\n` 连接（测试的期望格式）
  - `scenarios.js`：左 `#sc-list`（每个剧本一张卡：名字、步数、说明 / 读不了的错误；「回放」按钮 → `ask(<旧确认文案>)` → POST `api/sandbox/replay {name}`）；列表空时显示示例剧本说明（`docs/sandbox-scenarios/` 里的人名是占位的，换成自己的好友名再复制进 `sandbox/scenarios/`）；`#sc-progress` 回放进度 + 「停止回放」；`#mark-scenarios` 回放中写「回放中」、否则写报告数
  - 右 `#sc-reports` 报告列表（剧本名 + `dayTime(time)`，新的在上），点了 `GET api/sandbox/reports/<encodeURIComponent(name)>` → `#sc-reader.innerHTML = renderMarkdown(text)`；回放刚结束时重拉列表并自动打开最新一份
- [ ] **Step 4: 跑测试**：`.venv\Scripts\python.exe -m pytest -q` → PASS
- [ ] **Step 5: 提交** `feat(console): 剧本和报告页，报告在页面里读`

---

### Task 8: 真机团子页（原总览 + 实时画面）

**Files:**
- Write: `src/skydango/console/static/live.js`
- Modify: `console.html`（`#page-live`）、`console.css`（追加 `/* ---- 真机 ---- */`）
- Test: `tests/test_console_page.py`

**Interfaces:**
- Consumes: Task 4 公共函数；`S.state.launch / problems / orphan / emotes_allowed`；`run.error`（Task 2）

- [ ] **Step 1: 写失败的测试**

```python
def test_live_page():
    b = bundle()
    for id_ in ("launch", "opt-live", "opt-emotes", "opt-duration", "start", "problems", "live-frame", "cards", "log", "banners"):
        assert f'id="{id_}"' in b, id_
    for api in ("api/run/start", "api/run/stop", "api/logs", "live/status", "api/orphan/stop"):
        assert api in b
    line = next(l for l in b.splitlines() if l.strip().startswith("const CARDS="))
    assert '"心情"' in line and '"精力"' in line
    assert "emotes_allowed" in b and "沙盒在跑，先下线" in b and "Pages.live" in b
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**——搬旧第 528~632 行（状态里和启动 / 卡片 / 日志 / 横幅有关的部分）：
  - 停着：`#launch` 启动区（模式、真的发送 + 警告、动作 + `emotes_allowed` 说明、时长、「叫醒团子」、`#problems` 用 `problemList`）；沙盒在跑时整个表单置灰、按钮文字「沙盒在跑，先下线」
  - 跑着（`run.kind!=="sandbox"` 且 running/stopping）：两栏，左 `#live-frame`（`aspect-ratio:16/9`，src `live/`，进入运行时才设、离开时移除——照旧逻辑），右 `#cards`（`CARDS` 同旧）+ 底部 `#log`（`<details open>` 包住，`api/logs` 每秒增量拉，`/Traceback|ERROR|错误/` 标红）
  - `#banners`：强杀（`run.forced`）、孤儿团子（`orphan` + 「让它退出」）横幅照旧
  - 「停止」按钮只在左栏卡片上（Task 4），这页不再放
- [ ] **Step 4: 跑测试**：`.venv\Scripts\python.exe -m pytest -q` → PASS
- [ ] **Step 5: 提交** `feat(console): 真机团子页（启动、实时画面、状态卡片、日志）`

---

### Task 9: 设置页（分组目录、锚点、高亮）和设备页

**Files:**
- Write: `src/skydango/console/static/settings.js`、`src/skydango/console/static/device.js`
- Modify: `console.html`（`#page-settings`、`#page-device`）、`console.css`（追加 `/* ---- 设置 / 设备 ---- */`）
- Test: `tests/test_console_page.py`

**Interfaces:**
- Consumes: Task 4 公共函数；`Pages.settings.show(arg)` 的 `arg` 是设置 key（如 `secret.llm`）
- Produces: 设置项容器 `id="set-" + key.replaceAll(".", "-")`

- [ ] **Step 1: 写失败的测试**

```python
def test_settings_and_device_pages():
    b = bundle()
    for id_ in ("settings-toc", "groups", "savebar", "save", "discard", "settings-error", "check", "checks", "device-note"):
        assert f'id="{id_}"' in b, id_
    assert '"set-"+' in b.replace(" ", "") and 'replaceAll(".","-")' in b.replace(" ", "")
    for api in ("api/settings", "api/settings/test", "api/device"):
        assert api in b
    assert "Pages.settings" in b and "Pages.device" in b
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**——搬旧第 634~715 行：
  - 设置：左侧粘性目录 `#settings-toc`（`GROUPS` 每组一个链接，点了 `scrollIntoView`）；每个 `.row` 设 `id="set-"+f.key.replaceAll(".","-")`；`show(arg)`：设置没加载先 `loadSettings()`，有 `arg` 时滚到对应行并加 `.flash`（2 秒的樱花底渐隐动画，`prefers-reduced-motion` 时只描边不动画）。设备页「用 xxx」按钮 → `markDirty("device.serial", d)` 后 `go("settings","device.serial")`
  - 设备：只换样式；团子在跑时的置灰和说明照旧（`onState` 里更新）
- [ ] **Step 4: 跑测试**：`.venv\Scripts\python.exe -m pytest -q` → PASS
- [ ] **Step 5: 提交** `feat(console): 设置页加分组目录和跳转高亮，设备页换样式`

---

### Task 10: 浏览器核对、文档、收尾

**Files:**
- Modify: `CLAUDE.md`（「管理面板」一节：页面结构改成左栏 + 六页，`#overview` 并进「真机团子」，预检补查、出错原因、报告页；代码结构表里 `console/` 那行加 `reports.py`、静态文件拆分）
- 截图放 `tmp/console-redesign/`

这一任务由主会话（不是子代理）做，因为要用内嵌浏览器。

- [ ] **Step 1**: `.claude/launch.json` 的 `console` 配置起面板（`preview_start {name:"console"}`），逐项核对 spec §7「页面」1~6：缺 Key 预检和「去设置 →」跳转高亮；用 fake_claude 起沙盒（`config.toml` 临时设 `[brain] claude_path` 指向 `tests/fake_claude.py` 的包装——做法照 `tests/test_cli_sandbox.py`，核对完改回）、冒充发言、来去、快进、三栏各自滚动、大脑栏有轮次、下线；三种 `ask` 对话框取消不做、确定才做；造一个崩溃（比如临时把 Key 清掉再启动）看卡片上的原因；造一份报告文件看剧本页能读；1500px 和 900px 各截一张图
- [ ] **Step 2**: 有问题回到对应任务的文件修，修完重跑 `.venv\Scripts\python.exe -m pytest -q`
- [ ] **Step 3**: 更新 `CLAUDE.md`，提交 `docs: 管理面板改版`
- [ ] **Step 4**: 按 CLAUDE.md 规矩合并进 main 并推送
