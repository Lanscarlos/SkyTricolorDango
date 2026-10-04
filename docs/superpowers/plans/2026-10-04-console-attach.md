# 去掉 viewer 网页、管理面板接管终端起的团子 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `run` 不再出网页、总在 19391 开接口且端口被占就拒绝启动；管理面板认出终端起的团子并接管（看、控、停、读日志）。

**Architecture:** viewer 只剩 HTTP 接口，`/status` 多一节 `run` 当"我是团子"的身份证；`Runner` 多一种"接管"的槽（不是 Popen 出来的，靠探 `/status` 活着、按 pid 强杀、日志读 `agent.log`）；`ConsoleServer` 在槽空着时探端口、探到就接管，原来团子的「孤儿」逻辑删掉。

**Tech Stack:** Python 3.13 标准库 http.server / urllib；页面原生 JS；pytest。

**Spec:** `docs/superpowers/specs/2026-10-04-console-attach-design.md`

## Global Constraints

- 在自己的 worktree 里做（`using-git-worktrees`），**不在主目录切分支**；做完合并进 main 并推送。
- 测试：`.venv\Scripts\python.exe -m pytest -q`（worktree 里也用主目录的 `.venv`；全量放后台跑）。
- `[viewer] port` 默认 **19391**；`viewer.host`、`console.child_port` 进 `config.DEPRECATED`。
- `/status.run` 字段固定为：`pid` `run_dir` `live` `brain` `emotes` `duration` `started`（墙钟 `time.time()`）`console`（有没有 `--parent-pid`）。
- 面板**只认带 `run` 节（dict）的 `/status`** 为团子；探活超时 0.5 秒；接管的团子连续 **3** 次探不通 → `exited`。
- 端口被占时 `run` 打印：`19391 端口上已经有一个团子在跑（终端或管理面板起的），先停掉它；不是团子的话改 [viewer] port`（端口号取实际值），非 0 退出，不建设备。
- `--view` 隐藏参数给了打印：`--view 已经不用了：接口总是开着，画面在管理面板「真机团子」页`，照常运行。
- 接管的日志抽屉提示：`终端起的，日志来自 agent.log`；读不到文件时只显示这一句。
- 页面原生弹窗禁用（`ask()` / `toast()`）；颜色只用 `var(--…)`（已有测试查）。
- 用中文写注释、日志、提示。

## Review Focus

1. 面板自己起的子进程在 STARTING 阶段已经开了端口：探测不能把它当终端团子接管（只在槽 idle / exited / crashed 时探）。→ Task 4 `test_discover_skips_while_own_child_busy`
2. `agent.log` 里一条 INFO 带多行（traceback / 多行消息）：续行跟着它那一条走，DEBUG 的续行一起丢。→ Task 3 `test_logtail_keeps_continuation_lines_of_kept_records`
3. 终端团子退出后又起了一个新的（新 pid、新运行目录）：面板要重新接管、日志换成新文件，不能沿用旧的。→ Task 4 `test_reattach_after_exit_picks_new_run`
4. 面板退出（Ctrl+C）时接管的团子不能被停掉。→ Task 3 `test_close_leaves_attached_running`
5. 19391 被不是团子的程序占着：面板不接管、启动时报"被别的程序占着"，而不是"上次留下的团子"。→ Task 4 `test_start_reports_foreign_port_owner`

---

### Task 1: viewer 只剩接口，删掉网页和 `view` 命令

**Files:**
- Modify: `src/skydango/vision/viewer.py`（删 `PAGE`、`/` 路由、`STATIC`、`static_asset`、模块文档里网页的说法；加 `run_info`）
- Delete: `src/skydango/vision/static/brain_trace.js`、`brain_trace.css`；Move: `src/skydango/vision/static/stage.js` → `src/skydango/console/static/stage.js`（删空的 `vision/static/`）
- Modify: `src/skydango/console/server.py`（删 `/static/` 路由和 `static_asset` 导入）、`src/skydango/console/static/console.html:348`（`src="console/static/stage.js"`）
- Modify: `src/skydango/cli.py`（删 `cmd_view`、`_view`、`_view_frames`、`view` 子命令注册）；检查 `pyproject.toml` 的 package-data 里有没有 `vision/static`，有就删
- Test: `tests/test_viewer.py`、`tests/test_console_server.py`、`tests/test_console_page.py`

**Interfaces:**
- Produces: `Viewer.run_info: dict | None`（默认 None）；`Viewer.status()` 在 `run_info` 不为 None 时返回值多 `"run": dict(run_info)`。

- [ ] **Step 1: 改测试**
  - `test_viewer.py`：删 `test_http_server_serves_page_and_snapshot` 里取页面的部分（留 `/snapshot`）、`test_view_*` 四个、`test_page_*` 五个、`test_viewer_serves_brain_trace_assets`、`test_brain_trace_script_parses`、`test_viewer_page_script_parses`，以及 `from skydango.vision.viewer import PAGE`。新增：
    ```python
    def test_root_page_is_gone():  # spec §1：只剩接口
        v = viewer(); v.start()
        try:
            assert http_status(v.url) == 404
        finally:
            v.stop()

    def test_status_has_run_section_only_when_set():
        v = viewer(); v.start()
        try:
            assert "run" not in get_json(v.url + "status")
            v.run_info = {"pid": 7, "run_dir": "r", "live": True, "brain": True, "emotes": True, "duration": 0, "started": 1.0, "console": False}
            assert get_json(v.url + "status")["run"]["pid"] == 7
        finally:
            v.stop()
    ```
    （`http_status` / `get_json` 用文件里已有的请求辅助函数，没有就加两个带 `Host: 127.0.0.1:<port>` 的小函数。）
  - `test_console_server.py`：删 `test_serves_shared_brain_trace_assets`；新增 `test_serves_stage_js_from_console_static`：`GET /console/static/stage.js` 200、内容含 `Stage`；`GET /static/stage.js` 404。
  - `test_console_page.py:27-29`：断言改成 `'src="console/static/stage.js"'` 且在 `live.js` 之前；`brain_trace` 不出现。
  - 有测 `view` 子命令注册的（`test_view_command_is_registered`）删掉，加一条 `cli.main(["view"])` 报 argparse 错误（`SystemExit` 码 2）。

- [ ] **Step 2: 跑这三个文件确认新测试失败**
  Run: `.venv\Scripts\python.exe -m pytest -q tests/test_viewer.py tests/test_console_server.py tests/test_console_page.py`
  Expected: 新加的几条 FAIL（404 断言、`run` 节、stage.js 路径）。

- [ ] **Step 3: 实现**：按 Files 删改；`console/server.py` 的 `_console_static` 本来就按目录列表给文件，`stage.js` 挪进去即可。

- [ ] **Step 4: 跑全量**（后台）Expected: 全部 PASS（别的文件引用 `PAGE` / `static_asset` / `_view` 的一并清掉）。

- [ ] **Step 5: Commit** `refactor(viewer): 删掉 viewer 网页和 view 命令，只留给管理面板的接口`

---

### Task 2: `run` 总是开接口、端口被占拒绝启动

**Files:**
- Modify: `src/skydango/config.py`（`ViewerConfig.port = 19391`、删 `ViewerConfig.host`、删 `ConsoleConfig.child_port`；`DEPRECATED` 加 `"viewer.host"`、`"console.child_port"`）
- Modify: `src/skydango/cli.py`：`_viewer`、`cmd_run`、`run` 子命令参数、`_run_brain` 手动控制分支（去掉 `LOCAL_HOSTS` 判断和局域网 warning）、`cmd_console`（用 `cfg.viewer.port`）
- Modify: `src/skydango/console/runner.py:40-50`（`build_command` 不再带 `--view` `--viewer-port` `--no-browser`）
- Modify: `src/skydango/vision/viewer.py`（`LOCAL_HOSTS` 不再有人用就删；`_Server` 绑定 `127.0.0.1`）、`src/skydango/console/settings.py`（设置清单里有 `viewer.host` / `console.child_port` 的删掉）、`config.example.toml`
- Test: `tests/test_cli_brain.py`（`:243`、`:333-362` 一带）、`tests/test_console_runner.py:56-60`、`tests/test_config.py`（或现有放 DEPRECATED 测试的文件）

**Interfaces:**
- Produces: `cli._viewer(cfg: Config, brain: bool, on_shutdown: Callable[[], None], run_info: dict) -> Viewer`（不开浏览器、不打印"局域网"；绑定失败 `raise SystemExit(<Global Constraints 里那句>)`）。
- Produces: `cli._run_info(cfg: Config, run: RunDir, args) -> dict`（字段见 Global Constraints；`started=time.time()`、`pid=os.getpid()`、`run_dir=str(run.path.resolve())`、`console=args.parent_pid is not None`、`live=not cfg.reply.dry_run`、`brain=cfg.brain.enabled`、`emotes=not args.no_emotes`、`duration=float(args.duration or 0)`）。
- Consumes: Task 1 的 `Viewer.run_info`。

- [ ] **Step 1: 写测试**
  - `test_run_refuses_when_port_taken`：先用 socket 占住一个空闲端口，`Config(viewer=ViewerConfig(port=该端口))`，monkeypatch `cli._run_brain` / `cli._run_agent` 为记录调用的假函数、`cli._device` 为会 `pytest.fail` 的函数；`cli.cmd_run(cfg, args)` 抛 `SystemExit`，消息含 `已经有一个团子在跑`；假 `_run_brain` 没被调用。
  - `test_run_always_opens_api_with_run_info`：假 `_run_brain(cfg, run, world, duration, viewer, **kw)` 里断言 `viewer is not None`、`viewer.run_info["console"] is True`（传 `--parent-pid 4321`）、`viewer.run_info["live"] is False`（`--dry-run`）、`viewer.run_info["pid"] == os.getpid()`。
  - `test_view_flag_is_ignored_with_notice`：`cli.main(["run", "--view"])`（假 `_run_brain`）不报错，`capsys` 输出含 `--view 已经不用了`。
  - `test_old_flags_are_gone`：`cli.main(["run", "--no-browser"])` 和 `["run", "--viewer-port", "1"]` 都 `SystemExit` 码 2。
  - `test_manual_control_always_attached`：沿用现有大脑组装测试的做法，`viewer.control` 不为 None（原来局域网分支的测试 `:362` 删掉）。
  - `test_build_command_is_explicit` 期望改成 `[..., "--duration", "90.0", "--parent-pid", "42"]`。
  - 配置：`viewer.host`、`console.child_port` 出现在 toml 里只警告不报错；`Config().viewer.port == 19391`。

- [ ] **Step 2: 跑这些测试确认失败**
  Run: `.venv\Scripts\python.exe -m pytest -q tests/test_cli_brain.py tests/test_console_runner.py tests/test_config.py`

- [ ] **Step 3: 实现**
  - `run` 子命令：删 `--no-browser`、`--viewer-port`；`--view` 改 `help=argparse.SUPPRESS`。
  - `cmd_run`：建 `RunDir`、`attach_log` 之后**总是** `viewer = _viewer(cfg, cfg.brain.enabled, request_exit, _run_info(cfg, run, args))`，`args.view` 时先打印提示。`_viewer` 在 try 里，`finally` 照旧 `viewer.stop()`（viewer 可能是 None：绑定失败时）。
  - `_viewer` 的 `on_shutdown` 总是挂（只听本机了）；打印一行 `接口：http://127.0.0.1:<port>/（管理面板「真机团子」页看画面）`。

- [ ] **Step 4: 跑全量**（后台）Expected: PASS。

- [ ] **Step 5: Commit** `feat(run): 接口总是开在 19391、端口被占拒绝启动，去掉 --no-browser / --viewer-port`

---

### Task 3: `Runner` 接管终端起的团子 + 读 agent.log

**Files:**
- Create: `src/skydango/console/logtail.py`
- Modify: `src/skydango/console/runner.py`
- Test: `tests/test_console_logtail.py`（新）、`tests/test_console_runner.py`

**Interfaces:**
- Produces（`runner.py`）：
  - `probe_run(port: int, timeout: float = 0.5) -> dict | None`：GET `/status`（走 `LOCAL`），返回 `data["run"]`（必须是 dict），否则 None；任何异常 None。
  - `Runner.__init__` 多一个参数 `probe_run: Callable[[int], dict | None] = probe_run`、`poll: float = 1.0`（接管后的探活间隔，测试传小值）。
  - `Runner.attach(info: dict) -> bool`：槽在 IDLE / EXITED / CRASHED 时：清日志、`_kind="dango"`、`_source="terminal"`、`_state=RUNNING`、`_options=LaunchOptions(brain, live, emotes, duration)`（取自 info）、`_run_dir=info["run_dir"]`、`_attached_pid=info["pid"]`、`_attached_started=info["started"]`、日志尾巴 `LogTail(Path(run_dir) / "agent.log", log_lines)`，起后台线程 `_watch_attached`；返回 True。槽忙返回 False。
  - `status()` 多 `"source": "console" | "terminal"`；接管时 `pid` = info 的 pid，`uptime = time.time() - started`。
  - `stop()`：接管时后台线程 `_stop_attached`：`_shutdown(port)` 后每 0.5 秒 `probe_run`，`stop_timeout` 内探不通 → EXITED；超时 → `_kill(pid)`、`forced=True`、EXITED。
  - `kill()`：接管时直接 `_kill(pid)`。
  - `close()`：接管时**只放手**（停后台线程、槽回 IDLE），不发 shutdown；自己起的照旧。
  - `logs(after)`：接管时先把 `LogTail.read_new()` 的行按 `_line` 同样的方式编号进环形缓冲；文件读不到时 `{"next": 0, "lines": [], "note": "终端起的，日志来自 agent.log"}`，读得到也带同一个 `note`。
  - `_watch_attached`：每 `poll` 秒 `probe_run(port)`；结果不是 None 但 `pid` 变了（另一个团子）也算这个退出了；连续 3 次 None → `_state=EXITED`、`_exit_code=None`、线程结束。
- Produces（`logtail.py`）：`class LogTail: __init__(self, path: Path)`、`read_new(self) -> list[str] | None`（None = 读不到文件）。按字节偏移增量读，文件比偏移短就从 0 读；只读完整的行（结尾没换行的留到下次）；`HEAD = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} (DEBUG|INFO|WARNING|ERROR|CRITICAL) ")`，INFO 以上的一条留下，续行（不匹配 HEAD 的行）跟随它前面那条的去留；UTF-8 `errors="replace"`。

- [ ] **Step 1: 写 `LogTail` 测试**（`tests/test_console_logtail.py`）
  - `test_logtail_keeps_info_and_above`：写 DEBUG / INFO / WARNING 各一行 → `read_new()` 只有后两行。
  - `test_logtail_keeps_continuation_lines_of_kept_records`：ERROR 一行 + 两行 traceback 续行 + DEBUG 一行 + 一行续行 → 只留 ERROR 和它的两行续行。
  - `test_logtail_reads_incrementally_and_waits_for_full_line`：先写一行完整 + 半行，读到 1 行；补完半行，再读到 1 行。
  - `test_logtail_restarts_when_file_shrinks`、`test_logtail_missing_file_is_none`。

- [ ] **Step 2: 写 Runner 测试**（`tests/test_console_runner.py`，用假的 `probe_run` / `shutdown` / `kill`，`poll=0.05`）
  - `test_attach_shows_terminal_run`：`attach(INFO)` → `status()` 的 `state == "running"`、`source == "terminal"`、`pid == INFO["pid"]`、`options["live"] is True`、`run_dir` 对、`uptime > 0`。
  - `test_attach_refused_while_busy`：先 `start(SLEEPER…)`，`attach` 返回 False。
  - `test_attached_exits_after_three_failed_probes`：假 probe 先返回 INFO 两次、再一直 None → 最终 `exited`、`exit_code is None`。
  - `test_attached_stop_sends_shutdown_then_kills_by_pid`：probe 一直返回 INFO、`stop_timeout=0.3` → `shutdown` 被调用、`kill` 收到 INFO 的 pid、`forced is True`、`exited`。
  - `test_attached_stop_clean`：shutdown 后 probe 返回 None → `exited`、`kill` 没被调用、`forced is False`。
  - `test_close_leaves_attached_running`（Review Focus 4）：`attach` 后 `close()` 立即返回，`shutdown` / `kill` 都没被调用，`state == "idle"`。
  - `test_attached_logs_come_from_agent_log`：`run_dir` 指向 tmp 目录、写 `agent.log` 两行 INFO → `logs(0)["lines"]` 是这两行、`note` 是 Global Constraints 那句；之后追加一行，`logs(2)` 只返回新行。

- [ ] **Step 3: 跑确认失败** Run: `.venv\Scripts\python.exe -m pytest -q tests/test_console_logtail.py tests/test_console_runner.py`

- [ ] **Step 4: 实现** `logtail.py` 和 `Runner` 的接管分支。自己起的子进程路径一行不改（`source="console"`）。

- [ ] **Step 5: 跑这两个文件** Expected: PASS。

- [ ] **Step 6: Commit** `feat(console): Runner 能接管终端起的团子，日志读 agent.log`

---

### Task 4: `ConsoleServer` 探端口接管、去掉团子的孤儿，页面标出「终端起的」

**Files:**
- Modify: `src/skydango/console/server.py`、`src/skydango/cli.py`（`cmd_console` 的提示）
- Modify: `src/skydango/console/static/common.js:143-146`（运行卡片）、`live.js`（顶栏、日志抽屉 note、删「让它退出」）、`console.html`（日志抽屉放 note 的元素，如需要）
- Test: `tests/test_console_server.py`、`tests/test_console_page.py`、`tests/test_console_inner.py`（`forget` 相关）

**Interfaces:**
- Consumes: Task 3 的 `probe_run`、`Runner.attach`、`status()["source"]`、`logs()["note"]`。
- Produces: `ConsoleServer.discover() -> bool`：槽不在 BUSY 时调 `probe_run(self.child_port)`，有就 `runner.attach(info)`；返回接管了没有。`ConsoleServer.__init__` 多参数 `probe: Callable[[int], dict | None] = probe_run`、`port_free: Callable[[int], bool] = port_free`。
- Produces（`runner.py` 或 `server.py`）：`port_free(port: int) -> bool`：在 `127.0.0.1` 上试绑定再关掉。

- [ ] **Step 1: 写测试**（`FakeRunner` 加 `attach(info)` 记录调用、把 state 设 running；`make_server` 能传假 `probe` / `port_free`）
  - `test_discover_attaches_terminal_dango`：probe 返回 INFO → 第一次 `GET /api/state` 后 `runner.attached == [INFO]`、返回的 `run.state == "running"`。
  - `test_discover_skips_while_own_child_busy`（Review Focus 1）：FakeRunner state = `starting` → probe 不被调用。
  - `test_reattach_after_exit_picks_new_run`（Review Focus 3，用真 `Runner` + 假 probe，`poll` 小）：接管 A（pid 1、run_dir a）→ probe 改返回 None 直到 `exited` → 再返回 B（pid 2、run_dir b）→ `state()` 后 `run.pid == 2`、`run_dir == "b"`。
  - `test_start_attaches_instead_of_second_dango`：probe 返回 INFO 时 `POST /api/run/start` → 409，问题文字含 `终端起的`，`runner.started` 为空，`runner.attached` 有一条。
  - `test_start_reports_foreign_port_owner`（Review Focus 5）：probe None、`port_free` False → 409，问题文字含 `被别的程序占着`。
  - `test_orphan_stop_is_sandbox_only`：`POST /api/orphan/stop {}` → 400；`{"kind": "sandbox"}` 照旧。删 `test_orphan_stop_sends_shutdown`、`test_start_refused_while_orphan_holds_port`。
  - `test_logs_pass_note`：`/api/run/logs` 透传 runner 的 `note`。
  - `test_console_page.py`：`live.js` 不再含 `api/orphan/stop`；`common.js` 含 `终端起的`。
  - `forget`：删"上次留下的团子还在跑"的用例，加 probe 返回 INFO 时 forget 走 running 分支（转发）。

- [ ] **Step 2: 跑确认失败** Run: `.venv\Scripts\python.exe -m pytest -q tests/test_console_server.py tests/test_console_page.py tests/test_console_inner.py`

- [ ] **Step 3: 实现**
  - `__init__` 里 `self.orphan = …` 换成 `self.discover()`（构造完 runner 之后）；`state()`、`/api/status`（页面每 2 秒拉的那个）开头调 `discover()`，`state()` 不再给团子的 `orphan`。
  - `start_run`：原来 `probe_status(self.child_port)` 那段换成：`discover()` 接管了 → 409 `problem("已经有一个团子在跑（终端起的），已接上")`；没接管且 `not port_free(self.child_port)` → 409 `problem(f"{port} 端口被别的程序占着，团子起不来；改 [viewer] port")`。
  - `sandbox_problems` 里团子端口的检查换成 `discover()`（接管了就是"团子在运行，先停团子"那条，`_other_busy` 已经会给）。
  - `forget` 删 `probe_status(self.child_port)` 那段。`stop_orphan` 只留沙盒分支，别的 400。
  - `cmd_console`：`server.discover()` 之后 runner 状态是 running 且 source terminal 时打印 `接上了终端起的团子（pid …，运行目录 …）`；删原来「注意：…上次留下的团子」。
  - 页面：运行卡片 `sub` 前加 `终端起的`（`run.source === "terminal"`）；真机页顶栏同样标；日志抽屉顶部显示 `logs.note`（`textContent`）；删「让它退出」按钮和 `st.orphan` 分支。

- [ ] **Step 4: 跑全量**（后台）Expected: PASS。

- [ ] **Step 5: Commit** `feat(console): 探到终端起的团子就接管，去掉团子的「让它退出」`

---

### Task 5: 文档

**Files:**
- Modify: `CLAUDE.md`、`docs/game-ops.md`（如有 `--view` / `view` 的操作说明）、`config.example.toml`（若 Task 2 没改完）

- [ ] **Step 1: 改 CLAUDE.md**
  - 代码结构表 `vision/viewer.py` 一行：改成"团子的 HTTP 接口（给管理面板）：`/status`（带 `run` 节）…"。
  - 「识别可视化（`[viewer]`，`view` / `run --view`）」整节改写为「团子的接口（`[viewer]`）」：`run` 总在 `127.0.0.1:19391` 开、端口被占拒绝启动（同一时间一个团子）、`/status.run` 字段、管理面板接管终端起的团子（看 / 控 / 停 / 日志读 agent.log、面板退出不停它）、`--view` 已不用；大脑时间线、手动控制的说法改指管理面板「真机团子」页。删局域网 / 手机看的说法。
  - 「管理面板」一节：父子进程命令行去掉 `--view --viewer-port 19391 --no-browser`；"19391 已经有人响应…这时不让启动新的"改成接管；「统管大脑」里"调提示词时加 `--view`"改成看管理面板「真机团子」页的大脑控制台。
  - 常用命令删 `view`、`run --view` 两行。
- [ ] **Step 2: `grep -rn "\-\-view\|skydango view\|19399" CLAUDE.md docs/game-ops.md config.example.toml src` 只剩历史 spec / 进度文档**（不改历史记录）。
- [ ] **Step 3: Commit** `docs: viewer 网页去掉后的说明（团子接口、面板接管）`

---

### 收尾

- [ ] `requesting-code-review` 自查整条分支 → 合并进 main 并推送（`finishing-a-development-branch`，按 CLAUDE.md 直接合并）。
- [ ] 真机验证（晚上，spec §5 四步）留给用户。
