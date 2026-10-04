# 去掉 viewer 网页，管理面板接管终端起的团子

日期：2026-10-04

## 0. 起因和目标

管理面板（`console`）做出来以后，viewer 自己的网页（`run --view` / `view`）就重复了，还带来两个问题：

- 会话里在终端跑 `run --live --view`，每次都弹出 viewer 网页；而管理面板完全不知道团子在跑——
  面板只认自己起的子进程（固定端口 `[console] child_port` 19391），终端起的走 `[viewer] port` 19399，面板探不到。
- 面板起一个、终端再起一个，两边端口不同，**可以同时有两个团子在线**（共用设备、令牌、`.brain-claude/`）。

目标：

1. 不再有 viewer 网页：`run` 不开网页、不开浏览器；`view` 命令删掉。
2. 管理面板知道真机团子在跑，不管它是面板起的还是终端起的；终端起的在面板里能看画面 / 大脑 / 聊天、能手动控制、能停止。
3. 同一时间只能有一个团子。

不做：把"回放录像看识别框"搬进管理面板（`view --images` 的用途，以后需要再单独做）；改沙盒（沙盒只由面板起，端口 19392，不受影响）。

## 1. `run`：viewer 只剩接口

viewer 现在是两样东西：① 自己的网页（`/` 页面、`brain_trace.js/css`、开浏览器）；② 给管理面板转发用的 HTTP 接口。
这次删 ①，留 ②，并且 ② **总是开**：

- `run`（大脑模式和 `--no-brain` 都算）启动时总在 `127.0.0.1:[viewer] port`（默认改成 **19391**）开接口：
  `/status` `/snapshot` `/brain` `/chat` `/control` `/control/options` `/inner` `/inner/forget` `/shutdown`，行为和现在面板子进程的一样
  （本机 Host 校验、POST 要 `X-Skydango` 头 + JSON、`/shutdown` = Ctrl+C 走正常收尾、手动控制总是挂上）。
- **端口被占就拒绝启动**：开接口放在建运行目录之后、碰设备之前；绑定失败打印
  「19391 端口上已经有一个团子在跑（终端或管理面板起的），先停掉它；不是团子的话改 [viewer] port」并以非 0 退出，不建设备、不起大脑。
  这样面板起的、终端起的，同一时间只能有一个。
- 删掉：
  - viewer 的 `/` 页面（`PAGE`）、`STATIC` 里只给网页用的 `brain_trace.js` / `brain_trace.css`；
    `stage.js`（画框代码）管理面板还在用，挪进 `console/static/`，面板的引用改过去；
  - `view` 命令（`cmd_view`、`_view`、`_view_frames` 和它的参数）；
  - `run` 的 `--no-browser`、`--viewer-port`；`_viewer()` 里开浏览器、"只有本机能看 / 局域网"的分支。
- `--view` 留一个隐藏参数（`argparse.SUPPRESS`），给了就打一行「--view 已经不用了：接口总是开着，画面在管理面板「真机团子」页」，照常运行。
  旧会话 / 旧文档里写着 `run --live --view` 的不会因为未知参数卡住。
- 配置：
  - `[viewer] port` 默认 19399 → 19391，面板和 `run` 都用它（同一份 config.toml / console.toml）；
  - `[viewer] host`、`[console] child_port` 进 `config.DEPRECATED`（读到只警告）。局域网 / 手机看画面的用法随之取消（面板本来也只听本机）；
  - `[viewer] fps` / `width` / `quality` 保留（面板的画面用）。
- 大脑模式里"局域网模式关掉手动控制"的分支（`cli._run_brain` 里 `cfg.viewer.host in LOCAL_HOSTS`）去掉，手动控制总是挂上。

### `/status` 多一节 `run`

```json
{"seq": 12, "age": 0.1, "info": {...},
 "run": {"pid": 1234, "run_dir": "D:/…/runs/1004-2130-live-brain", "live": true, "brain": true,
         "emotes": true, "duration": 0, "started": 1791234567.8, "console": false}}
```

- `started` 是墙钟（`time.time()`），面板算运行了多久；`console` = 有没有 `--parent-pid`（面板起的）。
- 面板**只认带 `run` 节的响应**是团子：别的程序占了 19391、或者旧版本的团子，都不当团子接管（见 §3）。

## 2. 管理面板：接管终端起的团子

现在的「孤儿」（`ConsoleServer.orphan`、`/api/orphan/stop`、页面上的「让它退出」）在团子这边换成「接管」。沙盒的孤儿逻辑不变。

- **发现**：面板启动时，以及每次 `/api/status`（页面本来每 2 秒拉一次）在槽空着（idle / exited / crashed）时探一次 `/status`；
  有带 `run` 节的响应且不是自己的子进程 → 槽进入 `running`：kind 团子，新增 `source = "terminal"`（面板自己起的是 `"console"`），
  `options` 取自 `/status.run`（live / brain / emotes / duration），`pid`、`run_dir`、`uptime`（按 `started` 算）也取自它。
  探活要短超时（0.5 秒），探不通不报错。
- 这样真机团子页、侧栏运行卡片、`/live/*` 转发、手动控制、内心页实时合并和删条目、沙盒的"团子在运行，先停团子"都照常工作——它们只看槽的状态。
  `start_run` 里"端口上有上次留下的团子"的检查改成：探到了就直接接管、返回"已经有一个团子在跑（终端起的），已接上"（409，页面刷新后就是在跑的样子）。
- **停止**：`POST /shutdown`，等 `[console] stop_timeout`；还在就按 `/status.run.pid` 结束进程树（`kill_tree`），提示检查轮盘（同现在的 `forced`）。
  进程没了（`/status` 连续 3 次探不通，或 pid 不在了）→ 槽变 `exited`，退出码未知（`exit_code = None`），卡片写「终端起的团子已经退出」。
  面板起的子进程的停止流程不变。
- **日志抽屉**：终端起的拿不到 stdout，改读 `run_dir/agent.log` 的尾巴：只要 INFO 及以上的行（和终端看到的一样；文件是 DEBUG 全量），
  每次 `/api/run/logs` 从上次读到的位置往后读（按字节偏移，文件变短 = 换了文件就从头），行号接着编，最多留 `[console] log_lines` 行。
  抽屉顶上注明「终端起的，日志来自 agent.log」。读不到文件只显示这一句提示。
- **面板退出不停终端起的团子**（不是它的子进程，`Runner.close()` 对接管的直接放手）；面板起的照旧停。
- 侧栏运行卡片和真机团子页顶栏标「终端起的」；停止按钮照常可用。
- 删掉：`ConsoleServer.orphan` 的团子部分、`state()` 里团子的 `orphan`、`stop_orphan` 的团子分支、`live.js` 里「让它退出」的按钮
  （`/api/orphan/stop` 留给沙盒，`kind` 不是 sandbox 时返回 400）。
  `forget` 里"上次留下的团子还在跑"的分支不再需要（探到了就已经接管、走 running 分支）。
- `console` 启动时打印的「注意：19391 端口上有上次留下的团子」改成「接上了终端起的团子（pid …，运行目录 …）」。

## 3. 边界

- **19391 被别的程序占着**：`run` 拒绝启动（§1）；面板探到没有 `run` 节的响应或探不通 → 不接管，预检里加一条「19391 端口被别的程序占着，团子起不来；改 [viewer] port」（只在端口确实绑不上时报：面板用一次试绑定判断）。
- **竞争**：面板起子进程的同时终端也起了一个：后绑端口的那个拒绝启动、退出；如果退出的是面板的子进程，槽变 crashed，下一次 `/api/status` 探到终端那个就接管。不会有两个团子同时碰设备。
- **接管后终端团子被强杀**（Ctrl+C 两次、关窗口）：探不通 → exited；面板不负责还原轮盘（和现在终端跑一样，靠退出收尾）。
- **面板起的子进程**：`build_command` 不再传 `--view` / `--viewer-port` / `--no-browser`，只传 `--parent-pid`；端口来自同一份配置。

## 4. 测试

- `run`：端口被占时拒绝启动、非 0 退出、没建设备（假设备计数）；`--view` 给了只打提示照常跑；`/status` 带 `run` 节且字段对；没有 `/` 页面（404）。
- `Runner` / `ConsoleServer`：
  - 槽空着时探到带 `run` 节的 → running、source terminal、options / pid / run_dir / uptime 来自 `/status`；
  - 探到不带 `run` 节的 → 不接管；
  - 接管的停止：先 `/shutdown`，超时按 pid 杀；探不通 3 次 → exited；
  - 接管的日志：读 agent.log 只要 INFO 以上、增量读、文件变短从头读、读不到给提示；
  - `close()` 不停接管的、停自己起的；
  - `start_run` 遇到终端团子 → 409 + 接管；沙盒的互斥照常。
- 删掉 `test_viewer.py` 里页面和 `view` 的用例，保留接口的；`console` 页面测试里 `stage.js` 的新路径、没有「让它退出」按钮。
- 全量：`.venv\Scripts\python.exe -m pytest -q`。

## 5. 真机验证（晚上）

1. 终端 `python -m skydango run --live`：不弹网页；打开管理面板，侧栏显示在跑（终端起的），真机团子页画面、大脑控制台、聊天记录都有内容，日志抽屉有 INFO 行。
2. 面板上点停止：团子正常收尾（轮盘换回、镜头复位、日记写了），面板变成已退出。
3. 终端团子在跑时，再在面板点启动 / 在另一个终端 `run`：都被拒绝，没有第二个团子。
4. 面板起团子，关掉面板：团子按看门狗正常退出（同现在）；终端起团子，关掉面板：团子接着跑。

## 6. 文档

- CLAUDE.md：「识别可视化（`[viewer]`）」整节改写成团子的接口（端口、`/status.run`、面板接管）；代码结构表里 `viewer.py` 一行；
  常用命令删 `view`、`run --view`；「管理面板」一节的父子进程、孤儿、停止说法；「统管大脑」里"调提示词时加 `--view`"改成看管理面板。
- docs/game-ops.md、进度文档里写 `run --view` / `view` 的地方（只改说明怎么看，不改历史记录）。
