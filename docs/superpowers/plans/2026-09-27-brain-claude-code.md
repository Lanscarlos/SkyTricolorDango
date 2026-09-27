# 大脑改用 Claude Code + 眼睛 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把统管大脑从“程序里直接调 Anthropic API”换成常驻的无界面 Claude Code（订阅登录），身体的工具经本机 MCP 服务交给它；新增眼睛（一次性 Haiku 调用）把截图写成文字，大脑平时只收文字；顺手修完最终审查的问题。

**Architecture:** `brain/claude.py` 负责起 Claude Code 进程、stream-json 收发、和用户自己的 Claude Code 配置隔离（单独配置目录 + `claude setup-token` 令牌）。`brain/session.py` 是常驻大脑进程（挂了用 `--resume` 接回）。`brain/mcp_server.py` 用 `mcp` 包在后台线程起一个 127.0.0.1 的 MCP 服务，工具经 `ToolBox`（参数校验、每轮计数）→ `Body.call()` 在身体线程执行。`brain/eyes.py` 后台看图、缓存最新描述。`brain/loop.py` 改写成：醒来 → 拼纯文字消息 → `session.send()` 等 `result`。删掉 `client.py` / `context.py` / `budget.py`。

**Tech Stack:** Python 3.13、pytest、Claude Code 2.1.233（`claude -p` stream-json）、`mcp` 2.2.0（`MCPServer`）、uvicorn、OpenCV。

**Spec:** `docs/superpowers/specs/2026-09-27-brain-claude-code-design.md`（身体、事件、护栏、醒睡节奏另见 `2026-09-27-brain-design.md`）

## Global Constraints

- 运行测试：`python -m pytest -q`；单元测试不连模拟器、**不启动真的 Claude Code**（用 `tests/fake_claude.py` 假进程）、不联网。
- 代码风格：中文注释 / 日志 / 报错 / 给模型看的文字，和现有文件一致；注释只写“为什么”。
- **只有身体线程碰设备**：大脑的工具一律经 `Body.call()`；眼睛只读 `Body.last_frame`，不自己截图。
- 大脑子进程命令固定带：`-p --input-format stream-json --output-format stream-json --verbose --model <brain.model> --effort <brain.effort> --mcp-config <文件> --strict-mcp-config --tools "" --allowedTools mcp__sky --permission-mode dontAsk --disable-slash-commands --append-system-prompt-file <文件>`，接回时加 `--resume <session_id>`。
- 眼睛子进程命令固定带：`-p --input-format stream-json --output-format stream-json --verbose --model <brain.eyes_model> --effort low --tools "" --strict-mcp-config --permission-mode dontAsk --disable-slash-commands --system-prompt <眼睛的规则>`。
- 子进程环境：去掉 `ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN`、`CLAUDE_CODE_OAUTH_TOKEN`，再设 `CLAUDE_CODE_OAUTH_TOKEN=<从 brain.token_env 读的令牌>`、`CLAUDE_CONFIG_DIR=<brain.config_dir 的绝对路径>`。
- 子进程结束一律按进程树（Windows：`taskkill /F /T /PID`）；工作目录是运行目录下的空文件夹（不读项目的 AGENTS.md / CLAUDE.md）。
- **不要删** `responder.RULES` 的“身份”一节和 `_CLAIMS_HUMAN`；`say` 必须过 `clean_reply`；大脑提示词保留“身份”“底线”两节。
- 默认 dry-run：说话、动作、转视角、接受请求只打印；dry-run 不写记忆。`brain.enabled` 默认 `false`。
- 不做移动；不给大脑点屏幕 / 任意按键的工具。
- 临时文件放项目 `tmp/`；别用 `git stash`（和别的会话共用）。
- 提交信息：`feat: ...` / `fix: ...` / `refactor: ...` / `docs: ...` 中文短句，末尾带 `Co-Authored-By` 行（按执行时的会话提示写）。

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/skydango/chat/responder.py` | 改 | `_CLAIMS_HUMAN` 补漏 |
| `src/skydango/chat/llm.py`、`src/skydango/config.py`（`LlmConfig`） | 改 | `max_retries`（备用回复用短超时、不重试） |
| `src/skydango/brain/camera.py` | 改 | 方向键 try/finally 松开；每步记偏移；`around()` 环顾一圈 |
| `src/skydango/brain/body.py` | 改 | 感知出错隔离；牵手随 leave 清掉；陌生人规则限制；聊天加「」；shutdown 让排队命令失败；`fresh_frame()` / `capture_around()`；`look_at` 裁上次看的图 |
| `src/skydango/brain/events.py` | 改 | `subscribe()`（眼睛订阅事件） |
| `src/skydango/brain/claude.py` | 新建 | `ClaudeError`、`claude_env`、`resolve_claude`、`StreamProcess`、`check_result`、`one_shot` |
| `tests/fake_claude.py` | 新建 | 测试用假 Claude Code |
| `src/skydango/brain/session.py` | 新建 | `BrainSession`：常驻大脑进程 |
| `src/skydango/brain/eyes.py` | 新建 | `Eyes`、`eyes_command`、眼睛的提示词 |
| `src/skydango/brain/tools.py` | 改写 | `DESCRIPTIONS`、`ACTIONS`、`ToolBox`（每轮计数、look 走眼睛、look_around） |
| `src/skydango/brain/mcp_server.py` | 新建 | `build_server`、`SkyServer` |
| `src/skydango/brain/prompt.py` | 改 | 工具说明改成新语义；`brain_prompt()`；`SUMMARY_REQUEST` |
| `src/skydango/brain/loop.py` | 改写 | `Brain`：醒睡、拼消息、`session.send`、失败 / 额度 / 离线、`farewell()` |
| `src/skydango/brain/client.py`、`context.py`、`budget.py` 及测试 | 删 | |
| `src/skydango/cli.py` | 改 | `_brain_env`、`_run_brain`、`cmd_look` |
| `src/skydango/config.py`（`BrainConfig`）、`config.example.toml`、`.gitignore`、`pyproject.toml` | 改 | 配置 |
| `AGENTS.md`、`README.md` | 改 | 文档 |

---

### Task 1: 一并修复（身份过滤、视角、身体、规则、备用回复客户端）

**Files:**
- Modify: `src/skydango/chat/responder.py`（`_CLAIMS_HUMAN`）
- Modify: `src/skydango/config.py`（`LlmConfig.max_retries`）、`src/skydango/chat/llm.py`
- Modify: `src/skydango/brain/camera.py`
- Modify: `src/skydango/brain/body.py`
- Test: `tests/test_responder.py`、`tests/test_llm.py`、`tests/test_brain_camera.py`、`tests/test_brain_body.py`

**Interfaces:**
- Produces: `LlmConfig.max_retries: int = 2`（两个客户端都传给 SDK）；`Body` 新行为（见下）；`Body.stopped: bool`。

- [ ] **Step 1: Write the failing tests**

`tests/test_responder.py` 末尾追加：

```python
def test_claims_human_filter_catches_variants():
    from skydango.chat.responder import clean_reply

    for bad in ("我当然是真人", "我可是活人", "当然不是AI啦", "真人一个", "我是真人", "我不是机器人哦"):
        assert clean_reply(bad, 40) is None, bad
    for ok in ("你才是AI吧", "哈哈哈随你怎么想", "我是AI", "在呢"):
        assert clean_reply(ok, 40) == ok, ok
```

`tests/test_llm.py` 末尾追加：

```python
def test_max_retries_is_passed_to_sdk(monkeypatch):
    made = {}
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda **kw: made.update(kw) or types.SimpleNamespace(messages=FakeMessages())
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("SKYDANGO_TEST_KEY", "k")
    cfg = llm_cfg("claude-sonnet-5")
    cfg.max_retries = 0
    AnthropicClient(cfg)
    assert made["max_retries"] == 0
```

`tests/test_brain_camera.py` 末尾追加：

```python
def test_arrow_key_is_released_even_when_interrupted():
    c, dev, _ = cam(panel=False)

    def boom(s):
        raise KeyboardInterrupt

    c.sleep = boom
    with pytest.raises(KeyboardInterrupt):
        c.move("left", 3)
    assert ("hw_up", 105) in dev.calls  # 方向键一定松开，不然镜头会一直转
    assert c.offset["turn"] == 0  # 第一步就被打断：这一步没走完，不记


def test_offset_counts_each_finished_step():
    c, dev, _ = cam(panel=False)
    presses = []

    def sleep(s):
        presses.append(s)
        if len(presses) == 3:  # 每步两次 sleep（按住、松开后）：第 3 次是第 2 步按住时，这时出错
            raise RuntimeError("adb 失败")

    c.sleep = sleep
    with pytest.raises(RuntimeError):
        c.move("right", 3)
    assert c.offset["turn"] == 1


def test_around_captures_four_directions_and_turns_full_circle():
    c, dev, state = cam()
    shots = []
    frames = c.around(lambda: shots.append(len(shots)) or len(shots))
    assert frames == [1, 2, 3, 4]
    assert dev.calls.count(("hw_down", 106)) == 8  # 每 90°（2 步）一张，最后再转 90° 回到原来的朝向
    assert c.offset["turn"] == 0 and state["panel"] is True
```

`tests/test_brain_body.py`：把 `test_chat_messages_become_events` 里的期望改成带「」：

```python
    assert [e.text for e in events.drain()] == ["聊天  懒洋洋大王：「在吗」", "聊天  （看不出是谁）：「hi」"]
```

并在文件末尾追加：

```python
def test_sensing_error_does_not_block_chat_and_commands(clock):
    class BrokenEnv(FakeEnv):
        def observe(self, frame, now, panel_visible):
            raise RuntimeError("OCR 出错")

    b, _, reader, events = body(clock, env=BrokenEnv())
    reader.batches = [[msg("在吗")]]
    b.step()
    kinds = [e.kind for e in events.drain()]
    assert "chat" in kinds and "error" in kinds


def test_screenshot_error_with_empty_message(clock):
    b, device, _, events = body(clock)

    def broken():
        raise RuntimeError()

    device.screenshot = broken
    b.step()
    assert events.drain()[-1].text == "截图失败：RuntimeError"


def test_holding_is_cleared_when_friend_leaves(clock):
    env = FakeEnv()
    b, _, _, events = body(clock, env=env)
    env.near = ["懒洋洋大王"]
    b.step()
    b.holding = "懒洋洋大王"
    env.near = []
    b.step()
    assert b.holding is None
    assert any(e.kind == "released" for e in events.drain())


def test_strangers_can_only_get_candle(clock):
    social = FakeSocial()
    b, _, _, _ = body(clock, social=social)
    for kind in ("hand", "hug", "*"):
        with pytest.raises(ToolError, match="点火"):
            b.set_policy("stranger", kind, True)
    b.set_policy("stranger", "candle", True)
    b.set_policy("stranger", "hand", False)  # 不接总是可以的
    assert social.policy_calls == [("stranger", "candle", True), ("stranger", "hand", False)]


def test_shutdown_fails_queued_commands_and_refuses_new_ones(clock):
    b, _, _, _ = body(clock)
    results = []

    def caller():
        try:
            b.call(lambda: "做了")
        except ToolError as exc:
            results.append(str(exc))

    t = threading.Thread(target=caller)
    t.start()
    while b._commands.empty():
        pass
    b.shutdown()
    t.join(2)
    assert results and "停" in results[0]
    with pytest.raises(ToolError, match="停"):
        b.call(lambda: 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_responder.py tests/test_llm.py tests/test_brain_camera.py tests/test_brain_body.py -q`
Expected: FAIL（过滤漏掉变体；`max_retries` 没传；方向键没松开；`around` 不存在；聊天没加「」；感知出错连带聊天丢失；IndexError；牵手没清；陌生人规则没拦；shutdown 不处理排队命令）

- [ ] **Step 3: Write minimal implementation**

`src/skydango/chat/responder.py`：把 `_CLAIMS_HUMAN` 换成（注释保留原来那句）：

```python
# 陪玩可以不主动提自己是 AI，但不能声称自己是真人：提示词里写了，模型还是偶尔会说，这里硬拦
# 宁可多拦：“我当然是真人”“我可是活人”“当然不是AI啦”“真人一个”都要拦下
_CLAIMS_HUMAN = re.compile(r"我.{0,4}是.{0,2}(真人|人类|活人)|(真人|活人)一个|不是\s*(ai|机器人|人工智能|bot)", re.I)
```

`src/skydango/config.py` 的 `LlmConfig` 末尾加：

```python
    max_retries: int = 2  # SDK 自己的重试次数；大脑离线时的备用回复用 0（在身体线程里跑，不能卡太久）
```

`src/skydango/chat/llm.py`：`OpenAICompatClient.__init__` 里改成

```python
        self._client = OpenAI(
            base_url=cfg.base_url or None, api_key=_api_key(cfg), timeout=cfg.timeout, max_retries=cfg.max_retries
        )
```

`AnthropicClient.__init__` 里 `kwargs = {...}` 改成

```python
        kwargs = {"api_key": _api_key(cfg), "timeout": cfg.timeout, "max_retries": cfg.max_retries}
```

`src/skydango/brain/camera.py`：`move` / `reset` / `_press` 换成下面的实现，并新增 `around`：

```python
    def move(self, action: str, steps: int = 1) -> str:
        if action not in KEYS:
            raise ValueError(f"不认识的视角操作：{action}（可以用 {'、'.join(KEYS)}）")
        steps = max(1, min(int(steps), MAX_STEPS))
        with self._ready():
            for _ in range(steps):
                self._step(action)
        return self.describe()

    def reset(self) -> str:
        undo = [(UNDO[axis][1] if value > 0 else UNDO[axis][0], abs(value)) for axis, value in self.offset.items() if value]
        if not undo:
            return "镜头已经在原位"
        with self._ready():
            for action, n in undo:
                for _ in range(n):
                    self._step(action)
        return "镜头转回原位了（来回转会有一点偏差）"

    def around(self, capture: Callable[[], Any], turns: int = 4, steps: int = 2) -> list:
        """环顾一圈：每转 steps 步（约 90°）截一张，共 turns 张，最后转满一圈回到原来的朝向（偏移不变）。"""
        frames = []
        with self._ready():
            for i in range(turns):
                if i:
                    for _ in range(steps):
                        self._press("right")
                    self.sleep(0.3)  # 等镜头停稳再截
                frames.append(capture())
            for _ in range(steps):
                self._press("right")
        return frames

    def _step(self, action: str) -> None:
        """走一步并马上记下偏移：中途出错时复原也准。"""
        self._press(action)
        axis, sign = AXIS[action]
        self.offset[axis] += sign

    def _press(self, action: str) -> None:
        code = KEYS[action]
        if action.startswith("zoom"):
            self.device.hw_key(code)
            self.sleep(0.3)
            return
        self.device.hw_key_down(code)
        try:
            self.sleep(self.step)
        finally:
            self.device.hw_key_up(code)  # Ctrl+C / 出错时也要松开，不然镜头会一直转
        self.sleep(0.2)
```

文件头的 import 改成 `from collections.abc import Callable, Iterator` 保持，另加 `from typing import Any`。

`src/skydango/brain/body.py`：

1) 文件里加一个工具函数（放在 `class ToolError` 上面）：

```python
def _first_line(exc: BaseException) -> str:
    return (str(exc).splitlines() or [type(exc).__name__])[0]
```

2) `__init__` 末尾加：

```python
        self.stopped = False  # shutdown 之后不再接大脑的命令
```

3) `step()` 换成（感知出错不影响读聊天、执行命令、备用回复）：

```python
    def step(self) -> None:
        now = self.clock()
        frame = None
        try:
            frame = self.device.screenshot()
        except Exception as exc:  # 实测 adb 截图偶尔会连续失败几秒
            self.events.put("error", f"截图失败：{_first_line(exc)}")
        if frame is not None:
            self.last_frame = frame
            fresh: list[Message] = []
            try:
                fresh = self.reader.read(frame, now)
            except Exception:
                log.exception("读聊天出错")
            try:
                self._sense(frame, now)
            except Exception:
                log.exception("感知出错，这一圈跳过")
                self.events.put("error", "身体感知出错了（详见日志）")
            self._heard(fresh, frame, now)
        self._run_commands()
        self._fallback(now)

    def _sense(self, frame, now: float) -> None:
        if self.cfg.vision.mode == "log":
            self.panel.maybe_reopen(now)
            self._watch_panel(now)
        self._watch_screen(frame, now)
        if self.env is not None:
            self.env.observe(frame, now, panel_visible=self.reader.panel_closed_since is None)
            self._watch_people(now)
```

4) `shutdown()` 换成：

```python
    def shutdown(self) -> None:
        """退出时（不等大脑）：不再接命令、排队的命令全部失败、镜头转回原位、轮盘换回去。"""
        self.stopped = True
        while True:
            try:
                _, fut = self._commands.get_nowait()
            except queue.Empty:
                break
            if fut.set_running_or_notify_cancel():
                fut.set_exception(ToolError("身体已经停了"))
        if self.camera is not None and not self.cfg.reply.dry_run:
            try:
                log.info(self.camera.reset())
            except Exception:
                log.exception("镜头没转回原位")
        if self.emotes is not None:
            try:
                self.emotes.restore()
            except Exception:
                log.exception("恢复轮盘失败，请用 emotes wheel 检查")
```

5) `call()` 开头加：

```python
        if self.stopped:
            raise ToolError("身体已经停了")
```

6) `_heard` 里放进事件的那行改成带「」：

```python
            self.events.put("chat", f"聊天  {m.speaker or '（看不出是谁）'}：「{m.text}」")
```

7) `_watch_people` 里 `leave` 的循环改成：

```python
        for name in sorted(self._nearby - near):
            self.events.put("leave", f"{name} 走开了（{self.cfg.env.keep:.0f} 秒没看到名字）")
            if name == self.holding:  # 人都走开了，肯定没牵着了
                self.events.put("released", f"（推测）和 {name} 分开了")
                self.holding = None
```

8) `set_policy` 里 `if kind not in REQUEST_KINDS:` 那段后面加：

```python
        who = who.strip() or "*"
        if who == "stranger" and accept and kind != "candle":
            raise ToolError("陌生人只能接点火，牵手 / 拥抱 / 击掌 / 背背都不接陌生人的")
        self.social.set_policy(who, kind, accept)
        return "现在的规则：" + self.social.describe_policy()
```

（删掉原来的 `self.social.set_policy(who.strip() or "*", kind, accept)` 和它下面的 return。）

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/skydango/chat/responder.py src/skydango/chat/llm.py src/skydango/config.py src/skydango/brain/camera.py src/skydango/brain/body.py tests/test_responder.py tests/test_llm.py tests/test_brain_camera.py tests/test_brain_body.py
git commit -m "fix: 身份过滤补漏、方向键一定松开、身体感知出错隔离、退出不等大脑、陌生人只接点火"
```

---

### Task 2: 起 Claude Code 进程（brain/claude.py）+ 假 Claude Code

**Files:**
- Create: `src/skydango/brain/claude.py`
- Create: `tests/fake_claude.py`
- Test: `tests/test_brain_claude.py`（新建）

**Interfaces:**
- Produces:
  - `class ClaudeError(RuntimeError)`，属性 `limit: bool`（订阅额度用完）
  - `claude_env(token: str, config_dir: str | Path) -> dict[str, str]`
  - `resolve_claude(path: str) -> list[str]`（找不到抛 `RuntimeError`）
  - `StreamProcess(cmd: list[str], env: dict, cwd: Path)`：`send(content)`（str 或内容块列表）、`until_result(timeout, on_message=None) -> dict`、`alive() -> bool`、`close()`、`kill()`
  - `check_result(m: dict) -> str`（成功返回 `result` 文字，失败抛 `ClaudeError`）
  - `one_shot(cmd, env, cwd, content, timeout) -> str`
  - 假进程 `tests/fake_claude.py`：环境变量 `FAKE_CLAUDE_MODE`（`ok` / `hang` / `die` / `die_unless_resume` / `limit` / `nomcp`）、`FAKE_CLAUDE_LOG`（每次启动记 `{"args", "token", "config_dir", "api_key"}`，每条消息记 `{"message", "images"}`）；`ok` 时回 `收到：<消息文字>`

- [ ] **Step 1: Write the fake and the failing tests**

新建 `tests/fake_claude.py`：

```python
"""测试用的假 Claude Code：读 stream-json 消息，按 FAKE_CLAUDE_MODE 回应。

每次启动把参数和关心的环境变量记进 FAKE_CLAUDE_LOG（jsonl），每条收到的消息也记一行。
"""

import json
import os
import sys
import time

MODE = os.environ.get("FAKE_CLAUDE_MODE", "ok")
LOG = os.environ.get("FAKE_CLAUDE_LOG")


def emit(m):
    print(json.dumps(m, ensure_ascii=False), flush=True)


def record(entry):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


args = sys.argv[1:]
resumed = "--resume" in args
session = args[args.index("--resume") + 1] if resumed else "fake-session-1"
record({
    "args": args,
    "token": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"),
    "config_dir": os.environ.get("CLAUDE_CONFIG_DIR"),
    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
})
for line in sys.stdin:
    content = json.loads(line)["message"]["content"]
    if isinstance(content, str):
        text, images = content, 0
    else:
        text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        images = sum(b.get("type") == "image" for b in content)
    record({"message": text, "images": images})
    status = "failed" if MODE == "nomcp" else "connected"
    emit({"type": "system", "subtype": "init", "session_id": session, "mcp_servers": [{"name": "sky", "status": status}]})
    if MODE == "die" or (MODE == "die_unless_resume" and not resumed):
        sys.exit(1)
    if MODE == "hang":
        time.sleep(3600)
    if MODE == "limit":
        emit({"type": "result", "subtype": "success", "is_error": True, "result": "You've hit your limit · resets 5pm",
              "session_id": session, "num_turns": 1, "total_cost_usd": 0, "usage": {}})
        continue
    emit({"type": "assistant", "message": {"content": [{"type": "text", "text": "想：" + text[:20]}]}})
    emit({"type": "result", "subtype": "success", "is_error": False, "result": "收到：" + text, "session_id": session,
          "num_turns": 1, "total_cost_usd": 0.01, "usage": {"input_tokens": 10, "output_tokens": 2}})
```

新建 `tests/test_brain_claude.py`：

```python
import json
import sys
from pathlib import Path

import pytest

from skydango.brain.claude import ClaudeError, StreamProcess, check_result, claude_env, one_shot, resolve_claude

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]
IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}


def fake_env(tmp_path, mode="ok"):
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE=mode, FAKE_CLAUDE_LOG=str(tmp_path / "log.jsonl"))
    return env


def log_lines(tmp_path):
    return [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_claude_env_isolates_from_user_login(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-oat01-bad")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    env = claude_env("tok", tmp_path)
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok" and env["CLAUDE_CONFIG_DIR"] == str(tmp_path.resolve())


def test_resolve_claude_reports_missing():
    with pytest.raises(RuntimeError, match="找不到 Claude Code"):
        resolve_claude("definitely-not-a-command-xyz")


def test_round_trip_text_and_image(tmp_path):
    p = StreamProcess(FAKE, fake_env(tmp_path), tmp_path / "work")
    seen = []
    p.send("你好")
    assert check_result(p.until_result(10, seen.append)) == "收到：你好"
    assert [m["type"] for m in seen] == ["system", "assistant", "result"]
    p.send([{"type": "text", "text": "看图"}, IMG])
    assert check_result(p.until_result(10)) == "收到：看图"
    p.close()
    assert not p.alive()
    lines = log_lines(tmp_path)
    assert lines[0]["token"] == "tok" and lines[0]["api_key"] is None
    assert lines[2] == {"message": "看图", "images": 1}


def test_timeout_and_death(tmp_path):
    hang = StreamProcess(FAKE, fake_env(tmp_path, "hang"), tmp_path / "w1")
    hang.send("x")
    with pytest.raises(ClaudeError, match="没有结果"):
        hang.until_result(0.5)
    hang.kill()
    assert not hang.alive()
    die = StreamProcess(FAKE, fake_env(tmp_path, "die"), tmp_path / "w2")
    die.send("x")
    with pytest.raises(ClaudeError, match="退出了"):
        die.until_result(10)


def test_limit_is_flagged(tmp_path):
    p = StreamProcess(FAKE, fake_env(tmp_path, "limit"), tmp_path / "w")
    p.send("x")
    with pytest.raises(ClaudeError) as err:
        check_result(p.until_result(10))
    assert err.value.limit
    p.close()


def test_one_shot(tmp_path):
    assert one_shot(FAKE, fake_env(tmp_path), tmp_path / "w", [{"type": "text", "text": "描述"}, IMG], 10) == "收到：描述"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_brain_claude.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'skydango.brain.claude'`）

- [ ] **Step 3: Write minimal implementation**

新建 `src/skydango/brain/claude.py`：

```python
"""起 Claude Code 无界面进程（stream-json），逐行收发；和用户自己的 Claude Code 配置隔离。

实测（2.1.233）：沿用用户登录会把用户的插件、钩子、技能一起加载进来 → 子进程用单独的配置目录，
登录用 `claude setup-token` 生成的令牌（CLAUDE_CODE_OAUTH_TOKEN）。有 ANTHROPIC_API_KEY 时 `-p` 一定用它，所以要去掉。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

LIMIT_WORDS = ("limit", "上限", "额度")


class ClaudeError(RuntimeError):
    """Claude Code 这一轮没成：起不来、挂了、超时、返回错误。limit=True 表示订阅额度用完了。"""

    def __init__(self, message: str, limit: bool = False) -> None:
        super().__init__(message)
        self.limit = limit


def claude_env(token: str, config_dir: str | Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    env["CLAUDE_CONFIG_DIR"] = str(Path(config_dir).resolve())
    return env


def resolve_claude(path: str) -> list[str]:
    found = shutil.which(path)
    if not found:
        raise RuntimeError(f"找不到 Claude Code（{path}）：先装好，确认命令行里 claude 能用")
    return [found]


def user_message(content) -> str:
    return json.dumps(
        {"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": None}, ensure_ascii=False
    )


class StreamProcess:
    def __init__(self, cmd: list[str], env: dict[str, str], cwd: Path) -> None:
        cwd.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.messages: queue.Queue = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=20)
        threading.Thread(target=self._read, name="claude-out", daemon=True).start()
        threading.Thread(target=self._read_err, name="claude-err", daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.messages.put(json.loads(line))
            except ValueError:
                log.debug("Claude Code 输出了一行不是 JSON 的内容：%s", line[:200])
        self.messages.put(None)  # 输出结束（进程退出）

    def _read_err(self) -> None:
        for line in self.proc.stderr:
            self.stderr_tail.append(line.rstrip())

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, content) -> None:
        try:
            self.proc.stdin.write(user_message(content) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise ClaudeError(f"写不进 Claude Code：{exc}") from None

    def until_result(self, timeout: float, on_message: Callable[[dict], None] | None = None) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ClaudeError(f"等了 {timeout:.0f} 秒没有结果")
            try:
                m = self.messages.get(timeout=left)
            except queue.Empty:
                continue
            if m is None:
                raise ClaudeError("Claude Code 退出了：" + " | ".join(list(self.stderr_tail)[-3:]))
            if on_message is not None:
                on_message(m)
            if m.get("type") == "result":
                return m

    def close(self) -> None:
        """正常关：关掉 stdin 让它自己退出，5 秒不退就按进程树结束。"""
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.kill()

    def kill(self) -> None:
        if self.proc.poll() is not None:
            return
        if sys.platform == "win32":  # claude 会起子进程：按进程树结束，别留孤儿
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            self.proc.kill()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            log.warning("Claude Code 进程 %d 没能结束", self.proc.pid)


def check_result(m: dict) -> str:
    """result → 这一轮最后的文字；失败抛 ClaudeError（额度用完时 limit=True）。"""
    text = m.get("result") or ""
    if m.get("subtype") != "success" or m.get("is_error"):
        detail = text or str(m.get("subtype") or "未知错误")
        limit = m.get("api_error_status") == 429 or any(w in detail.lower() for w in LIMIT_WORDS)
        raise ClaudeError(f"Claude Code 这一轮失败：{detail}", limit=limit)
    return text


def one_shot(cmd: list[str], env: dict[str, str], cwd: Path, content, timeout: float) -> str:
    """眼睛用：起一个进程，发一条消息，拿到结果就关。"""
    p = StreamProcess(cmd, env, cwd)
    try:
        p.send(content)
        return check_result(p.until_result(timeout))
    finally:
        p.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_brain_claude.py -q`（跑 3 次，确认 hang / die 不偶发失败、没有残留 python 进程）
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/skydango/brain/claude.py tests/fake_claude.py tests/test_brain_claude.py
git commit -m "feat: 起 Claude Code 无界面进程（stream-json 收发、隔离配置、按进程树结束）"
```

---
### Task 3: 常驻大脑进程（brain/session.py）

**Files:**
- Create: `src/skydango/brain/session.py`
- Test: `tests/test_brain_session.py`（新建）

**Interfaces:**
- Consumes: `ClaudeError`、`StreamProcess`、`check_result`（Task 2）；`tests/fake_claude.py`
- Produces: `BrainSession(base_cmd: list[str], env: dict, cwd: Path, mcp_url: str, prompt: str, model: str, effort: str, turn_timeout: float, on_message: Callable[[dict], None] | None = None)`：
  `send(text: str) -> dict`（这一轮成功的 result；失败抛 `ClaudeError`，额度用完 `limit=True`）、`close()`、`command() -> list[str]`、属性 `session_id`、`on_message`。
  启动时在 `cwd` 写 `mcp.json`（服务名 `sky`）和 `prompt.md`。

- [ ] **Step 1: Write the failing test**

新建 `tests/test_brain_session.py`：

```python
import json
import sys
from pathlib import Path

import pytest

from skydango.brain.claude import ClaudeError, claude_env
from skydango.brain.session import BrainSession

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]


def session(tmp_path, mode="ok", timeout=10.0):
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE=mode, FAKE_CLAUDE_LOG=str(tmp_path / "log.jsonl"))
    return BrainSession(FAKE, env, tmp_path / "brain", "http://127.0.0.1:1/mcp", "规则", "sonnet", "low", timeout)


def starts(tmp_path):
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    return [l["args"] for l in lines if "args" in l]


def test_command_locks_tools_and_writes_config(tmp_path):
    s = session(tmp_path)
    s.send("你好")
    args = starts(tmp_path)[0]
    assert args[args.index("--tools") + 1] == "" and args[args.index("--allowedTools") + 1] == "mcp__sky"
    assert args[args.index("--permission-mode") + 1] == "dontAsk" and args[args.index("--model") + 1] == "sonnet"
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--verbose", "--append-system-prompt-file"):
        assert flag in args
    assert "--resume" not in args
    cfg = json.loads((tmp_path / "brain" / "mcp.json").read_text(encoding="utf-8"))
    assert cfg == {"mcpServers": {"sky": {"type": "http", "url": "http://127.0.0.1:1/mcp"}}}
    assert (tmp_path / "brain" / "prompt.md").read_text(encoding="utf-8") == "规则"
    s.close()


def test_one_process_many_turns(tmp_path):
    s = session(tmp_path)
    assert s.send("一")["result"] == "收到：一"
    assert s.send("二")["result"] == "收到：二"
    assert len(starts(tmp_path)) == 1 and s.session_id == "fake-session-1"
    s.close()


def test_resumes_same_session_after_crash(tmp_path):
    s = session(tmp_path, "die_unless_resume")
    with pytest.raises(ClaudeError):
        s.send("一")
    assert s.send("二")["result"] == "收到：二"
    second = starts(tmp_path)[1]
    assert second[second.index("--resume") + 1] == "fake-session-1"
    s.close()


def test_timeout_kills_process_and_limit_is_flagged(tmp_path):
    s = session(tmp_path, "hang", timeout=0.5)
    with pytest.raises(ClaudeError, match="没有结果"):
        s.send("一")
    assert not s._proc.alive()
    limited = session(tmp_path, "limit")
    with pytest.raises(ClaudeError) as err:
        limited.send("一")
    assert err.value.limit
    limited.close()


def test_missing_mcp_is_an_error_and_messages_are_forwarded(tmp_path):
    seen = []
    s = session(tmp_path, "nomcp")
    s.on_message = seen.append
    with pytest.raises(ClaudeError, match="MCP"):
        s.send("一")
    assert seen[0]["type"] == "system" and seen[-1]["type"] == "result"
    s.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_session.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'skydango.brain.session'`）

- [ ] **Step 3: Write minimal implementation**

新建 `src/skydango/brain/session.py`：

```python
"""常驻的大脑：一个无界面 Claude Code 进程，每次醒来写一条消息、等这一轮的 result；挂了 / 超时就用 --resume 接回原会话。

工具只有我们的 MCP 服务（sky）：--tools "" 关掉所有内置工具（Bash、读写文件……），--allowedTools mcp__sky 让它们不弹确认。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from .claude import ClaudeError, StreamProcess, check_result

log = logging.getLogger(__name__)

SERVER = "sky"


class BrainSession:
    def __init__(
        self,
        base_cmd: list[str],
        env: dict[str, str],
        cwd: Path,
        mcp_url: str,
        prompt: str,
        model: str,
        effort: str,
        turn_timeout: float,
        on_message: Callable[[dict], None] | None = None,
    ) -> None:
        self.base_cmd = base_cmd
        self.env = env
        self.cwd = cwd  # 空的专用目录：不让它读到项目的 AGENTS.md / CLAUDE.md
        self.mcp_url = mcp_url
        self.prompt = prompt
        self.model = model
        self.effort = effort
        self.turn_timeout = turn_timeout
        self.on_message = on_message  # 每条输出都给它（打日志用）
        self.session_id: str | None = None
        self._proc: StreamProcess | None = None
        self._mcp_ok: bool | None = None

    def command(self) -> list[str]:
        cmd = [
            *self.base_cmd, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", self.model, "--effort", self.effort,
            "--mcp-config", str(self.cwd / "mcp.json"), "--strict-mcp-config",
            "--tools", "", "--allowedTools", f"mcp__{SERVER}",
            "--permission-mode", "dontAsk", "--disable-slash-commands",
            "--append-system-prompt-file", str(self.cwd / "prompt.md"),
        ]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        return cmd

    def send(self, text: str) -> dict:
        """发一条消息，等这一轮的 result。成功返回 result；失败抛 ClaudeError（进程会被结束，下次用 --resume 接回）。"""
        self._ensure()
        self._mcp_ok = None
        try:
            self._proc.send(text)
            result = self._proc.until_result(self.turn_timeout, self._seen)
        except ClaudeError:
            self._proc.kill()
            raise
        self.session_id = result.get("session_id") or self.session_id
        check_result(result)
        if self._mcp_ok is False:
            raise ClaudeError("大脑的 MCP 工具没连上（没有手脚）")
        return result

    def close(self) -> None:
        if self._proc is not None:
            self._proc.close()

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.alive():
            return
        if self._proc is not None:
            log.warning("大脑进程不在了，%s", f"接回会话 {self.session_id}" if self.session_id else "重新开一个会话")
        self.cwd.mkdir(parents=True, exist_ok=True)
        config = {"mcpServers": {SERVER: {"type": "http", "url": self.mcp_url}}}
        (self.cwd / "mcp.json").write_text(json.dumps(config), encoding="utf-8")
        (self.cwd / "prompt.md").write_text(self.prompt, encoding="utf-8")
        self._proc = StreamProcess(self.command(), self.env, self.cwd)

    def _seen(self, m: dict) -> None:
        if m.get("type") == "system" and m.get("subtype") == "init":  # 每一轮开头都会有一条
            self.session_id = m.get("session_id") or self.session_id
            servers = {s.get("name"): s.get("status") for s in m.get("mcp_servers") or []}
            self._mcp_ok = servers.get(SERVER) in ("connected", "pending")
            if not self._mcp_ok:
                log.error("大脑的 MCP 服务没连上：%s", servers)
        elif m.get("type") == "system" and m.get("subtype") == "compact_boundary":
            log.info("大脑的对话记录自动压缩了一次")
        if self.on_message is not None:
            self.on_message(m)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_session.py -q`（跑 3 次）
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/skydango/brain/session.py tests/test_brain_session.py
git commit -m "feat: 常驻的大脑进程（Claude Code），挂了用 --resume 接回"
```

---

### Task 4: 眼睛（brain/eyes.py）+ 事件订阅

**Files:**
- Modify: `src/skydango/brain/events.py`（`subscribe`）
- Create: `src/skydango/brain/eyes.py`
- Test: `tests/test_brain_events.py`、`tests/test_brain_eyes.py`（新建）

**Interfaces:**
- Consumes: `fit`、`image_block`、`label_note`（`brain/images.py`）；`BrainConfig`（`auto_look_min`、`auto_look_max`、`image_size`、`jpeg_quality`、`eyes_model`）
- Produces:
  - `EventQueue.subscribe(fn: Callable[[str], None])`：每放一个事件就在放事件的线程里调 `fn(kind)`（出错只记日志）
  - `AUTO_LOOK_KINDS = {"arrive", "leave", "scene_change"}`；`EYES_SYSTEM`、`LOOK_REQUEST`、`AROUND_REQUEST`
  - `eyes_command(base: list[str], cfg: BrainConfig) -> list[str]`
  - `Eyes(cfg, describe: Callable[[list[dict]], str], frame: Callable[[], ndarray | None], labels: Callable[[], dict], blackout: Callable[[], bool], label_keep: float = 7.0, clock=time.monotonic)`：
    `notice(kind)`、`due(now) -> bool`、`tick(now) -> bool`、`run(stop)`、`describe_frame(frame, now) -> str`、`describe_around(frames, now) -> str`、`summary(now) -> str`、属性 `latest: tuple[str, float] | None`、`last_look: float`、`look_request: str`

- [ ] **Step 1: Write the failing tests**

`tests/test_brain_events.py` 末尾追加：

```python
def test_subscribers_hear_every_event(clock):
    q = EventQueue(clock=clock)
    heard = []
    q.subscribe(heard.append)
    q.subscribe(lambda kind: 1 / 0)  # 订阅者出错不影响放事件
    q.put("arrive", "懒懒 来到身边")
    q.put("chat", "x")
    assert heard == ["arrive", "chat"] and len(q) == 2
```

新建 `tests/test_brain_eyes.py`：

```python
import numpy as np

from skydango.brain.events import EventQueue
from skydango.brain.eyes import AROUND_REQUEST, Eyes, eyes_command
from skydango.config import BrainConfig


def frame():
    return np.full((1080, 1920, 3), (60, 90, 40), np.uint8)


class Describer:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, content):
        self.calls.append(content)
        if self.fail:
            raise RuntimeError("Haiku 没回")
        return f" 描述{len(self.calls)} "


def eyes(clock, blackout=False, labels=None, fail=False):
    d = Describer(fail)
    e = Eyes(BrainConfig(), d, frame, lambda: dict(labels or {}), lambda: blackout, clock=clock)
    return e, d


def test_looks_first_then_only_on_triggers_or_timeout(clock):
    e, _ = eyes(clock)
    assert e.tick(clock()) is True  # 从没看过：马上看
    assert e.tick(clock() + 30) is False
    e.notice("chat")
    assert e.tick(clock() + 30) is False  # 聊天不触发
    e.notice("arrive")
    assert e.tick(clock() + 10) is False  # 离上次不到 20 秒
    assert e.tick(clock() + 25) is True
    assert e.tick(clock() + 25 + 181) is True  # 超过 3 分钟兜底


def test_blackout_skips(clock):
    e, d = eyes(clock, blackout=True)
    assert e.tick(clock()) is False and d.calls == []


def test_describe_frame_sends_image_names_and_caches(clock):
    e, d = eyes(clock, labels={"懒洋洋大王": (1320, 300, 160, 44, clock())})
    text = e.describe_frame(frame(), clock())
    content = d.calls[0]
    assert text == "描述1" and content[0]["type"] == "image"
    assert "懒洋洋大王：名字在 (933, 200)" in content[1]["text"] and "地点和环境" in content[1]["text"]
    assert e.summary(clock() + 35) == "场景（35 秒前看的）：\n描述1"


def test_stale_names_are_not_passed(clock):
    e, d = eyes(clock, labels={"懒洋洋大王": (1320, 300, 160, 44, clock() - 60)})
    e.describe_frame(frame(), clock())
    assert "图里没认出好友的名字" in d.calls[0][1]["text"]


def test_describe_around_marks_directions(clock):
    e, d = eyes(clock)
    e.describe_around([frame()] * 4, clock())
    content = d.calls[0]
    texts = [b["text"] for b in content if b["type"] == "text"]
    assert texts[:4] == ["【前】", "【右】", "【后】", "【左】"] and texts[-1] == AROUND_REQUEST
    assert sum(b["type"] == "image" for b in content) == 4


def test_failure_is_swallowed_and_waits(clock):
    e, _ = eyes(clock, fail=True)
    assert e.tick(clock()) is True
    assert e.latest is None and e.summary(clock()) == "场景：还没有场景描述"
    assert e.tick(clock() + 5) is False  # 别每秒都重试


def test_event_queue_pokes_eyes(clock):
    q = EventQueue(clock=clock)
    e, _ = eyes(clock)
    e.last_look = clock()
    q.subscribe(e.notice)
    q.put("scene_change", "画面变化很大")
    assert e.tick(clock() + 21) is True


def test_eyes_command_is_locked_down():
    cmd = eyes_command(["claude"], BrainConfig())
    assert cmd[cmd.index("--model") + 1] == "haiku" and cmd[cmd.index("--tools") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--system-prompt", "--verbose"):
        assert flag in cmd
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_brain_events.py tests/test_brain_eyes.py -q`
Expected: FAIL（`AttributeError: 'EventQueue' object has no attribute 'subscribe'`、`ModuleNotFoundError: skydango.brain.eyes`）

注意：`BrainConfig` 在 Task 7 才加 `eyes_model`。这一步先在 `src/skydango/config.py` 的 `BrainConfig` 里加两个字段（Task 7 会整体改写 `BrainConfig`，保留它们）：

```python
    eyes_model: str = "haiku"  # 眼睛：把截图写成文字的模型（一次性 claude -p）
    eyes_timeout: float = 60.0  # 眼睛一次描述最多等多久（秒）
```

- [ ] **Step 3: Write minimal implementation**

`src/skydango/brain/events.py`：文件头加 `import logging`、`log = logging.getLogger(__name__)`；`__init__` 末尾加 `self._listeners: list[Callable[[str], None]] = []`；加方法并在 `put` 的 `with` 块**之后**通知：

```python
    def subscribe(self, fn: Callable[[str], None]) -> None:
        """每放一个事件就调 fn(kind)（在放事件的线程里调）；眼睛用它知道有人来了、画面变了。"""
        self._listeners.append(fn)
```

`put` 的最后（`with self._cond:` 块外面）加：

```python
        for fn in list(self._listeners):
            try:
                fn(kind)
            except Exception:
                log.exception("事件订阅者出错")
```

新建 `src/skydango/brain/eyes.py`：

```python
"""眼睛：把截图写成文字描述（一次性的 Haiku），后台按时机自动看，缓存最新一份。

不碰设备：只读身体最近一帧（Body.last_frame）和 EnvWatcher 认出的名字位置。
大脑平时只收这份文字；要原图才调 look(image=true)。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from ..config import BrainConfig
from .images import fit, image_block, label_note

log = logging.getLogger(__name__)

AUTO_LOOK_KINDS = {"arrive", "leave", "scene_change"}  # 这些事件发生后（隔够 auto_look_min）自动看一眼

EYES_SYSTEM = """你是一个《光遇》玩家的眼睛：看游戏截图，写成简短的中文文字，给另一个 AI 看。
- 只写看到的，不猜；看不清就说看不清。
- 人名只用给出的“名字位置”里的名字；没给名字的人都叫“陌生人”。
- 不客套，不给建议，不提问。"""

LOOK_REQUEST = """按这四项写，每项一两句：
地点和环境：像哪张图 / 什么地方、建筑、天气、白天还是晚上
好友：上面列出的每个人穿什么（斗篷、发型、面具、颜色）、在干什么；没有就写“没看到”
陌生人：大概几个、在干什么
画面状态：有没有弹窗、黑屏、看不懂的图标；正常就写“正常”"""

AROUND_REQUEST = """这是原地转一圈拍的四张图（前、右、后、左）。每个方向一两句：有什么地形 / 建筑、有没有人（认不出名字，只写几个人、穿什么、在干什么）。
最后一句总结现在在什么地方。"""


def eyes_command(base: list[str], cfg: BrainConfig) -> list[str]:
    return [
        *base, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--model", cfg.eyes_model, "--effort", "low", "--tools", "", "--strict-mcp-config",
        "--permission-mode", "dontAsk", "--disable-slash-commands", "--system-prompt", EYES_SYSTEM,
    ]


class Eyes:
    def __init__(
        self,
        cfg: BrainConfig,
        describe: Callable[[list[dict]], str],  # 内容块 → 文字（一次性 claude -p；测试里换成假的）
        frame: Callable[[], np.ndarray | None],  # 身体最近一帧
        labels: Callable[[], dict],  # 名字 → (x, y, w, h, 看到的时间)
        blackout: Callable[[], bool],
        label_keep: float = 7.0,  # 名字多久内看到过才算在这张图里
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.describe = describe
        self.frame = frame
        self.labels = labels
        self.blackout = blackout
        self.label_keep = label_keep
        self.clock = clock
        self.latest: tuple[str, float] | None = None
        self.last_look = float("-inf")
        self.look_request = LOOK_REQUEST
        self._poked = False
        self._lock = threading.Lock()  # 自动看和大脑要看可能撞上：同一时间只看一次

    def notice(self, kind: str) -> None:
        if kind in AUTO_LOOK_KINDS:
            self._poked = True

    def due(self, now: float) -> bool:
        if self.blackout():
            return False
        since = now - self.last_look
        return since >= self.cfg.auto_look_max or (self._poked and since >= self.cfg.auto_look_min)

    def tick(self, now: float) -> bool:
        """后台线程每秒调一次：到了时机就看一眼。返回这次有没有去看。"""
        if not self.due(now):
            return False
        frame = self.frame()
        if frame is None:
            return False
        try:
            self.describe_frame(frame, now)
        except Exception as exc:
            log.warning("眼睛这次没看成：%s", exc)
            self.last_look = now  # 别每秒都重试，等下一个时机
        return True

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(1.0):
            try:
                self.tick(self.clock())
            except Exception:
                log.exception("眼睛出错")

    def describe_frame(self, frame: np.ndarray, now: float) -> str:
        view = fit(frame, tuple(self.cfg.image_size))
        recent = {n: v for n, v in self.labels().items() if now - v[4] <= self.label_keep}
        note = label_note(recent, view.shape[1] / frame.shape[1])
        content = [image_block(view, self.cfg.jpeg_quality), {"type": "text", "text": note + "\n\n" + self.look_request}]
        with self._lock:
            text = self.describe(content).strip()
        self._keep(text, now)
        return text

    def describe_around(self, frames: list[np.ndarray], now: float) -> str:
        content: list[dict] = []
        for side, frame in zip(("前", "右", "后", "左"), frames):
            content += [{"type": "text", "text": f"【{side}】"}, image_block(fit(frame, tuple(self.cfg.image_size)), self.cfg.jpeg_quality)]
        content.append({"type": "text", "text": AROUND_REQUEST if len(frames) > 1 else self.look_request})
        with self._lock:
            text = self.describe(content).strip()
        self._keep(text, now)
        return text

    def summary(self, now: float) -> str:
        if self.latest is None:
            return "场景：还没有场景描述"
        text, t = self.latest
        return f"场景（{now - t:.0f} 秒前看的）：\n{text}"

    def _keep(self, text: str, now: float) -> None:
        self.latest = (text, now)
        self.last_look = now
        self._poked = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_brain_events.py tests/test_brain_eyes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/skydango/config.py src/skydango/brain/events.py src/skydango/brain/eyes.py tests/test_brain_events.py tests/test_brain_eyes.py
git commit -m "feat: 眼睛：后台把截图写成文字描述（Haiku），缓存最新一份"
```

---

### Task 5: 工具改成给 MCP 用（每轮计数、look 走眼睛、环顾四周）

**Files:**
- Modify: `src/skydango/brain/body.py`（`look_frame`、`fresh_frame()`、`capture_around()`、`look_at` 裁上次看的图）
- Rewrite: `src/skydango/brain/tools.py`
- Test: `tests/test_brain_body.py`、`tests/test_brain_tools.py`（改写）

**Interfaces:**
- Consumes: `Body`（Task 1 之后）、`Camera.around`（Task 1）、`Eyes.describe_frame / describe_around / summary / latest / last_look`（Task 4）
- Produces:
  - `Body.fresh_frame() -> ndarray`、`Body.capture_around() -> list[ndarray]`、属性 `look_frame`
  - `tools.DESCRIPTIONS: dict[str, str]`（顺序固定）、`TOOL_NAMES: list[str]`、`ACTIONS: set[str]`
  - `ToolBox(body, eyes=None, max_steps=6, max_says=2)`：`begin_turn()`、`status() -> str`（不计数）、`run(name, args) -> tuple[str | list[dict], bool]`（计数、从不抛异常）、属性 `acted: bool`、`used: list[str]`、`body`、`eyes`

- [ ] **Step 1: Write the failing tests**

`tests/test_brain_body.py` 顶部 import 加 `import base64` 和 `import cv2`，末尾追加：

```python
def test_look_at_crops_the_last_look(clock):
    b, device, _, _ = body(clock)
    b.look()
    device.frames = [np.zeros((720, 1280, 3), np.uint8)]  # 之后画面变黑了
    img, _ = b.look_at(100, 100, 200, 100)
    data = np.frombuffer(base64.standard_b64decode(img["source"]["data"]), np.uint8)
    assert cv2.imdecode(data, cv2.IMREAD_COLOR).mean() > 20  # 裁的是上次 look 的图，不是新截的黑图


def test_capture_around(clock):
    class AroundCamera(FakeCamera):
        def around(self, capture):
            return [capture() for _ in range(4)]

    dry, _, _, _ = body(clock, camera=AroundCamera())
    assert len(dry.capture_around()) == 1  # dry-run 不转，只看当前画面
    live, _, _, _ = body(clock, live=True, camera=AroundCamera())
    assert len(live.capture_around()) == 4
    live.blackout = True
    with pytest.raises(ToolError, match="黑"):
        live.capture_around()
```

把 `tests/test_brain_tools.py` 整个换成：

```python
import threading

from skydango.brain.body import ToolError
from skydango.brain.tools import ACTIONS, TOOL_NAMES, ToolBox
from skydango.config import Config


class FakeBody:
    def __init__(self):
        self.calls = []
        self.timeouts = []
        self.cfg = Config()

    def clock(self):
        return 100.0

    def call(self, fn, timeout=None):
        self.timeouts.append(timeout)
        return fn()

    def look(self):
        return [{"type": "image"}]

    def fresh_frame(self):
        return "帧"

    def capture_around(self):
        return ["帧"] * 4

    def look_at(self, x, y, w, h):
        self.calls.append(("look_at", x, y, w, h))
        return "ok"

    def status(self):
        return "状态"

    def chat_log(self, n):
        self.calls.append(("chat_log", n))
        return "log"

    def say(self, text):
        if text == "太快":
            raise ToolError("说得太快了")
        return f"说了{text}"

    def emote(self, name, force):
        self.calls.append(("emote", name, force))
        return "ok"

    def set_policy(self, who, kind, accept):
        self.calls.append(("policy", who, kind, accept))
        return "ok"

    def camera_move(self, action, steps):
        self.calls.append(("camera", action, steps))
        return "ok"

    def camera_reset(self):
        raise ToolError("没有视角控制")


class FakeEyes:
    def __init__(self):
        self.latest = None
        self.last_look = float("-inf")

    def describe_frame(self, frame, now):
        return f"描述{frame}"

    def describe_around(self, frames, now):
        return f"四周{len(frames)}张"

    def summary(self, now):
        return "旧描述"


def test_tool_names_and_actions():
    assert TOOL_NAMES == [
        "look", "look_at", "look_around", "status", "chat_log", "say", "emote", "set_request_policy", "camera", "camera_reset"
    ]
    assert ACTIONS == {"say", "emote", "set_request_policy", "camera", "camera_reset"}


def test_look_uses_eyes_unless_image_requested():
    eyes = FakeEyes()
    tb = ToolBox(FakeBody(), eyes)
    assert tb.run("look", {}) == ("描述帧", False)
    assert tb.run("look", {"image": True}) == ([{"type": "image"}], False)
    eyes.latest, eyes.last_look = ("旧", 99.0), 99.0
    assert tb.run("look", {}) == ("旧描述", False)  # 5 秒内刚看过：给缓存，不再花一次


def test_look_around_gets_longer_timeout():
    body = FakeBody()
    assert ToolBox(body, FakeEyes()).run("look_around", {}) == ("四周4张", False)
    assert body.timeouts[-1] == 30
    assert ToolBox(body).run("look_around", {})[1] is True  # 没开眼睛


def test_per_turn_limits():
    tb = ToolBox(FakeBody(), FakeEyes(), max_steps=3, max_says=1)
    assert tb.run("say", {"text": "a"}) == ("说了a", False)
    assert tb.run("say", {"text": "b"})[1] is True
    assert tb.run("status", {})[1] is False
    content, err = tb.run("status", {})
    assert err and content.startswith("这一轮做的事够多了")
    tb.begin_turn()
    assert tb.run("say", {"text": "c"}) == ("说了c", False)


def test_failed_say_does_not_use_quota():
    tb = ToolBox(FakeBody(), max_says=1)
    assert tb.run("say", {"text": "太快"}) == ("说得太快了", True)
    assert tb.run("say", {"text": "好"}) == ("说了好", False)


def test_acted_used_and_status_not_counted():
    tb = ToolBox(FakeBody())
    assert tb.status() == "状态" and tb.used == []
    tb.run("status", {})
    assert tb.acted is False
    tb.run("camera", {"action": "left"})
    assert tb.acted is True and tb.used == ["status", "camera"]


def test_concurrent_says_respect_limit():
    tb = ToolBox(FakeBody(), max_steps=50, max_says=2)
    results = []
    threads = [threading.Thread(target=lambda: results.append(tb.run("say", {"text": "x"}))) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(not err for _, err in results) == 2


def test_run_dispatches_with_defaults():
    body = FakeBody()
    tb = ToolBox(body)
    tb.run("chat_log", {})
    tb.run("emote", {"name": "鞠躬"})
    tb.run("camera", {"action": "left"})
    tb.run("set_request_policy", {"who": "*", "kind": "hug", "accept": False})
    tb.run("look_at", {"x": 1, "y": 2, "w": 30, "h": 40})
    assert body.calls == [
        ("chat_log", 20), ("emote", "鞠躬", False), ("camera", "left", 1), ("policy", "*", "hug", False), ("look_at", 1, 2, 30, 40)
    ]


def test_bad_arguments_and_errors_become_error_results():
    tb = ToolBox(FakeBody())
    content, err = tb.run("say", {})
    assert err and "缺少参数 text" in content
    content, err = tb.run("look_at", {"x": "1", "y": 0, "w": 10, "h": 10})
    assert err and "整数" in content
    assert tb.run("look_at", {"x": True, "y": 0, "w": 10, "h": 10})[1] is True  # bool 不算整数
    assert tb.run("camera_reset", {}) == ("没有视角控制", True)
    content, err = tb.run("fly", {})
    assert err and "没有这个工具" in content


def test_crash_in_body_does_not_escape():
    body = FakeBody()
    body.status = lambda: 1 / 0
    content, err = ToolBox(body).run("status", {})
    assert err and content.startswith("出错了")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_brain_body.py tests/test_brain_tools.py -q`
Expected: FAIL（`TOOL_NAMES` 不存在、`capture_around` 不存在、`look_at` 裁的是新截图……）

- [ ] **Step 3: Write minimal implementation**

`src/skydango/brain/body.py`：

1) `__init__` 里 `self.last_look = float("-inf")` 后面加 `self.look_frame = None  # 最近一次 look（原图）看的那张，look_at 裁它`

2) `look()` 里 `self.last_frame, self.last_look = frame, now` 改成：

```python
        self.last_frame, self.last_look, self.look_frame = frame, now, frame
```

3) `look_at` 里 `frame = self.device.screenshot()` 改成：

```python
        # 大脑给的坐标是按上次 look 那张图算的；人会走动，裁新截的图会对不上
        frame = self.look_frame if self.look_frame is not None else self.device.screenshot()
```

4) 在 `look_at` 后面加：

```python
    def fresh_frame(self):
        """眼睛马上要看：在身体线程里截一张新的。"""
        frame = self.device.screenshot()
        self.last_frame = frame
        return frame

    def capture_around(self) -> list:
        """环顾四周：转一圈，每 90° 截一张（dry-run 不转，只截当前画面）。"""
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在看不了")
        if self.camera is None or self.cfg.reply.dry_run:
            return [self.fresh_frame()]
        try:
            frames = self.camera.around(self.device.screenshot)
        finally:
            self._ref_thumb = None  # 自己转的镜头，不算画面大变
        self.last_frame = frames[0]
        return frames
```

把 `src/skydango/brain/tools.py` 整个换成：

```python
"""大脑的工具：说明（给 MCP 服务用）+ 执行（参数逐个校验、每轮计数，交给身体在身体线程里做）。出错不抛异常，原因交给大脑。"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable

from .body import REQUEST_KINDS, ToolError
from .camera import KEYS

log = logging.getLogger(__name__)

DESCRIPTIONS = {  # 顺序固定：MCP 工具列表按这个顺序注册
    "look": "看现在的画面。默认让眼睛马上看一眼，返回文字描述；image=true 时返回原图（1280×720）和认出的名字位置，"
            "只在文字不够用、要自己看细节时才要原图。",
    "look_at": "放大看局部原图：坐标按 look(image=true) 那张 1280×720 的图给（左上角 x、y，宽 w，高 h），裁的就是那一张。",
    "look_around": "环顾四周：原地转一圈（每 90° 看一次），眼睛描述前 / 右 / 后 / 左各有什么，最后转回原来的朝向。要十几秒，别常用。",
    "status": "看身体现在的状态：面板和输入框开没开、身边有谁、是不是牵着手、镜头、能做的动作、互动规则、刚说过的话。",
    "chat_log": "看最近 n 条聊天记录（含你自己说的，标成“我”），n 1~50，默认 20。",
    "say": "在游戏里发一句话。一次一句，口语，短。",
    "emote": "做一个动作（只能用 status 里“能做的动作”列出的）。牵着手时会被拦下，确定要松手才传 force=true。",
    "set_request_policy": (
        "改互动请求的规则（本次运行有效）。who：好友昵称，或 \"*\" 表示所有好友、\"stranger\" 表示陌生人；"
        f"kind：{' / '.join(REQUEST_KINDS)}（hand 牵手、hug 拥抱、highfive 击掌、piggyback 背背、candle 点火、* 所有）；"
        "accept：接不接。陌生人只能接点火。"
    ),
    "camera": (
        f"转视角 / 缩放，action：{' / '.join(KEYS)}（left、right 左右转，每步约 45°；up 往上看、down 往下看；zoom_in 拉近、zoom_out 拉远）。"
        "steps 1~4，默认 1。转之前身体会关掉聊天记录面板，转完再打开。"
    ),
    "camera_reset": "把镜头转回原位（按之前转过的反着转回去）。",
}
TOOL_NAMES = list(DESCRIPTIONS)
ACTIONS = {"say", "emote", "set_request_policy", "camera", "camera_reset"}  # 算“做了事”的工具（心跳退档用）
AROUND_TIMEOUT = 30.0  # 环顾一圈要关面板、转四次，比一般命令慢

_MISSING = object()


def _get(args: dict, key: str, default, kind: type, label: str):
    value = args.get(key, default)
    if value is _MISSING:
        raise ToolError(f"缺少参数 {key}")
    if (kind is int and isinstance(value, bool)) or not isinstance(value, kind):
        raise ToolError(f"参数 {key} 应该是{label}")
    return value


def _int(args: dict, key: str, default=_MISSING) -> int:
    return _get(args, key, default, int, "整数")


def _str(args: dict, key: str, default=_MISSING) -> str:
    return _get(args, key, default, str, "字符串")


def _bool(args: dict, key: str, default=_MISSING) -> bool:
    return _get(args, key, default, bool, " true / false")


class ToolBox:
    def __init__(self, body, eyes=None, max_steps: int = 6, max_says: int = 2) -> None:
        self.body = body
        self.eyes = eyes
        self.max_steps = max_steps
        self.max_says = max_says
        self._lock = threading.Lock()  # Claude Code 可能并行调工具：计数要加锁
        self.begin_turn()

    def begin_turn(self) -> None:
        """每次醒来开始时清零：这一轮最多 max_steps 次工具、max_says 句话。"""
        with self._lock:
            self.calls = 0
            self.says = 0
            self.acted = False
            self.used: list[str] = []

    def status(self) -> str:
        """醒来时拼消息用：不算这一轮的工具次数。"""
        try:
            return self.body.call(self.body.status)
        except ToolError as exc:
            return f"（状态读不到：{exc}）"

    def run(self, name: str, args: dict) -> tuple[str | list[dict], bool]:
        """执行一个工具，返回 (结果内容, 是不是出错)。"""
        with self._lock:
            self.calls += 1
            self.used.append(name)
            if self.calls > self.max_steps:
                return "这一轮做的事够多了，先停下，下次醒来再说", True
            if name == "say":
                if self.says >= self.max_says:
                    return "这一轮已经说得够多了，别刷屏", True
                self.says += 1  # 先占上名额：并行调 say 也不会超；没说成再还回去
        try:
            out = self._exec(name, args or {})
        except Exception as exc:
            if name == "say":
                with self._lock:
                    self.says -= 1
            if not isinstance(exc, ToolError):
                log.exception("工具 %s 出错", name)
                return f"出错了：{exc}", True
            return str(exc), True
        if name in ACTIONS:
            self.acted = True
        log.info("工具 %s %s → %s", name, json.dumps(args or {}, ensure_ascii=False), out if isinstance(out, str) else "[图片]")
        return out, False

    def _exec(self, name: str, a: dict):
        b = self.body
        if name == "look":
            image = _bool(a, "image", False)
            if image or self.eyes is None:
                return b.call(b.look)
            now = b.clock()
            if self.eyes.latest is not None and now - self.eyes.last_look < b.cfg.brain.look_min_interval:
                return self.eyes.summary(now)
            return self.eyes.describe_frame(b.call(b.fresh_frame), now)
        if name == "look_around":
            if self.eyes is None:
                raise ToolError("没开眼睛，看不了四周")
            frames = b.call(b.capture_around, timeout=AROUND_TIMEOUT)
            return self.eyes.describe_around(frames, b.clock())
        return b.call(self._bind(name, a))

    def _bind(self, name: str, a: dict) -> Callable[[], object]:
        b = self.body
        if name == "look_at":
            x, y, w, h = _int(a, "x"), _int(a, "y"), _int(a, "w"), _int(a, "h")
            return lambda: b.look_at(x, y, w, h)
        if name == "status":
            return b.status
        if name == "chat_log":
            n = _int(a, "n", 20)
            return lambda: b.chat_log(n)
        if name == "say":
            text = _str(a, "text")
            return lambda: b.say(text)
        if name == "emote":
            emote, force = _str(a, "name"), _bool(a, "force", False)
            return lambda: b.emote(emote, force)
        if name == "set_request_policy":
            who, kind, accept = _str(a, "who"), _str(a, "kind"), _bool(a, "accept")
            return lambda: b.set_policy(who, kind, accept)
        if name == "camera":
            action, steps = _str(a, "action"), _int(a, "steps", 1)
            return lambda: b.camera_move(action, steps)
        if name == "camera_reset":
            return b.camera_reset
        raise ToolError(f"没有这个工具：{name}")
```

注意：旧的 `brain/loop.py` 还 import `TOOLS`（Task 7 改写 loop 时删掉）。这一步先在 `tools.py` 末尾临时保留一行兼容，免得旧 loop 和它的测试挂掉：

```python
TOOLS = [{"name": n, "description": d, "input_schema": {"type": "object", "properties": {}}} for n, d in DESCRIPTIONS.items()]  # Task 7 删
```

`tests/test_brain_loop.py` 里用到 `ToolBoxSpy` 的旧测试不依赖 tools.py 的行为，照常能过。

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/skydango/brain/body.py src/skydango/brain/tools.py tests/test_brain_body.py tests/test_brain_tools.py
git commit -m "feat: 工具改成每轮计数、look 默认走眼睛、新增环顾四周"
```

---

### Task 6: MCP 服务（brain/mcp_server.py）

**Files:**
- Create: `src/skydango/brain/mcp_server.py`
- Modify: `pyproject.toml`（可选依赖 `brain`）
- Test: `tests/test_brain_mcp.py`（新建）

**Interfaces:**
- Consumes: `ToolBox.run`、`DESCRIPTIONS`、`TOOL_NAMES`（Task 5）；`image_block`（images.py）
- Produces: `to_mcp(out) -> list`、`build_server(toolbox) -> MCPServer`、`SkyServer(toolbox, host="127.0.0.1")`：`start(timeout=10)`、`stop()`、属性 `url`（`http://127.0.0.1:<端口>/mcp`）

- [ ] **Step 1: Write the failing test**

新建 `tests/test_brain_mcp.py`：

```python
import asyncio

import numpy as np
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from test_brain_tools import FakeBody, FakeEyes

from skydango.brain.images import image_block
from skydango.brain.mcp_server import SkyServer, to_mcp
from skydango.brain.tools import TOOL_NAMES, ToolBox


class ImageBody(FakeBody):
    def look(self):
        return [image_block(np.zeros((8, 8, 3), np.uint8)), {"type": "text", "text": "名字"}]


async def talk(url, calls):
    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]
            results = [await session.call_tool(name, args) for name, args in calls]
            return names, results


def test_to_mcp_converts_text_and_images():
    out = to_mcp([image_block(np.zeros((8, 8, 3), np.uint8)), {"type": "text", "text": "名字"}])
    assert out[1] == "名字" and type(out[0]).__name__ == "Image"
    assert to_mcp("描述") == ["描述"]


def test_tools_over_mcp():
    tb = ToolBox(ImageBody(), FakeEyes())
    server = SkyServer(tb)
    server.start()
    try:
        names, (status, look, emote, bad) = asyncio.run(talk(server.url, [
            ("status", {}), ("look", {"image": True}), ("emote", {"name": "鞠躬"}), ("say", {"text": "太快"}),
        ]))
    finally:
        server.stop()
    assert names == TOOL_NAMES
    assert not status.is_error and [c.text for c in status.content] == ["状态"]
    assert [c.type for c in look.content] == ["image", "text"]
    assert not emote.is_error and tb.body.calls[-1] == ("emote", "鞠躬", False)
    assert bad.is_error and bad.content[0].text == "说得太快了"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_mcp.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'skydango.brain.mcp_server'`）。如果报 `No module named 'mcp'`，先 `pip install --user mcp`。

- [ ] **Step 3: Write minimal implementation**

新建 `src/skydango/brain/mcp_server.py`：

```python
"""把身体的工具通过本机 MCP 服务交给 Claude Code（大脑）：只监听 127.0.0.1、随机端口、后台线程跑。

工具先过 ToolBox（参数校验、每轮计数），再经 Body.call() 在身体线程执行。同步的工具函数 mcp 会放到线程池里跑。
"""

from __future__ import annotations

import base64
import logging
import socket
import threading
import time

import uvicorn
from mcp.server.mcpserver import Image, MCPServer
from mcp.types import CallToolResult, TextContent

from .tools import DESCRIPTIONS, ToolBox

log = logging.getLogger(__name__)


def to_mcp(out) -> list:
    """ToolBox 的结果（文字，或图片块 + 文字块的列表）→ MCP 工具返回值。"""
    if isinstance(out, str):
        return [out]
    items: list = []
    for block in out:
        if block.get("type") == "image":
            items.append(Image(data=base64.standard_b64decode(block["source"]["data"]), format="jpeg"))
        elif block.get("type") == "text":
            items.append(block["text"])
    return items


def build_server(toolbox: ToolBox) -> MCPServer:
    srv = MCPServer("sky", instructions="光遇里的身体：看画面、说话、做动作、转视角。")

    def call(tool: str, **args):
        out, err = toolbox.run(tool, args)
        if err:  # 错误文字原样给模型，让它换个办法
            return CallToolResult(content=[TextContent(type="text", text=str(out))], is_error=True)
        return to_mcp(out)

    # 参数写在函数签名里（MCP 用它生成给模型看的参数说明），注册顺序 = DESCRIPTIONS 的顺序
    @srv.tool(name="look", description=DESCRIPTIONS["look"])
    def look(image: bool = False):
        return call("look", image=image)

    @srv.tool(name="look_at", description=DESCRIPTIONS["look_at"])
    def look_at(x: int, y: int, w: int, h: int):
        return call("look_at", x=x, y=y, w=w, h=h)

    @srv.tool(name="look_around", description=DESCRIPTIONS["look_around"])
    def look_around():
        return call("look_around")

    @srv.tool(name="status", description=DESCRIPTIONS["status"])
    def status():
        return call("status")

    @srv.tool(name="chat_log", description=DESCRIPTIONS["chat_log"])
    def chat_log(n: int = 20):
        return call("chat_log", n=n)

    @srv.tool(name="say", description=DESCRIPTIONS["say"])
    def say(text: str):
        return call("say", text=text)

    @srv.tool(name="emote", description=DESCRIPTIONS["emote"])
    def emote(name: str, force: bool = False):
        return call("emote", name=name, force=force)

    @srv.tool(name="set_request_policy", description=DESCRIPTIONS["set_request_policy"])
    def set_request_policy(who: str, kind: str, accept: bool):
        return call("set_request_policy", who=who, kind=kind, accept=accept)

    @srv.tool(name="camera", description=DESCRIPTIONS["camera"])
    def camera(action: str, steps: int = 1):
        return call("camera", action=action, steps=steps)

    @srv.tool(name="camera_reset", description=DESCRIPTIONS["camera_reset"])
    def camera_reset():
        return call("camera_reset")

    return srv


def _free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class SkyServer:
    def __init__(self, toolbox: ToolBox, host: str = "127.0.0.1") -> None:
        self.port = _free_port(host)
        self.url = f"http://{host}:{self.port}/mcp"
        app = build_server(toolbox).streamable_http_app()
        self._server = uvicorn.Server(uvicorn.Config(app, host=host, port=self.port, log_level="warning"))
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 10.0) -> None:
        logging.getLogger("mcp").setLevel(logging.WARNING)  # 每个请求都打 INFO，太吵
        self._thread = threading.Thread(target=self._server.run, name="mcp", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("MCP 服务没起来")
            time.sleep(0.05)
        log.info("MCP 服务：%s", self.url)

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(5)
```

`pyproject.toml` 的 `[project.optional-dependencies]` 里加一行，并把 `all` 改成包含它：

```toml
brain = ["mcp>=2.2", "uvicorn>=0.30"]
all = ["skydango[ocr,openai,anthropic,brain]"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_mcp.py -q`（跑 3 次）
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/skydango/brain/mcp_server.py pyproject.toml tests/test_brain_mcp.py
git commit -m "feat: 本机 MCP 服务，把身体的工具交给 Claude Code"
```

---

### Task 7: 切换：大脑循环、提示词、配置、命令行接上 Claude Code，删掉直接调 API 的代码

**Files:**
- Modify: `src/skydango/brain/prompt.py`
- Rewrite: `src/skydango/brain/loop.py`
- Modify: `src/skydango/brain/tools.py`（删掉 Task 5 留的兼容 `TOOLS`）
- Modify: `src/skydango/config.py`（`BrainConfig`）、`config.example.toml`、`.gitignore`
- Modify: `src/skydango/cli.py`（`_brain_env`、`_run_brain`、`cmd_look`）
- Delete: `src/skydango/brain/client.py`、`src/skydango/brain/context.py`、`src/skydango/brain/budget.py`、`tests/test_brain_client.py`、`tests/test_brain_context.py`、`tests/test_brain_budget.py`
- Test: `tests/test_brain_loop.py`（改写）、`tests/test_brain_prompt.py`、`tests/test_config.py`、`tests/test_cli_brain.py`（改写）

**Interfaces:**
- Consumes: `BrainSession`（Task 3）、`Eyes` / `eyes_command`（Task 4）、`ToolBox`（Task 5：`begin_turn`、`status()`、`acted`、`used`）、`SkyServer`（Task 6）、`one_shot` / `claude_env` / `resolve_claude` / `ClaudeError`（Task 2）
- Produces:
  - `prompt.brain_prompt(reply, store) -> str`、`prompt.SUMMARY_REQUEST`
  - `Brain(cfg, chat, session, toolbox, events, nearby, eyes=None, clock=time.monotonic, wall=time.time, run=None, store=None)`：`due(now)`、`wake(now, reason)`、`run(stop)`、`offline(now)`、`heartbeat(now)`、`message(now, events) -> str`、`farewell() -> bool`；`loop.log_brain_message(m)`
  - `cli._brain_env(cfg) -> tuple[list[str], dict[str, str]]`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py` 里的 `test_brain_section` 换成：

```python
def test_brain_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[brain]\nenabled = true\nheartbeat = [30, 60]\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.brain.enabled is True and cfg.brain.heartbeat == [30, 60]
    assert cfg.brain.model == "sonnet" and cfg.brain.eyes_model == "haiku" and cfg.brain.effort == "low"
    assert cfg.brain.token_env == "SKYDANGO_CLAUDE_TOKEN" and cfg.brain.config_dir == ".brain-claude"
    assert cfg.brain.turn_timeout == 120 and cfg.brain.limit_retry == 600
```

`tests/test_brain_prompt.py` 整个换成：

```python
from skydango.brain.prompt import SUMMARY_REQUEST, brain_prompt, memory_prompt, static_prompt
from skydango.chat.memory import MemoryStore
from skydango.config import ReplyConfig


def test_static_prompt_keeps_identity_rules_and_explains_tools():
    text = static_prompt(ReplyConfig(max_chars=40))
    assert "不要说“我是真人”" in text and "老实承认是 AI" in text  # 身份底线不能丢
    assert "不会发进游戏" in text  # 普通文字是想法，只有 say 才说话
    for tool in ("say", "look", "look_at", "look_around", "emote", "set_request_policy", "camera"):
        assert tool in text
    assert "image=true" in text and "40 个字" in text


def test_memory_prompt_uses_files(tmp_path):
    store = MemoryStore(tmp_path)
    (tmp_path / "profile.md").write_text("我是团子", encoding="utf-8")
    (tmp_path / "friends.md").write_text("## 懒洋洋大王\n- 叫他懒懒", encoding="utf-8")
    store.add_memos(["懒洋洋大王 10月3日考试"])
    text = memory_prompt(ReplyConfig(), store)
    assert text.startswith("我是团子") and "叫他懒懒" in text and "10月3日考试" in text


def test_brain_prompt_is_persona_then_rules():
    text = brain_prompt(ReplyConfig(persona="小团子"), None)
    assert text.startswith("小团子") and text.index("小团子") < text.index("## 你在做什么")


def test_summary_request_asks_for_summary_without_tools():
    assert "经过" in SUMMARY_REQUEST and "不要调用工具" in SUMMARY_REQUEST
```

`tests/test_brain_loop.py` 整个换成：

```python
import logging
import threading
import time

from skydango.brain.claude import ClaudeError
from skydango.brain.events import EventQueue
from skydango.brain.loop import Brain, log_brain_message
from skydango.brain.prompt import SUMMARY_REQUEST
from skydango.chat.memory import MemoryStore
from skydango.config import BrainConfig, ChatConfig


def ok(text="好"):
    return {"subtype": "success", "result": text, "num_turns": 2, "total_cost_usd": 0.01, "usage": {"input_tokens": 5}}


class FakeSession:
    def __init__(self, *results):
        self.results = list(results)
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        r = self.results.pop(0) if self.results else ok()
        if isinstance(r, Exception):
            raise r
        return r


class FakeToolBox:
    def __init__(self):
        self.acted = False
        self.used = []
        self.status_calls = 0

    def begin_turn(self):
        self.acted = False
        self.used = []

    def status(self):
        self.status_calls += 1
        return "面板开"


class FakeEyes:
    def summary(self, now):
        return "场景（3 秒前看的）：\n在雨林"


class FakeRun:
    def __init__(self):
        self.entries = []

    def record_brain(self, entry):
        self.entries.append(entry)


def make(clock, session, store=None, run=None, nearby=None, **cfg):
    events = EventQueue(clock=clock)
    tb = FakeToolBox()
    near = [] if nearby is None else nearby
    brain = Brain(BrainConfig(**cfg), ChatConfig(), session, tb, events, lambda now: list(near), eyes=FakeEyes(),
                  clock=clock, wall=lambda: 0.0, run=run, store=store)
    return brain, events, tb, near


def test_first_wake_is_immediate_and_sends_text_only(clock):
    session = FakeSession()
    brain, _, _, _ = make(clock, session)
    assert brain.due(clock()) == "heartbeat"
    brain.wake(clock(), "heartbeat")
    text = session.sent[0]
    assert "没有新事件" in text and "状态：面板开" in text and "在雨林" in text


def test_events_wait_for_debounce_and_go_into_message(clock):
    session = FakeSession()
    brain, events, _, _ = make(clock, session)
    brain.last_wake = clock()
    events.put("chat", "聊天  懒懒：「在吗」")
    assert brain.due(clock() + 0.3) is None
    assert brain.due(clock() + 1.0) == "events"
    brain.wake(clock() + 1.0, "events")
    assert "- 聊天  懒懒：「在吗」" in session.sent[0]


def test_failure_backs_off_then_goes_offline(clock):
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("挂了"), ok()))
    t0 = clock()
    brain.wake(t0, "heartbeat")
    assert brain.due(t0 + 5) is None  # 退避 10 秒
    assert not brain.offline(t0 + 100) and brain.offline(t0 + 121)
    brain.wake(t0 + 130, "heartbeat")
    assert not brain.offline(t0 + 131) and brain.failures == 0


def test_limit_waits_longer(clock):
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("hit your limit", limit=True)))
    brain.wake(clock(), "heartbeat")
    assert brain.backoff_until == clock() + 600


def test_heartbeat_backs_off_when_idle_and_resets(clock):
    brain, _, tb, near = make(clock, FakeSession())
    assert brain.heartbeat(clock()) == 90  # 身边没好友：从第二档起
    brain.wake(clock(), "heartbeat")
    assert brain.heartbeat(clock()) == 180
    near.append("懒洋洋大王")
    assert brain.heartbeat(clock()) == 90
    brain.wake(clock(), "events")
    assert brain.heartbeat(clock()) == 45


def test_record_brain_entry(clock):
    run = FakeRun()
    brain, _, tb, _ = make(clock, FakeSession(), run=run)
    tb.used = ["say"]
    brain.toolbox.begin_turn = lambda: None  # 保留上面放进去的 used
    brain.wake(clock(), "heartbeat")
    (entry,) = run.entries
    assert entry["tools"] == ["say"] and entry["total_cost_usd"] == 0.01 and entry["num_turns"] == 2


def test_thread_survives_errors():
    class BrokenOnce(FakeToolBox):
        def status(self):
            if self.status_calls == 0:
                self.status_calls += 1
                raise RuntimeError("friends.md 读不了")
            return super().status()

    events = EventQueue()
    brain = Brain(BrainConfig(), ChatConfig(), FakeSession(), BrokenOnce(), events, lambda now: [])
    stop = threading.Event()
    t = threading.Thread(target=brain.run, args=(stop,), daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while brain.failures == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert brain.failures == 1 and t.is_alive()  # 出错了但线程没死
    stop.set()
    t.join(3)
    assert not t.is_alive()


def test_farewell_writes_summary_to_inbox(clock, tmp_path):
    store = MemoryStore(tmp_path)
    session = FakeSession(ok("在雨林和懒懒 玩了一会儿"))
    brain, _, _, _ = make(clock, session, store=store)
    assert brain.farewell() is True
    assert session.sent == [SUMMARY_REQUEST] and "在雨林和懒懒 玩了一会儿" in store.inbox()
    brain.failing_since = 0.0
    assert brain.farewell() is False  # 正在失败：不再发


def test_log_brain_message(caplog):
    caplog.set_level(logging.INFO, logger="skydango.brain.loop")
    log_brain_message({"type": "assistant", "message": {"content": [{"type": "text", "text": "先看看"}]}})
    log_brain_message({"type": "user", "message": {"content": []}})
    assert "大脑想：先看看" in caplog.text
```

`tests/test_cli_brain.py` 整个换成：

```python
import json
import sys
from pathlib import Path

import pytest
from conftest import FakeDevice, scene
from test_brain_body import FakeReader

from skydango import cli
from skydango.brain.claude import claude_env
from skydango.chat.tracker import SelfFilter
from skydango.config import Config
from skydango.runlog import RunDir

FAKE = [sys.executable, str(Path(__file__).parent / "fake_claude.py")]


def test_run_brain_wires_everything(tmp_path, monkeypatch):
    log = tmp_path / "claude.jsonl"
    env = claude_env("tok", tmp_path / "cfg")
    env.update(FAKE_CLAUDE_MODE="ok", FAKE_CLAUDE_LOG=str(log))
    monkeypatch.setattr(cli, "_brain_env", lambda cfg: (FAKE, env))
    monkeypatch.setattr(cli, "_device", lambda cfg: FakeDevice([scene()]))
    monkeypatch.setattr(cli, "_build_reader", lambda cfg: (FakeReader(), SelfFilter(60, 0.8, "")))
    cfg = Config()
    cfg.run.dir = str(tmp_path / "runs")
    cfg.llm.provider = "echo"
    cfg.env.enabled = False
    cfg.reply.memory_dir = ""
    run = RunDir.create(cfg, "dry-brain")
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0)
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    starts = [l["args"] for l in lines if "args" in l]
    messages = [l["message"] for l in lines if "message" in l]
    assert any("--append-system-prompt-file" in a for a in starts)  # 大脑进程起来了
    assert any("没有新事件" in m for m in messages)  # 上线后马上醒了一次
    assert (run.path / "brain.jsonl").exists()
    mcp = json.loads((run.path / "brain" / "session" / "mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["sky"]["url"].startswith("http://127.0.0.1:")


def test_brain_env_needs_token(monkeypatch):
    monkeypatch.delenv("SKYDANGO_CLAUDE_TOKEN", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    with pytest.raises(RuntimeError, match="setup-token"):
        cli._brain_env(Config())


def test_run_and_look_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "cmd_run", lambda cfg, args: seen.update(brain=args.brain))
    monkeypatch.setattr(cli, "cmd_look", lambda cfg, args: seen.update(prompt=args.prompt))
    cli.main(["run", "--brain"])
    cli.main(["look", "--prompt", "q.txt"])
    assert seen == {"brain": True, "prompt": "q.txt"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py tests/test_brain_prompt.py tests/test_brain_loop.py tests/test_cli_brain.py -q`
Expected: FAIL（`BrainConfig` 没有新字段、`brain_prompt` / `SUMMARY_REQUEST` 不存在、`Brain` 还是旧的签名、`cli._brain_env` 不存在）

- [ ] **Step 3: Write minimal implementation**

**配置。** `src/skydango/config.py` 的 `BrainConfig` 整个换成：

```python
@dataclass
class BrainConfig:
    """统管大脑（brain/）：常驻的 Claude Code（订阅）收事件、调身体的工具；眼睛（Haiku）把画面写成文字。`run --brain` 打开。

    和用户自己的 Claude Code 隔离：单独配置目录 config_dir + `claude setup-token` 生成的令牌（放在用户环境变量 token_env）。
    """

    enabled: bool = False
    claude_path: str = "claude"
    token_env: str = "SKYDANGO_CLAUDE_TOKEN"
    config_dir: str = ".brain-claude"  # 大脑和眼睛用的单独 Claude Code 配置目录（gitignore）
    model: str = "sonnet"
    effort: str = "low"  # 要快
    eyes_model: str = "haiku"  # 眼睛：把截图写成文字的模型（一次性 claude -p）
    eyes_timeout: float = 60.0  # 眼睛一次描述最多等多久（秒）
    turn_timeout: float = 120.0  # 大脑一轮最多等多久（秒），超了结束进程、下次用 --resume 接回
    heartbeat: list[float] = field(default_factory=lambda: [45.0, 90.0, 180.0])  # 没事件时隔多久醒一次，闲着就退到下一档
    max_steps: int = 6  # 每次醒来最多调几次工具
    max_says: int = 2  # 每次醒来最多说几句
    look_min_interval: float = 5.0  # look 最多几秒一次（再调给缓存的描述）
    auto_look_min: float = 20.0  # 眼睛自动看：有人来 / 走 / 画面大变时，距上次看至少这么久
    auto_look_max: float = 180.0  # 超过这么久没看，眼睛一定看一次
    image_size: list[int] = field(default_factory=lambda: [1280, 720])
    jpeg_quality: int = 80
    look_at_max: int = 800  # look_at 返回图的最长边
    command_timeout: float = 15.0  # 身体执行一条命令最多等多久
    offline_fallback: float = 120.0  # 大脑连续失败这么久，聊天交给备用回复
    limit_retry: float = 600.0  # 订阅额度用完后多久再试
    camera_step: float = 0.25  # 转视角每步按住方向键的秒数（0.5 s 约 90°）
    scene_change: float = 0.25  # 缩略图平均差异（0~1）超过这个算画面大变
```

`config.example.toml` 的 `[brain]` 整节换成：

```toml
[brain]
# 统管大脑：常驻的 Claude Code（订阅）看眼睛写的场景描述、调身体的工具。`run --brain` 打开
# 先运行一次 claude setup-token，把生成的令牌放进用户环境变量：setx SKYDANGO_CLAUDE_TOKEN "<令牌>"
enabled = false
model = "sonnet"
effort = "low"
eyes_model = "haiku"           # 眼睛：把截图写成文字
heartbeat = [45, 90, 180]      # 没事件时多久醒一次（身边有好友从第一档起），闲着就退到下一档
```

`.gitignore` 末尾加：

```
# 大脑用的单独 Claude Code 配置目录（登录状态、会话记录）
.brain-claude/
```

**提示词。** `src/skydango/brain/prompt.py`：

1) `BRAIN_RULES` 里这两行：

```
- 看不清、不确定画面上发生了什么，就用 look / look_at 看，必要时 camera 转一下再看。看完用一句话记下看到了什么
  （在哪、什么天气、谁在旁边、穿什么、在干什么）：旧截图过一会儿会被删掉，只剩你记下的话。
```

换成：

```
- 每次醒来的消息里有“眼睛”最近写的场景描述（带多久前看的）。想知道现在的样子就用 look（眼睛马上再看一眼，给你文字）；
  文字不够用、要自己看细节，才用 look(image=true) 看原图、look_at 放大局部；想知道四周有什么用 look_around（要十几秒，别常用）。
  对话记录太长时会自动压缩，要紧的事（在哪、谁在、答应过什么）自己记在心里想的话里。
```

2) 同一节里 `截图里的人是谁，以 look 附带的“名字位置”为准，` 改成 `画面里的人是谁，以眼睛的描述和 look 附带的“名字位置”为准，`

3) `## 视角（camera）` 一节最后加一行：

```
- 只是想看看四周，用 look_around：它自己转一圈、描述完再转回来，不用你一步步转。
```

4) 删掉 `COMPACT_REQUEST` 和 `LOOK_QUESTION`，加：

```python
SUMMARY_REQUEST = """（身体）要下线了。用不超过 300 字写一份这次的经过，留给下次的你：在哪、和谁玩了什么、聊了什么、答应过什么、要注意的事。
只输出这份经过本身，这次不要调用工具。"""
```

5) 文件末尾加：

```python
def brain_prompt(reply: ReplyConfig, store: MemoryStore | None) -> str:
    """追加给 Claude Code 的系统提示词：先人设和记忆，再规则。启动时读一次（之后靠对话记录）。"""
    return memory_prompt(reply, store) + "\n\n" + static_prompt(reply)
```

并把文件开头的说明改成 `"""大脑的系统提示词（追加给 Claude Code）：人设和记忆 + 固定的规则（身份底线、怎么用工具、光遇常识）。"""`。

**工具。** `src/skydango/brain/tools.py`：删掉 Task 5 末尾那行兼容用的 `TOOLS = [...]`。

**大脑循环。** `src/skydango/brain/loop.py` 整个换成：

```python
"""大脑：常驻线程。有事件（攒一小会儿）或到了心跳就醒来，把事件、身体状态、眼睛最近的描述拼成一条文字消息，
交给常驻的 Claude Code（session），它自己调 MCP 工具，直到这一轮 result。失败退避；连续失败太久算离线（聊天交给备用回复）。"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

from ..chat.memory import format_date
from ..config import BrainConfig, ChatConfig
from .claude import ClaudeError
from .events import Event, EventQueue
from .prompt import SUMMARY_REQUEST

log = logging.getLogger(__name__)

BACKOFF = (10.0, 30.0, 60.0)  # 连续失败后隔多久再试


def log_brain_message(m: dict) -> None:
    """把大脑一轮里想了什么、调了什么工具打进日志（agent.log）。"""
    if m.get("type") != "assistant":
        return
    for block in (m.get("message") or {}).get("content") or []:
        if block.get("type") == "text" and (block.get("text") or "").strip():
            log.info("大脑想：%s", block["text"].strip())
        elif block.get("type") == "tool_use":
            log.debug("大脑调用 %s %s", block.get("name"), json.dumps(block.get("input"), ensure_ascii=False))


class Brain:
    def __init__(
        self,
        cfg: BrainConfig,
        chat: ChatConfig,
        session,  # brain.session.BrainSession：send(text) -> result
        toolbox,  # brain.tools.ToolBox：begin_turn()、status()、acted、used
        events: EventQueue,
        nearby: Callable[[float], list[str]],
        eyes=None,  # brain.eyes.Eyes：summary(now)
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        run=None,  # runlog.RunDir
        store=None,  # chat.memory.MemoryStore：live 时退出前把经过记进 inbox.md
    ) -> None:
        self.cfg = cfg
        self.chat = chat
        self.session = session
        self.toolbox = toolbox
        self.events = events
        self.nearby = nearby
        self.eyes = eyes
        self.clock = clock
        self.wall = wall
        self.run_dir = run
        self.store = store
        self.last_wake = float("-inf")  # 刚上线马上醒一次
        self._idle = 0  # 连着几次醒来什么都没做（心跳逐档退后）
        self.failures = 0
        self.failing_since: float | None = None
        self.backoff_until = float("-inf")

    # ---- 什么时候醒 ----
    def offline(self, now: float) -> bool:
        """聊天要不要交给备用回复：连续失败太久。"""
        return self.failing_since is not None and now - self.failing_since >= self.cfg.offline_fallback

    def heartbeat(self, now: float) -> float:
        beats = self.cfg.heartbeat
        start = 0 if self.nearby(now) else 1  # 身边有好友时醒得勤一点
        return beats[min(start + self._idle, len(beats) - 1)]

    def due(self, now: float) -> str | None:
        if now < self.backoff_until:
            return None
        if len(self.events) and now - self.events.last_put >= self.chat.debounce:
            return "events"
        if now - self.last_wake >= self.heartbeat(now):
            return "heartbeat"
        return None

    def run(self, stop: threading.Event) -> None:
        log.info("大脑上线（Claude Code：%s，effort=%s）", self.cfg.model, self.cfg.effort)
        while not stop.is_set():
            try:
                now = self.clock()
                reason = self.due(now)
                if reason is None:
                    if len(self.events):  # 在攒消息或在退避：稍等再看
                        stop.wait(0.2)
                    else:
                        self.events.wait(0.5)
                    continue
                self.wake(now, reason)
            except Exception as exc:  # 大脑线程不能悄悄死掉：记下来，歇一秒接着跑
                log.exception("大脑这一轮出错")
                self._failed(self.clock(), exc)
                stop.wait(1.0)
        log.info("大脑下线")

    # ---- 醒来一次 ----
    def message(self, now: float, events: list[Event]) -> str:
        w = self.wall()
        stamp = f"{format_date(w)} {time.strftime('%H:%M:%S', time.localtime(w))}"
        lines = [f"[{stamp}] " + ("事件：" if events else "没有新事件（定时醒来）")]
        lines += [f"- {e.line()}" for e in events]
        lines.append("状态：" + self.toolbox.status())
        lines.append(self.eyes.summary(now) if self.eyes is not None else "场景：（没开眼睛）")
        return "\n".join(lines)

    def wake(self, now: float, reason: str) -> None:
        self.last_wake = now
        events = self.events.drain()
        text = self.message(now, events)
        self.toolbox.begin_turn()
        try:
            result = self.session.send(text)
        except ClaudeError as exc:
            self._failed(self.clock(), exc)
            return
        self._ok()
        self._idle = 0 if (reason == "events" or self.toolbox.acted) else self._idle + 1
        self._log(result)

    def farewell(self) -> bool:
        """退出前让它写一份这次的经过，记进 inbox.md（只在 live、没在失败时）。"""
        if self.store is None or self.failing_since is not None:
            return False
        self.toolbox.begin_turn()
        try:
            result = self.session.send(SUMMARY_REQUEST)
        except ClaudeError as exc:
            log.warning("退出前写经过失败：%s", exc)
            return False
        text = " ".join((result.get("result") or "").split())
        if not text:
            return False
        self.store.add_memos([f"{format_date(self.wall())} 的经过：{text}"])
        log.info("这次的经过记进了 inbox.md（%d 字）", len(text))
        return True

    # ---- 失败 / 日志 ----
    def _failed(self, now: float, exc: Exception | None = None) -> None:
        self.failures += 1
        if self.failing_since is None:
            self.failing_since = now
        limit = bool(getattr(exc, "limit", False))
        delay = self.cfg.limit_retry if limit else BACKOFF[min(self.failures, len(BACKOFF)) - 1]
        self.backoff_until = now + delay
        what = "订阅额度用完了" if limit else "大脑这一轮失败"
        log.warning("%s（第 %d 次），%.0f 秒后再试：%s", what, self.failures, delay, exc)

    def _ok(self) -> None:
        self.failures = 0
        self.failing_since = None
        self.backoff_until = float("-inf")

    def _log(self, result: dict) -> None:
        text = (result.get("result") or "").strip()
        if text:
            log.info("大脑这一轮最后说：%s", text)
        if self.run_dir is None:
            return
        try:
            self.run_dir.record_brain({
                "subtype": result.get("subtype"),
                "num_turns": result.get("num_turns"),
                "total_cost_usd": result.get("total_cost_usd"),  # 订阅不按这个收费，只做参考
                "usage": result.get("usage"),
                "tools": list(self.toolbox.used),
                "text": text,
            })
        except Exception:
            log.exception("记大脑日志出错")
```

**命令行。** `src/skydango/cli.py`：

1) 在 `_run_agent` 前面加：

```python
def _brain_env(cfg: Config) -> tuple[list[str], dict[str, str]]:
    """大脑和眼睛的 Claude Code：命令 + 隔离的环境（单独配置目录 + claude setup-token 令牌）。"""
    from .brain.claude import claude_env, resolve_claude
    from .chat.llm import read_key

    try:
        token = read_key(cfg.brain.token_env)
    except RuntimeError:
        raise RuntimeError(
            f"没有找到大脑用的 Claude 令牌：先运行 claude setup-token，再 setx {cfg.brain.token_env} \"<令牌>\""
        ) from None
    return resolve_claude(cfg.brain.claude_path), claude_env(token, cfg.brain.config_dir)
```

2) `_run_brain` 整个换成：

```python
def _run_brain(cfg: Config, run: RunDir, no_emotes: bool = False, duration: float = 0.0) -> None:
    """统管大脑：身体在当前线程跑（独占设备）；大脑（常驻 Claude Code）和眼睛各一个后台线程；工具经本机 MCP 服务。"""
    import dataclasses
    import threading

    from .brain.body import Body
    from .brain.camera import Camera
    from .brain.claude import one_shot
    from .brain.events import EventQueue
    from .brain.eyes import Eyes, eyes_command
    from .brain.loop import Brain, log_brain_message
    from .brain.mcp_server import SkyServer
    from .brain.prompt import brain_prompt
    from .brain.session import BrainSession
    from .brain.tools import ToolBox
    from .chat.llm import make_llm
    from .chat.responder import Responder
    from .chat.sender import ChatSender

    base, claude_vars = _brain_env(cfg)  # 先检查令牌和 claude 命令，缺了早点报错
    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    reader.trace_path = run.rows_log
    store = notes = None
    if cfg.reply.memory_dir:  # dry-run 也读人设和记忆（看大脑的表现要用），但不写
        from .chat.memory import MemoryStore, NotesKeeper

        store = MemoryStore(cfg.reply.memory_dir)
        if not cfg.reply.dry_run:
            notes = NotesKeeper(make_llm(cfg.llm), store, cfg.reply.persona, cfg.reply.notes_every)
    live_store = None if cfg.reply.dry_run else store
    icons = _icon_classifier(cfg) if cfg.env.enabled else None
    env = _env_watcher(cfg, icons=icons) if cfg.env.enabled else None
    social = None
    if env and icons:
        from .game.social import SocialHandler

        social = SocialHandler(
            dev, cfg.social, icons, _friend_names(cfg), dry_run=cfg.reply.dry_run, panel_visible=reader.panel_visible
        )
    emotes = _build_emotes(cfg, dev, reader, no_emotes)
    if cfg.vision.mode == "log":
        panel_visible, panel_key = (lambda: reader.panel_visible(dev.screenshot())), cfg.vision.log_open_key
    else:
        panel_visible, panel_key = (lambda: False), 0
    camera = Camera(dev, cfg.brain.camera_step, panel_visible, panel_key)
    events = EventQueue()
    # 大脑离线时的备用回复：在身体线程里调，给短超时、不重试；不带记忆存储，免得和身体重复记聊天记录
    fallback = Responder(make_llm(dataclasses.replace(cfg.llm, timeout=10.0, max_retries=0)), cfg.reply)
    body = Body(
        cfg, dev, reader, ChatSender(dev, cfg.sender, _screen_size_fn(dev)), self_filter, events,
        env=env, social=social, emotes=emotes, camera=camera, fallback=fallback, store=live_store, notes=notes, run=run,
    )
    work = run.path / "brain"
    eyes = Eyes(
        cfg.brain,
        describe=lambda content: one_shot(eyes_command(base, cfg.brain), claude_vars, work / "eyes", content, cfg.brain.eyes_timeout),
        frame=lambda: body.last_frame,
        labels=lambda: dict(env.labels) if env else {},
        blackout=lambda: body.blackout,
        label_keep=cfg.env.interval * 2 + 1,
    )
    events.subscribe(eyes.notice)
    toolbox = ToolBox(body, eyes, cfg.brain.max_steps, cfg.brain.max_says)
    server = SkyServer(toolbox)
    server.start()
    session = BrainSession(
        base, claude_vars, work / "session", server.url, brain_prompt(cfg.reply, store),
        cfg.brain.model, cfg.brain.effort, cfg.brain.turn_timeout, on_message=log_brain_message,
    )
    brain = Brain(
        cfg.brain, cfg.chat, session, toolbox, events, nearby=env.nearby if env else (lambda now: []),
        eyes=eyes, run=run, store=live_store,
    )
    stop = threading.Event()
    brain_thread = threading.Thread(target=brain.run, args=(stop,), name="brain", daemon=True)
    eyes_thread = threading.Thread(target=eyes.run, args=(stop,), name="eyes", daemon=True)
    body.brain_offline = lambda now: brain.offline(now) or not brain_thread.is_alive()
    brain_thread.start()
    eyes_thread.start()
    try:
        body.run(duration, stop)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        stop.set()
        body.shutdown()  # 先复原镜头、恢复轮盘、让排队的命令失败：不等大脑
        brain_thread.join(timeout=5)
        if live_store is not None and not brain_thread.is_alive():
            try:
                brain.farewell()  # 把这次的经过记进 inbox.md
            except Exception:
                log.exception("退出前写经过失败")
        session.close()
        server.stop()
```

3) `cmd_look` 整个换成：

```python
def cmd_look(cfg: Config, args) -> None:
    """截一张图让眼睛（Haiku）描述一遍：在真实画面上调眼睛的提示词，看它认得准不准、会不会编名字。"""
    from .brain.claude import one_shot
    from .brain.eyes import Eyes, eyes_command
    from .vision.bubbles import roi_rect
    from .vision.chatlog import find_input_top

    base, claude_vars = _brain_env(cfg)
    dev = _device(cfg)
    frame = dev.screenshot()
    height, width = frame.shape[:2]
    area = roi_rect(cfg.vision.log_roi, width, height)
    panel = find_input_top(frame[:, area.x : area.x2]) is not None
    env = _env_watcher(cfg, background=False)
    env.observe(frame, 0.0, panel_visible=panel)
    Path("tmp").mkdir(exist_ok=True)
    imwrite("tmp/look.jpg", frame)
    eyes = Eyes(
        cfg.brain,
        describe=lambda content: one_shot(
            eyes_command(base, cfg.brain), claude_vars, Path("tmp/look-claude"), content, cfg.brain.eyes_timeout
        ),
        frame=lambda: frame,
        labels=lambda: dict(env.labels),
        blackout=lambda: False,
        clock=lambda: 0.0,
    )
    if args.prompt:
        eyes.look_request = Path(args.prompt).read_text(encoding="utf-8")
    started = time.perf_counter()
    text = eyes.describe_frame(frame, 0.0)
    print(text)
    print(f"\n{cfg.brain.eyes_model}：{time.perf_counter() - started:.1f} 秒；截图存在 tmp/look.jpg")
```

4) `main()` 里 `look` 的 help 改成 `"截一张图让眼睛（Claude Haiku）描述一遍（调眼睛的提示词，看它认得准不准）"`，`run --brain` 的 help 改成 `"用统管大脑（Claude Code）指挥：看画面、决定说什么做什么"`。

**删除。**

```bash
git rm src/skydango/brain/client.py src/skydango/brain/context.py src/skydango/brain/budget.py tests/test_brain_client.py tests/test_brain_context.py tests/test_brain_budget.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q`（`test_run_brain_wires_everything` 真的起线程和假进程，约 4 秒；跑 3 次确认稳定、没有残留 python 进程）
Expected: 全部 PASS；`grep -rn "brain.client\|brain.context\|brain.budget\|COMPACT_REQUEST\|LOOK_QUESTION" src tests` 没有结果

- [ ] **Step 5: Commit**

```bash
git add -A src/skydango tests config.example.toml .gitignore
git commit -m "feat: 大脑改用常驻的 Claude Code（订阅）+ 眼睛，删掉直接调 API 的实现"
```

---

### Task 8: 文档

**Files:**
- Modify: `AGENTS.md`、`README.md`

**Interfaces:**
- Consumes: 全部实现（照实写）

- [ ] **Step 1: AGENTS.md**

- “代码结构”表格里 `src/skydango/brain/` 那一行换成：

```markdown
| `src/skydango/brain/` | 统管大脑：`body.py` 身体（事件、命令队列、工具的护栏）、`loop.py` 大脑循环、`session.py` 常驻 Claude Code、`claude.py` 起进程 / 隔离、`mcp_server.py` + `tools.py` 工具、`eyes.py` 眼睛、`camera.py` 视角 |
```

- “运行目录”表格里 `brain.jsonl` 那一行换成：

```markdown
| `brain.jsonl` | 大脑每一轮：subtype、轮数、用量、total_cost_usd（订阅不按它收费，参考）、用了哪些工具、最后说了什么（只有 `--brain`）；`brain/` 下是 Claude Code 的工作目录（mcp.json、prompt.md） |
```

- “## 统管大脑”一节整个换成：

```markdown
## 统管大脑（`[brain]`，`run --brain`）

设计见 `docs/superpowers/specs/2026-09-27-brain-claude-code-design.md`（身体部分见 `2026-09-27-brain-design.md`）。
- 大脑 = 常驻的无界面 Claude Code（`claude -p` stream-json，订阅登录，`--model sonnet --effort low`）；身体的工具经本机 MCP 服务（`sky`）给它，
  `--tools ""` 关掉所有内置工具，只能调 look / look_at / look_around / status / chat_log / say / emote / set_request_policy / camera / camera_reset
- **和用户自己的 Claude Code 隔离**：单独配置目录 `.brain-claude/` + `claude setup-token` 生成的令牌（用户环境变量 `SKYDANGO_CLAUDE_TOKEN`）。
  沿用用户登录会把用户的插件、钩子、技能一起加载进大脑（实测）。子进程里去掉 `ANTHROPIC_API_KEY`（有它时 `-p` 一定用它）
- 眼睛 = 一次性 `claude -p --model haiku`：有人来 / 走、画面大变（隔 ≥20 秒）或 3 分钟没看时，把身体最近一帧写成文字；大脑醒来的消息里只有文字，要原图才 `look(image=true)`
- 身体线程独占设备；每轮最多 6 次工具、2 句话（`ToolBox` 计数）；`say` 照样过 `clean_reply`；牵着手时 `emote` 要 `force=true`；陌生人只能接点火
- 大脑一轮 120 秒没结果就结束进程、下次 `--resume` 接回；连续失败 120 秒或额度用完：聊天交给 `[llm]`（DeepSeek）备用回复，额度用完 10 分钟后再试
- 退出：身体先复原镜头、恢复轮盘（不等大脑）→ live 时让大脑写一份经过记进 `inbox.md` → 按进程树结束 Claude Code
- 前提：`pip install --user mcp`；运行一次 `claude setup-token` 并 `setx SKYDANGO_CLAUDE_TOKEN "<令牌>"`
```

- “常用命令”里两行换成：

```bash
python -m skydango run --brain [--live] [--duration 秒]  # 统管大脑（Claude Code，订阅）；先 claude setup-token、设 SKYDANGO_CLAUDE_TOKEN
python -m skydango look [--prompt 文件]    # 截一张图让眼睛（Haiku）描述（调眼睛的提示词、看它会不会编名字）
```

- [ ] **Step 2: README.md**

- “工作原理”表格里 `统管大脑（可选）` 那一行换成：

```markdown
| 统管大脑（可选） | `src/skydango/brain/` | `run --brain`：常驻的 Claude Code（订阅）当大脑，眼睛（Haiku）把画面写成文字，经本机 MCP 调身体的工具 |
```

- “逐步联调”里 `skydango look` 那一行换成：

```bash
skydango look                     # 截一张图让眼睛（Claude Haiku）描述（要先 claude setup-token、设 SKYDANGO_CLAUDE_TOKEN）
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: 大脑改用 Claude Code + 眼睛的用法"
```

---

### Task 9: 真机验证（和用户一起）

**Files:**
- Modify: `docs/game-ops.md`（视角实测）

- [ ] **Step 1: 前置（用户做）**

用户运行 `claude setup-token`（浏览器登录，生成长期令牌），再在 PowerShell 里 `setx SKYDANGO_CLAUDE_TOKEN "<令牌>"`。
建议用户顺手删掉之前误放进 `ANTHROPIC_API_KEY` 的登录令牌：`[Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', $null, 'User')`。
确认 `pip show mcp uvicorn` 都在。工作树里的 `config.toml` 已经是 DeepSeek 的 `[llm]`、`memory_dir` 指向主仓库的 `memory/`。

- [ ] **Step 2: 眼睛**

Run（工作树里，用工作树的代码）：`PYTHONPATH=src python -m skydango look`
在几种画面各跑一次（好友在旁、人多、室内、黑夜），看描述准不准、会不会编名字、耗时。
第一次运行还要确认：`.brain-claude/` 里没有用户的插件 / 技能（Claude Code 启动时没加载 superpowers 等）。

- [ ] **Step 3: 视角**

先按 C 关掉聊天记录面板，用 `tmp/camera_check.py` 验证转视角、`around()` 转一圈能回到原来的朝向：

```python
from skydango.brain.camera import Camera
from skydango.cli import _device
from skydango.config import load_config
from skydango.imageio import imwrite

cfg = load_config("config.toml")
dev = _device(cfg)
cam = Camera(dev, cfg.brain.camera_step, lambda: False, 0)
imwrite("tmp/cam_0.png", dev.screenshot())
frames = cam.around(dev.screenshot)
for i, f in enumerate(frames):
    imwrite(f"tmp/cam_around_{i}.png", f)
imwrite("tmp/cam_1.png", dev.screenshot())
```

Run: `PYTHONPATH=src python tmp/camera_check.py`，对比 `cam_0.png` 和 `cam_1.png` 的偏差、四张图是不是 90° 间隔；
面板开着时按一下 ← 看有没有反应。结果写进 `docs/game-ops.md` §2 的方向键那一行。

- [ ] **Step 4: dry-run 跑大脑**

Run: `PYTHONPATH=src python -m skydango run --brain --duration 300`
看 `runs/<时间>-dry-brain/agent.log`（“大脑想：”“工具 … →”）、`brain.jsonl`、`brain/session/` 下的文件。
确认：上线先醒一次、有人说话 1~2 秒后醒、眼睛有描述、每轮工具不超过 6 次、没有刷屏；跑完没有残留的 claude / python 进程
（`Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'claude|node|python' }`）。

- [ ] **Step 5: 好友在场时 live**

先告诉用户要开始 live，确认好友在线。
Run: `PYTHONPATH=src python -m skydango run --brain --live --duration 600`
检查：回复是否自然、牵手时没被动作松开、镜头退出后回到原位、轮盘复原、`memory/inbox.md` 里有这次的经过；跑完确认没有残留进程。

- [ ] **Step 6: Commit**

```bash
git add docs/game-ops.md
git commit -m "docs: game-ops 补充视角实测"
```
