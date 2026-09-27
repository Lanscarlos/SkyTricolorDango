# Agent 表情动作能力 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让聊天 Agent 由大模型决定在合适的时候做表情动作（可以边说边做，也可以只做动作），轮盘上现有的动作直接按数字键，白名单动作按需换到指定格子，退出时恢复轮盘。

**Architecture:** 模型在回复句首写 `[动作名]`，`responder.parse_reply` 拆成 `Reply(text, emote)`；每轮可用的动作由新模块 `game/emotes.py` 的 `EmotePlayer.available()` 在调用模型前算好（限速、换轮盘限速都在这里）。`Agent.step` 先做动作再发文字；`cli.py` 负责构造 / 启动 / 退出时恢复。

**Tech Stack:** Python 3.13、pytest、OpenCV（现有图标匹配）、adb sendevent（现有 `Device.hw_key`）。

**Spec:** `docs/superpowers/specs/2026-09-27-agent-emotes-design.md`

## Global Constraints

- 运行测试：`python -m pytest -q`（包不在 PATH 上，统一用 `python -m ...`）；单元测试不连模拟器。
- 代码风格：中文注释 / 日志 / 报错，和现有文件一致；注释密度跟周围代码一样（只写"为什么"）。
- **不要删** `responder.RULES` 的"身份"一节和 `_CLAIMS_HUMAN`：声称真人时整轮丢弃（文字和动作都不做）。
- 默认 dry-run：dry-run 下不按任何动作键、不开编辑界面（启动时读一次轮盘除外）。
- 做动作会松开牵手 → 开关：配置 `emotes.enabled` + 命令行 `run --no-emotes`。
- 锁定格子（`wheel.locked_slots`，默认 `[3, 8]`，是火 / 门等道具）里的东西**永远不出现在可用动作里**，也不会被换。
- 临时文件（截图等）放项目 `tmp/`，不放系统 Temp。
- 在 Git Bash 里调 adb 带 `/dev/...` 路径时先 `export MSYS_NO_PATHCONV=1`。
- 提交信息：`feat: ...` / `docs: ...` 中文短句，末尾带
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/skydango/chat/responder.py` | 改 | `Reply`、`parse_reply`、提示词里的"动作"一节、`Responder` 接收 `available_emotes` 回调、返回 `Reply` |
| `src/skydango/chat/__init__.py` | 改 | 导出 `Reply`、`parse_reply` |
| `src/skydango/config.py` | 改 | 新增 `EmoteConfig`，挂到 `Config.emotes` |
| `src/skydango/game/wheel.py` | 改 | `ensure` / `_victim` 支持 `candidates` 限定候选格子 |
| `src/skydango/game/emotes.py` | 新建 | `EmotePlayer`：启动读轮盘、可用动作、做动作（必要时关面板换轮盘）、dry-run 记账、退出恢复 |
| `src/skydango/agent.py` | 改 | 先做动作再发文字；只做动作不占发送限速；失败不影响文字 |
| `src/skydango/cli.py` | 改 | `run --no-emotes`、`_build_emotes`、退出恢复；`chat --emotes` 调提示词 |
| `config.example.toml` | 改 | `[emotes]` 示例 |
| `AGENTS.md`、`README.md`、`docs/game-ops.md` | 改 | 文档 |
| `tests/test_responder.py`、`tests/test_memory.py` | 改 | `Reply` 返回值、解析、提示词 |
| `tests/test_wheel.py` | 改 | `candidates` |
| `tests/test_emotes.py` | 新建 | `EmotePlayer` |
| `tests/test_agent.py` | 改 | Agent 动作流程 |
| `tests/test_cli_emotes.py` | 新建 | `_build_emotes` |

---

### Task 0: 前置检查

- [ ] **Step 1: 工作区干净**

Run: `git status --short`
Expected: 没有输出。若 `agent.py` / `responder.py` 等还有别人没提交的改动，**停下来问用户**，不要把它们混进本功能的提交。

- [ ] **Step 2: 测试基线**

Run: `python -m pytest -q`
Expected: 全部 passed（记下数量，之后每个任务结束都应 ≥ 这个数且全过）。

---

### Task 1: 真机确认——聊天记录面板开着时数字键能不能做动作

这一步决定 Task 3 里 `perform` 按数字键时要不要关面板。需要模拟器里开着光遇、角色站在地上、没牵手。

**Files:**
- Modify: `docs/game-ops.md`（§4 快捷轮盘一节）

- [ ] **Step 1: 确认连接和当前画面**

Run: `python -m skydango devices` 然后 `python -m skydango shot -o tmp/emote_check_0.png`
Expected: 设备已连接；看截图：输入框关着（底部没有全宽浅色输入栏）、没有编辑界面残留。

- [ ] **Step 2: 打开聊天记录面板**

如果截图里左侧没有聊天记录面板（底部没有"聊天……"胶囊框），按一次 C：

```bash
python -c "from skydango.cli import _device; from skydango.config import load_config; _device(load_config('config.toml')).hw_key(46)"
```

再 `python -m skydango shot -o tmp/emote_check_1.png`，确认面板开着。

- [ ] **Step 3: 面板开着时按数字键 1，连拍几张**

```bash
python -c "import time; from skydango.cli import _device; from skydango.config import load_config; from skydango.imageio import imwrite; d=_device(load_config('config.toml')); d.hw_key(2); [ (time.sleep(0.4), imwrite(f'tmp/emote_check_key_{i}.png', d.screenshot())) for i in range(5)]"
```

（Linux 键码 2 = 数字键 1，本机轮盘 1 号格是"鞠躬"。）

- [ ] **Step 4: 看结果**

逐张看 `tmp/emote_check_key_*.png`：角色有没有做鞠躬；面板是否还开着；输入框是否没被打开、没有打出"1"。
- **做了动作** → 结论 A：面板开着也能按数字键。Task 3 按原样实现。
- **没做动作** → 按 C 关面板再按一次数字键 1 对照（确认是面板的原因），然后按 C 恢复面板。结论 B：Task 3 用"结论 B"的变体。

- [ ] **Step 5: 写进 game-ops**

在 `docs/game-ops.md` §4「快捷轮盘」列表里、"在轮盘上点击 / 拖动……只能用数字键"那条后面加一条（按实际结论二选一）：

```markdown
- 聊天记录面板（C）开着时按数字键**能**直接做动作，面板保持打开（2026-09-27 实测）—— Agent 做轮盘上的动作不用关面板
```

或

```markdown
- 聊天记录面板（C）开着时按数字键**没反应**，要先按 C 关面板（2026-09-27 实测）—— Agent 做动作前后都要关 / 开面板
```

- [ ] **Step 6: Commit**

```bash
git add docs/game-ops.md
git commit -m "docs: 实测聊天记录面板开着时数字键做动作

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 回复里的动作标签（Responder）

**Files:**
- Modify: `src/skydango/chat/responder.py`
- Modify: `src/skydango/chat/__init__.py`
- Modify: `src/skydango/agent.py:94-116`（只改用 `reply.text`，完整动作流程在 Task 4）
- Modify: `src/skydango/cli.py`（`cmd_chat` 的 print，只改用 `Reply`）
- Test: `tests/test_responder.py`、`tests/test_memory.py`

**Interfaces:**
- Produces:
  - `Reply`（frozen dataclass）：`text: str | None = None`、`emote: str | None = None`；`render(prefix: str = "") -> str` 返回 `"[害羞]" + prefix + text` 这种形式（没有的部分省略）
  - `parse_reply(raw: str, max_chars: int, emotes: Sequence[str] = ()) -> Reply | None`
  - `build_system_prompt(cfg, profile="", friends="", notes="", emotes: Sequence[str] = ()) -> str`
  - `Responder(llm, cfg, store=None, notes=None, clock=time.time, available_emotes: Callable[[], list[str]] | None = None)`
  - `Responder.reply(incoming) -> Reply | None`（原来返回 `str | None`）
  - `Responder.system_prompt(emotes: Sequence[str] = ()) -> str`（原来的 `system` 属性）

- [ ] **Step 1: 写失败的测试**

在 `tests/test_responder.py` 顶部把 import 改成：

```python
from skydango.chat.reader import Message
from skydango.chat.responder import SKIP_TOKEN, Reply, Responder, build_system_prompt, clean_reply, parse_reply
from skydango.config import ReplyConfig
from skydango.vision.bubbles import Rect
```

把 `test_reply_history_alternates_and_trims` 里的两处断言改成比较 `.text`：

```python
    assert r.reply([msg("你好")]).text == "你好呀"
    assert r.reply([msg("啊啊啊")]) is None
    assert r.reply([msg("去哪"), msg("一起吗")]).text == "去霞谷吧"
```

文件末尾追加：

```python
EMOTES = ["鞠躬", "害羞", "欢呼"]


def test_parse_reply_splits_emote_tag():
    assert parse_reply("[害羞]哪有啦", 40, EMOTES) == Reply("哪有啦", "害羞")
    assert parse_reply("【欢呼】好耶", 40, EMOTES) == Reply("好耶", "欢呼")
    assert parse_reply("哪有啦 [害羞]", 40, EMOTES) == Reply("哪有啦", "害羞")
    assert parse_reply("[鞠躬]", 40, EMOTES) == Reply(None, "鞠躬")
    assert parse_reply("[鞠躬]\n晚安", 40, EMOTES) == Reply("晚安", "鞠躬")  # 标签单独一行
    assert parse_reply("回复：“[害羞]哪有啦”", 40, EMOTES) == Reply("哪有啦", "害羞")
    assert parse_reply("好呀", 40, EMOTES) == Reply("好呀", None)
    assert parse_reply("<skip>", 40, EMOTES) is None


def test_parse_reply_drops_unknown_emote_but_keeps_text():
    assert parse_reply("[跳舞]来了", 40, EMOTES) == Reply("来了", None)
    assert parse_reply("[跳舞]", 40, EMOTES) is None
    assert parse_reply("[害羞]哪有啦", 40, []) == Reply("哪有啦", None)  # 这一轮不让做动作


def test_parse_reply_claim_of_human_drops_emote_too():
    assert parse_reply("[害羞]我是真人啦", 40, EMOTES) is None


def test_parse_reply_truncates_only_text():
    assert parse_reply("[欢呼]" + "一" * 50, 10, EMOTES) == Reply("一" * 10, "欢呼")


def test_reply_render():
    assert Reply("哪有啦", "害羞").render("【AI】") == "[害羞]【AI】哪有啦"
    assert Reply(None, "鞠躬").render("【AI】") == "[鞠躬]"
    assert Reply("好呀").render() == "好呀"


def test_prompt_emote_section_only_when_available():
    plain = build_system_prompt(ReplyConfig())
    assert "## 动作" not in plain and "做动作" in plain  # 没动作可用：还是"动不了"
    with_emotes = build_system_prompt(ReplyConfig(), emotes=["鞠躬", "害羞"])
    assert "## 动作" in with_emotes and "鞠躬、害羞" in with_emotes and "[动作名]" in with_emotes


def test_responder_asks_available_emotes_each_turn_and_keeps_tag_in_history():
    llm = ScriptedLlm(["[害羞]哪有啦", "嗯嗯"])
    offers = [["害羞"], []]
    r = Responder(llm, ReplyConfig(), available_emotes=lambda: offers.pop(0))
    assert r.reply([msg("你真可爱")]) == Reply("哪有啦", "害羞")
    assert "## 动作" in llm.calls[0][0]
    assert r.reply([msg("真的")]) == Reply("嗯嗯", None)
    assert "## 动作" not in llm.calls[1][0]
    assert llm.calls[1][1][1] == {"role": "assistant", "content": "[害羞]哪有啦"}
```

在 `tests/test_memory.py` 里把比较回复文字的断言改成 `.text`（`is None` 的不用改）：

```python
    assert r.reply([msg("醒醒")]).text == "在呢"
```
```python
    assert r2.reply([msg("你还记得我吗")]).text == "记得呀"
```
```python
    assert r.reply([msg("我今天加班到九点")]).text == "辛苦啦"
```
```python
    assert Responder(llm, ReplyConfig(), store=store, notes=keeper).reply([msg("在吗")]).text == "在呢"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_responder.py tests/test_memory.py`
Expected: FAIL，`ImportError: cannot import name 'Reply'`

- [ ] **Step 3: 实现 responder.py**

import 部分改成：

```python
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
```

`RULES` 里"你现在只能打字聊天，动不了"那一行（两行里的第一行）换成占位符（第二行"被叫去就自然地推掉……"不动）：

```
- {move}别答应这些（不说“我跟着你”“我飞给你看”“走呗”），
```

在 `RULES` 定义之后加：

```python
MOVE_PLAIN = "你现在只能打字聊天，动不了：不能走、飞、跑图、跟着别人、做动作、弹琴、送东西。"
MOVE_WITH_EMOTES = "你现在只能打字聊天，除了下面“动作”一节列的几个动作，动不了：不能走、飞、跑图、跟着别人、弹琴、送东西。"

EMOTE_RULES = """
## 动作
你可以在游戏里做这几个动作：{names}。
- 想做就在那句话最前面写 [动作名]，比如“[害羞]哪有啦”；只做动作不说话就只写“[害羞]”。一次最多一个。
- 大多数时候不用做，自然的时候才做：见面、道别、被夸、特别开心，或者别人叫你做某个动作。别连着几轮都做。
- 只能用上面列的名字，别的动作做不了；别人要你做列表外的，就用文字接话（比如“这个我还没学会”），不要自己编动作名。
""".strip()
```

`build_system_prompt` 改成（新增 `emotes` 参数，末尾两行变化）：

```python
def build_system_prompt(
    cfg: ReplyConfig, profile: str = "", friends: str = "", notes: str = "", emotes: Sequence[str] = ()
) -> str:
    """profile / friends / notes 来自记忆目录里的文件（见 chat/memory.py）；人设文件优先于配置里的 persona。

    emotes：这一轮能做的动作；为空时不出现“动作”一节，模型也被告知动不了。
    """
    parts = [(profile or cfg.persona).strip()]
    people = "\n".join(f"- {name}：{note}" for name, note in cfg.friends.items())
    if people or friends:
        parts.append(
            "## 认识的人\n"
            + "\n\n".join(p for p in (people, friends.strip()) if p)
            + "\n\n称呼他们时用这里的叫法。朋友之间互相起的外号是在叫对方，别当成在叫你。"
        )
    if notes.strip():
        parts.append("## 长期记忆（之前聊天里记下的，可能不全；和上面冲突时以上面为准）\n" + notes.strip())
    move = MOVE_WITH_EMOTES if emotes else MOVE_PLAIN
    parts.append(RULES.format(max_chars=cfg.max_chars, skip=SKIP_TOKEN, move=move))
    if emotes:
        parts.append(EMOTE_RULES.format(names="、".join(emotes)))
    return "\n\n".join(parts)
```

把 `clean_reply` 里的前缀正则提成常量，并在它后面加 `Reply` 和 `parse_reply`：

```python
_PREFIX = re.compile(r"^(回复|答|AI|我)\s*[:：]\s*")


def clean_reply(raw: str, max_chars: int) -> str | None:
    text = (raw or "").strip()
    if not text or SKIP_TOKEN in text.lower():
        return None
    # 有的模型会带“回复：”前缀或引号，或多写几行解释，只取第一行正文
    text = next((line.strip() for line in text.splitlines() if line.strip()), "")
    text = _PREFIX.sub("", text)
    text = text.strip(_QUOTES + " *`")
    if not text:
        return None
    if _CLAIMS_HUMAN.search(text):
        log.warning("模型回复里声称自己是真人，不发: %s", text)
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


@dataclass(frozen=True)
class Reply:
    """一轮回复：要发的文字和 / 或要做的动作，至少有一个。"""

    text: str | None = None
    emote: str | None = None

    def render(self, prefix: str = "") -> str:
        """[害羞]哪有啦：存进聊天历史、打日志用；prefix（【AI】标识）只加在文字前面。"""
        return (f"[{self.emote}]" if self.emote else "") + (prefix + self.text if self.text else "")


_TAG_HEAD = re.compile(r"^[\[【]([^\[\]【】]{1,10})[\]】]\s*")
_TAG_TAIL = re.compile(r"\s*[\[【]([^\[\]【】]{1,10})[\]】]$")


def parse_reply(raw: str, max_chars: int, emotes: Sequence[str] = ()) -> Reply | None:
    """模型输出 → Reply。句首或句尾的 [动作名] / 【动作名】是动作标签；名字不在 emotes 里就只留文字。"""
    text = (raw or "").strip()
    if not text or SKIP_TOKEN in text.lower():
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = _PREFIX.sub("", lines[0]).strip(_QUOTES + " *`")
    emote = None
    tag = _TAG_HEAD.match(first) or _TAG_TAIL.search(first)
    if tag:
        first = (first[: tag.start()] + first[tag.end() :]).strip()
        name = tag.group(1).strip()
        if name in emotes:
            emote = name
        else:
            log.warning("模型用了这一轮不能做的动作「%s」，只发文字", name)
        if not first and len(lines) > 1:  # 标签单独一行，正文在下一行
            first = lines[1]
    if first and _CLAIMS_HUMAN.search(first):  # 整轮都不发：动作也不做
        log.warning("模型回复里声称自己是真人，不发: %s", first)
        return None
    body = clean_reply(first, max_chars) if first else None
    if body is None and emote is None:
        return None
    return Reply(body, emote)
```

`Responder.__init__` 加参数并保存（参数列表末尾加一个，其余不动）：

```python
        clock: Callable[[], float] = time.time,
        available_emotes: Callable[[], list[str]] | None = None,
    ) -> None:
        self.llm = llm
        self.cfg = cfg
        self.store = store
        self.notes = notes
        self.clock = clock
        # 每轮调用：这一轮能做哪些动作（限速中 / 关掉了就是空列表）
        self.available_emotes = available_emotes or (lambda: [])
```

把 `system` 属性换成方法：

```python
    def system_prompt(self, emotes: Sequence[str] = ()) -> str:
        # 每次都重新读记忆文件：用户改了人设 / 好友，或者笔记刚在后台更新过，不用重启就生效
        if self.store is None:
            return build_system_prompt(self.cfg, emotes=emotes)
        notes = self.store.notes()
        inbox = self.store.inbox()
        if inbox:
            notes = (notes + "\n\n" if notes else "") + "刚记下的：\n" + inbox
        return build_system_prompt(self.cfg, self.store.profile(), self.store.friends(), notes, emotes)
```

`_remember` 改成接收 `Reply | None`：

```python
    def _remember(self, user_content: str, reply: Reply | None, now: float) -> None:
        said = reply.render() if reply else SKIP_TOKEN  # 带着 [动作名]，模型记得自己做过什么
        self.history.append({"role": "user", "content": user_content})
        self.history.append({"role": "assistant", "content": said})
        self.last_turn_at = now
        if self.store is not None:
            try:
                self.store.history.append(user_content, said, now)
            except OSError:
                log.exception("写聊天记录失败")
        if self.notes is not None:  # 后台：挑出这一轮值得记的，攒够了再整理进长期记忆
            self.notes.turn_added(Turn(now, user_content, said))
        limit = max(0, self.cfg.history_turns) * 2
        if len(self.history) > limit:
            self.history = self.history[len(self.history) - limit :]
```

`reply` 改成：

```python
    def reply(self, incoming: list[Message]) -> Reply | None:
        """返回这一轮的回复（文字不含 AI 标识前缀）；不需要回复时返回 None。"""
        if not incoming:
            return None
        now = self.clock()
        user_content = format_incoming(incoming)
        if self.last_turn_at is not None and now - self.last_turn_at >= GAP_NOTE_AFTER:
            user_content = f"（距离上次聊天过了 {format_gap(now - self.last_turn_at)}）\n" + user_content
        emotes = self.available_emotes()
        try:
            raw = self.llm.complete(self.system_prompt(emotes), self._messages(user_content))
        except Exception:
            log.exception("调用大模型失败")
            return None
        reply = parse_reply(raw, self.cfg.max_chars, emotes)
        self._remember(user_content, reply, now)
        return reply
```

- [ ] **Step 4: 导出**

`src/skydango/chat/__init__.py`：

```python
from .responder import Reply, Responder, build_system_prompt, clean_reply, parse_reply
```

`__all__` 里按字母序加 `"Reply"`（放在 `"Message"` 后）和 `"parse_reply"`（放在 `"normalize"` 后）。

- [ ] **Step 5: 调用方先改用 `.text`（动作流程在 Task 4）**

`src/skydango/agent.py` 的 `step` 里，`reply is None` 分支之后那一行：

```python
        text = self.cfg.reply.disclosure_prefix + reply
```

改成：

```python
        if reply.text is None:  # Task 4 之前不会出现：没给模型可用动作
            return None
        text = self.cfg.reply.disclosure_prefix + reply.text
```

`src/skydango/cli.py` 的 `cmd_chat` 最后一行：

```python
        print("（不回复）" if reply is None else reply.render(cfg.reply.disclosure_prefix))
```

- [ ] **Step 6: 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部 passed。

- [ ] **Step 7: Commit**

```bash
git add src/skydango/chat/responder.py src/skydango/chat/__init__.py src/skydango/agent.py src/skydango/cli.py tests/test_responder.py tests/test_memory.py
git commit -m "feat: 回复里的 [动作名] 标签和提示词里的动作一节

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: EmotePlayer（做动作、换轮盘、恢复）+ 配置

**Files:**
- Modify: `src/skydango/config.py`（新增 `EmoteConfig`，`Config` 加字段）
- Modify: `src/skydango/game/wheel.py`（`ensure` / `_victim` 的 `candidates`）
- Create: `src/skydango/game/emotes.py`
- Modify: `config.example.toml`
- Test: `tests/test_wheel.py`、`tests/test_emotes.py`、`tests/test_config.py`

**Interfaces:**
- Consumes: `Wheel`（`slots`、`refresh()`、`assign(slot, name, force=False)`、`perform(name) -> int`、`slot_of(name)`、`cfg.locked_slots`、`cfg.ui_delay`、`library.templates`）；`Device`（`ime_shown()`、`key(code)`、`hw_key(code)`）；`KEYCODE_BACK`（`device/base.py`，值 4）
- Produces:
  - `EmoteConfig`：`enabled: bool = True`、`extra: list[str] = []`、`swap_slots: list[int] = []`、`min_interval: float = 20.0`、`swap_min_interval: float = 120.0`；`Config.emotes: EmoteConfig`
  - `Wheel.ensure(name, candidates: list[int] | None = None) -> int`、`Wheel._victim(candidates: list[int] | None = None) -> int`
  - `EmotePlayer(device, wheel, cfg: EmoteConfig, panel_visible: Callable[[], bool], panel_key: int, sleep=time.sleep, clock=time.monotonic)`
  - `EmotePlayer.start() -> None`、`available() -> list[str]`、`perform(name) -> int`、`pretend(name) -> None`、`restore() -> None`、`on_wheel() -> list[str]`；属性 `extra: list[str]`、`swap_slots: list[int]`、`original: dict[int, str]`

- [ ] **Step 1: 写失败的测试**

`tests/test_config.py` 末尾追加：

```python
def test_emotes_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[emotes]\nextra = ["拥抱"]\nswap_slots = [7]\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.emotes.extra == ["拥抱"] and cfg.emotes.swap_slots == [7]
    assert cfg.emotes.enabled is True and cfg.emotes.min_interval == 20.0
```

`tests/test_wheel.py` 末尾追加：

```python
def test_victim_limited_to_candidates(library):
    wheel, _, _ = make_wheel(library)
    assert wheel._victim([7, 3]) == 7  # 3 被锁定
    with pytest.raises(WheelError, match="锁定"):
        wheel._victim([3])
```

新建 `tests/test_emotes.py`：

```python
import numpy as np
import pytest
from conftest import FakeDevice
from test_wheel import icon

from skydango.config import EmoteConfig, WheelConfig
from skydango.device.base import KEYCODE_BACK
from skydango.game.emotes import EmotePlayer
from skydango.game.wheel import EmoteLibrary, Wheel, WheelError
from skydango.imageio import imwrite

PANEL_KEY = 46  # C


@pytest.fixture
def library(tmp_path):
    for name, kind in (("鞠躬", "circle"), ("欢呼", "cross"), ("指向", "person")):
        imwrite(tmp_path / f"{name}.png", icon(kind))
    return EmoteLibrary(tmp_path)


def make_player(library, cfg=None, panel=True, shown=False, slots=None):
    """真 Wheel，但打开编辑界面的 refresh / assign 换成直接改状态（编辑界面的流程在 test_wheel 里测）。"""
    device = FakeDevice([np.zeros((1080, 1920, 3), np.uint8)])
    device.shown = shown
    press = device.key

    def key(code):  # BACK 会关掉输入框
        press(code)
        if code == KEYCODE_BACK:
            device.shown = False

    device.key = key
    t = [100.0]
    wheel = Wheel(device, WheelConfig(), library, sleep=lambda s: None, clock=lambda: t[0])
    state = dict(slots or {1: "鞠躬", 2: None, 3: None, 4: None, 5: None, 6: None, 7: "欢呼", 8: None})

    def refresh():
        device.calls.append(("refresh",))
        wheel.slots = dict(state)
        return {s: (n, 1.0) for s, n in state.items()}

    def assign(slot, name, force=False):
        device.calls.append(("assign", slot, name))
        state[slot] = name
        wheel.slots[slot] = name

    wheel.refresh = refresh
    wheel.assign = assign
    cfg = cfg or EmoteConfig(extra=["指向"], swap_slots=[7])
    player = EmotePlayer(device, wheel, cfg, lambda: panel, PANEL_KEY, sleep=lambda s: None, clock=lambda: t[0])
    return player, device, t, state


def test_start_reads_wheel_with_panel_closed_then_reopens(library):
    player, device, _, _ = make_player(library)
    player.start()
    assert device.calls == [("hw_key", PANEL_KEY), ("refresh",), ("hw_key", PANEL_KEY)]
    assert player.original == {7: "欢呼"}


def test_start_leaves_panel_alone_when_closed(library):
    player, device, _, _ = make_player(library, panel=False)
    player.start()
    assert device.calls == [("refresh",)]


def test_unrecognized_swap_slot_is_not_used(library):
    player, _, _, _ = make_player(library, cfg=EmoteConfig(extra=["指向"], swap_slots=[6]))
    player.start()
    assert player.swap_slots == [] and player.available() == ["鞠躬", "欢呼"]


def test_config_filters_unknown_extra_and_locked_slots(library):
    player, _, _, _ = make_player(library, cfg=EmoteConfig(extra=["指向", "不存在"], swap_slots=[3, 7]))
    assert player.extra == ["指向"] and player.swap_slots == [7]


def test_available_skips_locked_slots(library):
    slots = {1: "鞠躬", 2: None, 3: "欢呼", 4: None, 5: None, 6: None, 7: None, 8: None}
    player, _, _, _ = make_player(library, cfg=EmoteConfig(), slots=slots)
    player.start()
    assert player.available() == ["鞠躬"]  # 3 号格锁定（道具），不给模型


def test_perform_on_wheel_presses_number_key_only(library):
    player, device, _, _ = make_player(library)
    player.start()
    device.calls.clear()
    assert player.perform("鞠躬") == 1
    assert device.calls == [("hw_key", 2)]  # KEY_1


def test_perform_closes_input_box_first(library):
    player, device, _, _ = make_player(library, shown=True)
    player.start()
    device.shown = True
    device.calls.clear()
    player.perform("鞠躬")
    assert device.calls == [("key", KEYCODE_BACK), ("hw_key", 2)]


def test_perform_swaps_extra_into_swap_slot_with_panel_closed(library):
    player, device, _, state = make_player(library)
    player.start()
    device.calls.clear()
    assert player.perform("指向") == 7
    assert device.calls == [("hw_key", PANEL_KEY), ("assign", 7, "指向"), ("hw_key", PANEL_KEY), ("hw_key", 8)]
    assert state[7] == "指向"


def test_perform_rejects_emote_not_on_wheel_or_whitelist(library):
    player, device, _, _ = make_player(library, cfg=EmoteConfig())
    player.start()
    device.calls.clear()
    with pytest.raises(WheelError):
        player.perform("指向")
    assert device.calls == []


def test_available_respects_intervals(library):
    player, _, t, _ = make_player(library)
    player.start()
    assert player.available() == ["鞠躬", "欢呼", "指向"]
    player.perform("鞠躬")
    assert player.available() == []  # min_interval 20 s 内
    t[0] += 21
    assert player.available() == ["鞠躬", "欢呼", "指向"]
    player.perform("指向")  # 换掉了 7 号格的欢呼
    t[0] += 21
    assert player.available() == ["鞠躬", "指向"]  # 换轮盘还在 120 s 限速内：只给轮盘上的
    t[0] += 120
    assert player.available() == ["鞠躬", "指向"]  # 欢呼不在白名单，换下来就没了


def test_pretend_counts_for_intervals_without_touching_device(library):
    player, device, t, _ = make_player(library)
    player.start()
    device.calls.clear()
    player.pretend("指向")
    assert device.calls == [] and player.available() == []
    t[0] += 21
    assert player.available() == ["鞠躬", "欢呼"]  # 假装换过轮盘：换轮盘也在限速


def test_restore_puts_original_back(library):
    player, device, _, state = make_player(library)
    player.start()
    player.perform("指向")
    device.calls.clear()
    player.restore()
    assert device.calls == [("hw_key", PANEL_KEY), ("assign", 7, "欢呼"), ("refresh",), ("hw_key", PANEL_KEY)]
    assert state[7] == "欢呼"


def test_restore_noop_when_unchanged(library):
    player, device, _, _ = make_player(library)
    player.start()
    device.calls.clear()
    player.restore()
    assert device.calls == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_config.py tests/test_wheel.py tests/test_emotes.py`
Expected: FAIL，`ImportError: cannot import name 'EmoteConfig'`

- [ ] **Step 3: 配置**

`src/skydango/config.py` 里 `WheelConfig` 之后加：

```python
@dataclass
class EmoteConfig:
    """聊天时做表情动作（game/emotes.py）。做动作会松开牵手：牵着手时用 `run --no-emotes`。"""

    enabled: bool = True  # 图标库是空的时候自动不做
    # 白名单：不在轮盘上时可以换上去的动作（图标库里的名字）
    extra: list[str] = field(default_factory=list)
    # 允许换的格子；为空就只用轮盘上现有的动作。启动时记下原来的动作，退出时换回去
    swap_slots: list[int] = field(default_factory=list)
    min_interval: float = 20.0  # 两次动作至少隔这么久（秒）
    swap_min_interval: float = 120.0  # 两次换轮盘至少隔这么久：每次换要关聊天面板 5~10 秒
```

`Config` 里在 `wheel` 后面加一行：

```python
    emotes: EmoteConfig = field(default_factory=EmoteConfig)
```

- [ ] **Step 4: wheel.py 的 candidates**

把 `_victim` 和 `ensure` 换成：

```python
    def _victim(self, candidates: list[int] | None = None) -> int:
        free = self.free_slots()
        if candidates is not None:
            free = [s for s in candidates if s in free]
        if not free:
            raise WheelError("能换的格子都被锁定了，没法换动作")
        # 最久没用过的先换；都没用过就按编号
        return min(free, key=lambda s: (self.last_used.get(s, float("-inf")), s))

    def ensure(self, name: str, candidates: list[int] | None = None) -> int:
        """保证动作在轮盘上，返回格子编号；不在就换掉一个最久没用的空闲格子（candidates 限定从哪些格子里挑）。"""
        if not self.slots:
            self.refresh()
        slot = self.slot_of(name)
        if slot is None:
            slot = self._victim(candidates)
            self.assign(slot, name)
        return slot
```

注意报错文案从"所有格子都被锁定了"改成了"能换的格子都被锁定了"——`test_victim_limited_to_candidates` 用 `match="锁定"`，两种都匹配。

- [ ] **Step 5: 新建 `src/skydango/game/emotes.py`**

```python
"""聊天时做表情动作：轮盘上有的直接按数字键；白名单里的先换到指定格子再做。

- 换轮盘要长按 Z，聊天记录面板（C）开着时长按 Z 没反应 → 换之前关面板、换完再打开
- 输入框开着时按键会变成打字 → 先按 BACK 关掉
- 启动时记下可换格子原来的动作，退出时换回去
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from ..config import EmoteConfig
from ..device.base import KEYCODE_BACK, Device
from .wheel import Wheel, WheelError

log = logging.getLogger(__name__)


class EmotePlayer:
    def __init__(
        self,
        device: Device,
        wheel: Wheel,
        cfg: EmoteConfig,
        panel_visible: Callable[[], bool],
        panel_key: int,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.device = device
        self.wheel = wheel
        self.cfg = cfg
        self.panel_visible = panel_visible
        self.panel_key = panel_key  # 开关聊天记录面板的键；0 表示没有面板要管
        self.sleep = sleep
        self.clock = clock
        known = wheel.library.templates
        self.extra = [n for n in cfg.extra if n in known]
        missing = [n for n in cfg.extra if n not in known]
        if missing:
            log.warning("emotes.extra 里这些动作图标库里没有，忽略: %s", "、".join(missing))
        self.swap_slots = [s for s in cfg.swap_slots if s not in wheel.cfg.locked_slots]
        if len(self.swap_slots) < len(cfg.swap_slots):
            log.warning("emotes.swap_slots 里有锁定的格子（wheel.locked_slots），不会换它们")
        self.original: dict[int, str] = {}  # 可换格子原来的动作，退出时换回去
        self.last_emote = float("-inf")
        self.last_swap = float("-inf")

    def start(self) -> None:
        """读一次轮盘，记下可换格子原来的动作。"""
        self._close_input()
        with self._panel_closed():
            self.wheel.refresh()
        for slot in list(self.swap_slots):
            name = self.wheel.slots.get(slot)
            if name is None:
                log.warning("格子 %d 里的动作认不出（图标库里没有），换了就换不回去，所以不用它", slot)
                self.swap_slots.remove(slot)
            else:
                self.original[slot] = name

    def on_wheel(self) -> list[str]:
        locked = self.wheel.cfg.locked_slots
        return [name for slot, name in sorted(self.wheel.slots.items()) if name and slot not in locked]

    def available(self) -> list[str]:
        """这一轮能给模型用的动作：动作限速中为空；换轮盘限速中只有轮盘上现有的。"""
        now = self.clock()
        if now - self.last_emote < self.cfg.min_interval:
            return []
        names = self.on_wheel()
        if self.swap_slots and now - self.last_swap >= self.cfg.swap_min_interval:
            names += [n for n in self.extra if n not in names]
        return names

    def perform(self, name: str) -> int:
        self._close_input()
        now = self.clock()
        if name not in self.on_wheel():
            if name not in self.extra or not self.swap_slots:
                raise WheelError(f"「{name}」不在轮盘上，也不在可以换上去的白名单里")
            with self._panel_closed():
                self.wheel.ensure(name, candidates=self.swap_slots)
            self.last_swap = now
        slot = self.wheel.perform(name)
        self.last_emote = now
        return slot

    def pretend(self, name: str) -> None:
        """dry-run：不按键，只记下时间，让限速和真跑时一样。"""
        now = self.clock()
        if name not in self.on_wheel():
            self.last_swap = now
        self.last_emote = now

    def restore(self) -> None:
        """把换过的格子换回启动时的动作，再读一遍核对。"""
        changed = {s: n for s, n in self.original.items() if self.wheel.slots.get(s) != n}
        if not changed:
            return
        self._close_input()
        with self._panel_closed():
            for slot, name in changed.items():
                try:
                    self.wheel.assign(slot, name)
                except WheelError as exc:
                    log.warning("格子 %d 没换回「%s」: %s", slot, name, exc)
            self.wheel.refresh()
        wrong = [f"{s}（应为{n}）" for s, n in changed.items() if self.wheel.slots.get(s) != n]
        if wrong:
            log.warning("这些格子没恢复，请用 emotes wheel 检查: %s", "、".join(wrong))
        else:
            log.info("轮盘已恢复: %s", "、".join(f"{s}={n}" for s, n in changed.items()))

    def _close_input(self) -> None:
        if self.device.ime_shown():
            self.device.key(KEYCODE_BACK)
            self.sleep(0.3)

    @contextmanager
    def _panel_closed(self) -> Iterator[None]:
        was_open = bool(self.panel_key) and self.panel_visible()
        if was_open:
            self.device.hw_key(self.panel_key)
            self.sleep(self.wheel.cfg.ui_delay)
        try:
            yield
        finally:
            if was_open:
                self.device.hw_key(self.panel_key)
                self.sleep(self.wheel.cfg.ui_delay)
```

**如果 Task 1 的结论是 B（面板开着时数字键没反应）**，`perform` 最后按键那段改成下面这样，并把 `test_perform_on_wheel_presses_number_key_only` 的期望改成 `[("hw_key", PANEL_KEY), ("hw_key", 2), ("hw_key", PANEL_KEY)]`、`test_perform_closes_input_box_first` 改成 `[("key", KEYCODE_BACK), ("hw_key", PANEL_KEY), ("hw_key", 2), ("hw_key", PANEL_KEY)]`、`test_perform_swaps_extra_into_swap_slot_with_panel_closed` 改成 `[("hw_key", PANEL_KEY), ("assign", 7, "指向"), ("hw_key", PANEL_KEY), ("hw_key", PANEL_KEY), ("hw_key", 8), ("hw_key", PANEL_KEY)]`：

```python
        with self._panel_closed():  # 实测面板开着时数字键没反应（game-ops §4）
            slot = self.wheel.perform(name)
```

- [ ] **Step 6: config.example.toml**

在 `[sender]` 一节之前插入：

```toml
[emotes]
# 聊天时让模型挑时机做动作。做动作会松开牵手：牵着手时用 `run --no-emotes`
enabled = true
# 可以换上轮盘的动作（图标库 emotes/ 里的名字），不在轮盘上时会换到 swap_slots 里再做
extra = []
# 允许换的格子；为空就只用轮盘上现有的动作。退出时换回原来的动作
swap_slots = []
min_interval = 20.0           # 两次动作至少隔几秒
swap_min_interval = 120.0     # 两次换轮盘至少隔几秒（换一次要关聊天面板 5~10 秒）
```

- [ ] **Step 7: 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部 passed（`test_example_config_loads` 也会加载新的 `[emotes]` 一节）。

- [ ] **Step 8: Commit**

```bash
git add src/skydango/config.py src/skydango/game/wheel.py src/skydango/game/emotes.py config.example.toml tests/test_config.py tests/test_wheel.py tests/test_emotes.py
git commit -m "feat: EmotePlayer：聊天时做动作、按需换轮盘、退出恢复

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Agent 接上动作

**Files:**
- Modify: `src/skydango/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `Reply`（Task 2）；`EmotePlayer.perform(name)`、`pretend(name)`（Task 3，Agent 只用这两个，测试里用鸭子类型的假对象）
- Produces: `Agent(..., run: RunDir | None = None, emotes: EmotePlayer | None = None)`；`Agent.emoted: list[str]`

- [ ] **Step 1: 写失败的测试**

`tests/test_agent.py` 顶部 import 加：

```python
from skydango.game.wheel import WheelError
```

把 `build` 换成下面这个（新增 `llm` / `emotes` 两个可选参数，老测试不用改）：

```python
def build(cfg: Config, frames, ocr_texts, clock, llm=None, emotes=None):
    device = FakeDevice(frames)
    self_filter = SelfFilter(cfg.chat.self_window, cfg.chat.similarity, cfg.reply.disclosure_prefix)
    reader = ChatReader(FakeOcr(ocr_texts), cfg.vision, cfg.ocr, cfg.chat, self_filter)
    available = emotes.available if emotes else None
    responder = Responder(llm or EchoClient(), cfg.reply, available_emotes=available)
    sender = ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None)
    agent = Agent(cfg, device, reader, responder, sender, self_filter, clock=clock, sleep=lambda s: None, emotes=emotes)
    if emotes:
        emotes.device = device
    return agent, device
```

文件末尾追加：

```python
class FixedLlm:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, messages):
        return self.reply


class FakeEmotes:
    def __init__(self, names=("鞠躬", "害羞"), fail=False):
        self.names = list(names)
        self.fail = fail
        self.pretended = []
        self.device = None

    def available(self):
        return list(self.names)

    def perform(self, name):
        if self.fail:
            raise WheelError("动作列表里没找到")
        self.device.calls.append(("emote", name))
        return 1

    def pretend(self, name):
        self.pretended.append(name)


def run_one_turn(agent, clock):
    agent.step()
    clock.advance(2)
    return agent.step()


def test_emote_before_text(clock):
    cfg = live_config()
    emotes = FakeEmotes()
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["你真可爱"] * 5, clock, FixedLlm("[害羞]哪有啦"), emotes)
    assert run_one_turn(agent, clock) == "【AI】哪有啦"
    assert device.calls == [("emote", "害羞"), ("tap", 64, 72), ("text", "【AI】哪有啦"), ("editor", 4)]
    assert agent.emoted == ["害羞"]


def test_emote_only_turn_does_not_use_send_quota(clock):
    cfg = live_config()
    emotes = FakeEmotes()
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["晚安"] * 5, clock, FixedLlm("[鞠躬]"), emotes)
    assert run_one_turn(agent, clock) is None
    assert device.calls == [("emote", "鞠躬")]
    assert agent.sent == [] and agent.limiter.allow(clock())


def test_dry_run_pretends_emote(clock):
    cfg = Config()  # 默认 dry_run
    emotes = FakeEmotes()
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["你真可爱"] * 5, clock, FixedLlm("[害羞]哪有啦"), emotes)
    assert run_one_turn(agent, clock) == "【AI】哪有啦"
    assert device.calls == [] and emotes.pretended == ["害羞"]


def test_emote_failure_still_sends_text(clock):
    cfg = live_config()
    emotes = FakeEmotes(fail=True)
    agent, device = build(cfg, [scene([(400, 200, 300, 50)])], ["你真可爱"] * 5, clock, FixedLlm("[害羞]哪有啦"), emotes)
    assert run_one_turn(agent, clock) == "【AI】哪有啦"
    assert ("text", "【AI】哪有啦") in device.calls
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_agent.py`
Expected: FAIL，`TypeError: Agent.__init__() got an unexpected keyword argument 'emotes'`

- [ ] **Step 3: 实现**

`src/skydango/agent.py` import 加：

```python
from .game.emotes import EmotePlayer
```

`Agent.__init__` 参数末尾加 `emotes: EmotePlayer | None = None,`，函数体末尾加：

```python
        self.emotes = emotes
        self.emoted: list[str] = []  # 做过（含 dry-run）的动作，方便测试和日志
```

`step` 里从 `batch, self.pending = self.pending, []` 到函数结尾换成：

```python
        batch, self.pending = self.pending, []
        reply = self.responder.reply(batch)
        dry = self.cfg.reply.dry_run
        if reply is None:
            log.info("模型选择不回复")
            if self.run_dir:
                self.run_dir.record_reply(batch, None, sent=False)
            return None
        if reply.emote:
            self._emote(reply.emote)
        if reply.text is None:  # 只做动作不说话：不占发送限速
            if self.run_dir:
                self.run_dir.record_reply(batch, reply.render(), sent=not dry)
            return None

        text = self.cfg.reply.disclosure_prefix + reply.text
        self.limiter.record(now)
        self.sent.append(text)
        if dry:
            log.info("[dry-run] 将会发送: %s", text)
        else:
            self.sender.send(text)
            self.self_filter.remember(text, self.clock())
        if self.run_dir:
            self.run_dir.record_reply(batch, reply.render(self.cfg.reply.disclosure_prefix), sent=not dry)
        return text

    def _emote(self, name: str) -> None:
        self.emoted.append(name)
        if self.emotes is None:
            return
        if self.cfg.reply.dry_run:
            log.info("[dry-run] 将会做动作: %s", name)
            self.emotes.pretend(name)
            return
        try:
            self.emotes.perform(name)
        except Exception:  # 换轮盘没找到图标、adb 出错……动作做不成，话照样说
            log.exception("做动作「%s」失败，跳过", name)
```

（Task 2 Step 5 临时加的 `if reply.text is None: return None` 被上面这段替换掉。）

- [ ] **Step 4: 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部 passed。

- [ ] **Step 5: Commit**

```bash
git add src/skydango/agent.py tests/test_agent.py
git commit -m "feat: Agent 先做动作再发文字，只做动作不占发送限速

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 命令行（run --no-emotes、chat --emotes、退出恢复）+ 文档

**Files:**
- Modify: `src/skydango/cli.py`
- Modify: `AGENTS.md`、`README.md`、`docs/game-ops.md`
- Test: `tests/test_cli_emotes.py`

**Interfaces:**
- Consumes: `EmotePlayer`（Task 3）、`Responder(available_emotes=...)`（Task 2）、`Agent(emotes=...)`（Task 4）
- Produces: `_build_emotes(cfg, dev, reader, no_emotes: bool) -> EmotePlayer | None`；`_run_agent(cfg, run, no_emotes: bool = False)`

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_cli_emotes.py`：

```python
import numpy as np
import pytest
from conftest import FakeDevice
from test_wheel import icon

from skydango import cli
from skydango.config import Config
from skydango.game.emotes import EmotePlayer
from skydango.game.wheel import WheelError
from skydango.imageio import imwrite


class PanelReader:
    def panel_visible(self, frame):
        return True


@pytest.fixture
def cfg(tmp_path):
    cfg = Config()
    cfg.vision.mode = "log"
    imwrite(tmp_path / "鞠躬.png", icon("circle"))
    cfg.wheel.library_dir = str(tmp_path)
    return cfg


def device():
    return FakeDevice([np.zeros((1080, 1920, 3), np.uint8)])


def test_no_emotes_flag_or_disabled(cfg):
    assert cli._build_emotes(cfg, device(), PanelReader(), no_emotes=True) is None
    cfg.emotes.enabled = False
    assert cli._build_emotes(cfg, device(), PanelReader(), no_emotes=False) is None


def test_empty_library(cfg, tmp_path):
    cfg.wheel.library_dir = str(tmp_path / "没有")
    assert cli._build_emotes(cfg, device(), PanelReader(), no_emotes=False) is None


def test_start_failure_disables_emotes(cfg, monkeypatch):
    def boom(self):
        raise WheelError("没能打开轮盘编辑界面")

    monkeypatch.setattr(EmotePlayer, "start", boom)
    assert cli._build_emotes(cfg, device(), PanelReader(), no_emotes=False) is None


def test_builds_player(cfg, monkeypatch):
    monkeypatch.setattr(EmotePlayer, "start", lambda self: None)
    player = cli._build_emotes(cfg, device(), PanelReader(), no_emotes=False)
    assert isinstance(player, EmotePlayer)
    assert player.panel_key == cfg.vision.log_open_key and player.panel_visible() is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_cli_emotes.py`
Expected: FAIL，`AttributeError: module 'skydango.cli' has no attribute '_build_emotes'`

- [ ] **Step 3: 实现 cli.py**

`cmd_chat` 里构造 `Responder` 那行换成：

```python
    names = [n.strip() for n in args.emotes.split(",") if n.strip()]  # 假装轮盘上有这些动作
    responder = Responder(make_llm(cfg.llm), cfg.reply, available_emotes=lambda: names)
```

`cmd_run` 里 `_run_agent(cfg, run)` 改成 `_run_agent(cfg, run, args.no_emotes)`。

在 `_run_agent` 前面加：

```python
def _build_emotes(cfg: Config, dev, reader, no_emotes: bool):
    """准备聊天时做动作：读一次轮盘。做不了（关掉了 / 图标库空 / 读轮盘失败）返回 None，聊天照常。"""
    if no_emotes or not cfg.emotes.enabled:
        log.info("这次不做动作")
        return None
    from .game.emotes import EmotePlayer
    from .game.wheel import EmoteLibrary, Wheel

    library = EmoteLibrary(cfg.wheel.library_dir)
    if not library.names:
        log.warning("图标库 %s 是空的，这次不做动作（先 emotes scan，把用得上的改名放进去）", cfg.wheel.library_dir)
        return None
    if cfg.vision.mode == "log":
        panel_visible, panel_key = (lambda: reader.panel_visible(dev.screenshot())), cfg.vision.log_open_key
    else:
        panel_visible, panel_key = (lambda: False), 0
    player = EmotePlayer(dev, Wheel(dev, cfg.wheel, library), cfg.emotes, panel_visible, panel_key)
    try:
        player.start()
    except Exception as exc:
        log.warning("读轮盘失败，这次不做动作: %s", exc)
        return None
    log.info("能做的动作: 轮盘上 %s；可以换上去的 %s", "、".join(player.on_wheel()) or "（无）", "、".join(player.extra) or "（无）")
    return player
```

`_run_agent` 改成：

```python
def _run_agent(cfg: Config, run: RunDir, no_emotes: bool = False) -> None:
    from .agent import Agent
    from .chat.llm import make_llm
    from .chat.responder import Responder
    from .chat.sender import ChatSender

    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    reader.trace_path = run.rows_log
    llm = make_llm(cfg.llm)
    store = notes = None
    if cfg.reply.memory_dir and not cfg.reply.dry_run:  # dry-run 的回复没真的发出去，不记
        from .chat.memory import MemoryStore, NotesKeeper

        store = MemoryStore(cfg.reply.memory_dir)
        if not store.profile():
            log.warning("还没有人设文件 %s/profile.md，先用配置里的 persona；可以运行 memory init 生成", store.dir)
        notes = NotesKeeper(llm, store, cfg.reply.persona, cfg.reply.notes_every)
    emotes = _build_emotes(cfg, dev, reader, no_emotes)
    responder = Responder(llm, cfg.reply, store=store, notes=notes, available_emotes=emotes.available if emotes else None)
    sender = ChatSender(dev, cfg.sender, _screen_size_fn(dev))
    agent = Agent(cfg, dev, reader, responder, sender, self_filter, run=run, emotes=emotes)
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        if emotes is not None:
            try:
                emotes.restore()
            except Exception:
                log.exception("恢复轮盘失败，请用 emotes wheel 检查")
```

`main` 里 `chat` 子命令加参数：

```python
    p.add_argument("--emotes", default="", help="假装轮盘上有这些动作（逗号分隔），看模型怎么用")
```

`run` 子命令加参数：

```python
    p.add_argument("--no-emotes", action="store_true", help="这次不做动作（牵着手时用：做动作会松开牵手）")
```

- [ ] **Step 4: 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部 passed。

- [ ] **Step 5: 文档**

`AGENTS.md`：
- 开头"当前能力"那句改成：`当前能力：读聊天记录 → 大模型回复 → 在游戏里发言（可以带表情动作）；快捷动作轮盘的读取 / 编辑 / 做动作。`
- 代码结构表在 `game/wheel.py` 那行后加：
  `| src/skydango/game/emotes.py | 聊天时做动作：可用动作 + 限速、轮盘上的直接按键、白名单动作换进 swap_slots、退出恢复 |`（路径用反引号，和表里其他行一致）
- 常用命令里 `run` 那行改成 `python -m skydango run [--echo] [--live] [--no-emotes]  # Agent；默认 dry-run，--echo 不调模型，牵着手时 --no-emotes`，
  并在 `say` 那行后加 `python -m skydango chat --emotes 鞠躬,害羞       # 终端里和人设聊天，假装轮盘上有这些动作`
- "工作约定"末尾加一条：`- 聊天时做动作会松开牵手；Agent 退出时会把 swap_slots 换回原样，改了换轮盘逻辑要保证这一点。`

`README.md`：在 `skydango run --live` 那行后加 `skydango run --no-emotes          # 不做表情动作（牵着手时用）`；在 emotes 命令那组后面加一段说明：

```markdown
聊天时做动作：`[emotes]` 配置。轮盘上（除锁定格外）的动作模型都能用；`extra` 里的动作会在需要时换进 `swap_slots`，退出时换回去。
轮盘上的动作要在图标库里有命名好的图标（启动时读一次轮盘靠它们认）。
```

`docs/game-ops.md` §4「副作用」一节末尾加：

```markdown
- Agent 聊天时做动作（`game/emotes.py`）：轮盘上的直接按数字键；白名单动作要换轮盘，先按 C 关聊天记录面板、编辑完再按 C 打开，
  期间别人发的消息留在面板里，重开后照样读得到。启动时读一次轮盘（同样要关 / 开面板），退出时把 `swap_slots` 换回原样。
```

- [ ] **Step 6: Commit**

```bash
git add src/skydango/cli.py tests/test_cli_emotes.py AGENTS.md README.md docs/game-ops.md
git commit -m "feat: run 接上表情动作（--no-emotes、退出恢复轮盘），chat --emotes 调提示词

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 端到端验证（需要用户先命名图标）

前提：用户已把轮盘上的动作和白名单动作从 `tmp/emscan` 复制到 `emotes/` 并改名，`config.toml` 的 `wheel.library_dir` 改回 `emotes`。没做就停下来提醒用户。

- [ ] **Step 1: 调提示词（不进游戏）**

Run: `python -m skydango chat --emotes 鞠躬,害羞,欢呼`，依次输入：`卡洛：你好呀`、`卡洛：你今天好可爱`、`卡洛：给我鞠个躬`、`卡洛：跳个舞`、`卡洛：我去吃饭了`、`卡洛：嗯`
Expected: 大多数回复不带动作；"鞠个躬"那句带 `[鞠躬]`；"跳个舞"不编动作名、用文字接话；没有输出 emoji 或多行。频率明显过高就调 `EMOTE_RULES` 的措辞，改完重跑 `python -m pytest -q`。

- [ ] **Step 2: 读轮盘**

Run: `python -m skydango emotes wheel`
Expected: 非锁定格子都认得出名字；认不出的格子不会出现在可用动作里（给用户看结果）。

- [ ] **Step 3: dry-run**

`config.toml` 里临时设 `[emotes] extra = [...]`、`swap_slots = [...]`（和用户商量用哪一格）。
Run: `python -m skydango run`，让好友说几句，Ctrl+C 退出。
Expected: 日志开头有"能做的动作: 轮盘上 …；可以换上去的 …"；有 `[dry-run] 将会做动作: …`；两次动作间隔 ≥ 20 秒；退出时轮盘没被改（dry-run 不换）。

- [ ] **Step 4: live（和用户约好、没牵手时）**

Run: `python -m skydango run --live`
Expected: 做轮盘上的动作时聊天面板不闪（Task 1 结论 A）；做白名单动作时面板关 / 开一次，之后新消息照常读到；每一步截图确认。Ctrl+C 退出后日志有"轮盘已恢复"。

- [ ] **Step 5: 核对轮盘**

Run: `python -m skydango emotes wheel`
Expected: 和 Step 2 的结果一致。把实测到的新坑写进 `docs/game-ops.md` §4，提交：

```bash
git add docs/game-ops.md
git commit -m "docs: Agent 做动作的真机验证记录

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
