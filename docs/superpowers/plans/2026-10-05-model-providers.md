# 模型供应商和按用处选模型 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把所有调模型的地方收进一个「供应商 + 用处」路由：除识图外默认走 DeepSeek，每处可选主 / 备模型，管理面板新增「模型」页维护供应商和选择。

**Architecture:** 新包 `src/skydango/models/`：`config.py` 把 `[providers.*]` / `[models.*]`（和旧字段换算）解析成 `ModelSetup`；`claude_code.py` / `openai_compat.py` 两种接入方式各一个后端；`gate.py` 每家一个闸；`registry.py` 按用处给出 `GatedCall`（主 → 备）和大脑会话。cli、大脑循环、工具箱、管理面板都改成只从 `Registry` 拿模型。

**Tech Stack:** Python 3.13、`openai` SDK（OpenAI 兼容）、Claude Code CLI（`claude -p` stream-json）、管理面板原生 JS（无框架）、pytest。

**Spec:** `docs/superpowers/specs/2026-10-05-model-providers-design.md`

## Global Constraints

- 注释、日志、提示词、面板文案一律中文，照周围代码的密度和说法写。
- `config.toml` 面板只读不写；面板改的写 `console.toml`，密钥按环境变量名写 `secrets.toml` 的 `[env]`。
- 供应商 id：`^[a-z0-9_-]+$`；模型写成 `供应商id/模型名`，按**第一个** `/` 拆。
- 接入方式只有 `claude-code` / `openai`；内部还有 `echo`（面板不列）。
- 默认值（spec §1.2 原样）：brain / memory / reflect 主 `deepseek/deepseek-chat` 备 `claude/sonnet`；reply / text_label 主 `deepseek/deepseek-chat` 无备；eyes / wardrobe 主 `claude/haiku`；image_label 主 `claude/sonnet`；
  temperature 都是 0.8；max_tokens：brain 4096、memory 4096、reflect 4096、reply 200、text_label 4096、eyes 2048、wardrobe 1024、image_label 8192。
- 内置供应商：`claude`（kind claude-code，models / vision = sonnet、haiku、opus）、`deepseek`（kind openai，`https://api.deepseek.com`，`DEEPSEEK_API_KEY`，models deepseek-chat、deepseek-reasoner，vision 空）。
- 关闸：claude-code 的 limit / auth；openai 的 401、403（auth）、402（limit）、429 且错误码 `insufficient_quota`（limit）。别的 429、超时、连接错、5xx 不关闸。
- 测试命令：`python -m pytest -q`（本机系统 Python 有 pytest；worktree 里 pytest 用的是 worktree 的 `src`）。在 worktree 里手动跑 `python -m skydango …` 要加 `PYTHONPATH=src`。
- 全量测试放后台跑，别前台挂着等；跑全量时别改文件。
- 每个任务做完就提交；提交信息用中文 `feat(models): …` / `refactor(brain): …` 这类前缀，末尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

## Review Focus

1. **本机现在的配置**：`config.toml` 只有 `[llm]`（DeepSeek）、`console.toml` 只有 `[console]` / `[appearance]`。换算后大脑必须是 `deepseek/deepseek-chat`，Key 从 `DEEPSEEK_API_KEY` 读，团子能起来。→ Task 2 `test_local_layout_brain_on_deepseek`。
2. **面板第一次保存**：面板一写 `[providers]`，内置默认就不再生效，旧 `[llm]` 的地址 / 模型如果没带进去，DeepSeek 就配丢了。保存的必须是当时生效的值。→ Task 10 `test_first_save_keeps_legacy_llm_values`。
3. **别的用处先撞墙**：随手记撞上 DeepSeek 402 关了闸，大脑也在 DeepSeek 上，下一轮应该直接切到备用，而不是自己再撞一次。→ Task 7 `test_gate_closed_elsewhere_switches_before_send`。
4. **模型名里带 `/`**（比如 Ollama 的 `qwen/qwen2.5`）：只按第一个 `/` 拆，不能报错。→ Task 2 `test_model_ref_splits_on_first_slash`。
5. **识图用处选了看不了图的模型**（比如在面板上把眼睛选成 deepseek）：这一处停用、团子照常起；大脑调 look_person 时代看返回「现在看不了图」，不能抛异常。→ Task 2 `test_vision_use_with_blind_model_is_disabled`，Task 8 `test_proxy_when_eyes_unavailable`。

---

### Task 1: Config 收 `[providers]` / `[models]` 并记每个键的来源

**Files:**
- Modify: `src/skydango/config.py`（`Config`、`_merge`、`load_config`）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Config.providers: dict[str, dict]`，`Config.models: dict[str, dict]`：原样的表，两个文件**按 id / 用处名合并**（同一 id 下 console.toml 的字段覆盖 config.toml 的）。
  - `Config.sources: dict[str, str]`：每个出现在配置文件里的叶子键的点路径 → `"config"` / `"console"`（比如 `"llm.model": "config"`、`"providers.deepseek.base_url": "console"`）。没出现过的键不在里面。

- [ ] **Step 1: 写失败的测试**

```python
def test_providers_and_models_merge_by_id(tmp_path):
    base = tmp_path / "config.toml"; over = tmp_path / "console.toml"
    base.write_text('[providers.deepseek]\nkind = "openai"\nbase_url = "https://a"\n[models.brain]\nmain = "deepseek/deepseek-chat"\n', encoding="utf-8")
    over.write_text('[providers.deepseek]\nbase_url = "https://b"\n[providers.gpt]\nkind = "openai"\n[models.brain]\nbackup = ""\n', encoding="utf-8")
    cfg = load_config(base, over)
    assert cfg.providers["deepseek"] == {"kind": "openai", "base_url": "https://b"}
    assert cfg.providers["gpt"] == {"kind": "openai"}
    assert cfg.models["brain"] == {"main": "deepseek/deepseek-chat", "backup": ""}
    assert cfg.sources["providers.deepseek.base_url"] == "console"
    assert cfg.sources["providers.deepseek.kind"] == "config"

def test_sources_record_legacy_keys(tmp_path):
    base = tmp_path / "config.toml"
    base.write_text('[llm]\nmodel = "deepseek-chat"\n[brain]\nmodel = "opus"\n', encoding="utf-8")
    cfg = load_config(base)
    assert cfg.sources["llm.model"] == "config" and cfg.sources["brain.model"] == "config"
    assert "brain.eyes_model" not in cfg.sources

def test_providers_must_be_tables(tmp_path):
    base = tmp_path / "config.toml"
    base.write_text('providers = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="providers"):
        load_config(base)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_config.py -q -k "providers or sources"`
Expected: FAIL（`Config` 没有 `providers` 字段 / `未知配置项: providers`）

- [ ] **Step 3: 实现**

`Config` 末尾加三个字段（`field(default_factory=dict)`）。`_merge(obj, data, path="", source="config")` 多一个 `source` 参数：每设一个叶子就记 `root.sources[f"{path}{key}"] = source`（递归时把根 `Config` 传下去，或者由 `load_config` 在 merge 之后用一个小的 `_leaves(data)` 函数把点路径全记一遍，二选一，后者更简单）。
顶层键 `providers` / `models` 特殊处理：值必须是「表的表」（否则 `ValueError("配置项 providers 应该是一个表，每家一个 [providers.<id>]")`），对每个 id `current.setdefault(id, {}).update(value[id])`。`load_config` 里 config.toml 记 `"config"`、overlay 记 `"console"`。`overlay_keys` 不用改。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py tests/test_config.py
git commit -m "feat(config): 收 [providers] / [models] 表，记每个键来自哪个文件"
```

---

### Task 2: `models/config.py`：解析供应商和用处、换算旧字段、校验

**Files:**
- Create: `src/skydango/models/__init__.py`（空，只写一行模块说明）
- Create: `src/skydango/models/config.py`
- Test: `tests/test_models_config.py`

**Interfaces:**
- Consumes: Task 1 的 `Config.providers` / `models` / `sources`；现有 `cfg.llm`、`cfg.brain.claude_path / token_env / config_dir / model / eyes_model / memory_model`、`cfg.inner.reflect_model`、`cfg.appearance.describe_model`、`cfg.assist.model`。
- Produces:

```python
KINDS = ("claude-code", "openai")          # 面板可选；另有内部 "echo"
@dataclass(frozen=True)
class ProviderConfig:
    id: str; kind: str
    models: tuple[str, ...] = (); vision: tuple[str, ...] = ()
    path: str = "claude"; token_env: str = "SKYDANGO_CLAUDE_TOKEN"; config_dir: str = ".brain-claude"   # claude-code
    base_url: str = ""; key_env: str = ""; timeout: float = 30.0; max_retries: int = 2                   # openai
    source: str = "default"                 # default / config / console（取这家任一字段最"新"的来源）
    def sees(self, model: str) -> bool      # model in vision
    def secret_env(self) -> str             # claude-code → token_env；openai → key_env；echo → ""

@dataclass(frozen=True)
class ModelRef:
    provider: str; model: str
    @staticmethod
    def parse(text: str) -> "ModelRef"      # 按第一个 "/" 拆；空、没有 "/"、任一边为空 → ValueError
    def __str__(self) -> str                # "provider/model"

@dataclass(frozen=True)
class Use:                                  # 用处的固定定义
    name: str; label: str; help: str; vision: bool
    main: str; backup: str; temperature: float; max_tokens: int

USES: tuple[Use, ...]                       # 顺序同 spec §1.2 的表：brain, memory, reflect, reply, text_label, eyes, wardrobe, image_label
USE_NAMES: tuple[str, ...]

@dataclass
class UseConfig:
    name: str; main: ModelRef | None; backup: ModelRef | None
    temperature: float; max_tokens: int
    source: str                             # default / config / console / legacy（旧字段换算来的）
    disabled: str = ""                      # 非空 = 主和备都用不了的原因（这一处停用）

@dataclass(frozen=True)
class Problem:
    text: str; use: str | None = None; provider: str | None = None

@dataclass
class ModelSetup:
    providers: dict[str, ProviderConfig]    # 含内部 "echo"（kind echo，models ("echo",)）
    uses: dict[str, UseConfig]
    problems: list[Problem]
    legacy: list[str]                       # 换算过的旧字段，启动时警告一行一个

def resolve(cfg: Config) -> ModelSetup
```

`Use.label` / `help` 照 spec §1.2 的「说明」列写，比如 brain = `("大脑", "每次醒来想、说、调工具")`。

- [ ] **Step 1: 写失败的测试**

```python
def _cfg(tmp_path, text="", console=""):
    base = tmp_path / "config.toml"; base.write_text(text, encoding="utf-8")
    over = tmp_path / "console.toml"; over.write_text(console, encoding="utf-8")
    return load_config(base, over)

def test_defaults(tmp_path):
    s = resolve(_cfg(tmp_path))
    assert set(s.providers) >= {"claude", "deepseek"}
    assert str(s.uses["brain"].main) == "deepseek/deepseek-chat" and str(s.uses["brain"].backup) == "claude/sonnet"
    assert str(s.uses["eyes"].main) == "claude/haiku" and s.uses["eyes"].backup is None
    assert s.uses["reply"].max_tokens == 200 and s.uses["brain"].max_tokens == 4096
    assert s.problems == [] and s.legacy == []

def test_local_layout_brain_on_deepseek(tmp_path):   # Review Focus 1
    s = resolve(_cfg(tmp_path, '[llm]\nprovider = "openai"\nbase_url = "https://api.deepseek.com"\nmodel = "deepseek-chat"\napi_key_env = "DEEPSEEK_API_KEY"\n',
                     '[console]\n[appearance]\nenabled = true\n'))
    assert str(s.uses["brain"].main) == "deepseek/deepseek-chat"
    assert s.providers["deepseek"].key_env == "DEEPSEEK_API_KEY"
    assert str(s.uses["reply"].main) == "deepseek/deepseek-chat"

def test_legacy_llm_model_added_to_models(tmp_path):
    s = resolve(_cfg(tmp_path, '[llm]\nmodel = "deepseek-v4"\ntemperature = 0.5\nmax_tokens = 300\n'))
    assert "deepseek-v4" in s.providers["deepseek"].models
    assert str(s.uses["reply"].main) == "deepseek/deepseek-v4"
    assert s.uses["reply"].temperature == 0.5 and s.uses["reply"].max_tokens == 300
    assert any("llm" in x for x in s.legacy)

def test_legacy_claude_models(tmp_path):
    s = resolve(_cfg(tmp_path, '[brain]\nmodel = "opus"\neyes_model = "sonnet"\nmemory_model = "haiku"\n[inner]\nreflect_model = "opus"\n'
                               '[appearance]\ndescribe_model = "sonnet"\n[assist]\nmodel = "opus"\n'))
    assert {n: str(s.uses[n].main) for n in ("brain", "eyes", "memory", "reflect", "wardrobe", "image_label")} == {
        "brain": "claude/opus", "eyes": "claude/sonnet", "memory": "claude/haiku", "reflect": "claude/opus",
        "wardrobe": "claude/sonnet", "image_label": "claude/opus"}
    assert s.uses["brain"].source == "legacy"

def test_models_table_beats_legacy(tmp_path):
    s = resolve(_cfg(tmp_path, '[brain]\nmodel = "opus"\n[models.brain]\nmain = "deepseek/deepseek-chat"\n'))
    assert str(s.uses["brain"].main) == "deepseek/deepseek-chat"

def test_llm_echo_and_anthropic(tmp_path):
    assert str(resolve(_cfg(tmp_path, '[llm]\nprovider = "echo"\n')).uses["reply"].main) == "echo/echo"
    s = resolve(_cfg(tmp_path, '[llm]\nprovider = "anthropic"\n'))
    assert str(s.uses["reply"].main) == "deepseek/deepseek-chat" and any("anthropic" in x for x in s.legacy)

def test_explicit_providers_replace_builtin(tmp_path):
    s = resolve(_cfg(tmp_path, '[providers.gpt]\nkind = "openai"\nbase_url = "https://x"\nkey_env = "OPENAI_API_KEY"\nmodels = ["gpt-4o"]\nvision = ["gpt-4o"]\n'
                               '[models.brain]\nmain = "gpt/gpt-4o"\nbackup = ""\n'))
    assert "deepseek" not in s.providers and "claude" not in s.providers and "echo" in s.providers
    assert s.uses["brain"].backup is None
    assert s.uses["memory"].disabled            # 默认的 deepseek 没了
    assert any(p.use == "memory" for p in s.problems)

def test_model_ref_splits_on_first_slash():      # Review Focus 4
    r = ModelRef.parse("ollama/qwen/qwen2.5")
    assert (r.provider, r.model) == ("ollama", "qwen/qwen2.5")
    for bad in ("", "deepseek", "/x", "x/"):
        with pytest.raises(ValueError):
            ModelRef.parse(bad)

def test_vision_use_with_blind_model_is_disabled(tmp_path):   # Review Focus 5
    s = resolve(_cfg(tmp_path, '[models.eyes]\nmain = "deepseek/deepseek-chat"\n'))
    assert s.uses["eyes"].disabled and "看图" in s.uses["eyes"].disabled
    assert any(p.use == "eyes" for p in s.problems)

def test_bad_provider_rows(tmp_path):
    s = resolve(_cfg(tmp_path, '[providers.Bad]\nkind = "openai"\n[providers.x]\nkind = "anthropic"\n[providers.y]\nkind = "openai"\nmodels = ["a"]\nvision = ["b"]\n'))
    texts = " ".join(p.text for p in s.problems)
    assert "Bad" in texts and "anthropic" in texts and "vision" in texts

def test_unknown_model_only_warns(tmp_path):
    s = resolve(_cfg(tmp_path, '[models.brain]\nmain = "deepseek/deepseek-v9"\n'))
    assert not s.uses["brain"].disabled
    assert any(p.use == "brain" and "deepseek-v9" in p.text for p in s.problems)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_models_config.py -q`
Expected: FAIL（`No module named 'skydango.models'`）

- [ ] **Step 3: 实现 `resolve`**

顺序：① 供应商：`cfg.providers` 为空 → 内置 `claude`（path / token_env / config_dir 取 `cfg.brain`）+ `deepseek`（base_url / key_env / timeout / max_retries 取 `cfg.llm`，`cfg.llm.model` 不在 models 里就补到最前），不为空 → 只用写了的（未知字段、id 不合法、kind 不在 `KINDS`、vision 不是 models 的子集都记 `Problem` 并跳过这家 / 这一项）；总是加内部 `echo`。
② 用处：从 `Use` 默认值出发；`cfg.sources` 里显式写了旧字段且 `cfg.models[use]` 没写 `main` 时按 spec §1.3 换算（source = `"legacy"`，`legacy` 列表记一句「[brain] model 已改到 [models.brain] main」）；`[llm] provider = "echo"` → reply 主 `echo/echo`；`"anthropic"` → 记 legacy 一句「不再支持，reply 用默认」；`cfg.llm.temperature / max_tokens` 显式写了 → reply 的；然后 `cfg.models[use]` 的 `main` / `backup` / `temperature` / `max_tokens` 覆盖（`backup = ""` = 没有备用；未知字段 → Problem）。
③ 校验每个用处的主 / 备：供应商不存在 → 这个引用作废 + Problem；看图用处的模型不在那家 `vision` → 作废 + Problem（文字里带「看不了图」）；模型不在那家 `models` 里 → 只记 Problem、照用。主和备都作废 → `disabled` = 原因。
`source` 取最后起作用的那层（`cfg.sources` 里 `models.<use>.main` 的来源，否则 legacy / default）。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_models_config.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/models tests/test_models_config.py
git commit -m "feat(models): 解析供应商和用处、换算旧字段、校验"
```

---

### Task 3: 错误类型和按供应商的闸

**Files:**
- Create: `src/skydango/models/errors.py`、`src/skydango/models/gate.py`
- Modify: `src/skydango/brain/claude.py`（`ClaudeError` 改继承 `ModelError`；删 `ClaudeGate`、`claude_down` 改成转调 `down_kind`）
- Test: `tests/test_models_gate.py`

**Interfaces:**
- Produces:

```python
# errors.py
class ModelError(RuntimeError):
    def __init__(self, message: str, *, down: str | None = None, provider: str = "") -> None  # down ∈ {"limit", "auth", None}
    down: str | None; provider: str
class ModelUnavailable(ModelError): ...       # 主和备都不能用（调用方按"停用"处理）
def down_kind(exc: BaseException) -> str | None   # ModelError → exc.down；别的 → None

# brain/claude.py
class ClaudeError(ModelError):
    def __init__(self, message: str, limit: bool = False, auth: bool = False, provider: str = "claude") -> None
    # .limit / .auth 保留；down = "limit" if limit else "auth" if auth else None

# gate.py
class ProviderGates:
    def ok(self, provider: str) -> bool
    def reason(self, provider: str) -> str | None
    def trip(self, provider: str, kind: str, detail: str) -> bool   # 刚关上返回 True；WARNING 一行「<provider> 额度 / 余额用完 / 认证失败：…，这次运行里改用备用」
    def closed(self) -> dict[str, str]
```

- [ ] **Step 1: 写失败的测试**

```python
def test_gate_per_provider():
    g = ProviderGates()
    assert g.ok("deepseek") and g.ok("claude")
    assert g.trip("deepseek", "limit", "402") is True
    assert g.trip("deepseek", "auth", "x") is False          # 已经关了，原因不改
    assert not g.ok("deepseek") and g.ok("claude")
    assert "用完" in g.reason("deepseek") and g.closed() == {"deepseek": g.reason("deepseek")}

def test_claude_error_is_model_error():
    e = ClaudeError("x", limit=True)
    assert isinstance(e, ModelError) and e.down == "limit" and e.limit and e.provider == "claude"
    assert down_kind(ClaudeError("x", auth=True)) == "auth"
    assert down_kind(RuntimeError("x")) is None and down_kind(ModelError("x")) is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_models_gate.py -q`
Expected: FAIL（导入失败）

- [ ] **Step 3: 实现**；`brain/claude.py` 删 `ClaudeGate`（用到它的地方 Task 9 一起改；这一步先让 `GatedLlm` / `gated_describe` 暂时接 `ProviderGates` + `provider="claude"`，保证全量测试还能过——它们在 Task 6 被 `GatedCall` 取代后删除）。

- [ ] **Step 4: 跑测试**

Run: `python -m pytest tests/test_models_gate.py tests/test_brain_claude.py -q`
Expected: PASS（`test_brain_claude.py` 里用 `ClaudeGate` 的测试改成 `ProviderGates` + `"claude"`）

- [ ] **Step 5: 提交** `feat(models): 错误类型和按供应商的闸`

---

### Task 4: Claude Code 后端

**Files:**
- Create: `src/skydango/models/claude_code.py`
- Modify: `src/skydango/brain/claude.py`（`StreamProcess`、`user_message`、`check_result`、`one_shot_message`、`one_shot`、`claude_env`、`resolve_claude`、`LIMIT_WORDS`、`AUTH_WORDS`、`ClaudeError` 搬到 `claude_code.py`；`brain/claude.py` 只留 `from ..models.claude_code import …` 的再导出 + `ClaudeLlm` / `GatedLlm` / `gated_describe` 到 Task 6 删）
- Test: `tests/test_models_claude_code.py`（复用 `tests/fake_claude.py`）

**Interfaces:**
- Consumes: Task 2 `ProviderConfig`，Task 3 `ClaudeError` / `ModelError`
- Produces:

```python
def claude_base(provider: ProviderConfig, environ: Mapping[str, str] = os.environ) -> tuple[list[str], dict[str, str]]
    # 令牌：environ[token_env] → Windows 用户环境变量（chat.llm._user_env）；没有 → ModelError("没有 Claude 令牌：先运行 claude setup-token，再 setx <token_env> …", down="auth", provider=id)
    # 找不到命令 → ModelError(down="auth")（同样算"这家用不了"）
def claude_command(base: list[str], model: str, system: str, *, effort: str = "low", persist: bool = True) -> list[str]
    # = 现在 eyes_command / wardrobe_command / ClaudeLlm.command 的参数；persist=False 时加 --no-session-persistence（原 assist_command）、不带 --effort
class ClaudeCodeBackend:
    def __init__(self, provider: ProviderConfig, model: str, cwd: Path, *, environ: Mapping[str, str] = os.environ, effort: str = "low", persist: bool = True) -> None
    def message(self, system: str, content: str | list[dict], *, history: list[dict] | None = None,
                max_tokens: int | None = None, timeout: float = 120.0) -> dict
    # 起一次性进程；history 非空时把 history 每条 content 和 content 用 "\n\n" 拼成一条（同现在 ClaudeLlm）；
    # 返回 {"result": 文字, "usage": result 消息里的 usage, "provider": id, "model": model}；失败抛 ClaudeError(provider=id)
```

- [ ] **Step 1: 写失败的测试**：用 `tests/fake_claude.py` 的假进程（看现有 `test_brain_claude.py` 怎么接）：
  - `test_command_has_model_and_system`：`claude_command(["claude"], "haiku", "S")` 含 `"--model", "haiku"`、`"--system-prompt", "S"`、`"--effort", "low"`；`persist=False` 时含 `--no-session-persistence`、不含 `--effort`。
  - `test_message_returns_result_and_usage`：假进程回 `{"type":"result","subtype":"success","result":"ok","usage":{"input_tokens":3}}` → `{"result":"ok","usage":{"input_tokens":3},"provider":"claude","model":"haiku"}`。
  - `test_history_joined`：`history=[{"role":"user","content":"a"}]`、`content="b"` → 发出的那条消息 content 是 `"a\n\nb"`。
  - `test_limit_error_carries_provider`：假进程回 429 → `ClaudeError`，`down == "limit"`、`provider == "claude"`。
  - `test_missing_token_is_auth_down`：`claude_base(provider, environ={})`（并 monkeypatch `_user_env` 返回 ""）→ `ModelError`，`down == "auth"`。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_models_claude_code.py -q` → FAIL（导入失败）

- [ ] **Step 3: 实现**（搬代码 + 新类；`brain/eyes.py` 的 `eyes_command`、`vision/wardrobe.py` 的 `wardrobe_command`、`vision/assist.py` 的 `assist_command` 先保留，Task 9 删）

- [ ] **Step 4: 跑测试**：`python -m pytest tests/test_models_claude_code.py tests/test_brain_claude.py tests/test_brain_session.py -q` → PASS（`test_brain_session.py` 不存在就跳过这个路径）

- [ ] **Step 5: 提交** `refactor(models): Claude Code 后端搬进 models/`

---

### Task 5: OpenAI 兼容后端（文字 + 看图）

**Files:**
- Create: `src/skydango/models/openai_compat.py`
- Modify: `src/skydango/chat/llm.py`（删 `OpenAICompatClient` / `AnthropicClient` / `_with_options`；留 `LlmClient`、`ChatMessage`、`_user_env`、`read_key`、`EchoClient`；`make_llm` Task 6 改）
- Test: `tests/test_models_openai.py`（`tests/test_llm.py` 里测 OpenAI / Anthropic 客户端的用例删掉或挪过来）

**Interfaces:**
- Consumes: `ProviderConfig`、`ModelError`、`chat.llm.read_key`
- Produces:

```python
def to_openai_content(content: str | list[dict]) -> str | list[dict]
    # Claude 块 → OpenAI 片段：{"type":"text","text":…} 原样；{"type":"image","source":{"media_type":m,"data":d}} → {"type":"image_url","image_url":{"url":f"data:{m};base64,{d}"}}
def classify(exc: BaseException) -> str | None
    # 读 exc.status_code（openai.APIStatusError）：401 / 403 → "auth"；402 → "limit"；429 且 (exc.code == "insufficient_quota" 或 body["error"]["code"] == "insufficient_quota") → "limit"；其余 None
def build_client(provider: ProviderConfig, *, api_key: str | None = None, timeout: float | None = None, max_retries: int | None = None, environ: Mapping[str, str] = os.environ)
    # openai.OpenAI(base_url, api_key or read_key(key_env)(读 environ), timeout or provider.timeout, max_retries 默认 provider.max_retries)；没 Key → ModelError(down="auth", provider=id)
class OpenAIBackend:
    def __init__(self, provider: ProviderConfig, model: str, *, temperature: float, max_tokens: int,
                 timeout: float | None = None, max_retries: int | None = None, client=None, environ=os.environ) -> None   # client 懒建
    def message(self, system: str, content: str | list[dict], *, history: list[dict] | None = None,
                max_tokens: int | None = None, timeout: float | None = None) -> dict
    # messages = [system] + history + [{"role":"user","content": to_openai_content(content)}]；
    # 返回 {"result", "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens, "cache_read_input_tokens": prompt_cache_hit_tokens 或 0}, "provider", "model"}；
    # timeout 和构造时不同就用 client.with_options(timeout=…)；SDK 异常 → ModelError(f"{id} 出错：{exc}", down=classify(exc), provider=id)
```

- [ ] **Step 1: 写失败的测试**（假客户端：一个有 `chat.completions.create(**kw)` 的对象，记下 kw，返回带 `choices[0].message.content` 和 `usage` 的 `SimpleNamespace`）
  - `test_text_message`：发出的 `messages[0] == {"role":"system","content":"S"}`、最后一条是 user、`model` / `temperature` / `max_tokens` 对；结果的 usage 换成 input/output 名字。
  - `test_image_blocks_become_image_url`：`content=[image_block(np.zeros((8,8,3),np.uint8)), {"type":"text","text":"看"}]` → user content 第一项 `type == "image_url"`、url 以 `data:image/jpeg;base64,` 开头。
  - `test_classify`：造 `openai.APIStatusError`（`httpx.Response(status, request=httpx.Request("POST","http://x"))`，body 带 code）：401 / 403 → "auth"、402 → "limit"、429 + insufficient_quota → "limit"、429 rate_limit → None、500 → None；`TimeoutError()` → None。
  - `test_sdk_error_wrapped`：假客户端抛 402 → `ModelError`，`down == "limit"`、`provider == "deepseek"`。
  - `test_missing_key_is_auth_down`：`OpenAIBackend(..., environ={}).message(...)`（monkeypatch `_user_env` 为空）→ `ModelError(down="auth")`。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_models_openai.py -q` → FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 跑测试**：`python -m pytest tests/test_models_openai.py tests/test_llm.py -q` → PASS

- [ ] **Step 5: 提交** `feat(models): OpenAI 兼容后端，能发图、按状态码判关闸`

---

### Task 6: Registry 和 GatedCall（主 → 备）

**Files:**
- Create: `src/skydango/models/registry.py`
- Modify: `src/skydango/chat/llm.py`（`make_llm(cfg: LlmConfig, api_key=None)` 删掉；所有调用方 Task 9 改成 Registry。先搜 `make_llm(`，测试里用到的一起改）
- Modify: `src/skydango/brain/claude.py`（删 `ClaudeLlm`、`GatedLlm`、`gated_describe`）
- Test: `tests/test_models_registry.py`

**Interfaces:**
- Consumes: Tasks 2–5
- Produces:

```python
class Registry:
    def __init__(self, setup: ModelSetup, gates: ProviderGates, workdir: Path, environ: Mapping[str, str] = os.environ,
                 backends: Callable[[ProviderConfig, str, UseConfig, Path], object] | None = None) -> None  # backends：测试注入假后端
    setup: ModelSetup; gates: ProviderGates
    def call(self, use: str, *, timeout: float | None = None, max_retries: int | None = None, cwd: Path | None = None) -> "GatedCall"
        # cwd 默认 workdir / use；claude-code 的 persist = (use != "image_label")
    def current(self, use: str) -> ModelRef | None     # 主那家闸开着 → 主；否则备（备那家也开着）；都不行 → None
    def describe(self, use: str) -> str                 # "deepseek/deepseek-chat（备 claude/sonnet）"；停用写 "停用：<原因>"
    def provider(self, name: str) -> ProviderConfig
    def sees(self, ref: ModelRef | None) -> bool
    def requirements(self, uses: Iterable[str]) -> list[Problem]
        # 这些用处的主 / 备用到的每家：claude-code 查令牌（claude_base 能不能过）、openai 查 Key；返回问题（不抛）
    def log_summary(self) -> None                       # 启动时每个用处 INFO 一行「模型：brain = …」；legacy / problems 各 WARNING 一行

class GatedCall:
    use: str; main: ModelRef | None; backup: ModelRef | None
    timeout: float | None                              # 可写（下线反思压超时：主和备都用新值、备用不重试）
    def available(self) -> bool                         # current(use) is not None
    def message(self, system: str, content: str | list[dict], *, history: list[dict] | None = None, max_tokens: int | None = None) -> dict
    def text(self, system: str, content: str | list[dict]) -> str
    def complete(self, system: str, messages: list[dict[str, str]], max_tokens: int | None = None) -> str   # LlmClient 接口：history = messages[:-1]
```

`GatedCall.message` 的规则（spec §2.2）：主那家闸开 → 调主；抛 `ModelError` 且 `down_kind` 非空 → `gates.trip(主那家, kind, str(exc))`，有备就这一笔改走备、没备原样抛；主的闸关着 → 走备；备那家闸也关 / 没备 / 主备都 `disabled` → 抛 `ModelUnavailable(f"{use} 用不了：…")`。走备时备出关闸类错误也 trip 备那家再抛。DEBUG 一行「brain 这一笔改走 claude/sonnet（deepseek 余额用完）」。`max_tokens` 默认 `UseConfig.max_tokens`。

- [ ] **Step 1: 写失败的测试**（注入假后端：按 `(provider id, model)` 返回一个可编程的对象，记录调用、可以设定抛什么）
  - `test_main_ok`：主成功 → 结果里 `provider == "deepseek"`，备没被调。
  - `test_main_down_switches_and_trips`：主抛 `ModelError(down="limit", provider="deepseek")` → 返回备的结果；`gates.ok("deepseek") is False`；再调一次**不**调主。
  - `test_main_other_error_raises`：主抛 `ModelError(down=None)` → 原样抛、闸开着、备没调。
  - `test_no_backup_raises_unavailable_after_trip`：没备、主 auth → 第一次抛那个 `ModelError`；第二次抛 `ModelUnavailable`。
  - `test_disabled_use`：eyes 选了 deepseek（setup 里 disabled）→ `available() is False`，`text()` 抛 `ModelUnavailable`。
  - `test_complete_passes_history_and_default_max_tokens`：`complete("S", [{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}])` → 后端收到 `history` 两条、content `"c"`、`max_tokens == 4096`（memory 用处）。
  - `test_timeout_setter`：`call.timeout = 7` 后后端收到 `timeout == 7`。
  - `test_describe`：`describe("brain") == "deepseek/deepseek-chat（备 claude/sonnet）"`；deepseek 闸关后 `current("brain") == ModelRef("claude", "sonnet")`。
  - `test_requirements`：`environ={}` 时 `requirements(["brain"])` 列出 deepseek 缺 Key 和 claude 缺令牌两条；`environ={"DEEPSEEK_API_KEY": "k"}` 时只剩 claude 那条。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_models_registry.py -q` → FAIL

- [ ] **Step 3: 实现**（默认 `backends`：claude-code → `ClaudeCodeBackend`、openai → `OpenAIBackend`、echo → 包一层 `EchoClient` 的后端）

- [ ] **Step 4: 跑测试**：`python -m pytest tests/test_models_registry.py -q` → PASS

- [ ] **Step 5: 提交** `feat(models): Registry 按用处给模型，主 → 备自动切换`

---

### Task 7: 大脑会话：按 kind 建、跨 kind 切备用

**Files:**
- Create: `src/skydango/brain/toolloop.py`（`brain/deepseek.py` 改名搬过来；`deepseek.py` 删掉，引用一起改）
- Create: `src/skydango/brain/sessions.py`
- Modify: `src/skydango/brain/loop.py`（`Brain` 构造参数、`_wake`、`_failed`、`trace_state`、`_log`）
- Modify: `src/skydango/brain/llm_tools.py`（`openai_tools(toolbox, *, blind=True)` → `openai_tools(toolbox)`：工具全给，会返回图的几个加 `question`，见 Task 8；这一步先只改签名，`question` 在 Task 8 加）
- Test: `tests/test_brain_toolloop.py`（由 `tests/test_brain_deepseek.py` 改名）、`tests/test_brain_loop.py`

**Interfaces:**
- Consumes: `Registry.provider` / `setup.uses["brain"]` / `gates`、`claude_base`、`build_client`、`classify`
- Produces:

```python
# toolloop.py
BLIND_NOTE = """你看不到画面：判断靠消息里的状态、聊天记录、身边人名地名（身体 OCR / YOLO 认的）和眼睛写的场景描述。
看图的工具（look(image=true)、look_at、look_person、check_friend、panel_read(image=true)）会请眼睛代看，返回文字；想看清什么写在 question 里。
- 前面几条消息是你最近几轮收到的和做了的：别重复你刚说过的话（意思一样也不行），接着上一句聊
- recall 查不到就说记不清，别顺着别人的话编
- 嘴上说要做动作（点头、鞠躬……）就真的调 emote，没调就别这么说
- 单字、语气词（嗯、哦、哈）不用每句都接"""
ASIDE_NOTE = …                                   # 原样搬
class ToolLoopBrain:                             # 原 DeepSeekBrain，构造参数同原来 + provider: str
    model: str; provider: str
    def send(self, text: str) -> dict            # 出错抛 ModelError(down=classify(exc), provider=provider)；超时 ModelError("超时", provider=…)；返回 dict 多 "provider" / "model"

# sessions.py
def make_session(registry: Registry, ref: ModelRef, *, prompt: str, toolbox, mcp_url: str, workdir: Path,
                 cfg: BrainConfig, addressee: bool, on_message) -> object
    # claude-code → BrainSession(base, env, workdir/"session", mcp_url, prompt, ref.model, cfg.effort, cfg.turn_timeout, on_message)，base/env 来自 claude_base
    # openai → ToolLoopBrain(build_client(provider), prompt + "\n\n" + BLIND_NOTE + (ASIDE_NOTE if addressee else ""), toolbox, openai_tools(toolbox),
    #           model=ref.model, temperature=use.temperature, max_tokens=use.max_tokens, max_steps=cfg.max_steps, turn_timeout=cfg.turn_timeout,
    #           on_message=on_message, history=cfg.history, provider=ref.provider)
    # BrainSession 也要有 .provider（= ref.provider）属性：在 BrainSession.__init__ 加 provider 参数，默认 "claude"

# loop.py：Brain 构造参数改
Brain(cfg, chat, session, toolbox, events, nearby, eyes=None, clock=…, wall=…, run=None, store=None, trace=None, slow=None,
      fallback: Callable[[], object] | None = None,   # 懒建备用会话（make_session 的 partial）
      gates: ProviderGates | None = None)
# 当前会话的 provider / model 取 session.provider / session.model
```

`[brain]` 配置：`BrainConfig` 删 `fallback`、`force_fallback`、`fallback_max_tokens`、`fallback_history`，加 `history: int = 8`；
`DEPRECATED` 加 `brain.fallback`、`brain.force_fallback`、`brain.fallback_max_tokens`（读到只警告）；`MOVED` 只警告不搬值，所以在它旁边加 `RENAMED = {"brain.fallback_history": "brain.history"}`，`_merge` 读到旧名时把值写进新字段并警告。

loop 规则：
- `_wake` 开头：`gates` 不为空、当前会话那家闸关了、还没切过、有 `fallback` → `_switch_to_fallback(f"{provider} 不能用了（{reason}），大脑切到 {备用 provider/model}")`（备用会话这时才建；建失败（`ModelError`）→ WARNING、不切，走原来的失败退避）。
- `send` 抛 `ModelError`（含 `ClaudeError`）：`kind = down_kind(exc)`；有 kind → `gates.trip(exc.provider or 当前 provider, kind, …)`，有备用且没切过就切；否则照原来的退避（`limit_retry` / `BACKOFF`）。
- `trace_state()["model"]` = `f"{session.provider}/{session.model}"`；`"provider_gates"` = `gates.closed()`（替掉 `"claude_gate"`）；`"on_fallback"` 照旧。
- `_log` 记 `brain.jsonl` 多 `"provider"` / `"model"`（从 result 取，取不到用当前会话的）。
- `force_fallback` 那段删掉。

- [ ] **Step 1: 写失败的测试**（`tests/test_brain_loop.py` 里已有假会话的写法照着用）
  - `test_switches_to_backup_on_limit`：主假会话 `provider="deepseek"`，`send` 抛 `ModelError(down="limit", provider="deepseek")`；`fallback` 返回的假会话 `provider="claude"`；`_wake` 一次后 `brain.session is backup`、`gates.ok("deepseek") is False`、`brain.on_fallback`；`fallback` 只被调一次。
  - `test_gate_closed_elsewhere_switches_before_send`（Review Focus 3）：先 `gates.trip("deepseek", "limit", "402")`，再 `_wake` → 主会话的 `send` 没被调过、备用的被调了一次。
  - `test_no_fallback_backs_off`：`fallback=None`、主抛 limit → `backoff_until == now + cfg.limit_retry`。
  - `test_other_error_no_switch`：主抛 `ModelError(down=None)` → 不切、按 `BACKOFF` 退避。
  - `test_brain_jsonl_has_provider`：成功一轮后 `run.record_brain` 收到的 dict 有 `provider == "deepseek"`、`model == "deepseek-chat"`。
  - `tests/test_brain_toolloop.py`：原 DeepSeek 测试改名；加 `test_sdk_402_is_limit_down`（假客户端抛 402 → `ModelError.down == "limit"`、`provider == "deepseek"`）；`test_blind_note_in_system`（`make_session` openai 分支的 system 以 prompt 开头、含 BLIND_NOTE 第一句、不含「Claude 额度不足」）。
  - `tests/test_config.py`：`test_fallback_history_renamed`（写 `[brain] fallback_history = 3` → `cfg.brain.history == 3`）；`test_fallback_keys_deprecated`（写 `force_fallback = true` 不报错）。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_loop.py tests/test_brain_toolloop.py tests/test_config.py -q` → FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 跑测试**：同上 → PASS

- [ ] **Step 5: 提交** `refactor(brain): 大脑会话按供应商建，主那家撞墙切备用（跨 Claude / OpenAI 兼容）`

---

### Task 8: 代看：看不了图的大脑调看图工具时交给眼睛

**Files:**
- Modify: `src/skydango/brain/llm_tools.py`（删 `BLIND_EXCLUDE`；`QUESTION_TOOLS = ("look", "look_at", "look_person", "check_friend", "panel_read")`，`openai_tools` 给这几个加 `("question", "string", "")`）
- Modify: `src/skydango/brain/tools.py`（`ToolBox` 加 `proxy` / `sees` 参数，`run` 里换图）
- Modify: `src/skydango/brain/eyes.py`（`Eyes.proxy`）
- Test: `tests/test_brain_llm_tools.py`、`tests/test_brain_tools.py`、`tests/test_brain_eyes.py`

**Interfaces:**
- Consumes: `ModelError` / `ModelUnavailable`
- Produces:

```python
# eyes.py
PROXY_REQUEST = "大脑看不到图，请你替它看：{question}\n只说图里看得到的，名字只用上面位置说明里给的，别编；两三句话。"
def Eyes.proxy(self, blocks: list[dict], question: str) -> str
    # content = blocks（工具原来返回的图 + 文字块，位置说明就在文字块里）+ [{"type":"text","text": PROXY_REQUEST.format(question=question.strip() or "描述一下这张图里有什么")}]
    # 调 self.describe(content)；available() 为假或抛 ModelError → 抛 ModelUnavailable("现在看不了图")
# tools.py
ToolBox(..., proxy: Callable[[list[dict], str], str] | None = None, sees: Callable[[], bool] = lambda: True)
PROXY_PREFIX = "（眼睛代看）"
```

`ToolBox.run`：`_exec` 返回 `list`（有图）且 `not self.sees()` 且 `self.proxy` 不为空 → `out = PROXY_PREFIX + self.proxy(out, str(args.get("question") or ""))`；`ModelError` → 返回 `("现在看不了图：" + str(exc), True)`。`look` 且 `image=true` 且看不了图时：`eyes.latest` 在 `look_min_interval` 内 → 直接返回 `eyes.summary(now)`（不再起代看）。沙盒 `text_only` 的分支不动（本来就不返回图）。

- [ ] **Step 1: 写失败的测试**
  - `test_openai_tools_all_present_with_question`：`openai_tools(tb)` 的名字含 look / look_at / look_person / look_around / check_friend；look_person 的参数有 `question`、`required == ["name"]`；panel_read 有 `image` 和 `question`；say 没有 `question`。
  - `test_proxy_replaces_image`：假身体的 `look_person` 返回 `[image_block(…), {"type":"text","text":"小明在 (300,200)"}]`；`ToolBox(..., proxy=lambda b, q: f"{len(b)}|{q}", sees=lambda: False)`，`run("look_person", {"name":"小明","question":"衣服"})` → `("（眼睛代看）2|衣服", False)`。
  - `test_sees_keeps_image`：`sees=lambda: True` → 返回原来的 list。
  - `test_proxy_when_eyes_unavailable`（Review Focus 5）：proxy 抛 `ModelUnavailable("现在看不了图")` → `run` 返回 `("现在看不了图：现在看不了图", True)` 或包含「看不了图」、`is_error is True`，不抛。
  - `test_eyes_proxy_builds_request`：`Eyes(describe=记录器, …).proxy([blk], "")` → describe 收到的最后一块文字含「描述一下这张图里有什么」、第一块是 blk。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_llm_tools.py tests/test_brain_tools.py tests/test_brain_eyes.py -q` → FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 跑测试** → PASS

- [ ] **Step 5: 提交** `feat(brain): 大脑看不了图时看图工具请眼睛代看`

---

### Task 9: cli 全部改接 Registry

**Files:**
- Modify: `src/skydango/cli.py`：`_run_brain`、`_wardrobe`、`_inner_mind`、`_final_reflection` 附近、`cmd_memory`（update）、`cmd_chat`、`--no-brain` 的组装（约 2600 行）、`cmd_look`、`perception label --assist`（约 2058）、`--objects`（约 2166）、`attrs-label`（约 1258）、`gesture-label`（约 1337）、`addressee label`（约 699）；删 `_fallback_brain`、`_gated_backup`；`_claude_base` / `_brain_env` 改成只查 mcp + 建 Registry
- Modify: `src/skydango/brain/eyes.py`（删 `eyes_command`）、`src/skydango/vision/wardrobe.py`（删 `wardrobe_command`）、`src/skydango/vision/assist.py`（删 `assist_command`；`Reviewer` 的 `run` 现在收 `GatedCall.message` 包的函数，返回 dict 照旧有 `result` / `usage`）
- Modify: `src/skydango/brain/backstage.py` 的调用处（`section` 的三个模型名传 `registry.describe(...)` 的主模型 `str(registry.current(use))`）、`src/skydango/brain/tools.py` 的 introspect 眼睛那句（`ToolBox` 加 `eyes_label: str = ""`，显示 `眼睛（{eyes_label}）`）
- Test: `tests/test_cli_brain.py`（或现有测 `_run_brain` 的文件，`grep -l "_run_brain" tests`）、`tests/test_cli_*.py` 相关用例

**Interfaces:**
- Consumes: Tasks 2–8
- Produces:

```python
def _registry(cfg: Config, workdir: Path, environ: Mapping[str, str] = os.environ) -> Registry
    # resolve(cfg) → Registry(setup, ProviderGates(), workdir, environ)；registry.log_summary()
```

改接对照（spec §2.5）：
- 记忆：`registry.call("memory", timeout=cfg.brain.memory_timeout)`；反思：`registry.call("reflect", timeout=cfg.inner.reflect_timeout)`（`_final_reflection` 里 `reflector.llm.timeout = _final_timeout(cfg)` 不变，`GatedCall.timeout` 接住）；
- 眼睛：`Eyes(describe=lambda content: eyes_call.text(EYES_SYSTEM, content), available=eyes_call.available, …)`，`eyes_call = registry.call("eyes", timeout=cfg.brain.eyes_timeout)`；
- 装扮描述：`registry.call("wardrobe", timeout=a.describe_timeout)`，`describe=lambda c: call.text(WARDROBE_SYSTEM, c)`、`available=call.available`；日志改「描述装扮：{registry.describe('wardrobe')}，每小时最多 N 次」；
- 备用回复：`Responder(registry.call("reply", timeout=10.0, max_retries=0), cfg.reply)`；
- 大脑：`session = make_session(registry, registry.current("brain") …)`；`registry.current("brain") is None` → `RuntimeError("大脑没有能用的模型：" + 原因)`；备用 `fallback=partial(make_session, registry, setup.uses["brain"].backup, …)`（备为 None 时 `fallback=None`）；
  `ToolBox(..., proxy=eyes.proxy, sees=lambda: registry.sees(ModelRef(brain.session.provider, brain.session.model)) and isinstance(brain.session, BrainSession), eyes_label=registry.describe("eyes"))`（ToolLoopBrain 一律代看：OpenAI 的 tool 消息放不了图）；
- `--no-brain` / `cmd_chat`：`registry.call("reply")` 和 `registry.call("memory")`；`--echo`：在建 Registry 之前 `cfg.models["reply"] = {"main": "echo/echo"}`（替掉原来改 `cfg.llm.provider`）；
- `cmd_memory update`：`registry.call("memory", timeout=cfg.brain.memory_timeout)`；
- `cmd_look`：`registry.call("eyes", timeout=cfg.brain.eyes_timeout)`；
- 看图标注四个命令：`call = registry.call("image_label", timeout=a.timeout, cwd=assist.assist_workdir())`，`Reviewer(lambda content: call.message(system, content), …)`；打印「交给 {registry.describe('image_label')} 核对」；
- `addressee label`：`registry.call("text_label", timeout=a.timeout)`，`lambda content: call.message(ae.SYSTEM, content)`；
- **status 显示在用的模型**（spec §4）：`Body` 加 `models_line: Callable[[], str] | None`，`status` 在「场合」前面加一行「模型：大脑 deepseek/deepseek-chat」，切过备用写「模型：大脑已切到 claude/sonnet（deepseek 余额用完）」，有停用的用处再加「（眼睛停用：…）」；cli 里用 `brain.session` 和 `registry.gates.closed()` 拼；
- **Claude 令牌只在用到时查**：`_brain_env` 只查 mcp；大脑模式启动时 `problems = registry.requirements(["brain"])`，brain 的主和备都过不了才抛 `RuntimeError`；别的用处的问题 `log.warning` 一行、照常起（用的时候会撞 `ModelError(down="auth")` → 关闸 → 停用）。

- [ ] **Step 1: 写失败的测试**
  - `test_run_brain_all_deepseek_without_claude_token`：照现有 `_run_brain` 测试的假世界写法，环境里只有 `DEEPSEEK_API_KEY`（monkeypatch `claude_base` 抛 `ModelError(down="auth")`、`build_client` 返回假客户端）→ 能组装起来；大脑会话是 `ToolLoopBrain`；日志里有眼睛那一处的 WARNING。
  - `test_run_brain_no_model_raises`：deepseek 和 claude 都没 Key / 令牌 → `RuntimeError`，消息含「大脑没有能用的模型」。
  - `test_echo_flag_uses_echo_reply`：`cmd_chat` 带 `--echo` 时 Responder 的模型是 echo（照现有 `cmd_chat` 测试改）。
  - `test_assist_uses_image_label`：monkeypatch `Registry.call` 记录用处名 → `perception label --assist` 走的是 `"image_label"`、`addressee label` 走的是 `"text_label"`。
  - `test_status_has_models_line`：`Body.status()` 带 `models_line` 时含「模型：大脑 deepseek/deepseek-chat」；`models_line=None` 时 status 逐字照旧。
  - 原来测 `_fallback_brain` / `_gated_backup` / `force_fallback` / `ClaudeGate` 的用例删掉或改写成上面这些。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests -q -k "cli or brain"` → FAIL（新用例）

- [ ] **Step 3: 实现**

- [ ] **Step 4: 全量测试（后台）**：`python -m pytest -q` → 全部 PASS；然后 `grep -rnE "cfg\.llm\b|eyes_model|memory_model|reflect_model|describe_model|assist\.model|ClaudeGate|GatedLlm|gated_describe|DeepSeekBrain|_fallback_brain|_gated_backup|force_fallback" src` 只剩 `config.py`（旧字段定义）和 `models/config.py`（换算）

- [ ] **Step 5: 提交** `refactor(cli): 所有调模型的地方改从 Registry 拿`

---

### Task 10: 管理面板后端：`/api/models`、测试按钮、预检

**Files:**
- Create: `src/skydango/console/models_view.py`
- Modify: `src/skydango/console/probes.py`（`test_llm` / `test_claude` 换成 `test_provider`）、`src/skydango/console/server.py`（路由；删 `/api/settings/test` 里 llm / claude 两支）、`src/skydango/console/preflight.py`、`src/skydango/console/settings.py`（删字段，见下）
- Test: `tests/test_console_models.py`、`tests/test_console_preflight.py`、`tests/test_console_settings.py`、`tests/test_console_server.py`

**Interfaces:**
- Consumes: `resolve`、`ProviderConfig`、`USES`、`claude_base`、`build_client`、`SettingsStore`（`config_path`、`console_path`、`secrets_path`、`_secret_source`）
- Produces:

```python
class ModelsView:
    def __init__(self, store: SettingsStore) -> None
    def view(self) -> dict
    # {"providers": [{"id","kind","path","config_dir","base_url","key_env","token_env","models":[{"name","vision"}],
    #                 "secret": 打码（同设置页）,"secret_source","source","used_by":[用处名]}],
    #  "uses": [{"name","label","help","vision","main","backup","source","problem"}],   # main/backup 是 "p/m" 或 ""
    #  "kinds": ["claude-code","openai"], "templates": TEMPLATES, "error": None 或 console.toml 坏了的原因}
    def save(self, body: dict) -> tuple[int, dict]
    # body = {"providers": [同 view 的结构（不含打码字段）], "uses": {"brain": {"main","backup"}, …}, "secrets": {"<provider id>": "新值" | "" 表示清除}}（没给的 id 不动）
    # 校验失败 400 {"ok": False, "text": 原因}；成功 200 {"ok": True, "restart": 团子 / 沙盒在跑}
    def test(self, body: dict) -> dict          # {"provider": 同上一家, "secret": 页面上没保存的值或 ""} → probes.test_provider

TEMPLATES = [  # 「+ 添加供应商」的预填（只填地址和模型名）
    {"label": "DeepSeek", "id": "deepseek", "kind": "openai", "base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY", "models": [["deepseek-chat", False], ["deepseek-reasoner", False]]},
    {"label": "ChatGPT", "id": "openai", "kind": "openai", "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "models": [["gpt-4o", True], ["gpt-4o-mini", True]]},
    {"label": "通义千问", "id": "qwen", "kind": "openai", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key_env": "DASHSCOPE_API_KEY", "models": [["qwen-plus", False], ["qwen-vl-plus", True]]},
    {"label": "Ollama（本机）", "id": "ollama", "kind": "openai", "base_url": "http://127.0.0.1:11434/v1", "key_env": "OLLAMA_API_KEY", "models": []},
    {"label": "Claude Code", "id": "claude", "kind": "claude-code", "path": "claude", "token_env": "SKYDANGO_CLAUDE_TOKEN", "config_dir": ".brain-claude", "models": [["sonnet", True], ["haiku", True], ["opus", True]]},
]

# probes.py
def test_provider(provider: ProviderConfig, secret: str, *, run: Callable = subprocess.run, client_factory: Callable | None = None) -> dict
    # claude-code：同原 test_claude（claude -p --model <models[0]> "只回复 ok"）；openai：build_client(provider, api_key=secret or None) 发 "只回复 ok"（models[0]）；
    # 勾了能看图的再用第一个能看图的模型发一张 64×64 纯红小图问「这张图是什么颜色？只回答颜色」，回答里有「红」算过；
    # 返回 {"ok": bool, "text": "通过：deepseek-chat 回复 ok" / 失败原因}；Ollama 这类没 Key 的：secret 为空、环境变量也没有时用 "ollama" 占位 Key
```

保存规则：
- 写 `console.toml` 的 `[providers]` **整张表**：页面提交的每一家都写（包括原来是默认 / config.toml 来的，值取页面上的）——**Review Focus 2**：第一次保存时页面显示的就是生效值（含旧 `[llm]` 换算出来的地址 / 模型），所以写进去不会丢。
- `[models.<用处>]`：只写和「默认值 + config.toml」算出来不同的；和它一样的删掉这一节（让 config.toml / 默认生效）。
- 删除供应商：提交里少了某家 → 还有用处引用它（主或备）就 400「还有 brain、memory 在用 deepseek」；它在 config.toml 里有定义 → 400「deepseek 写在 config.toml 里，面板不能删」。
- 校验同 `resolve`：id、kind、vision ⊆ models、看图用处选了看不了图的、引用不存在的 → 400，文字同 `Problem.text`。
- Key：`secrets` 里给了的按那家 `secret_env()` 写 / 删 `secrets.toml` 的 `[env]`（复用 `SettingsStore` 写 secrets 的那段，抽成 `SettingsStore.write_secrets(changes: dict[str, str | None])`）。

设置页（`settings.py` 的 `FIELDS`）删：`llm.provider`、`llm.base_url`、`llm.model`、`secret.llm`、`secret.claude`、`brain.claude_path`、`brain.model`、`brain.eyes_model`、`brain.memory_model`；`group` 里删 `llm`；`proactive.auto_look_busy` 说明里「眼睛（Haiku）」→「眼睛（识图模型）」，`appearance.describe` 说明里「让 Haiku」→「让识图模型」。`SettingsStore.secret(which)` 改成 `secret_env(name: str) -> str`（按环境变量名读：secrets.toml → 环境变量 → Windows 用户环境变量），调用处一起改。

预检（`preflight.py`）：
- 大脑模式：`registry.requirements(["brain"])`（用 `store.secret_env` 当 environ 查）→ brain 的主和备**都**不行才拦：`problem("大脑没有能用的模型：deepseek 缺 Key；claude 缺令牌", "models.brain")`；`mcp` 检查照旧。
- 普通模式：`reply` 的主不行就拦：`problem("普通模式的回复模型用不了：…", "models.reply")`。
- 别的用处有问题不拦（团子能起，那一处停用）。

- [ ] **Step 1: 写失败的测试**
  - `test_view_lists_builtin_and_uses`：空配置 → providers 有 claude / deepseek，`used_by` 里 deepseek 含 brain；uses 8 行、顺序同 `USE_NAMES`；Key 打码不含原值。
  - `test_first_save_keeps_legacy_llm_values`（Review Focus 2）：config.toml 只有 `[llm] base_url = "https://x"`、`model = "m1"` → 把 `view()` 的 providers 原样提交 `save` → 重新 `resolve` 后 `providers["deepseek"].base_url == "https://x"`、`"m1" in models`、`str(uses["reply"].main) == "deepseek/m1"`。
  - `test_save_writes_only_changed_uses`：只把 brain 主改成 `claude/sonnet` 保存 → console.toml 有 `[models.brain]`、没有 `[models.memory]`；再改回默认保存 → `[models.brain]` 没了。
  - `test_delete_in_use_rejected` / `test_delete_config_defined_rejected`：见上，400、文字对。
  - `test_save_rejects_blind_vision_use`：eyes 主选 `deepseek/deepseek-chat` → 400，含「看不了图」。
  - `test_secret_write_and_clear`：`secrets={"deepseek": "sk-1"}` → secrets.toml `[env] DEEPSEEK_API_KEY = "sk-1"`；`{"deepseek": ""}` → 删掉；`view()` 里不出现 `sk-1`。
  - `test_config_toml_untouched`：保存前后 config.toml 的 sha256 一样。
  - `test_test_provider_openai_and_vision`：`client_factory` 给假客户端（第一句回 "ok"、看图那句回 "红色"）→ `ok is True`；看图回 "蓝" → `ok is False`。
  - `test_test_provider_claude`：假 `run` 照原 `test_claude` 测试 → 通过 / 令牌缺失。
  - preflight：`test_brain_mode_blocks_only_when_brain_has_no_model`（只有 DEEPSEEK Key、没 Claude 令牌 → 不拦；都没有 → 一条，`setting == "models.brain"`）；`test_plain_mode_needs_reply`。
  - server：`GET /api/models` 200；`POST /api/models` 不带 `X-Skydango` 头被拒（同现有 `post_guard` 测试写法）；`POST /api/models/test` 走 `ModelsView.test`。
  - settings：`FIELDS` 里不再有上面删掉的键；`GROUPS` 不含 llm 的测试按现有写法改。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_console_models.py tests/test_console_preflight.py tests/test_console_settings.py tests/test_console_server.py -q` → FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 跑测试** → PASS

- [ ] **Step 5: 提交** `feat(console): 模型页后端：供应商增删改、用处选择、测试按钮、按用处预检`

---

### Task 11: 管理面板前端「模型」页

**Files:**
- Create: `src/skydango/console/static/models.js`
- Modify: `src/skydango/console/static/console.html`（左栏「设置」上面加 `<a href="#models" data-page="models">模型…</a>`；加 `<section id="page-models" class="page" hidden>`；`<script src="console/static/models.js">` 放在 `settings.js` 前面）
- Modify: `src/skydango/console/static/console.css`（模型页需要的样式，颜色只用 `var(--…)`）
- Modify: `src/skydango/console/static/settings.js`（`GROUPS` 删 llm；`PROBES` 删掉；在「统管大脑」组顶上放一行「模型在『模型』页设置 →」，点了 `go("models")`）
- Modify: `src/skydango/console/static/common.js`（`problemList` 的「去设置 →」：`setting` 以 `models.` 开头时 `go("models", 用处名)`）
- Test: `tests/test_console_page.py`（`JS` 列表加 `"models.js"`；加断言见下）

**Interfaces:**
- Consumes: Task 10 的 `GET/POST api/models`、`POST api/models/test`（相对路径）；`common.js` 的 `$` `el` `post` `getJSON` `ask` `toast` `Pages` `go`
- Produces: `Pages.models = {init, show(arg), hide}`；`show(use)` 滚到并高亮那一行（同设置页的樱花底闪一下，复用设置页那个 class）

页面（spec §3）：上半「供应商」卡片列表 + 「+ 添加供应商」（先选模板或「空白：Claude Code / OpenAI 兼容」）；下半「用处」表格（说明、主下拉、备下拉含「无」、看图标记、来源、恢复默认、问题标红）；底部一个「保存」。
未保存的修改放 `models.js` 自己的状态里（照 `settings.js` 的 `ST.dirty` 做法），切页再回来不丢；有未保存修改时离开页面不拦（同设置页）。
删除用 `ask()` 确认；结果用 `toast()`；Key 输入框 `type="password"`、`autocomplete="off"`、placeholder 是打码值。
下拉里的模型 = 页面上（含未保存）所有供应商的模型，看图的用处只列勾了能看图的；当前值不在列表里（比如供应商被删了）显示成「（已失效）p/m」并标红。
沿用 console.css 现有的卡片、表格、toggle、按钮样式，不另起视觉风格。

- [ ] **Step 1: 写失败的测试**（`tests/test_console_page.py`）
  - `JS` 列表加 `"models.js"` 后，现有的 `test_offline_relative_and_no_native_dialogs`、`test_colors_only_in_root` 自动覆盖它。
  - `test_models_page_wired`：`console.html` 里有 `href="#models"`、`id="page-models"`，`models.js` 的 script 在 `common.js` 之后、`settings.js` 之前；左栏里 `#models` 在 `#settings` 前面。
  - `test_models_js_uses_api`：`models.js` 文本里有 `api/models`、`api/models/test`、`Pages.models`，没有 `fetch("/`（绝对路径）。
  - 如果 `tests/` 里有用 node 跑 `common.js` 的测试（`grep -l "node" tests/test_console_*.py`），加一个 node 用例：`problemList([{text:"x", setting:"models.brain"}])` 的链接点了调 `go("models","brain")`。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_console_page.py -q` → FAIL

- [ ] **Step 3: 实现**

- [ ] **Step 4: 跑测试** → PASS

- [ ] **Step 5: 浏览器里看一眼**：`.claude/launch.json` 里加（没有就建）一项 `{"name": "console-models", "runtimeExecutable": "python", "runtimeArgs": ["-m", "skydango", "console", "--no-browser", "--port", "19395"], "port": 19395}`，并在 env 里给 `PYTHONPATH=src`（worktree 跑自己的代码）；`preview_start` 打开 `#models`：
  两家卡片都在、用处 8 行、改大脑主模型后「保存」成功（toast）、`console.toml` 里多了 `[models.brain]`；把它恢复默认再保存；`read_console_messages` 没有报错；截一张图给用户。测完 `preview_stop`，把 worktree 的 `console.toml` 删掉（别把测试值带走）。

- [ ] **Step 6: 提交** `feat(console): 「模型」页：维护供应商、给每个用处选主 / 备模型`

---

### Task 12: 文档、配置示例、收尾

**Files:**
- Modify: `CLAUDE.md`：「统管大脑」（大脑默认 DeepSeek、Claude 是备用；`force_fallback` / `fallback` 删了；代看；闸按供应商）、「管理面板」（七页 → 八页，加「模型」）、代码结构表（加 `src/skydango/models/` 一行，`brain/deepseek.py` 改成 `brain/toolloop.py`、加 `brain/sessions.py`、`console/models_view.py`）、「环境」里 `SKYDANGO_CLAUDE_TOKEN` 那句改成「用到 Claude 的用处才要」、「记忆」一节里「随手记和整理走 Claude Code」改成「走 `[models.memory]`（默认 DeepSeek）」、「常用命令」里 `run` 那行的「先 claude setup-token」改成按需
- Modify: `config.example.toml`：加 `[providers.*]` / `[models.*]` 示例（就用 spec §1.1 / §1.2 的写法），`[llm]` / `brain.model` 等旧项注释成「旧写法，还能读，换算见 docs/superpowers/specs/2026-10-05-model-providers-design.md §1.3」
- Modify: `docs/game-ops.md`：只有在里面写了「大脑是 sonnet」这类事实时才改（`grep -n "sonnet\|Haiku\|haiku\|DeepSeek" docs/game-ops.md`）

- [ ] **Step 1**：改文档
- [ ] **Step 2**：全量测试（后台）：`python -m pytest -q` → 全部 PASS（和基线 3632 passed 比，数量只多不少，少了的要说清是删了哪些过时用例）
- [ ] **Step 3**：`PYTHONPATH=src python -m skydango -c D:/Lanscarlos/Develop/SkyTricolorDango/config.toml memory show` 能跑（只读命令，验证配置换算不报错）；启动日志里有「模型：brain = deepseek/deepseek-chat（备 claude/sonnet）」这类行——用 `PYTHONPATH=src python -c "from skydango.config import load_all; from skydango.models.config import resolve; …"` 打印本机真实配置的 `resolve` 结果，贴给用户看
- [ ] **Step 4**：提交 `docs: 模型供应商和按用处选模型`
- [ ] **Step 5**：按 CLAUDE.md 合并：`git fetch && git rebase origin/main`，全量测试再过一遍（后台），`git push origin HEAD:main`；告诉用户 spec §6 的四步真机 / 沙盒验证要他晚上做，以及本机 `config.toml` 不用改、大脑下次启动就是 DeepSeek
