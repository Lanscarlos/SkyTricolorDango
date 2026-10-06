"""[providers.*] / [models.*] 解析成 ModelSetup：内置默认、旧字段换算、校验（问题清单给预检 / 面板）。

见 docs/superpowers/specs/2026-10-05-model-providers-design.md §1。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from urllib.parse import urlparse

from ..config import Config

KINDS = ("claude-code", "openai")  # 面板可选；另有内部 "echo"（--echo 用）
ID_RE = re.compile(r"^[a-z0-9_-]+$")
_RANK = {"default": 0, "legacy": 1, "config": 2, "console": 3}


@dataclass(frozen=True)
class Price:
    """元 / 百万 token 的空闲价：命中缓存的输入、没命中的输入、输出。est：价格页上没有、照别的模型估的。"""

    hit: float
    miss: float
    out: float
    est: bool = False


# DeepSeek 价格页（https://api-docs.deepseek.com/zh-cn/quick_start/pricing，2026-10-06 取）的空闲价；高峰 ×2。
# deepseek-chat / deepseek-reasoner 页上已经没有，按 Flash 的价估（spec 2026-10-06-model-usage §3.1）
DEEPSEEK_PRICES: dict[str, Price] = {
    "deepseek-flash": Price(0.02, 1, 4),
    "deepseek-v4-pro": Price(0.15, 4.5, 13.5),
    "deepseek-chat": Price(0.02, 1, 4, est=True),
    "deepseek-reasoner": Price(0.02, 1, 4, est=True),
}
DEEPSEEK_PEAK = 2.0
DEEPSEEK_PEAK_HOURS = ("09:00-12:00", "14:00-18:00")  # 北京时间周一到周五
_HOURS_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d-(?:(?:[01]\d|2[0-3]):[0-5]\d|24:00)$")


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    kind: str
    models: tuple[str, ...] = ()
    vision: tuple[str, ...] = ()
    # claude-code
    path: str = "claude"
    token_env: str = "SKYDANGO_CLAUDE_TOKEN"
    config_dir: str = ".brain-claude"
    # openai
    base_url: str = ""
    key_env: str = ""
    timeout: float = 30.0
    max_retries: int = 2
    # 单价（spec 2026-10-06-model-usage §3）：只对 openai 有意义，claude-code 走订阅不算钱
    prices: tuple[tuple[str, Price], ...] = ()
    peak: float = 1.0  # 高峰价 = 空闲价 × peak
    peak_hours: tuple[str, ...] = ()  # 北京时间周一到周五，"HH:MM-HH:MM"
    source: str = "default"  # default / config / console（这家任一字段最"新"的来源）

    def sees(self, model: str) -> bool:
        return model in self.vision

    def price(self, model: str) -> Price | None:
        return dict(self.prices).get(model)

    def secret_env(self) -> str:
        """这家的密钥放在哪个环境变量：claude-code 是令牌，openai 是 Key，echo 没有。"""
        if self.kind == "claude-code":
            return self.token_env
        if self.kind == "openai":
            return self.key_env
        return ""


_PROVIDER_KEYS = {f.name for f in fields(ProviderConfig)} - {"id", "source"}


def is_deepseek(p: ProviderConfig) -> bool:
    """DeepSeek 官方接口（查得到余额、内置单价）：OpenAI 兼容且地址主机是 api.deepseek.com。"""
    return p.kind == "openai" and (urlparse(p.base_url or "").hostname or "") == "api.deepseek.com"


def mark_est(model: str, hit: float, miss: float, out: float) -> Price:
    """和内置估价一模一样的数（面板保存时原样写回去的）仍然算估。"""
    known = DEEPSEEK_PRICES.get(model)
    est = known is not None and known.est and (known.hit, known.miss, known.out) == (hit, miss, out)
    return Price(hit, miss, out, est)


def parse_price(model: str, value) -> Price:
    """[命中, 没命中, 输出] → Price；不对抛 ValueError。"""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{model} 的单价要写成 [命中缓存的输入, 没命中的输入, 输出] 三个数")
    nums = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"{model} 的单价要是不小于 0 的数")
        nums.append(float(v))
    return mark_est(model, *nums)


def parse_pricing(pid: str, values: dict, problems: list[Problem]) -> None:
    """把 values 里的 prices / peak / peak_hours 换成 ProviderConfig 要的样子；坏的那项丢掉、记一条问题。"""
    if "prices" in values:
        raw, out = values.pop("prices"), []
        if not isinstance(raw, dict):
            problems.append(Problem(f"供应商 {pid} 的 prices 要写成 {{模型 = [命中, 没命中, 输出]}}", provider=pid))
        else:
            for model, v in raw.items():
                try:
                    out.append((str(model), parse_price(str(model), v)))
                except ValueError as exc:
                    problems.append(Problem(f"供应商 {pid}：{exc}", provider=pid))
        values["prices"] = tuple(out)
    if "peak" in values:
        v = values.pop("peak")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 1:
            problems.append(Problem(f"供应商 {pid} 的 peak（高峰倍数）要是不小于 1 的数", provider=pid))
        else:
            values["peak"] = float(v)
    if "peak_hours" in values:
        v = values.pop("peak_hours")
        hours = [str(h) for h in v] if isinstance(v, (list, tuple)) else None
        if hours is None or not all(_HOURS_RE.match(h) for h in hours):
            problems.append(Problem(f"供应商 {pid} 的 peak_hours 要写成 [\"09:00-12:00\", …]", provider=pid))
        else:
            values["peak_hours"] = tuple(hours)


def deepseek_pricing() -> dict:
    return {"prices": tuple(DEEPSEEK_PRICES.items()), "peak": DEEPSEEK_PEAK, "peak_hours": DEEPSEEK_PEAK_HOURS}


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @staticmethod
    def parse(text: str) -> "ModelRef":
        """"供应商id/模型名"，按第一个 "/" 拆（模型名里可以再有 "/"，比如 Ollama）。"""
        provider, sep, model = str(text or "").strip().partition("/")
        if not sep or not provider or not model:
            raise ValueError(f"模型要写成「供应商/模型名」：{text!r}")
        return ModelRef(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class Use:
    """一个用到模型的地方（固定定义）。"""

    name: str
    label: str
    help: str
    vision: bool
    main: str
    backup: str
    temperature: float
    max_tokens: int
    follows: str = ""  # 非空 = 跟着这个用处的模型走、不单独选（只用来单独记用量）：不读 [models.<它>]，面板 / status / 预检不单独报


USES: tuple[Use, ...] = (
    Use("brain", "大脑", "每次醒来想、说、调工具", False, "deepseek/deepseek-flash", "claude/sonnet", 1.0, 4096),
    Use("memory", "记忆", "随手记 inbox.md、整理 notes.md、memory update", False, "deepseek/deepseek-flash", "claude/sonnet", 0.8, 4096),
    Use("reflect", "反思", "反思、日记、性格", False, "deepseek/deepseek-flash", "claude/sonnet", 0.8, 4096),
    Use("reply", "回复", "普通 Agent（--no-brain）、chat 命令、大脑离线时的纯文字回复", False, "deepseek/deepseek-flash", "", 0.8, 200),
    Use("text_label", "文字标注", "文字标注（addressee label）", False, "deepseek/deepseek-flash", "", 0.8, 4096),
    Use("eyes", "眼睛", "眼睛（截图写成文字）、给大脑代看", True, "deepseek/deepseek-flash", "claude/haiku", 0.8, 2048),
    Use("wardrobe", "装扮描述", "装扮描述", True, "deepseek/deepseek-flash", "claude/haiku", 0.8, 1024),
    Use("image_label", "看图标注", "perception label --assist / --objects、attrs-label、gesture-label", True,
        "deepseek/deepseek-flash", "claude/sonnet", 0.8, 8192),
    # spec 2026-10-06-brain-compact §5：OpenAI 兼容大脑的历史压缩，用的就是大脑那个会话的模型
    Use("recap", "压缩", "大脑历史太长时写成前情提要（跟着大脑的模型，不单独选）", False, "", "", 0.3, 4096, follows="brain"),
)
USE_NAMES: tuple[str, ...] = tuple(u.name for u in USES)
USE_BY_NAME = {u.name: u for u in USES}

# 旧字段 → 用处（值是 claude 的模型名）
LEGACY_CLAUDE = {
    "brain.model": "brain",
    "brain.eyes_model": "eyes",
    "brain.memory_model": "memory",
    "inner.reflect_model": "reflect",
    "appearance.describe_model": "wardrobe",
    "assist.model": "image_label",
}


@dataclass
class UseConfig:
    name: str
    main: ModelRef | None
    backup: ModelRef | None
    temperature: float
    max_tokens: int
    source: str  # default / config / console / legacy（旧字段换算来的）
    disabled: str = ""  # 非空 = 主和备都用不了的原因（这一处停用）


@dataclass(frozen=True)
class Problem:
    text: str
    use: str | None = None
    provider: str | None = None
    warn: bool = False  # 只是提醒（模型不在列表里、照样用）：面板保存不拦


@dataclass
class ModelSetup:
    providers: dict[str, ProviderConfig]  # 含内部 "echo"
    uses: dict[str, UseConfig]
    problems: list[Problem] = field(default_factory=list)
    legacy: list[str] = field(default_factory=list)  # 换算过的旧字段，启动时警告一行一个


ECHO = ProviderConfig("echo", "echo", models=("echo",))


def _get(cfg: Config, dotted: str):
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _builtin(cfg: Config) -> dict[str, ProviderConfig]:
    llm, brain = cfg.llm, cfg.brain
    if llm.provider != "openai":  # 旧的 anthropic / echo：地址、Key、型号都不是 DeepSeek 的，别灌进来（Key 会被发到别家）
        llm = type(llm)()
    models = ["deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"]  # deepseek-flash（V4.1 Flash）能看图；后两个是旧名
    if llm.model and llm.model not in models:
        models.insert(0, llm.model)
    return {
        "claude": ProviderConfig(
            "claude", "claude-code", models=("sonnet", "haiku", "opus"), vision=("sonnet", "haiku", "opus"),
            path=brain.claude_path, token_env=brain.token_env, config_dir=brain.config_dir),
        "deepseek": ProviderConfig(
            "deepseek", "openai", models=tuple(models), vision=("deepseek-flash",),
            base_url=llm.base_url, key_env=llm.api_key_env, timeout=float(llm.timeout), max_retries=int(llm.max_retries),
            **deepseek_pricing()),
    }


def _explicit(cfg: Config, problems: list[Problem]) -> dict[str, ProviderConfig]:
    out: dict[str, ProviderConfig] = {}
    for pid, table in cfg.providers.items():
        if not ID_RE.match(pid):
            problems.append(Problem(f"供应商 id「{pid}」只能用小写字母、数字、- 和 _", provider=pid))
            continue
        if pid == "echo":
            problems.append(Problem("供应商 id「echo」是内部保留的，换个名字", provider=pid))
            continue
        kind = table.get("kind", "")
        if kind not in KINDS:
            problems.append(Problem(f"供应商 {pid} 的接入方式「{kind}」不支持（只有 {' / '.join(KINDS)}）", provider=pid))
            continue
        values: dict = {}
        for key, value in table.items():
            if key == "kind":
                continue
            if key not in _PROVIDER_KEYS:
                problems.append(Problem(f"供应商 {pid} 有不认识的字段「{key}」", provider=pid))
                continue
            values[key] = value
        models = tuple(str(m) for m in values.pop("models", ()) or ())
        vision = tuple(str(m) for m in values.pop("vision", ()) or ())
        extra = [m for m in vision if m not in models]
        if extra:
            problems.append(Problem(f"供应商 {pid} 的 vision 里有 models 没列的模型：{'、'.join(extra)}", provider=pid))
            vision = tuple(m for m in vision if m in models)
        source = max((cfg.sources.get(f"providers.{pid}.{k}", "default") for k in table), key=_RANK.get, default="default")
        priced = any(k in values for k in ("prices", "peak", "peak_hours"))
        parse_pricing(pid, values, problems)
        if not priced and kind == "openai":  # 面板保存过的 deepseek（整张写进 console.toml、没单价）照样按内置单价算
            probe = ProviderConfig(pid, kind, base_url=str(values.get("base_url") or ""))
            if is_deepseek(probe):
                values.update(deepseek_pricing())
        try:
            out[pid] = ProviderConfig(pid, kind, models=models, vision=vision, source=source, **values)
        except TypeError as exc:
            problems.append(Problem(f"供应商 {pid} 写得不对：{exc}", provider=pid))
    return out


def _legacy_uses(cfg: Config, legacy: list[str]) -> dict[str, tuple[str, str]]:
    """旧字段换算出来的主模型：用处 → (主, 来源说明)。"""
    out: dict[str, tuple[str, str]] = {}
    for dotted, use in LEGACY_CLAUDE.items():
        if dotted in cfg.sources:
            section, key = dotted.split(".")
            legacy.append(f"[{section}] {key} 已改到 [models.{use}] main")
            out[use] = f"claude/{_get(cfg, dotted)}"
    llm_keys = sorted(k for k in cfg.sources if k.startswith("llm."))
    if llm_keys:
        provider = cfg.llm.provider
        if "llm.provider" in cfg.sources and provider == "echo":
            out["reply"] = "echo/echo"
        elif "llm.provider" in cfg.sources and provider not in ("openai", "echo"):
            legacy.append(f"[llm] provider = \"{provider}\" 不再支持，reply 用默认")
        elif "llm.model" in cfg.sources:
            out["reply"] = f"deepseek/{cfg.llm.model}"
        legacy.append("[llm] 已改到 [providers.deepseek] / [models.reply]")
    for key in ("claude_path", "token_env", "config_dir"):
        if f"brain.{key}" in cfg.sources:
            new = "path" if key == "claude_path" else key
            legacy.append(f"[brain] {key} 已改到 [providers.claude] {new}")
    return out


def resolve(cfg: Config) -> ModelSetup:
    problems: list[Problem] = []
    legacy: list[str] = []
    providers = _explicit(cfg, problems) if cfg.providers else _builtin(cfg)
    providers["echo"] = ECHO
    old = _legacy_uses(cfg, legacy)

    for name in cfg.models:
        if name not in USE_BY_NAME:
            problems.append(Problem(f"[models.{name}] 不是认识的用处（有 {'、'.join(USE_NAMES)}）", use=name))

    uses: dict[str, UseConfig] = {}
    for use in USES:
        if use.follows:  # 跟着别的用处（在它后面算）：[models.<它>] 写了只警告
            if cfg.models.get(use.name):
                problems.append(Problem(f"[models.{use.name}] 不生效：{use.label}跟着{USE_BY_NAME[use.follows].label}的模型",
                                        use=use.name, warn=True))
            uses[use.name] = replace(uses[use.follows], name=use.name, source="follows")
            continue
        main, backup, source = use.main, use.backup, "default"
        temperature, max_tokens = use.temperature, use.max_tokens
        if use.name in old:
            main, source = old[use.name], "legacy"
        if use.name == "reply":
            if "llm.temperature" in cfg.sources:
                temperature = float(cfg.llm.temperature)
            if "llm.max_tokens" in cfg.sources:
                max_tokens = int(cfg.llm.max_tokens)
        table = cfg.models.get(use.name, {})
        for key, value in table.items():
            if key == "main":
                main = value
            elif key == "backup":
                backup = value
            elif key == "temperature":
                temperature = float(value)
            elif key == "max_tokens":
                max_tokens = int(value)
            else:
                problems.append(Problem(f"[models.{use.name}] 有不认识的字段「{key}」", use=use.name))
        if table:
            source = max((cfg.sources.get(f"models.{use.name}.{k}", "default") for k in table),
                         key=_RANK.get, default=source)
            if source == "default":
                source = "legacy" if use.name in old else "default"
        reasons: list[str] = []
        refs = [_check(use, "主", main, providers, problems, reasons), _check(use, "备", backup, providers, problems, reasons)]
        disabled = "" if any(refs) else "；".join(reasons) or "没有选模型"
        uses[use.name] = UseConfig(use.name, refs[0], refs[1], temperature, max_tokens, source, disabled)
    return ModelSetup(providers, uses, problems, legacy)


def _check(use: Use, which: str, text: str, providers: dict[str, ProviderConfig],
           problems: list[Problem], reasons: list[str]) -> ModelRef | None:
    """校验一个引用：作废返回 None（原因记进 problems / reasons），模型不在列表里只警告。"""
    if not text:
        return None
    where = f"{use.label}（{use.name}）的{which}模型"
    try:
        ref = ModelRef.parse(text)
    except ValueError as exc:
        problems.append(Problem(f"{where}：{exc}", use=use.name))
        reasons.append(str(exc))
        return None
    provider = providers.get(ref.provider)
    if provider is None:
        problems.append(Problem(f"{where} {ref}：没有供应商「{ref.provider}」", use=use.name, provider=ref.provider))
        reasons.append(f"没有供应商 {ref.provider}")
        return None
    if use.vision and provider.kind != "echo" and not provider.sees(ref.model):
        problems.append(Problem(f"{where} {ref} 看不了图（{use.label}要看图，只能选勾了「能看图」的）", use=use.name, provider=ref.provider))
        reasons.append(f"这是看图的用处，{ref} 看不了图")
        return None
    if provider.kind != "echo" and ref.model not in provider.models:
        problems.append(Problem(f"{where} {ref}：{ref.provider} 的模型列表里没有 {ref.model}（照样用）", use=use.name,
                                provider=ref.provider, warn=True))
    return ref
