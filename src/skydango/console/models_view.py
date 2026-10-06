"""管理面板「模型」页的后端（spec 2026-10-05-model-providers §3）：供应商增删改、每个用处选主 / 备、测试按钮。

写 console.toml 的 [providers] / [models]（config.toml 只读），Key / 令牌按环境变量名写 secrets.toml；浏览器只拿到打码的密钥。
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from ..config import load_config
from ..models.config import ID_RE, KINDS, USES, ModelSetup, ProviderConfig, resolve
from .settings import SettingsStore, mask
from .tomlfile import dumps, read, write_atomic

# 「+ 添加供应商」的预填（只填地址和模型名，Key 自己填）
TEMPLATES = [
    {"label": "DeepSeek", "id": "deepseek", "kind": "openai", "base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY",
     "models": [["deepseek-chat", False], ["deepseek-reasoner", False], ["deepseek-flash", True]]},
    {"label": "ChatGPT", "id": "openai", "kind": "openai", "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY",
     "models": [["gpt-4o", True], ["gpt-4o-mini", True]]},
    {"label": "通义千问", "id": "qwen", "kind": "openai", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "key_env": "DASHSCOPE_API_KEY", "models": [["qwen-plus", False], ["qwen-vl-plus", True]]},
    {"label": "Ollama（本机）", "id": "ollama", "kind": "openai", "base_url": "http://127.0.0.1:11434/v1", "key_env": "OLLAMA_API_KEY",
     "models": []},
    {"label": "Claude Code", "id": "claude", "kind": "claude-code", "path": "claude", "token_env": "SKYDANGO_CLAUDE_TOKEN",
     "config_dir": ".brain-claude", "models": [["sonnet", True], ["haiku", True], ["opus", True]]},
]

# console.toml 里旧设置页写过的模型字段：保存「模型」页时迁走（它们已经由这次写的 [providers] / [models] 表达了，
# 留着的话会被换算成 legacy、盖过页面上「恢复默认」的选择）
LEGACY_KEYS = ("brain.model", "brain.eyes_model", "brain.memory_model", "brain.claude_path", "brain.token_env", "brain.config_dir",
               "inner.reflect_model", "appearance.describe_model", "assist.model")

FIELDS = {  # 每种接入方式写进 console.toml 的字段（kind / models / vision 之外）
    "claude-code": ("path", "token_env", "config_dir"),
    "openai": ("base_url", "key_env", "timeout", "max_retries"),
}


def _ref(ref) -> str:
    return str(ref) if ref is not None else ""


class BadRequest(ValueError):
    """提交的内容不对：400，原因给页面。"""


def provider_from(row: dict) -> ProviderConfig:
    """页面上的一行供应商 → ProviderConfig；不对抛 BadRequest。"""
    if not isinstance(row, dict):
        raise BadRequest("供应商要是对象")
    pid = str(row.get("id") or "")
    if not ID_RE.match(pid) or pid == "echo":
        raise BadRequest(f"供应商 id「{pid}」只能用小写字母、数字、- 和 _（echo 是内部保留的）")
    kind = row.get("kind")
    if kind not in KINDS:
        raise BadRequest(f"供应商 {pid} 的接入方式只能是 {' / '.join(KINDS)}")
    models, vision = [], []
    for m in row.get("models") or []:
        name = str((m or {}).get("name") or "").strip() if isinstance(m, dict) else ""
        if not name:
            raise BadRequest(f"供应商 {pid} 有一行模型名是空的")
        if name in models:
            raise BadRequest(f"供应商 {pid} 的模型 {name} 写了两遍")
        models.append(name)
        if m.get("vision"):
            vision.append(name)
    values: dict = {}
    for key in FIELDS[kind]:
        if key in row and row[key] is not None and row[key] != "":
            values[key] = row[key]
    try:
        if "timeout" in values:
            values["timeout"] = float(values["timeout"])
        if "max_retries" in values:
            values["max_retries"] = int(values["max_retries"])
        for key in ("path", "token_env", "config_dir", "base_url", "key_env"):
            if key in values:
                values[key] = str(values[key]).strip()
        return ProviderConfig(pid, kind, models=tuple(models), vision=tuple(vision), **values)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"供应商 {pid} 写得不对：{exc}") from None


def provider_table(p: ProviderConfig) -> dict:
    """写进 console.toml 的 [providers.<id>]。"""
    table: dict = {"kind": p.kind, "models": list(p.models), "vision": list(p.vision)}
    for key in FIELDS[p.kind]:
        table[key] = getattr(p, key)
    return table


def _drop_legacy(console: dict) -> None:
    """迁走 console.toml 里的旧模型字段；[llm] 的 temperature / max_tokens 挪进 [models.reply]（页面不调这两项）。"""
    llm = console.pop("llm", None)
    if isinstance(llm, dict):
        keep = {k: llm[k] for k in ("temperature", "max_tokens") if k in llm}
        if keep:
            reply = console.setdefault("models", {}).setdefault("reply", {})
            for k, val in keep.items():
                reply.setdefault(k, val)
    for dotted in LEGACY_KEYS:
        section, key = dotted.split(".")
        table = console.get(section)
        if isinstance(table, dict):
            table.pop(key, None)
            if not table:
                console.pop(section)


class ModelsView:
    def __init__(self, store: SettingsStore, busy: Callable[[], bool] = lambda: False) -> None:
        self.store = store
        self.busy = busy

    def _setup(self) -> tuple[ModelSetup, str | None]:
        try:
            return resolve(self.store.effective()), None
        except ValueError as exc:  # console.toml 坏了：照样显示 config.toml 的
            return resolve(self.store._fallback()), str(exc)

    def _baseline(self) -> dict:
        """不算面板改的：默认值 + config.toml 算出来的每个用处（config.toml 坏了就只看默认值）。"""
        path = self.store.config_path
        try:
            return resolve(load_config(path if path.exists() else None)).uses
        except ValueError:
            return resolve(load_config(None)).uses

    # ---- 读 ----
    def view(self) -> dict:
        setup, error = self._setup()
        providers = []
        for p in setup.providers.values():
            if p.kind == "echo":
                continue
            env = p.secret_env()
            value, source = self.store._secret_source(env) if env else ("", "none")
            used_by = [u.name for u in setup.uses.values() if any(r is not None and r.provider == p.id for r in (u.main, u.backup))]
            row = {"id": p.id, "kind": p.kind, "models": [{"name": m, "vision": p.sees(m)} for m in p.models],
                   "secret": mask(value), "secret_source": source, "source": p.source, "used_by": used_by}
            for key in ("path", "token_env", "config_dir", "base_url", "key_env", "timeout", "max_retries"):
                row[key] = getattr(p, key)
            providers.append(row)
        base = self._baseline()
        uses = []
        for use in USES:
            u, b = setup.uses[use.name], base[use.name]
            issues = [p.text for p in setup.problems if p.use == use.name]
            uses.append({"name": use.name, "label": use.label, "help": use.help, "vision": use.vision,
                         "main": _ref(u.main), "backup": _ref(u.backup), "source": u.source,
                         "default_main": _ref(b.main), "default_backup": _ref(b.backup),  # 「恢复默认」= 默认值 + config.toml
                         "problem": u.disabled or (issues[0] if issues else "")})
        return {"providers": providers, "uses": uses, "kinds": list(KINDS), "templates": TEMPLATES, "error": error}

    # ---- 写 ----
    def save(self, body: dict) -> tuple[int, dict]:
        try:
            self._save(body)
        except BadRequest as exc:
            return 400, {"ok": False, "text": str(exc)}
        except ValueError as exc:  # console.toml / secrets.toml 坏了
            return 400, {"ok": False, "text": str(exc)}
        return 200, {"ok": True, "restart": bool(self.busy())}

    def _save(self, body: dict) -> None:
        rows, uses, secrets = body.get("providers"), body.get("uses") or {}, body.get("secrets") or {}
        if not isinstance(rows, list) or not isinstance(uses, dict) or not isinstance(secrets, dict):
            raise BadRequest("providers 要是列表、uses 和 secrets 要是对象")
        providers = [provider_from(r) for r in rows]
        ids = [p.id for p in providers]
        if len(set(ids)) != len(ids):
            raise BadRequest("有两家供应商 id 一样")
        store = self.store
        config_path = store.config_path if store.config_path.exists() else None
        console = read(store.console_path)
        before, _ = self._setup()
        old = {p.text for p in before.problems}

        # 删掉的供应商：config.toml 里定义的删不了；还有用处在用（按提交上来的选择算）也不行
        gone = [pid for pid, p in before.providers.items() if p.kind != "echo" and pid not in ids]
        in_config = (read(store.config_path) if config_path else {}).get("providers") or {}
        for pid in gone:
            if pid in in_config:
                raise BadRequest(f"{pid} 写在 config.toml 里，面板不能删")
        for pid in gone:
            users = [name for name, sel in uses.items() if isinstance(sel, dict)
                     and any(str(sel.get(k) or "").partition("/")[0] == pid for k in ("main", "backup"))]
            if users:
                raise BadRequest(f"还有 {'、'.join(users)} 在用 {pid}，先给它们换个模型")

        console["providers"] = {p.id: provider_table(p) for p in providers}
        _drop_legacy(console)
        # 用处：只写和「默认值 + config.toml」算出来不一样的；一样就删掉，让 config.toml / 默认生效
        base = self._baseline()
        models = console.setdefault("models", {})
        for name, sel in uses.items():
            if name not in base or not isinstance(sel, dict):
                raise BadRequest(f"没有用处「{name}」")
            main, backup = str(sel.get("main") or "").strip(), str(sel.get("backup") or "").strip()
            b = base[name]
            table = models.setdefault(name, {})
            if (main, backup) == (_ref(b.main), _ref(b.backup)):
                table.pop("main", None)
                table.pop("backup", None)
            else:
                table.update(main=main, backup=backup)
            if not table:
                models.pop(name)
        if not models:
            console.pop("models")

        text = dumps(console)
        with tempfile.TemporaryDirectory() as tmp:  # 先按新内容算一遍，新冒出来的问题（看不了图、供应商不存在……）就不存
            trial = Path(tmp) / "console.toml"
            trial.write_text(text, encoding="utf-8")
            after = resolve(load_config(config_path, trial))
        new = [p.text for p in after.problems if not p.warn and p.text not in old]
        if new:
            raise BadRequest("；".join(new))

        changes: dict[str, str | None] = {}
        by_id = {p.id: p for p in providers}
        for pid, value in secrets.items():
            p = by_id.get(pid)
            if p is None or not isinstance(value, str):
                raise BadRequest(f"没有供应商 {pid}，Key 存不了")
            if not p.secret_env():
                raise BadRequest(f"供应商 {pid} 没写 Key 的环境变量名")
            changes[p.secret_env()] = value.strip() or None  # 粘贴时常带空格、换行；空 = 清除
        if changes:
            store.write_secrets(changes)
        write_atomic(store.console_path, text)

    # ---- 测试按钮 ----
    def test(self, body: dict) -> tuple[int, dict]:
        from . import probes

        try:
            provider = provider_from(body.get("provider"))
        except BadRequest as exc:
            return 400, {"ok": False, "text": str(exc)}
        secret = body.get("secret") if isinstance(body.get("secret"), str) else ""
        return 200, probes.test_provider(provider, secret.strip() or self.store.secret_env(provider.secret_env()))
