"""面板的设置页：挑出来的常用配置项（FIELDS）、每项的当前值和来源、保存到 console.toml / secrets.toml。

config.toml 是用户手写的，这里只读不写；面板改的写 console.toml（叠加在 config.toml 之后），密钥按环境变量名写 secrets.toml。
浏览器永远拿不到完整密钥，只拿打码后的样子。
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config, console_paths, load_config, read_secrets
from .tomlfile import dumps, read, write_atomic


@dataclass(frozen=True)
class Field:
    key: str  # 点路径（device.serial）；secret.llm / secret.claude 是密钥；owner 同时写两处主人昵称
    label: str
    help: str
    kind: str  # bool / int / float / str / choice / exe / file / secret
    group: str  # launch / connect / llm / brain / features / identity
    choices: tuple[str, ...] = ()


FIELDS: tuple[Field, ...] = (
    Field("console.brain", "统管大脑", "关掉就是普通 Agent（DeepSeek 回复，调试用）", "bool", "launch"),
    Field("console.live", "真的发送", "打开后团子会在游戏里真的说话、做动作；关着只打印（dry-run）", "bool", "launch"),
    Field("console.emotes", "聊天时做动作", "关掉就只聊天、不做动作", "bool", "launch"),
    Field("console.duration", "运行时长（秒）", "0 = 一直跑，到点自己退出", "float", "launch"),
    Field("device.adb_path", "adb 路径", "MuMu 安装目录下的 shell\\adb.exe 最稳", "exe", "connect"),
    Field("device.serial", "设备名", "比如 emulator-5554；设备页能列出实际看到的设备", "str", "connect"),
    Field("device.switch_ime", "自动切输入法", "启动时切到 ADBKeyboard（团子打中文靠它）、停下时切回搜狗；设备页也能手动切", "bool", "connect"),
    Field("device.capture", "截图方式", "auto：有 MuMu 原生截图就用（约 9 ms），否则 adb screencap", "choice", "connect", ("auto", "mumu", "adb")),
    Field("llm.provider", "接口类型", "openai = OpenAI 兼容接口（DeepSeek 等）；echo 不调模型、原样回显", "choice", "llm",
          ("openai", "anthropic", "echo")),
    Field("llm.base_url", "接口地址", "比如 https://api.deepseek.com", "str", "llm"),
    Field("llm.model", "模型", "比如 deepseek-chat", "str", "llm"),
    Field("secret.llm", "API Key", "普通模式的聊天回复、记忆整理、大脑离线时的备用回复都用它", "secret", "llm"),
    Field("secret.claude", "Claude 令牌", "运行一次 claude setup-token 生成；大脑和眼睛用它（订阅）", "secret", "brain"),
    Field("brain.claude_path", "claude 路径", "命令行里 claude 能用就不用改", "exe", "brain"),
    Field("brain.model", "大脑模型", "默认 sonnet", "str", "brain"),
    Field("brain.eyes_model", "眼睛模型", "把画面写成文字的模型，默认 haiku", "str", "brain"),
    Field("brain.memory_model", "记忆整理模型", "随手记 inbox.md、整理 notes.md 用的模型（live 时），默认 sonnet", "str", "brain"),
    Field("proactive.enabled", "看场合主动开口", "关掉就回到只接话、不主动的老样子", "bool", "brain"),
    Field("proactive.quota_busy", "热闹时主动额度", "每个时间窗口（proactive.quota_window，默认 600 秒）里最多主动说几句（好友在身边、聊得热闹）", "int", "brain"),
    Field("proactive.quota_quiet", "安静时主动额度", "每个时间窗口（proactive.quota_window，默认 600 秒）里最多主动说几句（好友在身边、没怎么说话）", "int", "brain"),
    Field("proactive.min_gap", "主动开口间隔（秒）", "两句主动的话之间至少隔多久", "float", "brain"),
    Field("proactive.auto_look_busy", "好友在身边时多久看一次（秒）", "眼睛（Haiku）最久多久看一次画面；越短越费订阅额度", "float", "brain"),
    Field("reflex.enabled", "身体反射", "有人叫团子马上冒输入气泡、回礼、闲着做小动作；关掉就回到老样子", "bool", "brain"),
    Field("reflex.bubble", "替大脑冒输入气泡", "有人跟团子说话时马上打开输入框，头顶显示正在输入", "bool", "brain"),
    Field("inner.enabled", "内心账本", "给好友记关系卡（见过几次、上次什么时候）和每次上线的日子；关掉就回到老样子", "bool", "brain"),
    Field("inner.reflect", "反思", "定时想想刚才发生的事：心情、精力、闹别扭、惦记的事，下线写日记；关掉就只记账", "bool", "brain"),
    Field("inner.persona", "性格", "攒口头禅、老梗和看法，敢唱反调、会拒绝、熟人之间互损；关掉就回到第 2 期", "bool", "brain"),
    Field("call.enabled", "按 Q 喊一声", "大脑能按 Q 喊一声找附近的好友（稍远的好友头顶会亮出名字）；要配合 YOLO 感知层", "bool", "brain"),
    Field("call.auto", "自动喊一声", "好友刚“走开”、画面里还有没挂名字的人时，身体自己喊一声确认（1 分钟最多 3 次）", "bool", "brain"),
    Field("attention.search", "有意识地找", "闲着时找刚走开的好友、一个人待着时隔一会儿往没看过的方向看看；关掉只剩“有人说话 / 走近就转过去看”", "bool", "brain"),
    Field("lull.enabled", "冷场时的心理活动", "好友不说话了、聊着聊着走了时团子心里会犯嘀咕，按节点叫醒大脑；关掉就回到老样子", "bool", "brain"),
    Field("backstage.enabled", "幕后", "团子知道自己是 AI、卡洛做了她，能跟卡洛聊她自己怎么运作；好友要在 friends.md 里写「知道团子是 AI」才会跟他承认", "bool", "brain"),
    Field("vision.source", "读聊天的方式", "a11y：读游戏的无障碍节点（准、面板关着也能读好友头顶气泡），读不到自动退回 OCR；ocr：截图识别（原来的做法）", "choice", "features", ("a11y", "ocr")),
    Field("env.enabled", "识别环境", "后台 OCR 画面：身边有谁、在哪", "bool", "features"),
    Field("perception.enabled", "YOLO 感知层", "开发中；打开后替换定时整图 OCR，要先训练模型", "bool", "features"),
    Field("perception.model", "YOLO 模型", "模型文件路径", "file", "features"),
    Field("attrs.enabled", "人物复核（第二层）", "YOLO 人物框裁图再判一次：挡误框、捞低分框、认先祖 / 共享空间；要配合 YOLO 感知层和训练好的模型", "bool", "features"),
    Field("attrs.model", "外形头模型", "第二层的线性头文件路径（models/attrs-<日期>.npz）；重训后换模型会写这里", "file", "features"),
    Field("inbox.enabled", "难例收件箱", "live 运行时把可能认错的画面存进收件箱，外形头先筛、再人工核对、攒够了一键重训", "bool", "features"),
    Field("inbox.ask", "停团子时问要不要整理", "在管理面板停团子时，这次存了难例或还有没整理的运行就问一句「现在整理吗」", "bool", "features"),
    Field("inbox.retrain_min", "攒够几张提示重训", "收件箱里核对过的难例攒到这么多张才提示可以重训", "int", "features"),
    Field("places.enabled", "认地图", "要配合 YOLO 感知层和 places/ 图库", "bool", "features"),
    Field("appearance.enabled", "认装扮", "按外观接回没读到名字的好友、认回来的陌生人；要配合 YOLO 感知层", "bool", "features"),
    Field("appearance.describe", "描述装扮", "让 Haiku 把团子和身边人的装扮写成一句话（花额度）", "bool", "features"),
    Field("catalog.enabled", "收集装扮图鉴", "近处的人清楚的整身裁图存进 catalog/inbox/（只存图、不发输入）；要配合 YOLO 感知层", "bool", "features"),
    Field("friend_check.enabled", "好友树核对", "大脑的 check_friend：点人物打开好友树（未在真机核对）", "bool", "features"),
    Field("panels.enabled", "面板识别", "认出画面上开着的面板（只接大脑模式）", "bool", "features"),
    Field("reply.disclosure_prefix", "AI 前缀", "每句话前面加的前缀，比如【AI】；留空就不加", "str", "identity"),
    Field("owner", "主人昵称", "你在游戏里的昵称，精确匹配；主人发的 # 命令会被执行。留空 = 关闭", "str", "identity"),
)
KNOWN = {f.key: f for f in FIELDS}
OWNER_KEYS = ("reply.owner_name", "brain.owner_name")


def mask(value: str) -> str:
    if not value:
        return "没设置"
    if len(value) >= 12:
        return f"已设置（{value[:3]}…{value[-4:]}）"
    return "已设置"


def _get(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _has(data: dict, dotted: str) -> bool:
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return False
        data = data[part]
    return True


def _set(data: dict, dotted: str, value: Any) -> None:
    *parents, last = dotted.split(".")
    for part in parents:
        data = data.setdefault(part, {})
    data[last] = value


def _delete(data: dict, dotted: str) -> None:
    """删掉一个键，删完空了的表也一并删掉。"""
    *parents, last = dotted.split(".")
    chain = [data]
    for part in parents:
        if not isinstance(chain[-1].get(part), dict):
            return
        chain.append(chain[-1][part])
    chain[-1].pop(last, None)
    for i in range(len(parents) - 1, -1, -1):
        if not chain[i + 1]:
            chain[i].pop(parents[i], None)


def _check(f: Field, value: Any) -> tuple[str | None, Any]:
    """校验一个值：(错误, 规整后的值)。"""
    if f.kind == "bool":
        return (None, value) if isinstance(value, bool) else ("要是开或关", None)
    if f.kind == "int":
        return (None, value) if isinstance(value, int) and not isinstance(value, bool) else ("要是整数", None)
    if f.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "要是数字", None
        if f.key == "console.duration" and value < 0:
            return "不能小于 0", None
        return None, float(value)
    if f.kind == "choice":
        return (None, value) if value in f.choices else (f"只能是 {' / '.join(f.choices)}", None)
    return (None, value) if isinstance(value, str) else ("要是文字", None)


class SettingsStore:
    def __init__(self, config_path: Path, environ: Mapping[str, str] = os.environ) -> None:
        self.config_path = Path(config_path)
        self.console_path, self.secrets_path = console_paths(self.config_path)
        self.environ = environ

    # ---- 读 ----
    def effective(self) -> Config:
        """现在生效的配置（config.toml + console.toml）；文件坏了抛 ValueError。"""
        config = self.config_path if self.config_path.exists() else None
        overlay = self.console_path if self.console_path.exists() else None
        return load_config(config, overlay)

    def _fallback(self) -> Config:
        """console.toml 坏了也要能显示：退回只看 config.toml，再不行就默认值。"""
        try:
            return self.effective()
        except ValueError:
            try:
                return load_config(self.config_path if self.config_path.exists() else None)
            except ValueError:
                return Config()

    def _env_name(self, which: str, cfg: Config) -> str:
        return cfg.llm.api_key_env if which == "llm" else cfg.brain.token_env

    def _secrets(self) -> dict[str, str]:
        try:
            return read_secrets(self.secrets_path)
        except ValueError:
            return {}

    def _secret_source(self, name: str) -> tuple[str, str]:
        from ..chat.llm import _user_env

        stored = self._secrets().get(name, "")
        if stored:
            return stored, "secrets"
        found = self.environ.get(name, "") or _user_env(name)
        return (found, "env") if found else ("", "none")

    def secret(self, which: str) -> str:
        """llm / claude 现在能用的密钥：secrets.toml → 环境变量 → Windows 用户环境变量；没有返回空串。"""
        return self._secret_source(self._env_name(which, self._fallback()))[0]

    def _path_warning(self, f: Field, value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.config_path.parent / path
        if f.kind == "exe" and not path.is_file() and not shutil.which(value):
            return "找不到这个程序"
        if f.kind == "file" and not path.exists():
            return "文件不存在"
        return None

    def view(self) -> dict:
        error = None
        try:
            config_data = read(self.config_path)
        except ValueError as exc:
            config_data, error = {}, str(exc)
        try:
            console_data = read(self.console_path)
            self.effective()
        except ValueError as exc:
            console_data, error = {}, error or str(exc)
        try:
            read_secrets(self.secrets_path)
        except ValueError as exc:
            error = error or str(exc)
        cfg = self._fallback()
        fields = []
        for f in FIELDS:
            item = {"key": f.key, "label": f.label, "help": f.help, "kind": f.kind, "group": f.group,
                    "choices": list(f.choices), "warning": None}
            if f.kind == "secret":
                value, source = self._secret_source(self._env_name(f.key.split(".")[1], cfg))
                item.update(value=mask(value), source=source)
            else:
                dotted = OWNER_KEYS[0] if f.key == "owner" else f.key
                value = _get(cfg, dotted)
                source = "console" if _has(console_data, dotted) else "config" if _has(config_data, dotted) else "default"
                item.update(value=value, source=source, warning=self._path_warning(f, value))
                if f.key == "owner" and cfg.reply.owner_name != cfg.brain.owner_name:
                    item["warning"] = (f"config.toml 里两处主人昵称不一样：普通模式「{cfg.reply.owner_name}」、"
                                       f"大脑「{cfg.brain.owner_name}」，保存会统一")
            fields.append(item)
        return {"fields": fields, "error": error}

    # ---- 写 ----
    def save(self, values: Mapping[str, Any], revert: list[str] | tuple[str, ...] = ()) -> dict:
        """values：键 → 新值（密钥空串 = 不改）；revert：用回 config.toml 的值（密钥 = 清除）。有一个错就什么都不写。"""
        try:
            console_data = read(self.console_path)
            secrets_data = read(self.secrets_path)
            read_secrets(self.secrets_path)  # env 不是表、值不是字符串：先报出来，别写坏
            cfg = self.effective()
        except ValueError as exc:
            return {"ok": False, "errors": {"_file": str(exc)}, "warnings": {}}
        secrets_env = secrets_data.setdefault("env", {})
        errors: dict[str, str] = {}
        warnings: dict[str, str] = {}
        touched = set()
        for key, value in values.items():
            f = KNOWN.get(key)
            if f is None:
                errors[key] = "没有这一项"
                continue
            if f.kind == "secret":
                if not isinstance(value, str):
                    errors[key] = "要是文字"
                elif value := value.strip():  # 粘贴时常带空格、换行；只有空白 = 不改
                    secrets_env[self._env_name(key.split(".")[1], cfg)] = value
                    touched.add("secrets")
                continue
            error, value = _check(f, value)
            if error:
                errors[key] = error
                continue
            warning = self._path_warning(f, value)
            if warning:
                warnings[key] = warning
            for dotted in OWNER_KEYS if key == "owner" else (key,):
                _set(console_data, dotted, value)
            touched.add("console")
        for key in revert:
            f = KNOWN.get(key)
            if f is None:
                errors[key] = "没有这一项"
            elif f.kind == "secret":
                secrets_env.pop(self._env_name(key.split(".")[1], cfg), None)
                touched.add("secrets")
            else:
                for dotted in OWNER_KEYS if key == "owner" else (key,):
                    _delete(console_data, dotted)
                touched.add("console")
        if errors:
            return {"ok": False, "errors": errors, "warnings": warnings}
        if "console" in touched:
            write_atomic(self.console_path, dumps(console_data))
        if "secrets" in touched:
            if not secrets_env:
                secrets_data.pop("env")
            write_atomic(self.secrets_path, dumps(secrets_data))
        return {"ok": True, "errors": {}, "warnings": warnings}
