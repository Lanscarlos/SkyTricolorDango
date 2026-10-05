"""大模型客户端的公共部分：LlmClient 接口、读 Key、EchoClient。真正调模型从 models.registry 拿（按用处、主 → 备）。"""

from __future__ import annotations

import os
import re
import sys
from typing import Protocol


ChatMessage = dict[str, str]  # {"role": "user" | "assistant", "content": ...}


class LlmClient(Protocol):
    def complete(self, system: str, messages: list[ChatMessage], max_tokens: int | None = None) -> str: ...


def _user_env(name: str) -> str:
    """Windows 上 setx 设置的用户环境变量，已经开着的终端读不到，直接从注册表读。"""
    if sys.platform != "win32":
        return ""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, name)[0])
    except OSError:
        return ""


def read_key(env_name: str) -> str:
    key = (os.environ.get(env_name, "") or _user_env(env_name)) if env_name else ""
    if not key:
        raise RuntimeError(f"没有找到 API Key，请设置环境变量 {env_name}")
    return key


class EchoClient:
    """不调用模型，用来联调截屏 → 识别 → 发送整条链路。"""

    def complete(self, system: str, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        last = messages[-1]["content"] if messages else ""
        quoted = re.findall(r"「(.*)」", last)  # format_incoming 的格式："名字：「内容」" 或 "「内容」"
        return "收到：" + " / ".join(quoted or [last])

