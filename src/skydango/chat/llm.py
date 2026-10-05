"""大模型客户端。国内网络建议用 OpenAI 兼容接口（DeepSeek、通义千问、Kimi、本地 Ollama 等）。"""

from __future__ import annotations

import os
import re
import sys
from typing import Protocol

from ..config import LlmConfig

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


class OpenAICompatClient:
    """过渡用：LlmConfig 的 LlmClient 包一层 models.openai_compat.OpenAIBackend（Registry 接上后删）。"""

    def __init__(self, cfg: LlmConfig, api_key: str | None = None) -> None:
        from ..models.config import ProviderConfig
        from ..models.openai_compat import OpenAIBackend, build_client

        self.cfg = cfg
        provider = ProviderConfig("deepseek", "openai", models=(cfg.model,), base_url=cfg.base_url,
                                  key_env=cfg.api_key_env, timeout=cfg.timeout, max_retries=cfg.max_retries)
        self._client = build_client(provider, api_key=api_key)
        self.backend = OpenAIBackend(provider, cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                                     client=self._client)

    def with_timeout(self, timeout: float, max_retries: int | None = None) -> OpenAICompatClient:
        """换了超时（和重试次数）的副本，原来的不动（下线反思改走备用时用）。"""
        import copy

        opts: dict = {"timeout": timeout}
        if max_retries is not None:
            opts["max_retries"] = max_retries
        out = copy.copy(self)
        out._client = self._client.with_options(**opts)
        out.backend = copy.copy(self.backend)
        out.backend._client = out._client
        out.backend.timeout = timeout
        return out

    def complete(self, system: str, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        if not messages:
            return self.backend.message(system, "", max_tokens=max_tokens)["result"]
        return self.backend.message(system, messages[-1]["content"], history=list(messages[:-1]),
                                    max_tokens=max_tokens)["result"]  # 整理记忆时要比回复长


class EchoClient:
    """不调用模型，用来联调截屏 → 识别 → 发送整条链路。"""

    def complete(self, system: str, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        last = messages[-1]["content"] if messages else ""
        quoted = re.findall(r"「(.*)」", last)  # format_incoming 的格式："名字：「内容」" 或 "「内容」"
        return "收到：" + " / ".join(quoted or [last])


def make_llm(cfg: LlmConfig, api_key: str | None = None) -> LlmClient:
    """api_key：直接用这个 Key，不读环境变量（管理面板「测试大模型」用页面上还没保存的 Key）。"""
    if cfg.provider == "openai":
        return OpenAICompatClient(cfg, api_key)
    if cfg.provider == "echo":
        return EchoClient()
    raise ValueError(f"不支持的 llm.provider: {cfg.provider}")
