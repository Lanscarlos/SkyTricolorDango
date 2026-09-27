"""大模型客户端。国内网络建议用 OpenAI 兼容接口（DeepSeek、通义千问、Kimi、本地 Ollama 等）。"""

from __future__ import annotations

import os
import re
import sys
from typing import Protocol

from ..config import LlmConfig

ChatMessage = dict[str, str]  # {"role": "user" | "assistant", "content": ...}


class LlmClient(Protocol):
    def complete(self, system: str, messages: list[ChatMessage]) -> str: ...


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


def _api_key(cfg: LlmConfig) -> str:
    key = (os.environ.get(cfg.api_key_env, "") or _user_env(cfg.api_key_env)) if cfg.api_key_env else ""
    if not key:
        raise RuntimeError(f"没有找到 API Key，请设置环境变量 {cfg.api_key_env}")
    return key


class OpenAICompatClient:
    def __init__(self, cfg: LlmConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install \"skydango[openai]\"") from exc
        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url or None, api_key=_api_key(cfg), timeout=cfg.timeout)

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, cfg: LlmConfig) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install \"skydango[anthropic]\"") from exc
        self.cfg = cfg
        kwargs = {"api_key": _api_key(cfg), "timeout": cfg.timeout}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        resp = self._client.messages.create(
            model=self.cfg.model,
            system=system,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class EchoClient:
    """不调用模型，用来联调截屏 → 识别 → 发送整条链路。"""

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        last = messages[-1]["content"] if messages else ""
        quoted = re.findall(r"「(.*)」", last)  # format_incoming 的格式："名字：「内容」" 或 "「内容」"
        return "收到：" + " / ".join(quoted or [last])


def make_llm(cfg: LlmConfig) -> LlmClient:
    if cfg.provider == "openai":
        return OpenAICompatClient(cfg)
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg)
    if cfg.provider == "echo":
        return EchoClient()
    raise ValueError(f"不支持的 llm.provider: {cfg.provider}")
