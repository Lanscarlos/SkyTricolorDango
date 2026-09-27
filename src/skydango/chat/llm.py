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


def _api_key(cfg: LlmConfig) -> str:
    return read_key(cfg.api_key_env)


class OpenAICompatClient:
    def __init__(self, cfg: LlmConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install \"skydango[openai]\"") from exc
        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url or None, api_key=_api_key(cfg), timeout=cfg.timeout)

    def complete(self, system: str, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,  # 整理记忆时要比回复长
        )
        return resp.choices[0].message.content or ""


# 这些型号不再接受 temperature（传了返回 400）；不传 thinking 时默认会思考，回复的 max_tokens 只有 200，会被思考吃光
_NO_SAMPLING = ("claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-8")


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

    def complete(self, system: str, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        kwargs = {
            "model": self.cfg.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens or self.cfg.max_tokens,  # 整理记忆时要比回复长
        }
        if self.cfg.model.startswith(_NO_SAMPLING):
            kwargs["thinking"] = {"type": "disabled"}
        else:
            kwargs["temperature"] = self.cfg.temperature
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class EchoClient:
    """不调用模型，用来联调截屏 → 识别 → 发送整条链路。"""

    def complete(self, system: str, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
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
