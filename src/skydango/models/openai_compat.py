"""OpenAI 兼容接入方式（DeepSeek / ChatGPT / 通义 / Kimi / Ollama）：文字 + 看图，按状态码判这家还能不能用。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..chat import llm as chat_llm
from .errors import ModelError

if TYPE_CHECKING:
    from .config import ProviderConfig


def to_openai_content(content: str | list[dict]) -> str | list[dict]:
    """Claude 的内容块 → OpenAI 的消息片段：文字原样，图片换成 data URI 的 image_url。"""
    if isinstance(content, str):
        return content
    out: list[dict] = []
    for block in content:
        if block.get("type") == "image":
            src = block.get("source") or {}
            url = f"data:{src.get('media_type', 'image/jpeg')};base64,{src.get('data', '')}"
            out.append({"type": "image_url", "image_url": {"url": url}})
        elif block.get("type") == "text":
            out.append({"type": "text", "text": block.get("text", "")})
    return out


def _error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if code:
        return str(code)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("code"):
            return str(err["code"])
    return ""


def classify(exc: BaseException) -> str | None:
    """这家还能不能用：401 / 403 认证失败 → 'auth'；402（DeepSeek 余额不足）、429 insufficient_quota（OpenAI 额度用完）→ 'limit'；
    别的 429（限流）、超时、连接错、5xx → None（不关闸，照各功能自己的重试 / 退避）。"""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "limit"
    if status == 429 and _error_code(exc) == "insufficient_quota":
        return "limit"
    return None


def build_client(provider: "ProviderConfig", *, api_key: str | None = None, timeout: float | None = None,
                 max_retries: int | None = None, environ: Mapping[str, str] = os.environ):
    """这家的 openai.OpenAI 客户端。没 Key 算"这家用不了"（down="auth"）。"""
    key = api_key or (environ.get(provider.key_env, "") or chat_llm._user_env(provider.key_env) if provider.key_env else "")
    if not key:
        raise ModelError(f"{provider.id} 没有 Key：设置环境变量 {provider.key_env or '（没写 key_env）'}",
                         down="auth", provider=provider.id)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ModelError("pip install \"skydango[openai]\"", provider=provider.id) from exc
    return OpenAI(base_url=provider.base_url or None, api_key=key,
                  timeout=provider.timeout if timeout is None else timeout,
                  max_retries=provider.max_retries if max_retries is None else max_retries)


class OpenAIBackend:
    """一个用处的 OpenAI 兼容后端：客户端第一次用时才建（没 Key 时到调用那一刻才报）。"""

    def __init__(self, provider: "ProviderConfig", model: str, *, temperature: float, max_tokens: int,
                 timeout: float | None = None, max_retries: int | None = None, client=None,
                 environ: Mapping[str, str] = os.environ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = provider.timeout if timeout is None else timeout
        self.max_retries = max_retries
        self.environ = environ
        self._client = client

    def _get_client(self):
        if self._client is None:
            self._client = build_client(self.provider, timeout=self.timeout, max_retries=self.max_retries, environ=self.environ)
        return self._client

    def message(self, system: str, content: str | list[dict], *, history: list[dict] | None = None,
                max_tokens: int | None = None, timeout: float | None = None) -> dict:
        client = self._get_client()
        if timeout is not None and timeout != self.timeout:
            opts: dict = {"timeout": timeout}
            if self.max_retries is not None:
                opts["max_retries"] = self.max_retries
            client = client.with_options(**opts)
        messages = [{"role": "system", "content": system}, *(history or []),
                    {"role": "user", "content": to_openai_content(content)}]
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens)
        except Exception as exc:  # noqa: BLE001 SDK 的各种错误（状态码、超时、连接）一律包成 ModelError
            raise ModelError(f"{self.provider.id} 出错：{exc}", down=classify(exc), provider=self.provider.id) from exc
        usage = getattr(resp, "usage", None)
        return {
            "result": resp.choices[0].message.content or "",
            "usage": {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
            },
            "provider": self.provider.id,
            "model": self.model,
        }
