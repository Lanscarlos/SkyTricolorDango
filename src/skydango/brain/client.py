"""大脑调 Claude：anthropic SDK 的薄封装，返回普通 dict（测试时换成假的）。"""

from __future__ import annotations

from ..chat.llm import read_key
from ..config import BrainConfig


class BrainClient:
    def __init__(self, cfg: BrainConfig) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install \"skydango[anthropic]\"") from exc
        self.cfg = cfg
        kwargs = {"api_key": read_key(cfg.api_key_env), "timeout": cfg.timeout}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def create(
        self, system: list[dict], messages: list[dict], tools: list[dict] | None = None, max_tokens: int | None = None
    ) -> dict:
        kwargs = {
            "model": self.cfg.model,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "system": system,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.cfg.effort},
            "cache_control": {"type": "ephemeral"},  # 对话记录的缓存点自动跟着往后挪
        }
        if tools:
            kwargs["tools"] = tools
        return self._client.messages.create(**kwargs).to_dict()
