"""DeepSeek 备用大脑：Claude 订阅额度用完时顶上（spec 2026-10-03-deepseek-fallback-brain-design）。

`DeepSeekBrain.send(text)` 的签名和 brain.session.BrainSession 对齐，内部跑 OpenAI function-calling 循环，
工具经同一个 ToolBox.run 执行。失败抛 ClaudeError(msg, limit=False)：loop 里按 BACKOFF 退避，不回 Claude。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from ..chat.llm import read_key
from ..config import LlmConfig
from .claude import ClaudeError

FALLBACK_NOTE = """你现在是备用大脑（DeepSeek）：Claude 额度不足，眼睛（看图的）也不可用，看不到画面。
判断只靠消息里的状态、聊天记录、身边人名地名（身体 OCR / YOLO 认的）和旧场景描述；工具里没有看图 / 点人的，别硬调。"""


def build_client(llm: LlmConfig, api_key: str | None = None):
    """OpenAI 客户端（DeepSeek）。api_key 为 None 时从环境变量读。"""
    from openai import OpenAI

    return OpenAI(
        base_url=llm.base_url or None,
        api_key=api_key or read_key(llm.api_key_env),
        timeout=llm.timeout,
        max_retries=0,
    )


class DeepSeekBrain:
    def __init__(
        self,
        client,
        system: str,
        toolbox,
        tools: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        max_steps: int = 6,
        turn_timeout: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.system = system
        self.toolbox = toolbox
        self.tools = tools
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.turn_timeout = turn_timeout
        self.clock = clock
        self._names = {t["function"]["name"] for t in tools}

    def send(self, text: str) -> dict:
        messages: list[dict] = [{"role": "system", "content": self.system}, {"role": "user", "content": text}]
        deadline = self.clock() + self.turn_timeout
        final = ""
        rounds = 0
        while True:
            if self.clock() >= deadline:
                raise ClaudeError("超时")
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=self.tools,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                )
            except ClaudeError:
                raise
            except Exception as exc:
                raise ClaudeError(f"DeepSeek 出错：{exc}") from None
            message = resp.choices[0].message
            content = message.content or ""
            tool_calls = list(message.tool_calls or [])
            assistant = {"role": "assistant", "content": content}
            if tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]
            messages.append(assistant)
            final = content
            if not tool_calls or rounds >= self.max_steps + 2:
                break
            rounds += 1
            for tc in tool_calls:
                name = tc.function.name
                if name not in self._names:
                    out = f"没有这个工具：{name}"
                else:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except (ValueError, TypeError):
                        args = None
                    if args is None:
                        out = "工具参数不是合法 JSON"
                    else:
                        out, _err = self.toolbox.run(name, args)
                if not isinstance(out, str):
                    out = json.dumps(out, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        return {"result": final, "subtype": "success"}
