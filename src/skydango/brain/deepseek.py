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
判断只靠消息里的状态、聊天记录、身边人名地名（身体 OCR / YOLO 认的）和旧场景描述；工具里没有看图 / 点人的，别硬调。
- 前面几条消息是你最近几轮收到的和做了的：别重复你刚说过的话（意思一样也不行），接着上一句聊
- recall 查不到就说记不清，别顺着别人的话编
- 嘴上说要做动作（点头、鞠躬……）就真的调 emote，没调就别这么说
- 单字、语气词（嗯、哦、哈）不用每句都接"""

HISTORY_TEXT = 1500  # 历史里每轮的唤醒消息最多留这么多字（状态之类长的截掉）
HISTORY_CHARS = 12000  # 历史总共最多这么多字，超了从最老的丢


def _summary(done: list[tuple[str, dict | None, str, bool]], final: str) -> str:
    """这一轮做了什么，压成一句话放进历史（只给它自己看：说了什么、做了什么动作、调了什么、结果）。"""
    parts = []
    for name, args, out, is_error in done:
        args = args or {}
        if name == "say":
            text = str(args.get("text", ""))
            parts.append(f"想说「{text}」被拦下：{out[:60]}" if is_error else f"说了「{text}」")
        elif name == "emote":
            parts.append(f"做了动作 {args.get('name', '')}" + (f"（没做成：{out[:40]}）" if is_error else ""))
        else:
            brief = json.dumps(args, ensure_ascii=False)[:60]
            parts.append(f"调了 {name}({brief}) → {out[:60]}")
    if final.strip():
        parts.append(f"心里：{final.strip()[:200]}")
    return "你这一轮：" + ("；".join(parts) if parts else "什么都没做")


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
        on_message: Callable[[dict], None] | None = None,
        history: int = 0,
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
        self.on_message = on_message  # 把工具调用喂回大脑时间线（同 BrainSession.on_message，spec 2026-10-03-deepseek-fallback-brain）
        self._names = {t["function"]["name"] for t in tools}
        # 短期记忆：最近 history 轮（唤醒消息, 这一轮做了什么）。Claude 大脑是常驻会话；这里每轮重发，
        # 不带的话它不记得刚说过什么（10-03 晚：同一句话隔 8 秒说两遍、连说三句晚安）。只在内存里
        self.history = history
        self._past: list[tuple[str, str]] = []

    def _history_messages(self) -> list[dict]:
        out: list[dict] = []
        total = 0
        for user, assistant in reversed(self._past):
            user = user if len(user) <= HISTORY_TEXT else user[:HISTORY_TEXT] + "……（后面截掉了）"
            size = len(user) + len(assistant)
            if out and total + size > HISTORY_CHARS:
                break
            if not out and size > HISTORY_CHARS:
                continue
            out[:0] = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
            total += size
        return out

    def send(self, text: str) -> dict:
        messages: list[dict] = [{"role": "system", "content": self.system}, *self._history_messages(),
                                {"role": "user", "content": text}]
        done: list[tuple[str, dict | None, str, bool]] = []
        deadline = self.clock() + self.turn_timeout
        final = ""
        rounds = 0
        requests = 0
        usage: dict | None = None
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
            requests += 1
            u = getattr(resp, "usage", None)  # 取不到就不累加；有一次取到就给 dict
            if u is not None:
                usage = usage or {"input_tokens": 0, "output_tokens": 0}
                usage["input_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                usage["output_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
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
            if content.strip() and self.on_message is not None:  # 工具调用前的中间文字：时间线上也要看到
                self.on_message({"type": "assistant", "message": {"content": [{"type": "text", "text": content}]}})
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (ValueError, TypeError):
                    args = None
                if self.on_message is not None:  # 工具调用：喂给大脑时间线（TOOL_PREFIX 同 trace._steps）
                    self.on_message({"type": "assistant", "message": {"content": [
                        {"type": "tool_use", "id": tc.id, "name": f"mcp__sky__{name}", "input": args}
                    ]}})
                if name not in self._names:
                    out, is_error = f"没有这个工具：{name}", True
                elif args is None:
                    out, is_error = "工具参数不是合法 JSON", True
                else:
                    out, is_error = self.toolbox.run(name, args)
                if not isinstance(out, str):
                    out = json.dumps(out, ensure_ascii=False)
                if self.on_message is not None:
                    self.on_message({"type": "user", "message": {"content": [
                        {"type": "tool_result", "tool_use_id": tc.id, "content": out, "is_error": is_error}
                    ]}})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                done.append((name, args, out, is_error))
        if self.history > 0:
            self._past = (self._past + [(text, _summary(done, final))])[-self.history:]
        return {"result": final, "subtype": "success", "num_turns": requests, "usage": usage}
