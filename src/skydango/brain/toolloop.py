"""OpenAI 兼容的大脑（DeepSeek / ChatGPT……）：function-calling 循环，每轮重发、带最近几轮短期记忆。

`ToolLoopBrain.send(text)` 的签名和 brain.session.BrainSession 对齐，工具经同一个 ToolBox.run 执行。
失败抛 ModelError：down 按状态码判（402 余额用完 / 401 认证……，loop 据此关这家的闸、切备用），别的错按 BACKOFF 退避。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from ..models.errors import ModelError
from ..models.openai_compat import classify

BLIND_NOTE = """你看不到画面：判断靠消息里的状态、聊天记录、身边人名地名（身体 OCR / YOLO 认的）和眼睛写的场景描述。
看图的工具（look(image=true)、look_at、look_person、check_friend、panel_read(image=true)）会请眼睛代看，返回文字；想看清什么写在 question 里。
- 前面几条消息是你最近几轮收到的和做了的：别重复你刚说过的话（意思一样也不行），接着上一句聊
- recall 查不到就说记不清，别顺着别人的话编
- 嘴上说要做动作（点头、鞠躬……）就真的调 emote，没调就别这么说
- 单字、语气词（嗯、哦、哈）不用每句都接"""

ASIDE_NOTE = "\n- 标着“跟别人说”的话默认不接"  # [addressee] 开着时接在 BLIND_NOTE 后面（10-03 晚它的说听比 0.9~1.15，比 Claude 更爱接）

HISTORY_TEXT = 1500  # 历史里每轮的唤醒消息最多留这么多字（状态之类长的截掉）
SILENT = "（没说话）"  # 历史里一轮什么都没输出时的 assistant 占位
HISTORY_TOOL = 200  # 历史里每个工具返回最多留这么多字
HISTORY_CHARS = 12000  # 历史总共最多这么多字，超了从最老的丢


def _replay(turn: list[dict], seq: int) -> list[dict]:
    """这一轮的 assistant / tool 消息，压成历史里重放的样子：保留真的工具调用（换成不重复的 id）、
    工具返回截到 HISTORY_TOOL 字、去掉没执行的调用（轮数到顶时最后一次）。
    10-06 沙盒：以前压成一句纯文字「你这一轮：说了「…」」，DeepSeek 照着学，回话写成文字、不调 say，话没发出去。"""
    answered = {m["tool_call_id"] for m in turn if m["role"] == "tool"}
    ids: dict[str, str] = {}
    out: list[dict] = []
    for m in turn:
        if m["role"] == "tool":
            content = m["content"] if len(m["content"]) <= HISTORY_TOOL else m["content"][:HISTORY_TOOL] + "……"
            out.append({"role": "tool", "tool_call_id": ids[m["tool_call_id"]], "content": content})
            continue
        calls = [c for c in m.get("tool_calls") or [] if c["id"] in answered]
        for c in calls:
            ids[c["id"]] = f"h{seq}_{len(ids)}"
        item = {"role": "assistant", "content": m.get("content") or ""}
        if calls:
            item["tool_calls"] = [{**c, "id": ids[c["id"]]} for c in calls]
        elif not item["content"].strip():
            continue
        out.append(item)
    if not out:  # 什么都没说、没调：也留一条 assistant，别让两条 user 挨着（deepseek-reasoner 不收）
        out.append({"role": "assistant", "content": SILENT})
    return out


def _size(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") + sum(len(c["function"]["arguments"] or "") for c in m.get("tool_calls") or [])
               for m in messages)


class ToolLoopBrain:
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
        provider: str = "deepseek",
    ) -> None:
        self.client = client
        self.system = system
        self.toolbox = toolbox
        self.tools = tools
        self.model = model
        self.provider = provider
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
        self._past: list[tuple[str, list[dict]]] = []  # (唤醒消息, 这一轮的 assistant / tool 消息，见 _replay)
        self._seq = 0  # 历史里工具调用 id 的轮次编号

    def close(self) -> None:
        """同 BrainSession.close：没有常驻进程，什么都不用关。"""

    def _history_messages(self) -> list[dict]:
        out: list[dict] = []
        total = 0
        for user, turn in reversed(self._past):
            user = user if len(user) <= HISTORY_TEXT else user[:HISTORY_TEXT] + "……（后面截掉了）"
            size = len(user) + _size(turn)
            if out and total + size > HISTORY_CHARS:
                break
            if not out and size > HISTORY_CHARS:
                continue
            out[:0] = [{"role": "user", "content": user}, *turn]
            total += size
        return out

    def send(self, text: str) -> dict:
        messages: list[dict] = [{"role": "system", "content": self.system}, *self._history_messages(),
                                {"role": "user", "content": text}]
        start = len(messages)
        deadline = self.clock() + self.turn_timeout
        final = ""
        rounds = 0
        requests = 0
        usage: dict | None = None
        while True:
            if self.clock() >= deadline:
                raise ModelError("超时", provider=self.provider)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=self.tools,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 SDK 的各种错误一律包成 ModelError
                raise ModelError(f"{self.provider} 出错：{exc}", down=classify(exc), provider=self.provider) from None
            requests += 1
            u = getattr(resp, "usage", None)  # 取不到就不累加；有一次取到就给 dict
            if u is not None:
                usage = usage or {"input_tokens": 0, "output_tokens": 0}
                usage["input_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                usage["output_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
                hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)  # DeepSeek 的缓存命中（算钱用）
                if hit:
                    usage["cache_read_input_tokens"] = usage.get("cache_read_input_tokens", 0) + hit
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
                if content.strip() and self.on_message is not None:  # 最后的文字也上时间线（不调工具的轮次才看得到它写了什么）
                    self.on_message({"type": "assistant", "message": {"content": [{"type": "text", "text": content}]}})
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
        if self.history > 0:
            self._seq += 1
            self._past = (self._past + [(text, _replay(messages[start:], self._seq))])[-self.history:]
        return {"result": final, "subtype": "success", "num_turns": requests, "usage": usage,
                "provider": self.provider, "model": self.model}
