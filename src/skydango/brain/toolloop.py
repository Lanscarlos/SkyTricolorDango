"""OpenAI 兼容的大脑（DeepSeek / ChatGPT……）：function-calling 循环，每轮重发、带最近几轮短期记忆。

`ToolLoopBrain.send(text)` 的签名和 brain.session.BrainSession 对齐，工具经同一个 ToolBox.run 执行。
开着 [brain.compact] 时历史一次上线内只往后接（DeepSeek 前缀缓存一路命中），太长在后台压成前情提要（spec 2026-10-06-brain-compact）。
失败抛 ModelError：down 按状态码判（402 余额用完 / 401 认证……，loop 据此关这家的闸、切备用），别的错按 BACKOFF 退避。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..models.errors import ModelError
from ..models.openai_compat import classify
from .compact import RECAP_ACK, InboxWatch, clean_recap, compact_request, inbox_note, recap_message, until_of, wake_stamp

log = logging.getLogger(__name__)

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
COMPACT_TEMPERATURE = 0.3  # 写前情提要要稳，不要发挥


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
        compact=None,  # config.CompactConfig：None = 滑动 history 轮（原来的样子，逐字照旧）
        inbox: Callable[[], str] | None = None,  # 读 inbox.md（compact 开着时才看）：中途新记的行接在下一轮唤醒消息末尾
        meter=None,  # models.usage.UsageMeter：每次压缩记一笔 recap
        backup: bool = False,  # 这个会话是 [models.brain] 的备用（用量那一笔的主 / 备）
        recap_path: Path | None = None,  # runs/<…>/brain/recap.jsonl：每次换上 / 失败一行
        on_note: Callable[[str], None] | None = None,  # 大脑时间线上的「── 压缩：… ──」
        wall: Callable[[], float] = time.time,
        spawn: Callable[[Callable[[], None]], None] | None = None,  # 怎么跑压缩任务：默认起一个后台线程（测试注入）
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
        # 只往后接（spec 2026-10-06-brain-compact）：_past 不截，靠压缩收；状态只在大脑线程里改
        self.compact = compact
        self.recap = ""  # 最新的前情提要正文（没压过是空）：最终反思也拿它当材料
        self.compactions = 0
        self.sliding = False  # 压缩失败 / 卡住：先退回滑动 history 轮（前面带前情提要）
        self._recap_head = ""  # 历史开头那条前情提要消息（recap_message 拼好的）
        self._inbox = InboxWatch(inbox) if compact is not None and inbox is not None else None
        self.meter = meter
        self.backup = backup
        self.recap_path = recap_path
        self.on_note = on_note
        self.wall = wall
        self._spawn = spawn or (lambda fn: threading.Thread(target=fn, name="brain-compact", daemon=True).start())
        self._lock = threading.Lock()
        self._done: dict | None = None  # 压缩线程放结果的槽，下一次 send 开头换上
        self._pending = False  # 起了压缩、结果还没换上
        self._retry_at = float("-inf")  # 压缩失败后到这个时候（clock）才再试

    def close(self) -> None:
        """同 BrainSession.close：没有常驻进程，什么都不用关。"""

    @staticmethod
    def _turn_messages(user: str, turn: list[dict]) -> list[dict]:
        """历史里的一轮：唤醒消息截到 HISTORY_TEXT 字 + _replay 压过的样子。"""
        user = user if len(user) <= HISTORY_TEXT else user[:HISTORY_TEXT] + "……（后面截掉了）"
        return [{"role": "user", "content": user}, *turn]

    def _history_messages(self, past: list[tuple[str, list[dict]]] | None = None) -> list[dict]:
        """滑动：从最近的往前装，总共不超过 HISTORY_CHARS 字。"""
        out: list[dict] = []
        total = 0
        for user, turn in reversed(self._past if past is None else past):
            msgs = self._turn_messages(user, turn)
            size = len(msgs[0]["content"]) + _size(turn)
            if out and total + size > HISTORY_CHARS:
                break
            if not out and size > HISTORY_CHARS:
                continue
            out[:0] = msgs
            total += size
        return out

    def _recap_messages(self) -> list[dict]:
        if not self._recap_head:
            return []
        return [{"role": "user", "content": self._recap_head}, {"role": "assistant", "content": RECAP_ACK}]

    def _append_messages(self) -> list[dict]:
        """只往后接：system + 前情提要 + 全部的轮（存下就不再变）。"""
        out = [{"role": "system", "content": self.system}, *self._recap_messages()]
        for user, turn in self._past:
            out += self._turn_messages(user, turn)
        return out

    def _prefix(self) -> list[dict]:
        if self.compact is None:
            return [{"role": "system", "content": self.system}, *self._history_messages()]
        if self.sliding:
            recent = self._past[-self.history:] if self.history > 0 else []
            return [{"role": "system", "content": self.system}, *self._recap_messages(), *self._history_messages(recent)]
        return self._append_messages()

    def send(self, text: str) -> dict:
        if self.compact is not None:
            self._take()
        mode = "sliding" if self.sliding else "append"
        if self._inbox is not None:
            note = inbox_note(self._inbox.fresh())
            if note:  # 接在事件后面、「状态：」前面：历史里唤醒消息截到 HISTORY_TEXT 字，接在最末尾会被截掉
                head, sep, tail = text.partition("\n状态：")
                text = f"{head}\n\n{note}{sep}{tail}" if sep else f"{text}\n\n{note}"
        messages: list[dict] = [*self._prefix(), {"role": "user", "content": text}]
        start = len(messages)
        deadline = self.clock() + self.turn_timeout
        final = ""
        rounds = 0
        requests = 0
        usage: dict | None = None
        prompt_tokens: int | None = None  # 这一轮最后一次请求的输入（判要不要压）；那次没给就是 None
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
            given = getattr(u, "prompt_tokens", None) if u is not None else None
            prompt_tokens = int(given) if isinstance(given, (int, float)) and not isinstance(given, bool) else None
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
        if self.compact is not None:
            self._seq += 1
            self._past.append((text, _replay(messages[start:], self._seq)))
            self._after(messages, prompt_tokens)
        elif self.history > 0:
            self._seq += 1
            self._past = (self._past + [(text, _replay(messages[start:], self._seq))])[-self.history:]
        result = {"result": final, "subtype": "success", "num_turns": requests, "usage": usage,
                  "provider": self.provider, "model": self.model}
        if self.compact is not None:
            result.update(history_mode=mode, recap=self.compactions)
        return result

    # ---- 压缩（spec 2026-10-06-brain-compact §2 §4）：起、跑、换上都在这里；状态只在大脑线程里改 ----
    def _after(self, messages: list[dict], prompt_tokens: int | None) -> None:
        """一轮结束：要不要起压缩；压缩卡着、历史已经到两倍预算就先退回滑动。"""
        c = self.compact
        if self._pending:
            if prompt_tokens is not None and prompt_tokens >= 2 * c.budget and not self.sliding:
                self.sliding = True
                log.warning("压缩还没回来、大脑历史已经 %d token，先只带最近 %d 轮", prompt_tokens, self.history)
            return
        if self.clock() < self._retry_at or len(self._past) <= max(1, c.keep_turns):
            return
        if self.sliding:  # 滑动的请求本来就小，不看 prompt_tokens；拿完整历史现拼（不走缓存，一次性的）
            source = self._append_messages()
        elif prompt_tokens is None or prompt_tokens < c.budget:
            return
        else:  # 接着这一轮最后一次请求发：前缀全在缓存里
            source = list(messages)
            last = source[-1]
            if last.get("role") == "assistant" and (last.get("tool_calls") or not (last.get("content") or "").strip()):
                # 轮数到顶、没执行的调用：只留文字；最后什么都没说：占位（空的 assistant 接口可能不收）
                source[-1] = {"role": "assistant", "content": last.get("content") or SILENT}
        keep = max(1, c.keep_turns)  # 0 / 负数（设置页能填）当 1：至少留这一轮原话
        cut = len(self._past) - keep
        first_kept = self._past[cut][0]
        request = [*source, {"role": "user", "content": compact_request(first_kept, keep, c.recap_max)}]
        info = {"cut": cut, "until": until_of(wake_stamp(first_kept))}
        self._pending = True
        log.info("大脑历史%s，起压缩：压掉 %d 轮、留 %d 轮原话",
                 f" {prompt_tokens} token" if prompt_tokens is not None else "", cut, keep)
        self._spawn(lambda: self._job(request, info))

    def _job(self, request: list[dict], info: dict) -> None:
        """压缩线程：结果（成功或失败）放进槽里，下一次 send 开头换上。ModelError 不关闸（大脑自己撞上了由 loop 关）。"""
        start = self.clock()
        usage: dict | None = None
        text, error = "", ""
        try:
            kwargs = {"model": self.model, "messages": request, "tools": self.tools,
                      "temperature": COMPACT_TEMPERATURE, "max_tokens": self.max_tokens, "timeout": self.turn_timeout}
            if self.tools:
                kwargs["tool_choice"] = "none"
            resp = self.client.chat.completions.create(**kwargs)
            u = getattr(resp, "usage", None)
            if u is not None:
                usage = {"input_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                         "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                         "cache_read_input_tokens": int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)}
            message = resp.choices[0].message
            if message.tool_calls:
                error = "压缩时调了工具"
            else:
                text = clean_recap(message.content, self.compact.recap_max)
                if not text:
                    error = "写出来是空的"
        except Exception as exc:  # noqa: BLE001 SDK 的各种错误、超时
            error = f"{self.provider} 出错：{exc}"
        finally:
            if self.meter is not None:
                try:
                    self.meter.record("recap", self.provider, self.model, backup=self.backup, usage=usage, ok=not error)
                except Exception:  # noqa: BLE001 记账出错不影响压缩
                    log.debug("记压缩用量出错", exc_info=True)
            with self._lock:
                self._done = {**info, "ok": not error, "text": text, "error": error, "usage": usage, "seconds": self.clock() - start}

    def _take(self) -> None:
        """send 开头：压缩有结果了就换上（成功）或退回滑动（失败）。"""
        with self._lock:
            done, self._done = self._done, None
        if done is None:
            return
        self._pending = False
        n = self.compactions + 1
        if done["ok"]:
            self._past = self._past[done["cut"]:]
            self.compactions = n
            self.recap = done["text"]
            self._recap_head = recap_message(done["text"], n, done["until"], max(1, self.compact.keep_turns))
            self.sliding = False
            note = (f"── 压缩：第 {n} 次，压掉 {done['cut']} 轮 → 前情提要 {len(done['text'])} 字，"
                    f"用时 {done['seconds']:.0f} 秒 ──\n{done['text']}")
            log.info("大脑历史压好了：第 %d 次，压掉 %d 轮 → 前情提要 %d 字（%.0f 秒）", n, done["cut"], len(done["text"]), done["seconds"])
        else:
            self.sliding = True
            self._retry_at = self.clock() + self.compact.retry
            note = f"── 压缩失败：{done['error']}；先只带最近 {self.history} 轮，{self.compact.retry:.0f} 秒后再试 ──"
            log.warning("大脑历史压缩失败（%s），先只带最近 %d 轮，%.0f 秒后再试", done["error"], self.history, self.compact.retry)
        if self.on_note is not None:
            try:
                self.on_note(note)
            except Exception:  # noqa: BLE001
                log.debug("压缩记到时间线出错", exc_info=True)
        if self.recap_path is not None:
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.wall())), "n": n, "ok": done["ok"],
                   "turns": done["cut"], "chars": len(done["text"]), "seconds": round(done["seconds"], 1), "usage": done["usage"]}
            row.update({"text": done["text"]} if done["ok"] else {"error": done["error"]})
            try:
                with Path(self.recap_path).open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                log.warning("写 recap.jsonl 出错", exc_info=True)
