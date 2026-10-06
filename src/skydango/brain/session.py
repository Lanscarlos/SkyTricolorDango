"""常驻的大脑：一个无界面 Claude Code 进程，每次醒来写一条消息、等这一轮的 result；挂了 / 超时就用 --resume 接回原会话。

工具只有我们的 MCP 服务（sky）：--tools "" 关掉所有内置工具（Bash、读写文件……），--allowedTools mcp__sky 让它们不弹确认。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from .claude import ClaudeError, StreamProcess, check_result

log = logging.getLogger(__name__)

SERVER = "sky"


class BrainSession:
    def __init__(
        self,
        base_cmd: list[str],
        env: dict[str, str],
        cwd: Path,
        mcp_url: str,
        prompt: str,
        model: str,
        effort: str,
        turn_timeout: float,
        on_message: Callable[[dict], None] | None = None,
        provider: str = "claude",
        on_rate_limit: Callable[[dict], None] | None = None,
    ) -> None:
        self.provider = provider  # 供应商 id（[providers.<id>]）：loop 据此看闸、记 brain.jsonl
        self.base_cmd = base_cmd
        self.env = env
        # 空的专用目录：不让它读到项目的 AGENTS.md / CLAUDE.md。转成绝对路径：子进程的 cwd 就是它，
        # 传给 --mcp-config / --append-system-prompt-file 的相对路径会被再拼一遍（run.dir 默认是相对的 runs/）
        self.cwd = Path(cwd).resolve()
        self.mcp_url = mcp_url
        self.prompt = prompt
        self.model = model
        self.effort = effort
        self.turn_timeout = turn_timeout
        self.on_message = on_message  # 每条输出都给它（打日志用）
        self.on_rate_limit = on_rate_limit  # 限额事件的 rate_limit_info（spec 2026-10-06-model-usage §5.2）
        self.session_id: str | None = None
        self._proc: StreamProcess | None = None
        self._mcp_ok: bool | None = None

    def command(self) -> list[str]:
        cmd = [
            *self.base_cmd, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", self.model, "--effort", self.effort,
            "--mcp-config", str(self.cwd / "mcp.json"), "--strict-mcp-config",
            "--tools", "", "--allowedTools", f"mcp__{SERVER}",
            "--permission-mode", "dontAsk", "--disable-slash-commands",
            "--append-system-prompt-file", str(self.cwd / "prompt.md"),
        ]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        return cmd

    def send(self, text: str) -> dict:
        """发一条消息，等这一轮的 result。成功返回 result；失败抛 ClaudeError（进程会被结束，下次用 --resume 接回）。"""
        self._ensure()
        self._mcp_ok = None
        try:
            self._proc.send(text)
            result = self._proc.until_result(self.turn_timeout, self._seen)
        except ClaudeError:
            self._proc.kill()
            raise
        self.session_id = result.get("session_id") or self.session_id
        check_result(result)
        if self._mcp_ok is False:
            raise ClaudeError("大脑的 MCP 工具没连上（没有手脚）")
        return result

    def close(self) -> None:
        if self._proc is not None:
            self._proc.close()

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.alive():
            return
        if self._proc is not None:
            log.warning("大脑进程不在了，%s", f"接回会话 {self.session_id}" if self.session_id else "重新开一个会话")
        self.cwd.mkdir(parents=True, exist_ok=True)
        config = {"mcpServers": {SERVER: {"type": "http", "url": self.mcp_url}}}
        (self.cwd / "mcp.json").write_text(json.dumps(config), encoding="utf-8")
        (self.cwd / "prompt.md").write_text(self.prompt, encoding="utf-8")
        self._proc = StreamProcess(self.command(), self.env, self.cwd)

    def _seen(self, m: dict) -> None:
        if m.get("type") == "system" and m.get("subtype") == "init":  # 每一轮开头都会有一条
            self.session_id = m.get("session_id") or self.session_id
            servers = {s.get("name"): s.get("status") for s in m.get("mcp_servers") or []}
            self._mcp_ok = servers.get(SERVER) in ("connected", "pending")
            if not self._mcp_ok:
                log.error("大脑的 MCP 服务没连上：%s", servers)
        elif m.get("type") == "system" and m.get("subtype") == "compact_boundary":
            log.info("大脑的对话记录自动压缩了一次")
        elif m.get("type") == "rate_limit_event" and self.on_rate_limit is not None:
            try:
                self.on_rate_limit(m.get("rate_limit_info") or {})
            except Exception:  # noqa: BLE001 记账出错不影响大脑
                log.debug("交出限额事件出错", exc_info=True)
        if self.on_message is not None:
            self.on_message(m)
