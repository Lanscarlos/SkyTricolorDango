"""起 Claude Code 无界面进程（stream-json），逐行收发；和用户自己的 Claude Code 配置隔离。

实测（2.1.233）：沿用用户登录会把用户的插件、钩子、技能一起加载进来 → 子进程用单独的配置目录，
登录用 `claude setup-token` 生成的令牌（CLAUDE_CODE_OAUTH_TOKEN）。有 ANTHROPIC_API_KEY 时 `-p` 一定用它，所以要去掉。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

LIMIT_WORDS = ("limit", "上限", "额度")


class ClaudeError(RuntimeError):
    """Claude Code 这一轮没成：起不来、挂了、超时、返回错误。limit=True 表示订阅额度用完了。"""

    def __init__(self, message: str, limit: bool = False) -> None:
        super().__init__(message)
        self.limit = limit


def claude_env(token: str, config_dir: str | Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    env["CLAUDE_CONFIG_DIR"] = str(Path(config_dir).resolve())
    # Windows 系统locale 不是 UTF-8（这台机器是 GBK）时子进程的 stdout 默认编码会跟着系统走，
    # 和这边按 UTF-8 读行对不上，中文会花掉：强制子进程用 UTF-8 收发。
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def resolve_claude(path: str) -> list[str]:
    found = shutil.which(path)
    if not found:
        raise RuntimeError(f"找不到 Claude Code（{path}）：先装好，确认命令行里 claude 能用")
    return [found]


def user_message(content) -> str:
    return json.dumps(
        {"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": None}, ensure_ascii=False
    )


class StreamProcess:
    def __init__(self, cmd: list[str], env: dict[str, str], cwd: Path) -> None:
        cwd.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.messages: queue.Queue = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=20)
        threading.Thread(target=self._read, name="claude-out", daemon=True).start()
        threading.Thread(target=self._read_err, name="claude-err", daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.messages.put(json.loads(line))
            except ValueError:
                log.debug("Claude Code 输出了一行不是 JSON 的内容：%s", line[:200])
        self.messages.put(None)  # 输出结束（进程退出）

    def _read_err(self) -> None:
        for line in self.proc.stderr:
            self.stderr_tail.append(line.rstrip())

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, content) -> None:
        try:
            self.proc.stdin.write(user_message(content) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise ClaudeError(f"写不进 Claude Code：{exc}") from None

    def until_result(self, timeout: float, on_message: Callable[[dict], None] | None = None) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ClaudeError(f"等了 {timeout:.0f} 秒没有结果")
            try:
                m = self.messages.get(timeout=left)
            except queue.Empty:
                continue
            if m is None:
                raise ClaudeError("Claude Code 退出了：" + " | ".join(list(self.stderr_tail)[-3:]))
            if on_message is not None:
                on_message(m)
            if m.get("type") == "result":
                return m

    def close(self) -> None:
        """正常关：关掉 stdin 让它自己退出，5 秒不退就按进程树结束。"""
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.kill()

    def kill(self) -> None:
        if self.proc.poll() is not None:
            return
        if sys.platform == "win32":  # claude 会起子进程：按进程树结束，别留孤儿
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            self.proc.kill()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            log.warning("Claude Code 进程 %d 没能结束", self.proc.pid)


def check_result(m: dict) -> str:
    """result → 这一轮最后的文字；失败抛 ClaudeError（额度用完时 limit=True）。"""
    text = m.get("result") or ""
    if m.get("subtype") != "success" or m.get("is_error"):
        detail = text or str(m.get("subtype") or "未知错误")
        limit = m.get("api_error_status") == 429 or any(w in detail.lower() for w in LIMIT_WORDS)
        raise ClaudeError(f"Claude Code 这一轮失败：{detail}", limit=limit)
    return text


def one_shot(cmd: list[str], env: dict[str, str], cwd: Path, content, timeout: float) -> str:
    """眼睛用：起一个进程，发一条消息，拿到结果就关。"""
    p = StreamProcess(cmd, env, cwd)
    try:
        p.send(content)
        return check_result(p.until_result(timeout))
    finally:
        p.close()
