"""Claude Code 接入方式：起无界面的 claude -p 进程（stream-json），逐行收发；和用户自己的 Claude Code 配置隔离。

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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from ..chat.llm import _user_env
from .errors import ModelError

if TYPE_CHECKING:
    from .config import ProviderConfig

log = logging.getLogger(__name__)

LIMIT_WORDS = ("limit", "上限", "额度")
AUTH_WORDS = ("authentication", "invalid api key", "invalid x-api-key", "oauth token")
# 子进程不继承的（ANTHROPIC_* 另外整个去掉）：用户自己的登录令牌、改走 Bedrock / Vertex 的开关
_DROP_ENV = ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")


class ClaudeError(ModelError):
    """Claude Code 这一轮没成：起不来、挂了、超时、返回错误。limit=True 表示订阅额度用完了，auth=True 表示认证失败。"""

    def __init__(self, message: str, limit: bool = False, auth: bool = False, provider: str = "claude") -> None:
        super().__init__(message, down="limit" if limit else "auth" if auth else None, provider=provider)
        self.limit = limit
        self.auth = auth


def claude_env(token: str, config_dir: str | Path) -> dict[str, str]:
    # ANTHROPIC_* 一律去掉：除了密钥，还有 ANTHROPIC_BASE_URL / ANTHROPIC_MODEL 这类改地址、改型号的
    # （10-03 晚起团子的会话把 BASE_URL 指到了 DeepSeek，子进程继承后拿着 Claude 令牌去打 DeepSeek，全部 401）
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith("ANTHROPIC_") and k not in _DROP_ENV}
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
        status = m.get("api_error_status")
        limit = status == 429 or any(w in detail.lower() for w in LIMIT_WORDS)
        # 两个都成立时只算额度
        auth = not limit and (status == 401 or any(w in detail.lower() for w in AUTH_WORDS))
        raise ClaudeError(f"Claude Code 这一轮失败：{detail}", limit=limit, auth=auth)
    return text


def one_shot_message(cmd: list[str], env: dict[str, str], cwd: Path, content, timeout: float) -> dict:
    """起一个进程，发一条消息，拿到结果就关；返回完整的 result 消息（辅助标注要里面的 usage）。失败抛 ClaudeError。"""
    p = StreamProcess(cmd, env, cwd)
    try:
        p.send(content)
        m = p.until_result(timeout)
        check_result(m)
        return m
    finally:
        p.close()


def one_shot(cmd: list[str], env: dict[str, str], cwd: Path, content, timeout: float) -> str:
    """眼睛用：起一个进程，发一条消息，拿到结果就关。"""
    return check_result(one_shot_message(cmd, env, cwd, content, timeout))


def claude_base(provider: "ProviderConfig", environ: Mapping[str, str] = os.environ) -> tuple[list[str], dict[str, str]]:
    """这家 Claude Code 的命令 + 环境（单独配置目录 + claude setup-token 令牌）。缺令牌 / 找不到命令都算"这家用不了"（down="auth"）。"""
    token = environ.get(provider.token_env, "") or _user_env(provider.token_env)
    if not token:
        raise ModelError(
            f"没有 Claude 令牌：先运行 claude setup-token，再 setx {provider.token_env} \"<令牌>\"",
            down="auth", provider=provider.id)
    try:
        base = resolve_claude(provider.path)
    except RuntimeError as exc:
        raise ModelError(str(exc), down="auth", provider=provider.id) from None
    return base, claude_env(token, provider.config_dir)


def claude_command(base: list[str], model: str, system: str, *, effort: str = "low", persist: bool = True) -> list[str]:
    """一次性 claude -p 的命令：不给任何工具。persist=False（看图标注）：--no-session-persistence，不在配置目录的
    projects/ 下留会话记录 —— 每批一个，跑一次上百个、从来不 --resume；也不带 --effort（同原来的 assist_command）。"""
    cmd = [*base, "-p"]
    if not persist:
        cmd.append("--no-session-persistence")
    cmd += ["--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--model", model]
    if persist:
        cmd += ["--effort", effort]
    return cmd + ["--tools", "", "--strict-mcp-config", "--permission-mode", "dontAsk", "--disable-slash-commands",
                  "--system-prompt", system]


class ClaudeCodeBackend:
    """一个用处的 Claude Code 后端：每次起一个一次性进程。Claude Code 没有 max_tokens 参数，忽略。

    base / env：测试里直接给（假进程）；不给就第一次用时按 claude_base 现取。"""

    def __init__(self, provider: "ProviderConfig", model: str, cwd: Path, *, environ: Mapping[str, str] = os.environ,
                 effort: str = "low", persist: bool = True, base: list[str] | None = None,
                 env: dict[str, str] | None = None) -> None:
        self.provider = provider
        self.model = model
        self.cwd = cwd
        self.environ = environ
        self.effort = effort
        self.persist = persist
        self._base, self._env = base, env

    def _ready(self) -> tuple[list[str], dict[str, str]]:
        if self._base is None or self._env is None:
            self._base, self._env = claude_base(self.provider, self.environ)
        return self._base, self._env

    def message(self, system: str, content: str | list[dict], *, history: list[dict] | None = None,
                max_tokens: int | None = None, timeout: float = 120.0) -> dict:
        base, env = self._ready()
        if history:
            # 一次性进程只发一条消息：把前面几条和这条拼起来（同原来的 ClaudeLlm，NotesKeeper 只发一条）
            parts = [m["content"] for m in history]
            content = "\n\n".join([*parts, content]) if isinstance(content, str) else content
        cmd = claude_command(base, self.model, system, effort=self.effort, persist=self.persist)
        try:
            m = one_shot_message(cmd, env, self.cwd, content, timeout)
        except ClaudeError as exc:
            exc.provider = self.provider.id
            raise
        return {"result": m.get("result") or "", "usage": m.get("usage") or {}, "provider": self.provider.id, "model": self.model}
