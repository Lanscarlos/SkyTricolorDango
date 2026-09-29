"""设置页的「测试大模型」「测试 Claude 令牌」：点了才测（会花一点额度），用页面上的值（可能还没保存）。"""

from __future__ import annotations

import dataclasses
import subprocess
import time
from collections.abc import Callable

from ..brain.claude import claude_env, resolve_claude
from ..chat.llm import make_llm
from ..config import LlmConfig

__test__ = False  # 函数名以 test_ 开头，别让 pytest 收集

CLAUDE_TIMEOUT = 60


def test_llm(cfg: LlmConfig, api_key: str, make: Callable = make_llm, clock: Callable[[], float] = time.monotonic) -> dict:
    if cfg.provider == "echo":
        return {"ok": True, "text": "echo 不调模型"}
    if not api_key:
        return {"ok": False, "text": "还没填 API Key"}
    probe = dataclasses.replace(cfg, max_retries=0, timeout=20.0)
    start = clock()
    try:
        reply = make(probe, api_key=api_key).complete("你是连通性测试。", [{"role": "user", "content": "只回复 ok"}], max_tokens=10)
    except Exception as exc:
        return {"ok": False, "text": f"失败：{exc}"}
    return {"ok": True, "text": f"成功，{(clock() - start) * 1000:.0f} ms：{(reply or '').strip()[:50]}"}


def test_claude(claude_path: str, config_dir: str, token: str, run: Callable = subprocess.run) -> dict:
    """用大脑的隔离配置目录和令牌跑一次 claude -p（和大脑起进程是同一套环境变量）。"""
    if not token:
        return {"ok": False, "text": "还没填 Claude 令牌"}
    try:
        exe = resolve_claude(claude_path)
        env = claude_env(token, config_dir)
        version = run(exe + ["--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=CLAUDE_TIMEOUT)
        if version.returncode != 0:
            return {"ok": False, "text": f"失败：claude --version 出错：{(version.stderr or version.stdout).strip()[:200]}"}
        reply = run(exe + ["-p", "--model", "haiku", "只回复 ok"], capture_output=True, text=True, encoding="utf-8",
                    errors="replace", env=env, timeout=CLAUDE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": f"失败：{CLAUDE_TIMEOUT} 秒没回应"}
    except Exception as exc:
        return {"ok": False, "text": f"失败：{exc}"}
    if reply.returncode != 0:
        return {"ok": False, "text": f"失败：{(reply.stderr or reply.stdout).strip()[:200]}"}
    return {"ok": True, "text": f"成功：{version.stdout.strip()} · {reply.stdout.strip()[:50]}"}
