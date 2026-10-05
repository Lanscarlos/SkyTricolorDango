"""「模型」页每家供应商的「测试」：点了才测（会花一点额度），用页面上的值（可能还没保存）。"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

import numpy as np

from ..brain.images import image_block
from ..models.claude_code import claude_env, resolve_claude
from ..models.config import ProviderConfig
from ..models.openai_compat import OpenAIBackend, build_client

__test__ = False  # 函数名以 test_ 开头，别让 pytest 收集

CLAUDE_TIMEOUT = 60
OPENAI_TIMEOUT = 20.0
PLACEHOLDER_KEY = "ollama"  # Ollama 这类本机服务不要 Key，但 SDK 不许空
COLOR_QUESTION = "这张图是什么颜色？只回答颜色"


def _red() -> dict:
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :, 2] = 255  # BGR：纯红
    return image_block(img)


def test_provider(provider: ProviderConfig, secret: str, *, run: Callable = subprocess.run,
                  client_factory: Callable | None = None, clock: Callable[[], float] = time.monotonic) -> dict:
    """发一句「只回复 ok」；勾了能看图的再用第一个能看图的模型问一张 64×64 纯红小图是什么颜色（回答里有「红」算过）。"""
    if not provider.models:
        return {"ok": False, "text": "还没列模型"}
    if provider.kind == "claude-code":
        return _test_claude(provider, secret, run)
    if provider.kind != "openai":
        return {"ok": False, "text": f"不认识的接入方式：{provider.kind}"}
    factory = client_factory or build_client
    start = clock()
    try:
        client = factory(provider, api_key=secret or PLACEHOLDER_KEY, timeout=OPENAI_TIMEOUT, max_retries=0)
        model = provider.models[0]
        reply = OpenAIBackend(provider, model, temperature=0.0, max_tokens=10, client=client).message(
            "你是连通性测试。", "只回复 ok")["result"].strip()
        text = f"通过：{model} 回复 {reply[:30]}（{(clock() - start) * 1000:.0f} ms）"
        if provider.vision:
            seeing = provider.vision[0]
            color = OpenAIBackend(provider, seeing, temperature=0.0, max_tokens=10, client=client).message(
                "你是连通性测试。", [_red(), {"type": "text", "text": COLOR_QUESTION}])["result"].strip()
            if "红" not in color:
                return {"ok": False, "text": f"{text}；看图没过：{seeing} 说红图是「{color[:30]}」"}
            text += f"；看图 {seeing} 认出了红色"
    except Exception as exc:  # noqa: BLE001 测试按钮：什么错都报给页面
        return {"ok": False, "text": f"失败：{exc}"}
    return {"ok": True, "text": text}


def _test_claude(provider: ProviderConfig, token: str, run: Callable) -> dict:
    """用这家的隔离配置目录和令牌跑一次 claude -p（和大脑起进程是同一套环境变量）。"""
    if not token:
        return {"ok": False, "text": "还没填 Claude 令牌"}
    try:
        exe = resolve_claude(provider.path)
        env = claude_env(token, provider.config_dir)
        version = run(exe + ["--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=CLAUDE_TIMEOUT)
        if version.returncode != 0:
            return {"ok": False, "text": f"失败：claude --version 出错：{(version.stderr or version.stdout).strip()[:200]}"}
        reply = run(exe + ["-p", "--model", provider.models[0], "只回复 ok"], capture_output=True, text=True, encoding="utf-8",
                    errors="replace", env=env, timeout=CLAUDE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": f"失败：{CLAUDE_TIMEOUT} 秒没回应"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": f"失败：{exc}"}
    if reply.returncode != 0:
        return {"ok": False, "text": f"失败：{(reply.stderr or reply.stdout).strip()[:200]}"}
    return {"ok": True, "text": f"成功：{version.stdout.strip()} · {provider.models[0]} 回复 {reply.stdout.strip()[:50]}"}
