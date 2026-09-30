"""内心页的接口处理（spec 2026-09-30-inner-viewer §2）：viewer 的 /inner、/inner/forget 和管理面板共用；之后沙盒也复用。纯函数，不碰网络。"""

from __future__ import annotations

import copy

KINDS = ("catchphrase", "joke", "opinion")  # 网页上能删的性格条目：口头禅 / 老梗 / 看法

# 没有身体时 /inner 的样子（团子没在跑）：和“没开反思”一样，都是空的
EMPTY_SNAPSHOT: dict = {
    "running": False, "at": None, "mood": None, "energy": None, "grudge": None,
    "wants": [], "soft": [], "persona": None, "log": [],
}


def empty_snapshot() -> dict:
    return copy.deepcopy(EMPTY_SNAPSHOT)


def _text(req: dict, key: str) -> str:
    value = req.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} 要是文字")
    return value.strip()


def parse_forget(req) -> tuple[str, str, str, str]:
    """删条目的请求 → (kind, text, who, topic)。类别不认识、该有的字段是空的 → ValueError。"""
    if not isinstance(req, dict):
        raise ValueError("请求要是一个 JSON 对象")
    kind = req.get("kind")
    if kind not in KINDS:
        raise ValueError(f"不认识的类别：{kind!r}")
    text, who, topic = _text(req, "text"), _text(req, "who"), _text(req, "topic")
    if kind == "opinion":
        if not topic:
            raise ValueError("看法要给话题（topic）")
    elif not text:
        raise ValueError("要给条目原文（text）")
    return kind, text, who, topic


def forget_result(reason: str) -> dict:
    """Body.forget 的返回（空 = 删掉了）→ 给网页的 JSON。"""
    return {"ok": True} if not reason else {"ok": False, "error": reason}
