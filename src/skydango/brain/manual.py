"""手动控制：可视化网页上直接让身体说话、做动作、转视角 / 环视、看人、盯人 / 停下（`run --brain --view`），手动试身体的功能用。

- 手动的总是真执行（身体方法传 live=True）：大脑 dry-run 时也一样；身体的护栏照旧（不能自称真人、限速、走动时牵手要 force……）
- 经 Body.call 交给身体线程，和大脑的工具排队执行
- 做成了就放一条 manual 事件告诉大脑（它会醒一次），免得它看到自己没说过的话犯糊涂
设计见 docs/superpowers/specs/2026-09-28-viewer-manual-control-design.md
"""

from __future__ import annotations

import json
import logging
import time

from .camera import KEYS, MAX_STEPS
from .calling import call_available
from .tools import AROUND_TIMEOUT, CALL_TIMEOUT, RESET_TIMEOUT, ToolError, finish_call

log = logging.getLogger(__name__)

CAMERA_NAMES = {"left": "左转", "right": "右转", "up": "抬头", "down": "低头", "zoom_in": "拉近", "zoom_out": "拉远"}


def _text(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缺少 {key}（要一段文字）")
    return value.strip()


def _int(args: dict, key: str, default: int | None = None) -> int:
    value = args.get(key, default)
    if value is None:
        raise ValueError(f"缺少 {key}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} 应该是整数")
    return value


class ManualControl:
    def __init__(self, body, eyes=None, events=None) -> None:
        self.body = body
        self.eyes = eyes  # brain.eyes.Eyes：没开感知层时环视完交给它描述
        self.events = events  # brain.events.EventQueue：做成了告诉大脑
        self.sleep = time.sleep  # 喊一声等呼喊窗口时用；测试换掉

    def options(self) -> dict:
        """网页填按钮用：现在能做的动作、视角操作、看人开没开、说话字数上限、是不是 dry-run。"""
        b = self.body
        cfg = b.cfg
        return {
            "emotes": list(b.emotes.available()) if b.emotes is not None else [],
            "camera": list(KEYS) if b.camera is not None else [],
            "max_steps": MAX_STEPS,
            "friend_check": b.friend_checker is not None and cfg.friend_check.enabled,
            "max_chars": cfg.reply.max_chars,
            "dry_run": cfg.reply.dry_run,
            "panels": getattr(b, "panels", None) is not None,
            # 盯人要镜头 + 感知层（people()）；认不出这个人等身体回话
            "track": b.camera is not None and hasattr(getattr(b, "env", None), "people"),
            "max_track_seconds": cfg.track.max_seconds,
            "call": call_available(cfg, getattr(b, "env", None)),  # 按 Q 喊一声：要开 [call] 和感知层
        }

    def run(self, action: str, args: dict) -> dict:
        """做一件事，返回 {"ok", "text"}；参数不对抛 ValueError（网页回 400，不交给身体）。"""
        fn, what, timeout = self._bind(action, args if isinstance(args, dict) else {})
        try:
            out = self.body.call(fn, timeout=timeout)
            # 没开感知层时环视拿回的是几张图：交给眼睛描述（在这个线程里，不占身体）
            if action == "call":  # 身体按完键就回来了：在这个线程里等呼喊窗口结束（约 6 秒），不占身体
                text = finish_call(self.body, out, self.sleep)
            elif action == "look_around" and not isinstance(out, str):
                text = self._around(out)
            else:
                text = self._describe(out)
        except ToolError as exc:
            result = {"ok": False, "text": str(exc)}
        except Exception as exc:
            log.exception("手动操作 %s 出错", action)
            result = {"ok": False, "text": f"出错了：{exc}"}
        else:
            result = {"ok": True, "text": text}
        log.info("手动：%s %s → %s", action, json.dumps(args, ensure_ascii=False), result["text"])
        if result["ok"] and self.events is not None:
            try:
                self.events.put("manual", f"主人在面板上手动让团子{what}：{result['text']}")
            except Exception:
                log.debug("手动操作的事件没放进去", exc_info=True)
        return result

    # ---- 内部 ----
    def _bind(self, action: str, a: dict):
        b = self.body
        if action == "say":
            text = _text(a, "text")
            return (lambda: b.say(text, live=True)), f"说了「{text}」", None
        if action == "emote":
            name = _text(a, "name")
            return (lambda: b.emote(name, live=True)), f"做了动作「{name}」", None
        if action == "camera":
            move = a.get("action")
            if move not in KEYS:
                raise ValueError(f"不认识的视角操作 {move!r}，可以用：{'、'.join(KEYS)}")
            steps = _int(a, "steps", 1)
            if not 1 <= steps <= MAX_STEPS:
                raise ValueError(f"步数要在 1～{MAX_STEPS} 之间")
            return (lambda: b.camera_move(move, steps, live=True)), f"转了视角（{CAMERA_NAMES[move]} ×{steps}）", None
        if action == "camera_reset":
            return (lambda: b.camera_reset(live=True)), "把视角复位", RESET_TIMEOUT
        if action == "look_around":
            if hasattr(getattr(b, "env", None), "sweep"):  # 打开了感知层：转一圈交给 YOLO
                return (lambda: b.sweep_around(live=True)), "环视了一圈", AROUND_TIMEOUT
            return (lambda: b.capture_around(live=True)), "环视了一圈", AROUND_TIMEOUT
        if action == "check_friend":
            x, y = _int(a, "x"), _int(a, "y")
            return (lambda: b.check_friend_at(x, y, live=True)), f"点了画面上 ({x}, {y}) 的人", None
        if action == "track":
            name, seconds = _text(a, "name"), _int(a, "seconds", 30)
            top = b.cfg.track.max_seconds
            if not 1 <= seconds <= top:
                raise ValueError(f"秒数要在 1～{top} 之间")
            return (lambda: b.track(name, seconds, live=True)), f"盯着{name}（{seconds} 秒）", None
        if action == "stop_task":
            return b.stop_task, "停下正在做的事", None
        if action == "call":
            return (lambda: b.call_out("manual", live=True)), "喊了一声", CALL_TIMEOUT
        if action == "panel_read":
            return (lambda: b.panel_read()), "读了面板", None
        if action == "panel_close":
            return (lambda: b.panel_close(live=True)), "关了面板", None
        raise ValueError(f"不认识的操作 {action!r}")

    def _around(self, frames: list) -> str:
        if self.eyes is None:
            return f"转完了，截了 {len(frames)} 张（没开眼睛，不描述）"
        return self.eyes.describe_around(frames, self.body.clock())

    @staticmethod
    def _describe(out) -> str:
        if isinstance(out, str):
            return out
        if isinstance(out, list):  # check_friend 的图片 + 文字块：网页只要文字（截图已存盘）
            return "\n".join(block.get("text", "") for block in out if isinstance(block, dict) and block.get("type") == "text")
        return str(out)
