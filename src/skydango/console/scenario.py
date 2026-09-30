"""沙盒剧本（brain-sandbox spec §6）：`sandbox/scenarios/*.toml` 的读、校验、写，和步骤 → `/sandbox/op` 请求。

一步一个动作键：say（配 who）/ come / leave / strangers / place / scene / notice / skip / time / reflect / offline / online；
外加可选 expect（只进报告，这一版不检查）、wait（这一步后多等几秒真实时间）。载入即校验，错了指出第几步。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..sandbox.clock import is_time_text, parse_duration
from .tomlfile import _string, _value

ACTIONS = ("say", "come", "leave", "strangers", "place", "scene", "notice", "skip", "time", "reflect", "offline", "online")
EXTRAS = ("who", "expect", "wait")
TOP_KEYS = ("name", "note", "start", "steps")
START_KEYS = ("memory", "time", "nearby", "strangers", "place", "scene")
MEMORY = ("reset", "keep")
START_CHOICES = ("resume", "sleep")
MAX_STRANGERS = 20


class ScenarioError(ValueError):
    pass


@dataclass
class Start:
    memory: str = "reset"  # reset = 先从 memory/ 重置；keep = 用沙盒现在的记忆
    time: str = ""  # 空 = 接着上次；resume / sleep / HH:MM / YYYY-MM-DD HH:MM
    nearby: list[str] = field(default_factory=list)
    strangers: int = 0
    place: str = ""
    scene: str = ""


@dataclass
class Step:
    action: str  # ACTIONS 之一
    value: object  # 那个键的值
    who: str = ""  # 只配 say
    expect: str = ""
    wait: float = 0.0  # 秒，真实时间


@dataclass
class Scenario:
    name: str
    note: str
    start: Start
    steps: list[Step]


def _strangers(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_STRANGERS:
        raise ScenarioError(f"{where}：陌生人数要是 0~{MAX_STRANGERS} 的整数，写的是 {value!r}")
    return value


def _text(value, where: str, what: str, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ScenarioError(f"{where}：{what}要写成字符串，写的是 {value!r}")
    if not allow_empty and not value.strip():
        raise ScenarioError(f"{where}：{what}不能是空的")
    return value


def _start_time(value, where: str) -> str:
    value = _text(value, where, "起始时间")
    if value and value not in START_CHOICES and not is_time_text(value):
        raise ScenarioError(f"{where}：时间写法不对：{value!r}（resume / sleep / HH:MM / YYYY-MM-DD HH:MM）")
    return value


def _parse_start(data) -> Start:
    where = "[start]"
    if not isinstance(data, dict):
        raise ScenarioError(f"{where} 要是一个表")
    unknown = [k for k in data if k not in START_KEYS]
    if unknown:
        raise ScenarioError(f"{where}：不认识的键 {'、'.join(unknown)}")
    s = Start()
    if "memory" in data:
        if data["memory"] not in MEMORY:
            raise ScenarioError(f"{where}：memory 只能是 reset 或 keep，写的是 {data['memory']!r}")
        s.memory = data["memory"]
    if "time" in data:
        s.time = _start_time(data["time"], where)
    if "nearby" in data:
        nearby = data["nearby"]
        if not isinstance(nearby, list) or not all(isinstance(n, str) and n.strip() for n in nearby):
            raise ScenarioError(f"{where}：nearby 要写成名字列表，如 [\"小明\"]")
        s.nearby = list(nearby)
    if "strangers" in data:
        s.strangers = _strangers(data["strangers"], where)
    for key in ("place", "scene"):
        if key in data:
            setattr(s, key, _text(data[key], where, key))
    return s


def _seconds(value, where: str, what: str) -> float:
    """时长：30s / 10m / 2h，或者直接写秒数。"""
    if isinstance(value, bool):
        raise ScenarioError(f"{where}：{what}写法不对：{value!r}")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ScenarioError(f"{where}：{what}不能是负数")
        return float(value)
    try:
        return parse_duration(value)
    except ValueError:
        raise ScenarioError(f"{where}：{what}的时长写法不对：{value!r}（30s / 10m / 2h）") from None


def _parse_step(i: int, data) -> Step:
    where = f"第 {i} 步"
    if not isinstance(data, dict):
        raise ScenarioError(f"{where}：要写成 [[steps]] 表")
    unknown = [k for k in data if k not in ACTIONS and k not in EXTRAS]
    if unknown:
        raise ScenarioError(f"{where}：不认识的键 {'、'.join(unknown)}")
    actions = [k for k in data if k in ACTIONS]
    if not actions:
        raise ScenarioError(f"{where}：没有动作（{' / '.join(ACTIONS)} 选一个）")
    if len(actions) > 1:
        raise ScenarioError(f"{where}：一步里有两个动作（{'、'.join(actions)}），拆成两步")
    action = actions[0]
    value = data[action]
    who = ""
    if "who" in data:
        if action != "say":
            raise ScenarioError(f"{where}：who 只配 say 用（come / leave 直接写名字）")
        who = _text(data["who"], where, "who", allow_empty=False)
    if action == "say":
        if not who:
            raise ScenarioError(f"{where}：say 要配 who（谁说的）")
        value = _text(value, where, "说的话", allow_empty=False)
    elif action in ("come", "leave"):
        value = _text(value, where, "名字", allow_empty=False)
    elif action == "strangers":
        value = _strangers(value, where)
    elif action in ("place", "scene"):
        value = _text(value, where, action)
    elif action == "notice":
        value = _text(value, where, "新鲜事", allow_empty=False)
    elif action == "skip":
        _seconds(value, where, "skip")
        if isinstance(value, (int, float)):
            value = float(value)
    elif action == "time":
        value = _text(value, where, "时间")
        if not is_time_text(value):
            raise ScenarioError(f"{where}：时间写法不对：{value!r}（HH:MM 或 YYYY-MM-DD HH:MM）")
    elif action in ("reflect", "offline"):
        if value is not True:
            raise ScenarioError(f"{where}：{action} 只能写 true")
    elif action == "online":
        value = _text(value, where, "上线时间")
        if value not in START_CHOICES and not is_time_text(value):
            raise ScenarioError(f"{where}：上线的时间写法不对：{value!r}（sleep / resume / HH:MM / YYYY-MM-DD HH:MM）")
    expect = _text(data["expect"], where, "expect") if "expect" in data else ""
    wait = _seconds(data["wait"], where, "wait") if "wait" in data else 0.0
    return Step(action, value, who=who, expect=expect, wait=wait)


def _check_online(steps: list[Step]) -> None:
    """online 前面要有 offline；offline 后面紧跟的只能是 online 或者结尾。"""
    online = True
    for i, step in enumerate(steps, 1):
        if not online and step.action != "online":
            raise ScenarioError(f"第 {i - 1} 步：offline 下线之后，下一步只能是 online（再上线）或者结尾")
        if step.action == "online" and online:
            raise ScenarioError(f"第 {i} 步：online 前面没有 offline（沙盒已经在线）")
        if step.action == "offline":
            online = False
        elif step.action == "online":
            online = True


def parse(data: dict, default_name: str = "") -> Scenario:
    unknown = [k for k in data if k not in TOP_KEYS]
    if unknown:
        raise ScenarioError(f"不认识的键 {'、'.join(unknown)}（只认 name / note / [start] / [[steps]]）")
    name = _text(data.get("name", default_name), "剧本", "name") or default_name
    note = _text(data.get("note", ""), "剧本", "note")
    start = _parse_start(data.get("start", {}))
    raw = data.get("steps", [])
    if not isinstance(raw, list):
        raise ScenarioError("steps 要写成 [[steps]]")
    steps = [_parse_step(i, s) for i, s in enumerate(raw, 1)]
    _check_online(steps)
    return Scenario(name, note, start, steps)


def load(path: Path) -> Scenario:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise ScenarioError(f"{path.name} 读不了：{e}") from None
    except tomllib.TOMLDecodeError as e:
        raise ScenarioError(f"{path.name} 格式不对：{e}") from None
    return parse(data, path.stem)


def dumps(s: Scenario) -> str:
    """写成 spec §6 的格式：name / note + [start] + 一串 [[steps]]。"""
    out = [f"name = {_string(s.name)}", f"note = {_string(s.note)}", "", "[start]"]
    st = s.start
    out += [
        f"memory = {_string(st.memory)}",
        f"time = {_string(st.time)}",
        f"nearby = {_value(list(st.nearby))}",
        f"strangers = {_value(st.strangers)}",
        f"place = {_string(st.place)}",
        f"scene = {_string(st.scene)}",
    ]
    for step in s.steps:
        out += ["", "[[steps]]"]
        if step.who:
            out.append(f"who = {_string(step.who)}")
        out.append(f"{step.action} = {_value(step.value)}")
        if step.expect:
            out.append(f"expect = {_string(step.expect)}")
        if step.wait:
            out.append(f"wait = {_value(float(step.wait))}")
    return "\n".join(out) + "\n"


def to_op(step: Step) -> dict | None:
    """一步 → `/sandbox/op` 的请求；offline / online 由回放引擎处理，返回 None。"""
    a, v = step.action, step.value
    if a == "say":
        return {"op": "say", "who": step.who, "text": v}
    if a in ("come", "leave"):
        return {"op": a, "who": v}
    if a == "strangers":
        return {"op": "strangers", "n": v}
    if a == "place":
        return {"op": "place", "name": v}
    if a in ("scene", "notice"):
        return {"op": a, "text": v}
    if a == "skip":
        return {"op": "skip", "seconds": float(v) if isinstance(v, (int, float)) else parse_duration(v)}
    if a == "time":
        return {"op": "time", "at": v}
    if a == "reflect":
        return {"op": "reflect"}
    if a in ("offline", "online"):
        return None
    raise ValueError(f"不认识的动作：{a}")
