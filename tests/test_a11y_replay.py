"""本机录像回放：`a11y --watch N --save tmp/a11y-live/raw.jsonl` 存下的原始快照逐行喂给读聊天。

录像里有好友昵称和聊天原话，只在本机（tmp/ 不进 git）；没有录像、没有 config.toml / 记忆目录就跳过。
`python -m pytest -q -s tests/test_a11y_replay.py` 能看到读到了哪些消息。
"""

from pathlib import Path

import pytest

from skydango.chat.a11yreader import A11yChatReader
from skydango.chat.tracker import SelfFilter
from skydango.config import ChatConfig, OcrConfig
from skydango.device.a11y import parse_line
from skydango.vision.a11yui import bubble_key

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "tmp" / "a11y-live" / "raw.jsonl"


def _friends() -> list[str]:
    from skydango.chat.memory import MemoryStore
    from skydango.config import load_config

    if not (ROOT / "config.toml").exists():
        pytest.skip("没有 config.toml")
    cfg = load_config(ROOT / "config.toml")
    mem = Path(cfg.reply.memory_dir)
    if not mem.is_absolute():
        mem = ROOT / mem
    if not mem.is_dir():
        pytest.skip(f"没有记忆目录 {mem}")
    return MemoryStore(mem).friend_names()


def test_replay_local_recording():
    if not RAW.exists():
        pytest.skip(f"没有本机录像 {RAW}")
    friends = _friends()
    holder = [None]
    reader = A11yChatReader(lambda: holder[0], lambda: friends, ChatConfig(), OcrConfig(), SelfFilter(30, 0.8))
    got = []
    t0 = None
    for i, line in enumerate(RAW.read_bytes().splitlines()):
        snap = parse_line(line, i * 0.2)
        if snap is None:
            continue
        if t0 is None:
            t0 = snap.device_ms
        now = (snap.device_ms - t0) / 1000 if snap.device_ms else i * 0.2  # 设备时间（毫秒），没有就按 0.2 秒一份
        holder[0] = snap
        for m in reader.read(None, now):
            got.append(m)
            print(f"{now:8.2f} [{m.source}] {m.speaker}：{m.text}")
    last: dict[tuple[str, str], float] = {}
    for m in got:
        assert m.speaker
        assert m.source in ("panel", "bubble")
        k = (m.speaker, bubble_key(m.text))
        assert k not in last or m.seen_at - last[k] > 20, f"20 秒内重复报：{m.speaker}：{m.text}"
        last[k] = m.seen_at
