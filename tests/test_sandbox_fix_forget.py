"""大脑沙盒审查意见的回归测试：沙盒停着时删性格条目用沙盒时间（审查 8）。"""


from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from test_brain_tools import FakeBody
from test_console_sandbox import make, post
from test_console_server import no_registry, request, upstream  # noqa: F401

from skydango.console.sandbox_view import reset
from skydango.sandbox.clock import save


# ---- 8. 沙盒停着时删性格条目：流水账那一条用沙盒时间 ----
def test_offline_sandbox_forget_logs_sandbox_time(tmp_path, upstream):  # noqa: F811
    from skydango.inner.persona import Persona, Trait
    from skydango.inner.store import InnerStore

    s = make(tmp_path, upstream)
    try:
        inner = tmp_path / "sandbox" / "memory" / "inner"
        InnerStore(inner).write_persona(Persona(catchphrases=[Trait("害", since=time.time())]))
        later = time.time() + 3 * 86400
        save(tmp_path / "sandbox" / "clock.json", later)
        assert post(s, "api/inner/forget", {"kind": "catchphrase", "text": "害", "source": "sandbox"}) == (200, {"ok": True})
        (row,) = [json.loads(line) for line in (inner / "mind_log.jsonl").read_text("utf-8").splitlines()]
        assert row["kind"] == "forget" and row["t"] >= later
        assert request(s.url + "api/inner?source=sandbox")[0] == 200
    finally:
        s.stop()
