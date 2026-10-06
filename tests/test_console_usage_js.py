"""「模型用量」卡片（console/static/usage.js，spec 2026-10-06-model-usage §6.2）：格式化、卡片内容、拉取失败时留着上一次。"""

import importlib.resources
import json
import shutil
import subprocess

import pytest

STATIC = importlib.resources.files("skydango.console") / "static"

DATA = {
    "ok": True, "source": "live", "started": 0,
    "run": {"rows": [
        {"use": "brain", "label": "大脑", "model": "deepseek/deepseek-flash", "backup": False, "calls": 41, "fails": 2,
         "input": 512000, "output": 9100, "cache_read": 460000, "cache_write": 0, "cost": 0.091, "est": False},
        {"use": "eyes", "label": "眼睛", "model": "claude/haiku", "backup": True, "calls": 3, "fails": 0,
         "input": 950, "output": 40, "cache_read": 0, "cache_write": 0, "cost": None, "est": False},
        {"use": "reply", "label": "回复", "model": "other/m", "backup": False, "calls": 1, "fails": 0,
         "input": 10, "output": 1, "cache_read": 0, "cache_write": 0, "cost": None, "est": False},
    ], "cost": 0.091, "est": True},
    "today": {"rows": [], "cost": 0.85, "est": True},
    "providers": [
        {"id": "claude", "kind": "claude-code", "closed": "额度 / 余额用完", "balance": None,
         "rate": {"status": "rejected", "type": "five_hour", "windows": {"five_hour": {"utilization": 0.62, "resets_at": 1700000000}}, "at": 1}},
        {"id": "deepseek", "kind": "openai", "closed": None, "rate": None,
         "balance": {"available": True, "items": [{"currency": "CNY", "total": 12.34, "granted": 1.0, "topped_up": 11.34}], "at": 1, "error": None}},
        {"id": "other", "kind": "openai", "closed": None, "rate": None, "balance": None},
    ],
    "uses": [{"use": "eyes", "label": "眼睛", "main": "deepseek/deepseek-flash", "current": "claude/haiku", "backup": True, "disabled": None},
             {"use": "wardrobe", "label": "装扮描述", "main": "x/y", "current": None, "backup": False, "disabled": "看不了图"}],
}


def _node(js: str):
    node = shutil.which("node") or pytest.skip("没有 node")
    src = f"const U=require({json.dumps(str(STATIC / 'usage.js'))});(async()=>{{const out=await (async()=>{{{js}}})();process.stdout.write(JSON.stringify(out))}})()"
    return json.loads(subprocess.run([node, "-e", src], capture_output=True, text=True, encoding="utf-8", check=True).stdout)


def test_fmt():
    assert _node("return [950, 1000, 12300, 4560000, 0].map(U.fmtTokens)") == ["950", "1k", "12.3k", "4.56M", "0"]
    assert _node("return [null, 0, 0.0042, 1.5].map(U.fmtYuan)") == ["—", "¥0", "¥0.004", "¥1.50"]


def test_usage_card_rows():
    m = _node(f"return U.usageModel({json.dumps(DATA)})")
    claude, ds, other = m["providers"]
    assert claude["state"] == "关了：额度 / 余额用完" and claude["bad"] is True
    assert ds["state"] == "开着" and ds["bad"] is False and ds["money"] == "余额 ¥12.34"
    assert "赠送 ¥1.00" in ds["moneyTitle"] and "充值 ¥11.34" in ds["moneyTitle"]
    assert other["money"] == "" and other["quota"] is None
    rows = {r["title"]: r for r in m["rows"]}
    assert rows["大脑"]["model"] == "deepseek-flash" and rows["大脑"]["backup"] is False
    assert rows["大脑"]["calls"] == "41 次（失败 2）" and rows["大脑"]["cost"] == "¥0.091"
    assert rows["大脑"]["tokens"] == "入 512k（命中 460k） 出 9.1k"
    assert rows["眼睛"]["backup"] is True and rows["眼睛"]["cost"] == "订阅"
    assert rows["回复"]["cost"] == "没填单价"
    assert m["scope"] == "这次运行" and m["totals"] == {"run": "¥0.091", "today": "¥0.85", "est": True}
    assert m["notes"] == ["眼睛现在走备用 claude/haiku", "装扮描述停用：看不了图"]


def test_usage_card_rate():
    m = _node(f"return U.usageModel({json.dumps(DATA)})")
    q = m["providers"][0]["quota"]
    assert q["frac"] == 0.62 and q["text"].startswith("5 小时 62%") and "重置" in q["text"]
    no_util = {**DATA, "providers": [{"id": "claude", "kind": "claude-code", "closed": None, "balance": None,
                                       "rate": {"status": "allowed_warning", "at": 1}}]}
    q = _node(f"return U.usageModel({json.dumps(no_util)})")["providers"][0]["quota"]
    assert q == {"text": "额度快用完", "frac": None}


def test_usage_card_stopped_shows_today():
    stopped = {**DATA, "source": None, "run": None, "today": DATA["run"]}
    m = _node(f"return U.usageModel({json.dumps(stopped)})")
    assert m["scope"] == "今天" and len(m["rows"]) == 3 and m["totals"]["run"] is None
    sandbox = {**DATA, "source": "sandbox"}
    assert _node(f"return U.usageModel({json.dumps(sandbox)}).scope") == "这次沙盒"


def test_usage_card_falls_back():   # Review Focus 5：拉取失败时留着上一次的内容
    got = _node(f"""
      const seen = []; let n = 0;
      const answers = [{json.dumps(DATA)}, null, {{ok: false, text: "x"}}];
      const m = U.mountUsage({{}}, "api/usage", {{
        every: 0, render: (c, data) => seen.push(data.source),
        getJSON: async () => {{ const a = answers[n++]; if (a === null) throw new Error("503"); return a; }} }});
      await m.tick(); await m.tick(); await m.tick(); m.stop();
      return seen;
    """)
    assert got == ["live"]
