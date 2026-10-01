"""管理面板网页（spec 2026-10-01-console-redesign）：骨架、离线、脚本能解析、路由、主题变量。各页任务往这里加自己的页面测试。"""
import importlib.resources
import json
import re
import shutil
import subprocess

import pytest

STATIC = importlib.resources.files("skydango.console") / "static"
JS = ["common.js", "markdown.js", "inner.js", "sandbox.js", "live.js", "scenarios.js", "settings.js", "device.js"]


def bundle() -> str:  # 页面 + 样式 + 全部脚本，页面断言都对它做
    return "\n".join((STATIC / n).read_text(encoding="utf-8") for n in ["console.html", "console.css", *JS])


def test_skeleton():
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    for id_ in ("side", "runcard", "nav", "work", "dialog", "toasts", *(f"page-{p}" for p in ("sandbox", "live", "inner", "scenarios", "settings", "device"))):
        assert f'id="{id_}"' in page, id_
    assert "<script>" not in page  # 没有内联脚本
    for n in JS:
        assert f'src="console/static/{n}"' in page
    assert page.index('src="static/brain_trace.js"') < page.index('src="console/static/common.js"') < page.index('src="console/static/sandbox.js"')


def test_nav_has_pages_and_marks():
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    for p in ("sandbox", "live", "inner", "scenarios", "settings", "device"):
        assert f'data-page="{p}"' in page and f'id="mark-{p}"' in page, p
    assert page.index('href="static/brain_trace.css"') < page.index('href="console/static/console.css"')


def test_offline_relative_and_no_native_dialogs():
    b = bundle()
    assert "http://" not in b and "https://" not in b
    assert not re.search(r"""fetch\(\s*[`"']/""", b)
    assert '"X-Skydango":"1"' in b.replace(" ", "")
    assert not re.search(r"\b(confirm|prompt|alert)\(", b)


def test_scripts_parse():
    node = shutil.which("node") or pytest.skip("没有 node")
    for n in JS:
        assert subprocess.run([node, "--check", str(STATIC / n)]).returncode == 0, n


def test_routes():  # 在 node 里跑 common.js 的 parseHash（common.js 末尾：typeof module!=="undefined" 时 module.exports={parseHash}，且顶层不碰 document）
    node = shutil.which("node") or pytest.skip("没有 node")
    js = f"const {{parseHash}}=require({json.dumps(str(STATIC / 'common.js'))});console.log(JSON.stringify(['#settings/secret.llm','#overview','#nope','','#inner'].map(parseHash)))"
    out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out == [{"page": "settings", "arg": "secret.llm"}, {"page": "live", "arg": None}, {"page": "sandbox", "arg": None},
                   {"page": "sandbox", "arg": None}, {"page": "inner", "arg": None}]


def test_theme_variables():
    css = (STATIC / "console.css").read_text(encoding="utf-8")
    for v in ("--paper:#f6f1e7", "--card:#fffdf8", "--ink:#3b332b", "--sakura:#e58aab", "--matcha:#7fae7d", "--brick:#c4573f"):
        assert v in css.replace(" ", "")


def test_colors_only_in_root():  # 约束：颜色只在 :root 定义，别处只用 var(--…)
    css = (STATIC / "console.css").read_text(encoding="utf-8")
    rest = re.sub(r":root\s*\{[^}]*\}", "", css)
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b", rest)


def test_common_api_exposed():  # 后面各页直接用这些全局名字（没有 document 时也要能载入）
    node = shutil.which("node") or pytest.skip("没有 node")
    names = ["$", "el", "getJSON", "post", "pad2", "hhmm", "dayTime", "span", "fmtUptime", "ask", "toast", "problemList",
             "Pages", "go", "parseHash", "S", "onState", "BUSY", "refresh"]
    js = (f"require({json.dumps(str(STATIC / 'common.js'))});"
          f"console.log(JSON.stringify({json.dumps(names)}.filter(n=>globalThis[n]===undefined)))")
    assert json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, check=True).stdout) == []


def test_time_helpers():  # 照搬旧实现
    node = shutil.which("node") or pytest.skip("没有 node")
    js = (f"require({json.dumps(str(STATIC / 'common.js'))});"
          "console.log(JSON.stringify([pad2(3),span(30),span(600),span(7200),span(86400*3),fmtUptime(5),fmtUptime(125),fmtUptime(3725),fmtUptime(null)]))")
    out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out == ["03", "不到 1 分钟", "10 分钟", "2 小时", "3 天", "5 秒", "2 分钟", "1 小时 2 分", ""]


def test_inner_page():
    b = bundle()
    for id_ in ("inner-source-toggle", "inner-source", "inner-now", "inner-curve", "inner-log", "inner-persona", "inner-cards", "inner-days", "inner-refresh", "inner-range", "inner-changed"):
        assert f'id="{id_}"' in b, id_
    for s in ("api/inner", "api/inner/forget", "Inner.renderNow", "Pages.inner", "IN.lastState"):
        assert s in b
    for text in ("（没开反思）", "实时取不到，显示的是上次保存的", "团子正在启动 / 停止，稍等再删", "只看有改动的", "删了团子就不会再用它（不能撤销）"):
        assert text in b


def test_inner_reload_decision():  # onState 每秒都来：连续相同的 running 不重读；5 秒定时器才重读
    node = shutil.which("node") or pytest.skip("没有 node")
    js = (
        "globalThis.Pages={};globalThis.onState=()=>{};"
        f"require({json.dumps(str(STATIC / 'inner.js'))});"
        "const f=Inner.innerShouldReload;"
        "console.log(JSON.stringify([f('running','running',false,true),f('running','running',true,true),"
        "f('running','running',true,false),f('idle','running',false,true),f(null,'idle',false,false),f('idle','idle',true,true)]))"
    )
    out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out == [False, True, False, True, True, False]


def test_sandbox_page():
    b = bundle()
    for id_ in ("sb-head", "sb-start", "sb-problems", "sb-clock", "sb-replay-bar", "sb-chat", "sb-say", "sb-brain", "sb-now", "sb-nearby", "sb-scene", "sb-rec", "sb-save", "sb-reset"):
        assert f'id="{id_}"' in b, id_
    assert 'mountBrainTrace($("sb-brain"),"sandbox/brain")' in b.replace(" ", "")
    for api in ("api/sandbox/start", "api/sandbox/stop", "api/sandbox/reset", "api/sandbox/info", "sandbox/state", "sandbox/op", "api/sandbox/save", "api/sandbox/record/new", "api/sandbox/replay"):
        assert api in b
    assert "会用 memory/ 覆盖沙盒记忆" in b and "Pages.sandbox" in b and "sandboxSummary" in b
    assert "1.15fr 1.1fr .85fr" in (STATIC / "console.css").read_text(encoding="utf-8")


def _sandbox_js(expr: str):  # 在 node 里载入 common.js + sandbox.js（没有 document），算 expr
    node = shutil.which("node") or pytest.skip("没有 node")
    js = (f"require({json.dumps(str(STATIC / 'common.js'))});require({json.dumps(str(STATIC / 'sandbox.js'))});"
          f"console.log(JSON.stringify({expr}))")
    return json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)


def test_sandbox_line_kinds():  # 聊天行的样式：反思 = 虚线框，别的事件 = 分隔线；团子说的靠右；被拦的删除线
    rows = [{"kind": "event", "text": "── 反思：心情 平常 → 开心 ──"}, {"kind": "event", "text": "── 开始反思 ──"},
            {"kind": "event", "text": "── 小明来了 ──"}, {"kind": "heard", "who": "小明"}, {"kind": "said", "who": "团子"},
            {"kind": "act", "who": "团子"}, {"kind": "blocked", "who": "团子"}, {"kind": "blocked", "who": ""}]
    assert _sandbox_js(f"{json.dumps(rows, ensure_ascii=False)}.map(Sandbox.lineKind)") == [
        "refl", "ev", "ev", "msg", "msg me", "act", "msg me blocked", "msg blocked"]


def test_sandbox_summary_text():  # 左栏卡片：「10月1日 11:28 · 开心 · 精神」，缺哪样就不写哪样
    t = "new Date(2026,9,1,11,28).getTime()/1000"
    assert _sandbox_js(f"[Sandbox.summaryText({t},'开心','精神'),Sandbox.summaryText({t},'',''),Sandbox.summaryText({t},null,'困')]") == [
        "10月1日 11:28 · 开心 · 精神", "10月1日 11:28", "10月1日 11:28 · 困"]
    assert _sandbox_js("[sandboxSummary(),typeof sandboxClock]") == [None, "function"]  # 沙盒没在跑：没有摘要
