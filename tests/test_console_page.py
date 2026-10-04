"""管理面板网页（spec 2026-10-01-console-redesign）：骨架、离线、脚本能解析、路由、主题变量。各页任务往这里加自己的页面测试。"""
import importlib.resources
import json
import re
import shutil
import subprocess

import pytest

STATIC = importlib.resources.files("skydango.console") / "static"
JS = ["common.js", "markdown.js", "inner.js", "brainlog.js", "chatlog.js", "sandbox.js", "livectl.js", "live.js", "scenarios.js", "emotenames.js", "labeling.js", "settings.js", "device.js"]


def bundle() -> str:  # 页面 + 样式 + 全部脚本，页面断言都对它做
    return "\n".join((STATIC / n).read_text(encoding="utf-8") for n in ["console.html", "console.css", *JS])


def test_skeleton():
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    for id_ in ("side", "runcard", "nav", "work", "dialog", "toasts", *(f"page-{p}" for p in ("sandbox", "live", "inner", "scenarios", "settings", "device"))):
        assert f'id="{id_}"' in page, id_
    assert "<script>" not in page  # 没有内联脚本
    for n in JS:
        assert f'src="console/static/{n}"' in page
    assert page.index('src="console/static/common.js"') < page.index('src="console/static/brainlog.js"') < page.index('src="console/static/sandbox.js"')
    assert page.index('src="console/static/chatlog.js"') < page.index('src="console/static/sandbox.js"')
    assert 'src="console/static/stage.js"' in page and page.index('src="console/static/stage.js"') < page.index('src="console/static/live.js"')
    assert page.index('src="console/static/chatlog.js"') < page.index('src="console/static/live.js"')
    assert "brain_trace" not in page and 'src="static/' not in page  # viewer 网页删了，画框脚本挪进 console/static/


def test_nav_has_pages_and_marks():
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    for p in ("sandbox", "live", "inner", "scenarios", "labeling", "settings", "device"):
        assert f'data-page="{p}"' in page and f'id="mark-{p}"' in page, p
    assert 'href="console/static/console.css"' in page


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
    assert 'mountBrainConsole($("sb-brain"),"sandbox/brain")' in b.replace(" ", "")
    for api in ("api/sandbox/start", "api/sandbox/stop", "api/sandbox/reset", "api/sandbox/info", "sandbox/state", "sandbox/op", "api/sandbox/save", "api/sandbox/record/new", "api/sandbox/replay"):
        assert api in b
    assert "会用 memory/ 覆盖沙盒记忆" in b and "Pages.sandbox" in b and "sandboxSummary" in b
    css = (STATIC / "console.css").read_text(encoding="utf-8")
    assert ".85fr 1.25fr 1fr" in css  # 左团子、中大脑、右聊天
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    assert page.index('class="pane sb-right"') < page.index('class="pane sb-brainpane"') < page.index('class="pane sb-chatpane"')


def _sandbox_js(expr: str):  # 在 node 里载入 common.js + sandbox.js（没有 document），算 expr
    node = shutil.which("node") or pytest.skip("没有 node")
    js = (f"require({json.dumps(str(STATIC / 'common.js'))});require({json.dumps(str(STATIC / 'chatlog.js'))});"
          f"require({json.dumps(str(STATIC / 'sandbox.js'))});console.log(JSON.stringify({expr}))")
    return json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)


def _chatlog_js(expr: str):  # 在 node 里载入 chatlog.js（没有 document），算 expr
    node = shutil.which("node") or pytest.skip("没有 node")
    js = f"const C=require({json.dumps(str(STATIC / 'chatlog.js'))});console.log(JSON.stringify({expr}))"
    return json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)


def test_chat_line_kinds():  # 聊天行的样式：反思 = 虚线框，别的事件 = 分隔线；团子说的（sb-me）靠左；被拦的删除线
    rows = [{"kind": "event", "text": "── 反思：心情 平常 → 开心 ──"}, {"kind": "event", "text": "── 开始反思 ──"},
            {"kind": "event", "text": "── 小明来了 ──"}, {"kind": "heard", "who": "小明"}, {"kind": "said", "who": "团子"},
            {"kind": "act", "who": "团子"}, {"kind": "blocked", "who": "团子"}, {"kind": "blocked", "who": ""}]
    assert _chatlog_js(f"{json.dumps(rows, ensure_ascii=False)}.map(C.lineKind)") == [
        "refl", "ev", "ev", "msg", "msg me", "act", "msg me blocked", "msg blocked"]


def test_sandbox_summary_text():  # 左栏卡片：「10月1日 11:28 · 开心 · 精神」，缺哪样就不写哪样
    t = "new Date(2026,9,1,11,28).getTime()/1000"
    assert _sandbox_js(f"[Sandbox.summaryText({t},'开心','精神'),Sandbox.summaryText({t},'',''),Sandbox.summaryText({t},null,'困')]") == [
        "10月1日 11:28 · 开心 · 精神", "10月1日 11:28", "10月1日 11:28 · 困"]
    assert _sandbox_js("[sandboxSummary(),typeof sandboxClock]") == [None, "function"]  # 沙盒没在跑：没有摘要


def test_scenarios_page():
    b = bundle()
    for id_ in ("sc-list", "sc-progress", "sc-reports", "sc-reader"):
        assert f'id="{id_}"' in b, id_
    for api in ("api/sandbox/scenarios", "api/sandbox/replay", "api/sandbox/replay/stop", "api/sandbox/reports"):
        assert api in b
    assert "回放时每一步会真的调 Claude，花额度" in b and "renderMarkdown(" in b and "Pages.scenarios" in b


def test_live_page():  # spec 2026-10-01-console-live-page：顶栏 + 三栏（团子 / 画面 + 大脑 / 聊天记录）+ 日志抽屉，不再用 iframe
    b = bundle()
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    live = page.split('id="page-live"', 1)[1].split('id="page-inner"', 1)[0]
    for id_ in ("launch", "opt-live", "opt-emotes", "opt-duration", "start", "problems", "banners", "lv-run", "live-chip", "rundir",
                "lv-log-btn", "lv-now", "cards", "lv-ctl", "lv-canvas", "lv-stage-none", "lv-legend", "lv-brain", "lv-brain-none",
                "lv-chat", "lv-drawer", "log"):
        assert f'id="{id_}"' in live, id_
    assert "<iframe" not in live and "live-frame" not in b
    assert live.index('id="lv-now"') < live.index('id="lv-canvas"') < live.index('id="lv-brain"') < live.index('id="lv-chat"')
    for api in ("api/run/start", "api/logs", "live/status", "live/snapshot", "live/chat", "live/brain", "api/inner", "api/orphan/stop"):
        assert api in b, api
    line = next(l for l in b.splitlines() if re.match(r"\s*const FACTS\s*=", l))
    assert '"身边的好友"' in line and '"心情"' in line
    assert "emotes_allowed" in b and "沙盒在跑，先下线" in b and "Pages.live" in b and "globalThis.LiveView" in b


def test_settings_and_device_pages():
    b = bundle()
    for id_ in ("settings-toc", "groups", "savebar", "save", "discard", "settings-error", "check", "checks", "device-note",
                "ime-read", "imes", "ime-msg"):
        assert f'id="{id_}"' in b, id_
    assert '"set-"+' in b.replace(" ", "") and 'replaceAll(".","-")' in b.replace(" ", "")
    for api in ("api/settings", "api/settings/test", "api/device", "api/device/ime"):
        assert api in b
    assert "Pages.settings" in b and "Pages.device" in b


def _brainlog_js(expr: str):  # 在 node 里载入 brainlog.js（没有 document），算 expr
    node = shutil.which("node") or pytest.skip("没有 node")
    js = f"const B=require({json.dumps(str(STATIC / 'brainlog.js'))});console.log(JSON.stringify({expr}))"
    return json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)


NL = chr(10)
PROMPT = NL.join(["[2026年10月1日（周四） 19:19:23] 事件：", "- 聊天  懒洋洋大王：「那我考考你」", "- 懒洋洋大王 来到身边",
                  "状态：聊天记录面板开 / 身边的好友：懒洋洋大王", "场景（4 秒前看的）：", "看不清"])


def test_brainlog_split_prompt():  # 事件一行一条拆出来，状态 / 场景整块折叠
    got = _brainlog_js(f"B.splitPrompt({json.dumps(PROMPT, ensure_ascii=False)})")
    assert got == {"head": "事件：", "events": ["聊天 懒洋洋大王：「那我考考你」", "懒洋洋大王 来到身边"],
                   "rest": NL.join(["状态：聊天记录面板开 / 身边的好友：懒洋洋大王", "场景（4 秒前看的）：", "看不清"])}
    beat = NL.join(["[2026年10月1日 19:16:53] 没有新事件（定时醒来）", "状态：安静"])
    assert _brainlog_js(f"B.splitPrompt({json.dumps(beat, ensure_ascii=False)})") == {"head": "没有新事件（定时醒来）", "events": [], "rest": "状态：安静"}
    assert _brainlog_js("B.splitPrompt('')") == {"head": "", "events": [], "rest": ""}


def test_brainlog_step_lines():  # 每种步骤一行：say 直接显示那句话，别的工具 名字 + 紧凑参数，返回出错标出来
    steps = [{"kind": "thinking", "text": "想想"}, {"kind": "text", "text": "等他回话"},
             {"kind": "tool", "name": "say", "input": {"text": "老登就是番茄"}},
             {"kind": "tool", "name": "emote", "input": {"name": "鞠躬", "n": 1}},
             {"kind": "result", "text": "已发送"}, {"kind": "result", "text": "被拒绝", "error": True}]
    got = _brainlog_js(f"{json.dumps(steps, ensure_ascii=False)}.map(B.stepLine)")
    assert got == [{"cls": "think", "mark": "…", "text": "想想"}, {"cls": "note", "mark": "»", "text": "等他回话"},
                   {"cls": "say", "mark": "▶", "text": "say “老登就是番茄”"},
                   {"cls": "tool", "mark": "▶", "text": 'emote {"name":"鞠躬","n":1}'},
                   {"cls": "ret", "mark": "↳", "text": "已发送"}, {"cls": "err", "mark": "↳", "text": "被拒绝"}]


def test_brainlog_turn_head_and_idle():  # 轮头：时间 · 原因 · 做了什么 · 耗时 · tokens；没做事的心跳算 idle
    t = {"start": 0, "reason": "events", "end": 1, "error": None, "tools": ["say"], "steps": [{"kind": "tool"}],
         "seconds": 2.94, "result": {"tokens": {"input": 4, "output": 80}}}
    head = _brainlog_js(f"B.turnHead({json.dumps(t)})")
    assert head.endswith(" · 新消息 / 事件 · say×1 · 2.9s · in 4 / out 80") and len(head.split(" · ")[0]) == 8
    idle = {"start": 0, "reason": "heartbeat", "end": 1, "error": None, "tools": [], "steps": [], "seconds": 1.7, "result": None}
    assert _brainlog_js(f"[B.isIdle({json.dumps(idle)}),B.isIdle({json.dumps(t)})]") == [True, False]
    live = {**t, "end": None, "tools": [], "steps": []}
    assert "进行中" in _brainlog_js(f"B.turnHead({json.dumps(live)})")


def test_brainlog_text_only():  # 大脑的话、工具返回可能带尖括号：只用 textContent，不拼 HTML
    src = (STATIC / "brainlog.js").read_text(encoding="utf-8")
    assert "innerHTML" not in src and "insertAdjacentHTML" not in src


def test_live_manual_control():
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    ctl = page.split('id="lv-ctl"', 1)[1].split("</section>", 1)[0]
    for id_ in ("lc-warn", "lc-none", "lc-busy", "lc-say-text", "lc-count", "lc-say-go", "lc-emote-name", "lc-emote-go", "lc-camera",
                "lc-steps", "lc-reset", "lc-around", "lc-pick", "lc-pick-tip", "lc-call", "lc-call-tip", "lc-track-name", "lc-track-pick", "lc-track-sec",
                "lc-track-go", "lc-stop", "lc-track-tip", "lc-panels", "lc-panel-read", "lc-panel-close", "lc-log"):
        assert f'id="{id_}"' in ctl, id_
    assert page.index('src="console/static/livectl.js"') < page.index('src="console/static/live.js"')
    js = (STATIC / "livectl.js").read_text(encoding="utf-8")
    for part in ("live/control/options", 'post("live/control"', "404", "3000", "5000", "ask(", "Stage.nameAt(", "LiveView.pick(", "globalThis.LiveCtl"):
        assert part in js, part
    assert "innerHTML" not in js


def test_live_control_line():  # 操作记录一行：做了什么 → 结果
    node = shutil.which("node") or pytest.skip("没有 node")
    js = (f"const C=require({json.dumps(str(STATIC / 'livectl.js'))});console.log(JSON.stringify(["
          "C.controlLine('say',{text:'hi'},{text:'ok'}),C.controlLine('camera',{action:'left',steps:2},{text:'ok'}),"
          "C.controlLine('track',{name:'ming',seconds:30},{text:'ok'}),C.controlLine('check_friend',{x:1,y:2},{text:'ok'}),C.controlLine('call',{},{text:'ok'})]))")
    out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out == ["说「hi」 → ok", "左转 ×2 → ok", "盯着ming（30 秒） → ok", "看人 (1, 2) → ok", "喊一声 → ok"]


def _fn_body(js: str, head: str) -> str:  # 取 live.js 里某个函数的函数体（到下一个顶层 function 为止）
    return js.split(head, 1)[1].split("\nfunction ", 1)[0]


def test_live_agent_mode_has_no_manual_control():  # 普通 Agent 不挂 control：别一直写「身体还没准备好」、每 3 秒白问
    live = (STATIC / "live.js").read_text(encoding="utf-8")
    ctl = (STATIC / "livectl.js").read_text(encoding="utf-8")
    assert "LiveCtl.start(brain)" in _fn_body(live, "function begin(")
    assert "普通 Agent 没有手动控制" in ctl


def test_live_wake_resets_pause():  # 上次暂停着停下的：下次叫醒画面照常拉
    begin = _fn_body((STATIC / "live.js").read_text(encoding="utf-8"), "function begin(")
    assert "L.paused = false" in begin and '$("lv-pause").textContent = "暂停"' in begin


def test_live_crash_drawer_loads_log():  # 打开页面时已经出错停下：抽屉自动打开时把日志也拉一次（不然是空的）
    live = (STATIC / "live.js").read_text(encoding="utf-8")
    assert "pullLogs(true)" in _fn_body(live, "function render(")
    assert "force" in _fn_body(live, "async function pullLogs(")
