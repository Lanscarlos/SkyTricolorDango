/* 真机团子页（spec 2026-10-01-console-redesign §4.2）：停着 = 启动表单 + 预检问题；跑着 = 左实时画面（iframe live/）、右状态卡片 + 日志；
 * 强杀 / 孤儿团子横幅。「停止」只在左栏卡片上（common.js），这页不放。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const L = {launchLoaded: false, framed: false, info: null, logNext: 0, follow: true, timer: 0, logTimer: 0, startProblems: false};
const CARDS=["身边的好友","陌生人","互动请求","开着的面板","牵着手","正在做","刚说过","场合","心情","精力","聊天面板","画面"];
const CHIP = {idle: "没在跑", starting: "正在启动", running: "运行中", stopping: "在收尾", crashed: "出错停下了", exited: "没在跑"};

function liveWarn() { $("live-warn").hidden = !$("opt-live").checked; }

function render() {
  const st = S.state; if (!st) return;
  const run = st.run, busy = BUSY.includes(run.state), isSb = run.kind === "sandbox";
  const mine = busy && !isSb;                       // 真机团子在跑（含启动中 / 收尾中）
  const chip = $("live-chip");
  chip.className = "chip" + (mine && run.state === "running" ? " live dot" : run.state === "crashed" && !isSb ? " bad" : "");
  chip.textContent = isSb ? (busy ? "沙盒在跑" : "没在跑") : (CHIP[run.state] || run.state);
  // 启动选项：第一次按服务器记住的填，之后用户自己改
  if (!L.launchLoaded && st.launch) {
    const l = st.launch;
    document.querySelector(`input[name=mode][value=${l.brain ? "brain" : "agent"}]`).checked = true;
    $("opt-live").checked = l.live; $("opt-emotes").checked = l.emotes; $("opt-duration").value = l.duration > 0 ? l.duration : "";
    L.launchLoaded = true; liveWarn();
  }
  $("launch").hidden = mine;
  const off = busy && isSb;                         // 沙盒在跑：表单整个置灰
  for (const x of $("launch").querySelectorAll("input,button")) x.disabled = off;
  if (!st.emotes_allowed) { $("opt-emotes").checked = false; $("opt-emotes").disabled = true; }  // config.toml 关了动作：只能关不能开
  $("emotes-note").textContent = st.emotes_allowed ? "" : "config.toml 里关掉了";
  $("start").textContent = off ? "沙盒在跑，先下线" : "叫醒团子";
  if (mine) L.startProblems = false;
  if (!L.startProblems) problemList($("problems"), mine ? [] : st.problems || []);
  // 横幅：强杀 / 上次留下的团子
  const banners = $("banners"); banners.textContent = "";
  if (run.forced && !busy && !isSb) {
    const b = el("div", "banner bad");
    b.append(el("span", "", "强制结束了，轮盘可能没换回，请用 python -m skydango emotes wheel 检查。")); banners.append(b);
  }
  if (st.orphan) {
    const b = el("div", "banner warn");
    b.append(el("span", "", "上次留下的团子还在运行（占着子进程端口），先让它退出再叫醒新的。"));
    const btn = el("button", "btn sm", "让它退出");
    btn.type = "button";
    btn.onclick = async () => { btn.disabled = true; await post("api/orphan/stop"); refresh(); };
    b.append(btn); banners.append(b);
  }
  $("rundir").textContent = mine && run.run_dir ? "运行目录 " + run.run_dir : "";
  // 跑着：两栏；iframe 进入运行时才设 src、离开时移除
  $("watch").hidden = !mine;
  const framed = mine && (run.state === "running" || run.state === "stopping");
  if (framed && !L.framed) { $("live-frame").src = "live/"; L.framed = true; }
  if (!framed && L.framed) { $("live-frame").removeAttribute("src"); L.framed = false; L.info = null; }
  $("live-frame").hidden = !framed; $("live-wait").hidden = framed;
  if (mine && !L.info) renderCards();
}

function renderCards() {
  const dl = $("cards"); dl.textContent = ""; const info = L.info;
  if (!info) {
    const d = el("div", "card lone"); d.append(el("dt", "", "团子的状态"), el("dd", "", "叫醒之后这里显示身边有谁、在做什么。")); dl.append(d); return;
  }
  for (const key of CARDS) {
    if (!(key in info)) continue;
    const v = info[key], d = el("div", "card"), dd = el("dd");
    if (Array.isArray(v)) {
      if (key === "身边的好友") dd.append(el("span", "count", String(v.length)), "个");
      const ul = el("ul"); for (const x of v) ul.append(el("li", "", String(x))); dd.append(ul);
    } else dd.textContent = String(v);
    d.append(el("dt", "", key), dd); dl.append(d);
  }
}
async function pullStatus() {
  const run = S.state && S.state.run;
  if (!run || run.state !== "running" || run.kind === "sandbox") return;
  try { const r = await fetch("live/status", {cache: "no-store"}); if (r.ok) { L.info = (await r.json()).info; renderCards(); } } catch (e) {}
}

async function pullLogs() {
  const run = S.state && S.state.run;
  if (!run || run.kind === "sandbox" || !BUSY.includes(run.state)) return;
  try {
    const r = await getJSON(`api/logs?after=${L.logNext}`), box = $("log");
    if (r.next < L.logNext) { L.logNext = 0; box.textContent = ""; return; }   // 重新启动过：从头来
    for (const line of r.lines) box.append(el("span", /Traceback|ERROR|错误/.test(line) ? "e" : /WARNING/.test(line) ? "w" : "", line + "\n"));
    while (box.childNodes.length > 1000) box.firstChild.remove();
    L.logNext = r.next; if (r.lines.length && L.follow) box.scrollTop = box.scrollHeight;
  } catch (e) {}
}

async function start(e) {
  e.preventDefault();
  const d = $("opt-duration").value.trim();
  const body = {brain: document.querySelector("input[name=mode]:checked").value === "brain", live: $("opt-live").checked,
    emotes: $("opt-emotes").checked, duration: d === "" ? 0 : Number(d)};
  $("start").disabled = true;
  const r = await post("api/run/start", body);
  if (!r.data.ok) { L.startProblems = true; problemList($("problems"), r.data.problems || [r.data.text || r.data.error || "启动失败"]); }
  else { L.startProblems = false; L.logNext = 0; $("log").textContent = ""; }
  $("start").disabled = false;
  await refresh();
}

Pages.live = {
  init() {
    $("opt-live").onchange = liveWarn;
    $("launch").onsubmit = start;
    $("log").addEventListener("scroll", () => { const b = $("log"); L.follow = b.scrollTop + b.clientHeight >= b.scrollHeight - 8; });
    onState(render);
  },
  show() {
    L.startProblems = false; render(); pullLogs(); pullStatus();
    clearInterval(L.timer); clearInterval(L.logTimer);
    L.timer = setInterval(pullStatus, 2000); L.logTimer = setInterval(pullLogs, 1000);
  },
  hide() { clearInterval(L.timer); clearInterval(L.logTimer); },
};
})();
