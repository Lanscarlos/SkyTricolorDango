/* 真机团子页（spec 2026-10-01-console-live-page）：顶栏（停着 = 启动选项，跑着 = 运行信息）+ 日志抽屉 + 三栏：
 * 左 现在（Inner.renderNow，api/inner）+ 身边和状态（live/status）+ 手动控制（livectl.js）；
 * 中 画面（live/snapshot 长轮询，stage.js 画框）+ 大脑控制台（brainlog.js，live/brain）；右 聊天记录（live/chat 长轮询，chatlog.js）。
 * 画面和聊天只在这一页开着、团子在跑时拉；停下后三栏留着最后的内容，下次叫醒才清空。
 * 给 livectl.js：globalThis.LiveView（最新快照、在画面上点一下、画十字）。「停止」只在左栏卡片上（common.js）。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const L = {launchLoaded: false, startProblems: false, on: false, gen: 0, shown: false, kind: "", state: "",
  seq: 0, snap: null, img: null, paused: false, boxes: true, mark: null, hover: null, pick: null, times: [],
  chatV: 0, brain: true, trace: null, info: null, innerOk: false, logNext: 0, follow: true, drain: false, timers: []};
const FACTS = ["身边的好友", "陌生人", "互动请求", "开着的面板", "牵着手", "正在做", "刚说过", "场合", "聊天面板", "心情", "精力"];
const WITH_INNER = ["心情", "精力"];  // 「现在」里已经有了，内心层开着时不重复
const CHIP = {idle: "没在跑", starting: "正在启动", running: "运行中", stopping: "在收尾", crashed: "出错停下了", exited: "没在跑"};

function liveWarn() { $("live-warn").hidden = !$("opt-live").checked; }
function sleep(ms) { return new Promise(ok => setTimeout(ok, ms)); }
function alive(gen) { return gen === L.gen && L.on && L.shown; }

/* ---- 顶栏、横幅（每秒随 /api/state） ---- */
function render() {
  const st = S.state; if (!st) return;
  const run = st.run, busy = BUSY.includes(run.state), isSb = run.kind === "sandbox", mine = busy && !isSb;
  if (!L.launchLoaded && st.launch) {  // 启动选项：第一次按服务器记住的填，之后用户自己改
    const l = st.launch;
    document.querySelector(`input[name=mode][value=${l.brain ? "brain" : "agent"}]`).checked = true;
    $("opt-live").checked = l.live; $("opt-emotes").checked = l.emotes; $("opt-duration").value = l.duration > 0 ? l.duration : "";
    L.launchLoaded = true; liveWarn();
  }
  $("launch").hidden = mine; $("lv-run").hidden = !mine;
  const off = busy && isSb;  // 沙盒在跑：表单整个置灰
  for (const x of $("launch").querySelectorAll("input,button")) x.disabled = off;
  if (!st.emotes_allowed) { $("opt-emotes").checked = false; $("opt-emotes").disabled = true; }  // config.toml 关了动作：只能关不能开
  $("emotes-note").textContent = st.emotes_allowed ? "" : "config.toml 里关掉了";
  $("start").textContent = off ? "沙盒在跑，先下线" : "叫醒团子";
  if (mine) L.startProblems = false;
  if (!L.startProblems) problemList($("problems"), mine ? [] : st.problems || []);
  const chip = $("live-chip");
  chip.className = "chip" + (run.state === "running" ? " live dot" : "");
  chip.textContent = CHIP[run.state] || run.state;
  const o = run.options || {};
  $("lv-mode").textContent = [run.source === "terminal" ? "终端起的" : "", o.brain === false ? "普通 Agent" : "统管大脑", o.live ? "真的发送" : "只打印",
    run.uptime != null ? `已运行 ${fmtUptime(run.uptime)}` : ""].filter(Boolean).join(" · ");
  $("rundir").textContent = mine && run.run_dir ? run.run_dir : "";
  renderBanners(st, run, busy, isSb);
  if (!isSb && run.state === "crashed" && L.state !== "crashed") {  // 出错停下：日志自己打开；页面是停下之后才打开的，日志还没拉过，拉一次
    openDrawer(true); if (!L.logNext) pullLogs(true);
  }
  L.state = isSb ? "" : run.state;
  const on = !isSb && (run.state === "running" || run.state === "stopping");
  if (on && !L.on) begin(o.brain !== false);
  else if (!on && L.on) end();
}
function renderBanners(st, run, busy, isSb) {
  const banners = $("banners"); banners.textContent = "";
  if (run.forced && !busy && !isSb) {
    const b = el("div", "banner bad");
    b.append(el("span", "", "强制结束了，轮盘可能没换回，请用 python -m skydango emotes wheel 检查。")); banners.append(b);
  }
}

/* ---- 进入 / 离开运行：清空上一次的内容、挂大脑、开手动控制 ---- */
function begin(brain) {
  L.on = true; L.brain = brain; L.gen++; L.seq = 0; L.paused = false; $("lv-pause").textContent = "暂停"; L.snap = null; L.chatV = 0; L.info = null; L.mark = null; L.times = [];
  L.logNext = 0; $("log").textContent = "";  // 新的一次运行（含重新接管终端起的团子）：日志从头拉，旧那次的不留
  const chat = $("lv-chat"); chat.textContent = ""; chat.append(el("p", "none", "还没有聊天"));
  $("lv-stage-none").hidden = false; $("lv-stage-none").textContent = "画面出来之后显示在这里";
  const ctx = $("lv-canvas").getContext("2d"); ctx.clearRect(0, 0, $("lv-canvas").width, $("lv-canvas").height);
  renderFacts(); renderNow(null);
  if (L.trace) { L.trace.stop(); L.trace = null; }
  const box = $("lv-brain"), none = $("lv-brain-none"); box.textContent = ""; box.hidden = true;
  if (brain && typeof mountBrainConsole === "function") { none.hidden = true; L.trace = mountBrainConsole(box, "live/brain"); }
  else { none.hidden = false; none.textContent = "普通 Agent 没有大脑"; }
  if (globalThis.LiveCtl) LiveCtl.start(brain);
  if (L.shown) loops();
}
function end() {  // 停下：内容留着，只停掉拉取
  L.on = false; L.gen++;
  if (L.trace) { L.trace.stop(); L.trace = null; }
  if (globalThis.LiveCtl) LiveCtl.stop();
  LiveView.pick(null);
  $("lv-fps").textContent = "停了"; $("lv-fps").className = "lv-fps";
}
function loops() { L.gen++; const g = L.gen; stageLoop(g); chatLoop(g); pullStatus(); pullInner(); }

/* ---- 画面 ---- */
async function stageLoop(gen) {
  if (typeof Stage === "undefined") { $("lv-stage-none").textContent = "画面脚本加载不了（console/static/stage.js）"; return; }
  while (alive(gen) && !L.paused) {
    try {
      const r = await fetch(`live/snapshot?after=${L.seq}`, {cache: "no-store"});
      if (!alive(gen)) return;
      if (r.status === 204) continue;
      if (!r.ok) throw new Error(r.status);
      const s = await r.json(); if (!alive(gen)) return;
      const img = new Image();
      await new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; img.src = "data:image/jpeg;base64," + s.image; });
      if (!alive(gen)) return;
      L.seq = s.seq; L.snap = s; L.img = img; redraw(); $("lv-stage-none").hidden = true;
      const now = performance.now(); L.times.push(now); L.times = L.times.filter(t => now - t < 2000);
      $("lv-fps").textContent = `${(L.times.length / 2).toFixed(1)} 帧/秒 · ${s.width}×${s.height}`; $("lv-fps").className = "lv-fps";
    } catch (e) {
      if (!alive(gen)) return;
      $("lv-fps").textContent = "连不上"; $("lv-fps").className = "lv-fps bad"; await sleep(1000);
    }
  }
}
function redraw() { if (L.snap && L.img) Stage.draw($("lv-canvas"), L.img, L.snap, {boxes: L.boxes, mark: L.mark, hover: L.hover}); }
function canvasPoint(e) { const c = $("lv-canvas"), r = c.getBoundingClientRect(); return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height]; }

/* ---- 聊天记录 ---- */
async function chatLoop(gen) {
  while (alive(gen)) {
    try {
      const r = await fetch(`live/chat?after=${L.chatV}&wait=2`, {cache: "no-store"});
      if (!alive(gen)) return;
      if (r.status === 404) {  // 普通 Agent 没有聊天记录；大脑模式是身体还没建好（端口先通、聊天记录后挂），或者可视化开在局域网
        if (!L.brain) { const box = $("lv-chat"); box.textContent = ""; box.append(el("p", "none", "普通 Agent 没有聊天记录")); return; }
        await sleep(3000); continue;
      }
      if (!r.ok) throw new Error(r.status);
      const d = await r.json(); if (!alive(gen)) return;
      if (d.v < L.chatV) { $("lv-chat").textContent = ""; }  // 团子重启过：从头来
      Chat.append($("lv-chat"), d.lines || []); L.chatV = d.v;
    } catch (e) { if (!alive(gen)) return; await sleep(1000); }
  }
}

/* ---- 左栏：现在 + 身边和状态 ---- */
function renderNow(d) {
  const box = $("lv-now");
  if (d && d.now) Inner.renderNow(box, d.now, d);
  else {
    box.textContent = ""; box.append(el("h3", "", "现在"));
    const b = el("div", "pane-body"); b.append(el("p", "none", L.on ? "（内心层没开，或者还没取到）" : "叫醒之后这里是团子现在的样子")); box.append(b);
  }
  const h = box.querySelector("h3"), link = el("button", "linkish sb-inner-link", "完整内心 →");
  link.type = "button"; link.onclick = () => { Inner.setSource("dango"); go("inner"); };
  h.append(el("span", "grow"), link);
}
async function pullInner() {
  if (!L.on) return;
  try { const d = await getJSON("api/inner"); if (!L.on) return; L.innerOk = !!(d && d.now); renderNow(d); }
  catch (e) { /* 下次再取 */ }
}
function renderFacts() {
  const dl = $("cards"); dl.textContent = ""; const info = L.info;
  if (!info) { dl.append(el("dt", "", "状态"), el("dd", "none", L.on ? "取状态中…" : "叫醒之后这里是身边有谁、在做什么")); return; }
  for (const key of FACTS) {
    if (!(key in info) || (L.innerOk && WITH_INNER.includes(key))) continue;
    const v = info[key], dd = el("dd");
    if (Array.isArray(v)) {
      if (!v.length) dd.textContent = "没有";
      else if (key === "身边的好友") for (const x of v) dd.append(el("span", "lv-tag", String(x)));
      else dd.textContent = v.map(String).join("、");
    } else dd.textContent = String(v);
    dl.append(el("dt", "", key), dd);
  }
}
async function pullStatus() {
  if (!L.on) return;
  try { const r = await fetch("live/status", {cache: "no-store"}); if (r.ok && L.on) { L.info = (await r.json()).info; renderFacts(); } } catch (e) {}
}

/* ---- 日志抽屉 ---- */
function openDrawer(open) {
  $("lv-drawer").classList.toggle("open", open); $("lv-log-btn").setAttribute("aria-expanded", String(open));
  if (open) { const b = $("log"); b.scrollTop = b.scrollHeight; }
}
async function pullLogs(force) {  // force：没在跑也拉一次（出错停下后才打开页面）
  const run = S.state && S.state.run;
  if (!run || run.kind === "sandbox") return;
  if (force) L.drain = false;
  else if (!BUSY.includes(run.state)) { if (!L.drain) return; L.drain = false; }  // 刚停下：再拉最后一次，把收尾 / traceback 读进来
  else L.drain = true;
  try {
    const r = await getJSON(`api/logs?after=${L.logNext}`), box = $("log"), note = $("log-note");
    note.textContent = r.note || ""; note.hidden = !r.note;  // 终端起的团子：日志来自 agent.log
    if (r.next < L.logNext) { L.logNext = 0; box.textContent = ""; return; }  // 重新启动过：从头来
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

globalThis.LiveView = {
  snapshot: () => L.snap,
  pick(fn) { L.pick = fn || null; $("lv-canvas").classList.toggle("picking", !!L.pick); },
  mark(m) { L.mark = m || null; redraw(); },
};

function bind() {
  $("opt-live").onchange = liveWarn;
  $("launch").onsubmit = start;
  $("log").addEventListener("scroll", () => { const b = $("log"); L.follow = b.scrollTop + b.clientHeight >= b.scrollHeight - 8; });
  $("lv-log-btn").onclick = () => openDrawer(!$("lv-drawer").classList.contains("open"));
  $("lv-log-close").onclick = () => openDrawer(false);
  document.addEventListener("keydown", e => { if (e.key === "Escape" && $("lv-drawer").classList.contains("open")) openDrawer(false); });
  $("lv-pause").onclick = () => {
    L.paused = !L.paused; $("lv-pause").textContent = L.paused ? "继续" : "暂停";
    if (!L.paused && L.on && L.shown) { L.gen++; const g = L.gen; stageLoop(g); chatLoop(g); }
  };
  $("lv-boxes").onclick = () => { L.boxes = !L.boxes; $("lv-boxes").textContent = L.boxes ? "隐藏框" : "显示框"; redraw(); };
  $("lv-save").onclick = () => {
    if (!L.snap) return;
    const a = el("a"); a.download = `dango-${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}.png`;
    a.href = $("lv-canvas").toDataURL("image/png"); a.click();
  };
  $("lv-legend-btn").onclick = () => {
    const g = $("lv-legend"), open = g.hidden; g.hidden = !open; $("lv-legend-btn").setAttribute("aria-expanded", String(open));
  };
  if (typeof Stage !== "undefined") for (const [k, name] of Object.entries(Stage.NAMES)) {
    const s = el("span"), i = el("i"); i.style.background = Stage.COLORS[k]; s.append(i, name); $("lv-legend").append(s);
  }
  const c = $("lv-canvas");
  c.addEventListener("mousemove", e => { L.hover = canvasPoint(e); redraw(); });
  c.addEventListener("mouseleave", () => { L.hover = null; redraw(); });
  c.addEventListener("click", e => {
    if (!L.pick || !L.snap || typeof Stage === "undefined") return;
    const [x, y] = Stage.toFrame(e.clientX, e.clientY, c.getBoundingClientRect(), L.snap.width, L.snap.height), fn = L.pick, s = L.snap;
    LiveView.pick(null); fn(x, y, s);
  });
}

Pages.live = {
  init() { bind(); renderNow(null); renderFacts(); onState(render); },
  show() {
    L.shown = true; L.startProblems = false; render(); pullLogs();
    for (const t of L.timers) clearInterval(t);
    L.timers = [setInterval(pullStatus, 2000), setInterval(pullInner, 5000), setInterval(pullLogs, 1000)];
    if (L.on) loops();
  },
  hide() { L.shown = false; L.gen++; for (const t of L.timers) clearInterval(t); L.timers = []; },
};
})();
