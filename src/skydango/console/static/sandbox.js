/* 沙盒页（spec 2026-10-01-console-redesign §4.1；沙盒本身见 2026-09-30-brain-sandbox）：顶栏（停着 = 启动表单，跑着 = 沙盒时间牌）+ 三栏工作台。
 * 左：现在（Inner.renderNow）+ 身边 / 团子看到的（/sandbox/op）；中：大脑控制台（brainlog.js）；右：/sandbox/state 长轮询的聊天记录 + 冒充（团子在左、别人在右）。
 * 给别处用：window.sandboxSummary()（左栏卡片）、window.sandboxClock()（内心页）、
 *   S.replay + onReplay(fn) + pullReplay()（api/sandbox/replay 的进度和录制状态，剧本页共用）、document 上的 "scenarios-changed" 事件（另存为之后）。
 * 顶层不碰 document：node 里 require 它能测 Sandbox.lineKind / summaryText（tests/test_console_page.py）。 */
(function () {
"use strict";
const SB = {v: 0, gen: 0, state: null, recv: 0, info: null, trace: null, running: false, rows: 0, sig: "",
  inner: null, whoPicked: false, sceneDirty: false, starting: false, wasPlaying: false};
const STRANGER = "__stranger";
const STATUS = {starting: "正在启动…", stopping: "正在下线（最终反思、写日记）…"};
const replayListeners = [];
S.replay = null;

/* ---- 纯函数（node 里测） ---- */
function lineKind(r) {  // 聊天行用哪种样式
  if (r.kind === "event") return /^──\s*反思/.test(r.text || "") ? "refl" : "ev";
  if (r.kind === "act") return "act";
  if (r.kind === "said") return "msg me";
  if (r.kind === "blocked") return r.who === "团子" ? "msg me blocked" : "msg blocked";
  return "msg";
}
function sbDate(t) { const d = new Date(t * 1000); return `${d.getMonth() + 1}月${d.getDate()}日 周${"日一二三四五六"[d.getDay()]}`; }
function summaryText(t, mood, energy) { return [dayTime(t), mood, energy].filter(Boolean).join(" · "); }

/* ---- 小工具 ---- */
function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 隐私模式之类：不记 */ } }
function sbRun() { const r = S.state && S.state.run; return r && r.kind === "sandbox" ? r : null; }
function sbNow() {  // 沙盒时间：跑着时两次长轮询之间自己往前走；停着就停在最后看到的那一刻
  if (!SB.state) return Date.now() / 1000;
  return SB.running ? SB.state.wall + (Date.now() / 1000 - SB.recv) : SB.state.wall;
}
function replaying() { return !!(S.replay && S.replay.progress && S.replay.progress.running); }
function pageOn(name) { const p = typeof document !== "undefined" && $("page-" + name); return !!p && !p.hidden; }
async function sbOp(body, quiet) {  // quiet：成功了不弹提示（说话：聊天记录里自己会出来）
  const r = await post("sandbox/op", body);
  if (!r.data.ok) toast(r.data.text || "没做成", "bad");
  else if (!quiet && r.data.text) toast(r.data.text, "ok");
  return !!r.data.ok;
}

/* ---- 启动前：起始时间下限、说话人、预检问题 ---- */
async function sbLoadInfo() { try { SB.info = await getJSON("api/sandbox/info"); renderSbInfo(); } catch (e) { /* 面板停了：左栏卡片会说 */ } }
function renderSbInfo() {
  const info = SB.info; if (!info) return;
  const f = $("sb-floor"); f.hidden = !info.floor_text;
  f.textContent = info.floor_text ? `沙盒时间最早只能从 ${info.floor_text} 开始（上次停在那儿，账本不能倒着走）` : "";
  const sel = $("sb-say").elements.who, keep = SB.whoPicked ? sel.value : lsGet("sb-who") || "";
  sel.textContent = "";
  for (const n of info.friends || []) { const o = el("option", "", n); o.value = n; sel.append(o); }
  const st = el("option", "", "陌生人…"); st.value = STRANGER; sel.append(st);
  if (info.owner_name && !(info.friends || []).includes(info.owner_name)) {
    const o = el("option", "", `主人 ${info.owner_name}（可以发 # 命令）`); o.value = info.owner_name; sel.append(o);
  }
  sel.value = [...sel.options].some(o => o.value === keep) ? keep : sel.options[0].value;  // 没选过：第一个好友
  sbWho();
  const dl = $("sb-friend-list"); dl.textContent = "";
  for (const n of info.friends || []) { const o = el("option"); o.value = n; dl.append(o); }
  renderProblems(info.problems || [], !!info.orphan);
}
function renderProblems(list, orphan) {
  const box = $("sb-problems");
  SB.problems = list;
  problemList(box, list);
  if (orphan) {
    const b = el("button", "btn sm", "让它退出"); b.type = "button";
    b.onclick = async () => {
      b.disabled = true;
      const r = await post("api/orphan/stop", {kind: "sandbox"});
      toast(r.data.ok ? "叫它退出了" : "没叫动（它可能已经退了）", r.data.ok ? "ok" : "warn");
      setTimeout(sbLoadInfo, 1500);
    };
    const row = el("div", "sb-orphan"); row.append(b); box.append(row); box.hidden = false;
  }
  renderStartButton();
}
function renderStartButton() {
  const r = S.state && S.state.run, other = !sbRun() && r && BUSY.includes(r.state);
  $("sb-go").disabled = SB.starting || replaying() || !!other || !!(SB.problems && SB.problems.length);
  for (const x of $("sb-start").querySelectorAll("input")) x.disabled = !!other;
}
function sbWho() { const f = $("sb-say").elements; f.name.hidden = f.who.value !== STRANGER; }

/* ---- 聊天记录 ---- */
function sbLine(r) {
  const k = lineKind(r), time = el("time", "", hhmm(r.t));
  if (k === "ev") { const d = el("div", "sb-ev"); d.append(time, " ", (r.text || "").replace(/^──\s*|\s*──$/g, "")); return d; }
  if (k === "refl") { const d = el("div", "sb-refl"); d.append(time, " ", r.text || ""); return d; }
  if (k === "act") { const d = el("div", "sb-act", r.text || ""); d.title = dayTime(r.t); return d; }
  const d = el("div", "sb-" + k.split(" ").join(" sb-")), who = el("div", "who");
  who.append(r.kind === "heard" ? r.who || "（不知道是谁）" : r.kind === "said" ? "团子" : r.who === "团子" ? "团子（没说出去）" : "（被拦下）", " ", time);
  d.append(who, el("div", "b", r.text || ""));
  if (r.kind === "blocked" && r.why) d.append(el("div", "why", r.why));
  return d;
}
function sbAppend(lines) {
  const box = $("sb-chat"); if (!lines.length) return;
  const follow = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;  // 原本就在底部才跟到底
  if (!SB.rows) box.textContent = "";
  for (const r of lines) { box.append(sbLine(r)); SB.rows++; }
  while (box.childNodes.length > 500) box.firstChild.remove();
  if (follow) box.scrollTop = box.scrollHeight;
}
function clearChat() { SB.v = 0; SB.rows = 0; $("sb-chat").textContent = ""; }

/* ---- /sandbox/state：时钟、身边、场景 ---- */
function sbApply(d) {
  if (d.version < SB.v) clearChat();  // 沙盒重启过：从头来
  SB.state = d; SB.recv = Date.now() / 1000; SB.v = d.version; sbAppend(d.lines || []);
  renderClock();
  const e = d.energy || {}, en = $("sb-energy"); en.textContent = "";
  if (e.level) { en.append(el("small", "", "精力"), el("b", "", e.level)); if (e.note && e.note !== e.level) en.title = e.note; }
  $("sb-limit").hidden = !d.limit; $("sb-limit").textContent = d.limit || "";
  $("sb-quiet").textContent = d.thinking ? "大脑在想…" : d.reflecting ? "在反思…" : d.idle ? "安静" : "有动静";
  const tags = $("sb-friends"); tags.textContent = "";
  if (!(d.friends || []).length) tags.append(el("span", "note", "没有好友在身边"));
  for (const n of d.friends || []) {
    const t = el("span", "sb-friend", n), x = el("button", "", "×");
    x.type = "button"; x.title = `${n} 走了`; x.setAttribute("aria-label", `${n} 走了`);
    x.onclick = () => sbOp({op: "leave", who: n}); t.append(x); tags.append(t);
  }
  $("sb-strangers").textContent = String(d.strangers || 0);
  if (document.activeElement !== $("sb-place")) $("sb-place").value = d.place || "";
  if (document.activeElement !== $("sb-scene-text") && !SB.sceneDirty) $("sb-scene-text").value = d.scene || "";
}
function renderClock() {
  if (!SB.state) return;
  const t = sbNow();
  $("sb-time").textContent = hhmm(t); $("sb-date").textContent = sbDate(t);
  $("sb-time").title = `沙盒比现在快 ${span(SB.state.wall - Date.now() / 1000)}`;
}
async function sbLoop(gen) {
  while (gen === SB.gen) {
    try {
      const r = await fetch(`sandbox/state?after=${SB.v}&wait=${SB.state ? 5 : 0}`, {cache: "no-store"});  // 第一次马上要；之后最多等 5 秒
      if (!r.ok) throw new Error(r.status);
      const d = await r.json(); if (gen !== SB.gen) return; sbApply(d);
    } catch (e) { if (gen !== SB.gen) return; await new Promise(ok => setTimeout(ok, 1000)); }
  }
}

/* ---- 右栏「现在」 ---- */
function renderSbNow(now, d) {
  const box = $("sb-now");
  if (now === undefined) { box.textContent = ""; box.append(el("h3", "", "现在")); const b = el("div", "pane-body"); b.append(el("p", "none", "（沙盒没在跑）")); box.append(b); }
  else Inner.renderNow(box, now, d);
  const h = box.querySelector("h3"), link = el("button", "linkish sb-inner-link", "完整内心 →");
  link.type = "button"; link.onclick = () => { Inner.setSource("sandbox"); go("inner"); };
  h.append(el("span", "grow"), link);
}
async function pullSbNow() {  // 沙盒在跑时每 5 秒（不管页面开没开：左栏卡片的心情也用它）
  if (!SB.running) return;
  try { const d = await getJSON("api/inner?source=sandbox"); if (!SB.running) return; d.clock = sbNow(); SB.inner = d; renderSbNow(d.now || {}, d); }
  catch (e) { /* 下次再取 */ }
}

/* ---- 大脑控制台 ---- */
function mountBrain() {
  $("sb-brain-none").hidden = true;
  if (typeof mountBrainConsole !== "function") return;
  SB.trace = mountBrainConsole($("sb-brain"), "sandbox/brain");
  const c = $("sb-brain").querySelector(".brain-acted");  // 「只看做了事的」默认勾上；用户改过就记住
  if (c) {
    c.checked = lsGet("sb-acted") !== "0"; c.dispatchEvent(new Event("change"));
    c.addEventListener("change", () => lsSet("sb-acted", c.checked ? "1" : "0"));
  }
}
function unmountBrain() {
  if (SB.trace) { SB.trace.stop(); SB.trace = null; }
  const b = $("sb-brain"); b.hidden = true; b.textContent = ""; $("sb-brain-none").hidden = false;
}

/* ---- 每秒随 /api/state ---- */
function renderSandbox() {
  const r = S.state && S.state.run; if (!r) return;
  const mine = sbRun(), st = mine ? mine.state : "idle", busy = BUSY.includes(st), running = st === "running" || st === "stopping";
  const other = !mine && BUSY.includes(r.state), sig = `${r.kind || "dango"}:${r.state}`;
  if (running && !SB.running) {
    SB.running = true; SB.gen++; SB.state = null; SB.inner = null; clearChat(); sbLoop(SB.gen); mountBrain(); pullSbNow();
  }
  if (!running && SB.running) {
    SB.running = false; SB.gen++; unmountBrain(); renderSbNow(); $("sb-quiet").textContent = "";
    const tg = $("sb-friends"); tg.textContent = ""; tg.append(el("span", "note", "没有好友在身边"));
    $("sb-strangers").textContent = "0"; $("sb-place").value = ""; $("sb-scene-text").value = ""; SB.sceneDirty = false;
    $("sb-energy").textContent = ""; $("sb-limit").hidden = true;
  }
  if (sig !== SB.sig) { SB.sig = sig; if (!busy) sbLoadInfo(); }  // 停下了、团子起来了 / 停了：预检问题会变
  $("sb-start").hidden = busy; $("sb-clock").hidden = !running;
  const status = $("sb-status"); status.hidden = !STATUS[st]; status.textContent = STATUS[st] || "";
  renderClock(); renderStartButton(); renderTools();
  const world = st === "running" && !replaying();
  for (const id of ["sb-say-set", "sb-nearby", "sb-scene", "sb-ff"]) $(id).disabled = !world;
}
function renderTools() {  // 顶栏右边：录制状态、新录制、另存为、重置记忆
  const d = S.replay || {}, rec = d.recording || {}, playing = replaying(), mine = sbRun();
  const r = S.state && S.state.run, anyBusy = !!r && BUSY.includes(r.state);
  const chip = $("sb-rec");
  if (playing) { chip.className = "chip warn dot"; chip.textContent = `回放中「${d.progress.name}」`; chip.title = ""; }
  else if (rec.active) { chip.className = "chip rec"; chip.textContent = `录制中 · ${rec.steps} 步`; chip.title = rec.memory === "keep" ? "用的是当时的记忆（不是从 memory/ 重置开始）" : ""; }
  else { chip.className = "chip"; chip.textContent = "没在录"; chip.title = mine ? "下一个操作开始录" : "启动沙盒就开始录"; }
  $("sb-save").disabled = playing || !rec.steps;
  $("sb-rec-new").disabled = playing;
  $("sb-reset").disabled = anyBusy || playing;
  $("sb-reset").title = anyBusy ? (mine ? "先下线沙盒再重置" : "团子在运行，先停团子") : playing ? "正在回放剧本" : "";
}

/* ---- 回放进度（剧本页也用：S.replay / onReplay / pullReplay） ---- */
function onReplay(fn) { replayListeners.push(fn); if (S.replay) try { fn(S.replay); } catch (e) { console.error(e); } }
async function pullReplay() {
  let d; try { d = await getJSON("api/sandbox/replay"); } catch (e) { return; }
  S.replay = d; renderReplay(d);
  for (const fn of replayListeners) { try { fn(d); } catch (e) { console.error(e); } }
}
function renderReplay(d) {
  const p = d.progress || {}, playing = !!p.running;
  $("sb-replay-bar").hidden = !playing;
  if (playing) {
    $("sb-replay-text").textContent = `回放『${p.name}』第 ${p.step}/${p.total} 步`;
    $("sb-replay-fill").style.width = `${p.total ? Math.round(p.step / p.total * 100) : 0}%`;
  }
  $("sb-cols").classList.toggle("sb-replaying", playing);
  if (SB.wasPlaying && !playing) toast(`回放「${p.name || ""}」结束了${d.report ? "，报告在「剧本和报告」页" : ""}`, "ok");
  SB.wasPlaying = playing;
  renderTools(); renderStartButton(); renderSandbox();
}

/* ---- 按钮 ---- */
function bind() {
  $("sb-say").elements.who.onchange = e => { SB.whoPicked = true; lsSet("sb-who", e.target.value); sbWho(); };
  $("sb-start").onsubmit = async e => {
    e.preventDefault();
    const when = document.querySelector("input[name=sb-when]:checked").value;
    const start = when === "custom" ? $("sb-when-at").value.trim() : when;
    if (!start) { toast("自定义时间写 HH:MM 或 YYYY-MM-DD HH:MM", "warn"); $("sb-when-at").focus(); return; }
    SB.starting = true; renderStartButton();
    const r = await post("api/sandbox/start", {start});
    SB.starting = false;
    if (!r.data.ok) {
      if (r.data.problems) renderProblems(r.data.problems, !!r.data.orphan);
      else toast(r.data.text || "没启动成", "bad");
    }
    renderStartButton(); refresh();
  };
  $("sb-when-at").onfocus = () => { document.querySelector("input[name=sb-when][value=custom]").checked = true; };
  $("sb-reset").onclick = async () => {
    const ok = await ask("会用 memory/ 覆盖沙盒记忆，沙盒里聊出的交情、日记、性格都会没。\n（真的 memory/ 不会被改。）确定重置？", {ok: "重置", danger: true});
    if (!ok) return;
    const r = await post("api/sandbox/reset");
    toast(r.data.ok ? r.data.text : r.data.text || "没重置成", r.data.ok ? "ok" : "bad");
    sbLoadInfo(); pullReplay();
  };
  $("sb-rec-new").onclick = async () => {
    const rec = (S.replay && S.replay.recording) || {};
    if (rec.steps && !(await ask("丢掉现在录的这一段，从这里重新开始录？", {ok: "重新录"}))) return;
    await post("api/sandbox/record/new");
    toast("重新开始录了（沙盒下次启动的选项会写成剧本的开头）", "ok"); pullReplay();
  };
  $("sb-save").onclick = async () => {
    const v = await ask("另存为剧本（存进 sandbox/scenarios/）", {ok: "存", fields: [
      {name: "name", label: "剧本名（中英文、数字、- _）"},
      {name: "note", label: "一句话说明这个剧本看什么（可以空着）"}]});
    if (!v) return;
    const name = v.name.trim(), note = v.note;
    if (!name) { toast("剧本名不能空着", "warn"); return; }
    let r = await post("api/sandbox/save", {name, note});
    if (r.status === 409 && r.data.exists) {
      if (!(await ask(r.data.text, {ok: "覆盖", danger: true}))) return;
      r = await post("api/sandbox/save", {name, note, overwrite: true});
    }
    if (!r.data.ok) { toast(r.data.text || "没存成", "bad"); return; }
    toast(`存好了：${r.data.path}${r.data.warning ? "。" + r.data.warning : ""}`, r.data.warning ? "warn" : "ok");
    document.dispatchEvent(new CustomEvent("scenarios-changed", {detail: {name}}));
  };
  $("sb-replay-stop").onclick = async () => {
    const r = await post("api/sandbox/replay/stop");
    toast(r.data.text || (r.data.ok ? "这一步做完就停" : "没停成"), r.data.ok ? "ok" : "bad"); pullReplay();
  };
  $("sb-say").onsubmit = async e => {
    e.preventDefault(); const f = e.target.elements;
    const who = f.who.value === STRANGER ? f.name.value.trim() : f.who.value, text = f.text.value.trim();
    if (!who) { toast("先填陌生人的名字", "warn"); f.name.focus(); return; }
    if (!text) return;
    if (await sbOp({op: "say", who, text}, true)) f.text.value = "";
    f.text.focus();
  };
  for (const b of $("sb-ff").querySelectorAll("[data-skip]")) b.onclick = () => sbOp({op: "skip", seconds: Number(b.dataset.skip)});
  for (const b of $("sb-ff").querySelectorAll("[data-at]")) b.onclick = () => sbOp({op: "time", at: b.dataset.at});
  $("sb-at-go").onclick = async () => { const at = $("sb-at").value.trim(); if (at && await sbOp({op: "time", at})) $("sb-at").value = ""; };
  $("sb-come").onclick = async () => { const who = $("sb-come-name").value.trim(); if (who && await sbOp({op: "come", who})) $("sb-come-name").value = ""; };
  for (const b of $("sb-nearby").querySelectorAll("[data-strangers]")) b.onclick = () => {
    const n = Math.max(0, Math.min(20, ((SB.state && SB.state.strangers) || 0) + Number(b.dataset.strangers))); sbOp({op: "strangers", n});
  };
  $("sb-place-go").onclick = () => sbOp({op: "place", name: $("sb-place").value.trim()});
  $("sb-scene-text").oninput = () => { SB.sceneDirty = true; };
  $("sb-scene-go").onclick = async () => { if (await sbOp({op: "scene", text: $("sb-scene-text").value.trim()})) SB.sceneDirty = false; };
  $("sb-notice-go").onclick = async () => { const text = $("sb-notice").value.trim(); if (text && await sbOp({op: "notice", text})) $("sb-notice").value = ""; };
  const enter = (input, btn) => { $(input).onkeydown = e => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); $(btn).click(); } }; };
  enter("sb-at", "sb-at-go"); enter("sb-come-name", "sb-come"); enter("sb-place", "sb-place-go"); enter("sb-notice", "sb-notice-go");
}

let infoTimer = null;
function sandboxSummary() {
  const r = sbRun(); if (!r || r.state !== "running" || !SB.state) return null;
  const now = SB.inner && SB.inner.now, mood = now && now.mood && now.mood.level;
  const energy = (SB.state.energy && SB.state.energy.level) || (now && now.energy && now.energy.level);
  return summaryText(sbNow(), mood, energy);
}
function sandboxClock() { return SB.running && SB.state ? sbNow() : Math.max(Date.now() / 1000, (SB.info && SB.info.floor) || 0); }

Object.assign(globalThis, {Sandbox: {lineKind, summaryText}, sandboxSummary, sandboxClock, onReplay, pullReplay});
Pages.sandbox = {
  init() {
    bind(); renderSbNow();
    setInterval(pullSbNow, 5000);
    setInterval(() => { if (pageOn("sandbox") || pageOn("scenarios") || replaying()) pullReplay(); }, 1500);
  },
  show() {  // 可能被重复调用（点当前导航项）：只重拉，定时器不重复建
    sbLoadInfo(); pullReplay();
    if (!infoTimer) infoTimer = setInterval(() => { if (pageOn("sandbox") && !SB.running && !BUSY.includes((sbRun() || {}).state)) sbLoadInfo(); }, 10000);
  },
  hide() { if (infoTimer) { clearInterval(infoTimer); infoTimer = null; } },
};
onState(renderSandbox);
})();
