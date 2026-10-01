/* 剧本和报告页（spec 2026-10-01-console-redesign §4.4）：左 剧本列表 + 回放进度，右 报告列表 + 报告正文（renderMarkdown）。
 * 回放进度不自己轮询：用 sandbox.js 的 S.replay / onReplay / pullReplay；另存为剧本后收 document 的 "scenarios-changed"。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const SC = {scripts: null, reports: null, open: "", wasPlaying: false};

function playing() { return !!(S.replay && S.replay.progress && S.replay.progress.running); }
function busyOther() {
  const run = S.state && S.state.run;
  return !!run && BUSY.includes(run.state) && run.kind !== "sandbox";
}

async function loadScripts() {
  try { const d = await getJSON("api/sandbox/scenarios"); SC.scripts = d.scenarios || []; } catch (e) { return; }
  renderScripts();
}
function renderScripts() {
  const box = $("sc-list"); box.textContent = "";
  const list = SC.scripts || [];
  if (!list.length) {
    const d = el("div", "sc-example");
    d.append("sandbox/scenarios/ 里还没有剧本。示例剧本在 ", el("code", "", "docs/sandbox-scenarios/"),
      "，里面的人名是占位的，换成自己的好友名再复制进 ", el("code", "", "sandbox/scenarios/"), "。也可以在沙盒页玩一遍，「另存为剧本」。");
    box.append(d);
    return;
  }
  for (const x of list) {
    const card = el("div", "sc-card" + (x.error ? " bad" : ""));
    const head = el("div", "sc-head");
    head.append(el("span", "sc-name", x.name));
    if (!x.error) head.append(el("span", "sc-meta", `${x.steps} 步`));
    const btn = el("button", "btn sm go", "回放");
    btn.type = "button";
    btn.disabled = !!x.error || playing() || busyOther();
    btn.onclick = () => startReplay(x);
    head.append(btn);
    card.append(head);
    const note = x.error ? "读不了：" + x.error : x.note;
    if (note) card.append(el("p", "sc-note", note));
    box.append(card);
  }
}
async function startReplay(x) {
  const reset = x.memory !== "keep" ? "\n剧本开头会用 memory/ 覆盖沙盒记忆（真的 memory/ 不会被改）。" : "\n剧本用沙盒现在的记忆接着跑。";
  const yes = await ask(`回放「${x.name}」：沙盒在跑会先下线（写日记），再按剧本重新起。${reset}\n回放时每一步会真的调 Claude，花额度。继续？`, {ok: "回放"});
  if (!yes) return;
  const r = await post("api/sandbox/replay", {name: x.name});
  if (!r.data.ok) toast(r.data.text || "回放没开始", "bad"); else toast(`开始回放「${x.name}」`, "ok");
  pullReplay();
}

function renderProgress() {
  const box = $("sc-progress"), p = (S.replay && S.replay.progress) || {};
  if (!playing()) { box.hidden = true; box.textContent = ""; return; }
  if (!box.firstChild) {
    const bar = el("div", "sc-bar"); bar.append(el("i"));
    const row = el("div", "sc-row"), stop = el("button", "btn sm halt", "停止回放");
    stop.type = "button";
    stop.onclick = async () => {
      const r = await post("api/sandbox/replay/stop");
      toast(r.data.text || (r.data.ok ? "这一步做完就停" : "没停成"), r.data.ok ? "ok" : "bad"); pullReplay();
    };
    row.append(el("span"), stop);
    box.append(el("div"), bar, row);
  }
  box.hidden = false;
  box.children[0].textContent = `回放中「${p.name}」`;
  box.querySelector("i").style.width = (p.total ? Math.round(p.step / p.total * 100) : 0) + "%";
  box.querySelector(".sc-row span").textContent = `第 ${p.step} / ${p.total} 步`;
}

async function loadReports(openLatest) {
  let d;
  try { d = await getJSON("api/sandbox/reports"); } catch (e) { return; }
  SC.reports = d.reports || [];
  renderReports();
  if (openLatest && SC.reports.length) openReport(SC.reports[0].name);
}
function renderReports() {
  const box = $("sc-reports"); box.textContent = "";
  const list = SC.reports || [];
  if (!list.length) box.append(el("p", "none", "还没有报告（回放一个剧本就有了）"));
  for (const r of list) {
    const b = el("button", "sc-report");
    b.type = "button";
    b.setAttribute("aria-current", r.name === SC.open ? "true" : "false");
    b.append(el("span", "sc-name", r.scenario), el("span", "sc-meta", dayTime(r.time)));
    b.onclick = () => openReport(r.name);
    box.append(b);
  }
  renderMark();
}
async function openReport(name) {
  SC.open = name;
  renderReports();
  const rd = $("sc-reader");
  try {
    const r = await fetch("api/sandbox/reports/" + encodeURIComponent(name), {cache: "no-store"});
    const d = await r.json();
    if (SC.open !== name) return;  // 读的时候又点了别的
    if (!r.ok || typeof d.text !== "string") { rd.textContent = d.text || "读不出这份报告"; return; }
    rd.innerHTML = renderMarkdown(d.text);
  } catch (e) { if (SC.open === name) rd.textContent = "读不出这份报告（面板没回应）"; }
}

function renderMark() {
  const m = $("mark-scenarios");
  m.textContent = playing() ? "回放中" : (SC.reports && SC.reports.length ? `${SC.reports.length} 份报告` : "");
}

function onReplayData() {
  const now = playing();
  renderProgress(); renderMark();
  if (SC.wasPlaying && !now) { loadScripts(); loadReports(true); }
  else if (SC.wasPlaying !== now) renderScripts();  // 回放按钮跟着开始 / 结束禁用 / 恢复
  SC.wasPlaying = now;
}

Pages.scenarios = {
  init() {
    onReplay(onReplayData);
    onState(() => { const b = busyOther(); if (SC.scripts && b !== SC.busy) { SC.busy = b; renderScripts(); } });
    document.addEventListener("scenarios-changed", loadScripts);
    loadReports(false);
  },
  show() { loadScripts(); loadReports(false); pullReplay(); },
};
})();
