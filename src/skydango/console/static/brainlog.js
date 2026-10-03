/* 沙盒页的大脑控制台：和 brain_trace.js 同一个 /brain 长轮询数据，换成终端那种按时间往下滚的日志。
 * 每一轮：轮头（时间 · 原因 · 做了什么 · 耗时 · tokens）→ 收到的事件一行一条 → 状态 / 场景折叠 → 思考、说、工具调用、工具返回各一行。
 * 新的一轮加在底部；原本在底部就跟到底，往上翻着时出「回到最新」。点轮头复制这一轮。
 * 只用 textContent 放数据（大脑的话、工具返回都可能带尖括号）。viewer 照旧用 brain_trace.js，这里不碰它。
 * 顶层不碰 document：node 里 require 它能测 splitPrompt / stepLine / turnHead / isIdle（tests/test_console_page.py）。 */
(function () {
"use strict";
const REASONS = {events: "新消息 / 事件", background: "周围的变化", heartbeat: "心跳", farewell: "退出前总结", outside: "轮外"};
const FOLD = 6;  // 超过这么多行的思考 / 返回折起来

/* ---- 纯函数（node 里测） ---- */
function kilo(n) { return n == null ? "—" : n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n); }
function hms(t) { return t ? new Date(t * 1000).toTimeString().slice(0, 8) : "--:--:--"; }
function splitPrompt(p) {  // "[时间] 事件：\n- 一条\n- 一条\n状态：…" → 轮头那句、事件逐条、剩下的整块（状态 / 场景）
  if (!p) return {head: "", events: [], rest: ""};
  const lines = String(p).split("\n"), events = [];
  const head = lines[0].replace(/^\[[^\]]*\]\s*/, "").trim();
  let i = 1;
  while (i < lines.length && lines[i].startsWith("- ")) { events.push(lines[i].slice(2).replace(/\s+/g, " ").trim()); i++; }
  return {head, events, rest: lines.slice(i).join("\n").trim()};
}
function stepLine(s) {
  if (s.kind === "thinking") return {cls: "think", mark: "…", text: s.text || ""};
  if (s.kind === "text") return {cls: "note", mark: "»", text: s.text || ""};
  if (s.kind === "tool") {
    const input = s.input || {};
    if (s.name === "say" && typeof input.text === "string") return {cls: "say", mark: "▶", text: `say “${input.text}”`};
    const args = Object.keys(input).length ? " " + JSON.stringify(input) : "";
    return {cls: "tool", mark: "▶", text: `${s.name}${args}`};
  }
  return {cls: s.error ? "err" : "ret", mark: "↳", text: s.text || ""};
}
function toolCounts(t) {
  const c = {}; for (const n of t.tools || []) c[n] = (c[n] || 0) + 1;
  const s = Object.entries(c).map(([n, k]) => `${n}×${k}`).join(" ");
  return s || ((t.steps || []).some(x => x.kind === "text") ? "（只说了话）" : "（什么都没做）");
}
function turnHead(t) {
  const parts = [hms(t.start), REASONS[t.reason] || t.reason];
  if (t.error) parts.push(`失败：${t.error}`);
  else if (t.end === null) parts.push("进行中…" + ((t.tools || []).length ? " " + toolCounts(t) : ""));
  else {
    parts.push(toolCounts(t));
    if (t.seconds != null) parts.push(`${t.seconds.toFixed(1)}s`);
    if (t.result) { const k = t.result.tokens || {}; parts.push(`in ${kilo(k.input)} / out ${kilo(k.output)}`); }
  }
  return parts.join(" · ");
}
function isIdle(t) { return !(t.end === null || t.error || (t.steps || []).some(s => s.kind === "tool" || s.kind === "text")); }
function copyText(t) {
  const lines = [turnHead(t), "== 收到 ==", t.prompt || ""];
  for (const s of t.steps || []) { const l = stepLine(s); lines.push(`${l.mark} ${l.text}`); }
  const r = t.result, k = (r && r.tokens) || {};
  if (r) lines.push("== 结果 ==", `${r.subtype ?? "—"} · ${r.num_turns ?? "—"} 步 · 缓存读 ${kilo(k.cache_read)} · 缓存写 ${kilo(k.cache_write)}` + (r.cost != null ? ` · 参考 $${r.cost.toFixed(4)}` : ""));
  return lines.join("\n");
}

/* ---- 画 ---- */
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function line(cls, mark, text) {
  const rows = text.split("\n");
  if (rows.length <= FOLD) { const d = el("div", "bc-line " + cls); d.append(el("span", "m", mark), el("span", "x", text)); return d; }
  const det = el("details", "bc-fold bc-line " + cls), sm = el("summary");
  sm.append(el("span", "m", mark), el("span", "x", `${rows[0]}  …（共 ${rows.length} 行）`));
  det.append(sm, el("pre", "", text)); return det;
}
function turnEl(t, onCopy) {
  const d = el("div", "bc-turn" + (t.error ? " err" : "") + (t.end === null ? " live" : "") + (isIdle(t) ? " idle" : ""));
  const th = el("div", "bc-th", turnHead(t)); th.title = "点一下复制这一轮"; th.onclick = () => onCopy(t, th); d.append(th);
  if (isIdle(t)) return d;  // 什么都没做：只留轮头一行
  if (t.reason !== "outside") {
    const p = splitPrompt(t.prompt);
    for (const e of p.events) d.append(line("in", "‹", e));
    if (!p.events.length && p.head) d.append(line("dim", "‹", p.head));
    if (p.rest) {
      const det = el("details", "bc-fold bc-ctx"), sm = el("summary"), first = p.rest.split("\n")[0];
      sm.append(el("span", "m", "▸"), el("span", "x", first.length > 48 ? first.slice(0, 48) + "…" : first));
      det.append(sm, el("pre", "", p.rest)); d.append(det);
    }
  }
  for (const s of t.steps || []) { const l = stepLine(s); d.append(line(l.cls, l.mark, l.text)); }
  if (t.error) d.append(line("fail", "✗", `失败：${t.error}`));
  return d;
}

function mountBrainConsole(root, url) {
  const B = {turns: new Map(), els: new Map(), version: 0, boot: null, acted: false, stopped: false, ctrl: null};
  root.classList.add("brain-console"); root.textContent = "";
  const head = el("div", "bc-head"), stateEl = el("span", "bc-state", "连接中…"), label = el("label"), check = el("input", "brain-acted");
  check.type = "checkbox"; label.append(check, "只看做了事的");
  head.append(el("b", "", "大脑"), stateEl, label);
  const log = el("div", "bc-log"), jump = el("button", "bc-jump", "↓ 回到最新");
  jump.type = "button"; jump.hidden = true;
  root.append(head, log, jump);
  const atBottom = () => log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  log.addEventListener("scroll", () => { if (atBottom()) jump.hidden = true; });
  jump.onclick = () => { log.scrollTop = log.scrollHeight; jump.hidden = true; };

  async function copy(t, th) {
    const text = copyText(t);
    try { await navigator.clipboard.writeText(text); }
    catch (e) { const a = el("textarea"); a.value = text; document.body.append(a); a.select(); document.execCommand("copy"); a.remove(); }
    th.classList.add("copied"); setTimeout(() => th.classList.remove("copied"), 1200);
  }
  function order(a, b) { return (a.start || 0) - (b.start || 0) || a.id - b.id; }
  function render(changed) {
    const follow = atBottom();
    for (const t of changed) {
      const neu = turnEl(t, copy), old = B.els.get(t.id);
      neu.hidden = B.acted && isIdle(t);
      if (old) old.replaceWith(neu); B.els.set(t.id, neu);
    }
    [...B.turns.values()].sort(order).forEach((t, i) => { const e = B.els.get(t.id); if (log.children[i] !== e) log.insertBefore(e, log.children[i] || null); });
    if (!changed.length) return;
    if (follow) log.scrollTop = log.scrollHeight; else jump.hidden = false;
  }
  function state(st) {
    let text = `${st.model ?? "?"} / ${st.effort ?? "?"} · 已醒 ${st.turns ?? 0} 轮 · `, cls = "";
    if (st.on_fallback) { text += "备用大脑（DeepSeek）"; cls = "bad"; }
    else if (st.offline) { text += "已转备用回复（DeepSeek）"; cls = "bad"; }
    else if (st.retry_in != null) { text += `连续失败 ${st.failures} 次，${Math.ceil(st.retry_in)} 秒后重试`; cls = "warn"; }
    else text += "在线";
    stateEl.textContent = text; stateEl.className = "bc-state" + (cls ? " " + cls : "");
  }
  function merge(d) {
    if ((B.boot && d.boot !== B.boot) || d.version < B.version) {  // 程序重启过：清空后从头拉
      B.turns.clear(); for (const e of B.els.values()) e.remove(); B.els.clear(); B.boot = d.boot; B.version = 0; return;
    }
    B.boot = d.boot; B.version = d.version;
    for (const t of d.turns) B.turns.set(t.id, t);
    if (d.oldest != null) for (const id of [...B.turns.keys()]) if (id !== 0 && id < d.oldest) { B.turns.delete(id); B.els.get(id)?.remove(); B.els.delete(id); }
    state(d.state || {}); render(d.turns.filter(t => B.turns.has(t.id)));
  }
  check.onchange = () => { B.acted = check.checked; for (const [id, e] of B.els) e.hidden = B.acted && isIdle(B.turns.get(id)); };
  (async function loop() {
    let seen = false;
    while (!B.stopped) {
      try {
        B.ctrl = new AbortController();
        const r = await fetch(`${url}?after=${B.version}`, {cache: "no-store", signal: B.ctrl.signal});
        if (r.status === 404 && !seen) return;
        if (!r.ok) throw new Error(r.status);
        const d = await r.json(); if (B.stopped) return;
        seen = true; root.hidden = false; merge(d);
      } catch (e) {
        if (B.stopped) return;
        if (seen) { stateEl.textContent = "连不上（程序停了？）"; stateEl.className = "bc-state bad"; }
        await new Promise(ok => setTimeout(ok, 1000));
      }
    }
  })();
  return {stop() { B.stopped = true; if (B.ctrl) B.ctrl.abort(); }};
}

globalThis.mountBrainConsole = mountBrainConsole;
if (typeof module !== "undefined") module.exports = {splitPrompt, stepLine, turnHead, isIdle};
})();
