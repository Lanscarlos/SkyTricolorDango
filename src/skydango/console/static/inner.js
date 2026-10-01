/* 内心页（spec 2026-10-01-console-redesign §4.3）。renderNow 只往传进来的容器里画，沙盒页右栏复用。 */
(function () {
const IN = {data: null, source: "dango", hours: 24, loading: false, lastState: null};
const MOOD_COLOR = {"开心": "var(--m-happy)", "平常": "var(--m-plain)", "低落": "var(--m-low)", "烦": "var(--m-cross)"};
const ENERGY_COLOR = {"精神": "var(--matcha)", "还行": "var(--stick)", "有点累": "var(--amber)", "困": "var(--m-low)"};
const KIND_LABEL = {catchphrase: "口头禅", joke: "老梗", opinion: "看法"};
const WANT_LABEL = {"惦记": "惦记", "想做": "想", "小心思": "小心思"};
const SVGNS = "http:\/\/www.w3.org/2000/svg";
function svg(tag, attrs) { const e = document.createElementNS(SVGNS, tag); for (const k in attrs || {}) e.setAttribute(k, attrs[k]); return e; }
function nowSec() { return (IN.data && IN.data.clock) || Date.now() / 1000; }
function innerMine(r) { return !!r && (r.kind || "dango") === (IN.source === "sandbox" ? "sandbox" : "dango"); }  // 子进程槽里跑的正是这页在看的那个
function innerBusy() { const r = S.state && S.state.run; return innerMine(r) && (r.state === "starting" || r.state === "stopping"); }
function innerClock() {  // 沙盒的"现在"是沙盒时间（Task 6 提供 sandboxClock）；没有就用真实时间
  if (IN.source === "sandbox" && typeof window.sandboxClock === "function") { const t = window.sandboxClock(); if (t) return t; }
  return Date.now() / 1000;
}
function setSource(src) {
  IN.source = src === "sandbox" ? "sandbox" : "dango";
  for (const b of $("inner-source-toggle").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.src === IN.source));
  $("inner-title").textContent = IN.source === "sandbox" ? "沙盒里团子的内心" : "团子的内心";
}
function innerOff(d) { return d.source === "live" ? "（没开反思）" : "（还没有）"; }
function innerUrl(source) { return source && source !== "dango" ? `api/inner?source=${encodeURIComponent(source)}` : "api/inner"; }
async function loadInner(source) {
  IN.source = source || "dango"; if (IN.loading) return; IN.loading = true;
  const asked = IN.source;
  try {
    const d = await getJSON(innerUrl(asked));
    if (IN.source === asked) { d.clock = innerClock(); IN.data = d; renderInner(d); }  // 期间切了来源：这份是旧的，丢掉
  }
  catch (e) { if (IN.source === asked) { const c = $("inner-source"); c.className = "chip warn"; c.textContent = "读不到内心数据（面板停了？）"; } }
  finally { IN.loading = false; }
  if (IN.source !== asked) return loadInner(IN.source);  // 来源变了：重新读
}
function renderInner(d) {
  const c = $("inner-source");
  if (d.source === "live") { c.className = "chip live dot"; c.textContent = "实时"; }
  else if (d.source === "files_fallback") { c.className = "chip warn dot"; c.textContent = "实时取不到，显示的是上次保存的"; }
  else { c.className = "chip dot"; c.textContent = d.saved_at ? `上次保存于 ${hhmm(d.saved_at)}` : "上次保存的（还没有保存过）"; }
  renderNow($("inner-now"), d.now, d);
  renderCurve($("inner-curve"), d.log || [], d);
  renderLog($("inner-log"), d.log || []);
  renderPersona($("inner-persona"), d.persona, d);
  renderCards($("inner-cards"), d.cards || []);
  renderDays($("inner-days"), d.days || [], d.diaries || []);
}

/* 1. 现在 */
function renderNow(container, now, d) {
  container.textContent = ""; container.append(el("h3", "", "现在"));
  const body = el("div", "pane-body"); container.append(body);
  now = now || {}; d = d || {source: "files"};
  if (!now.mood) { body.append(el("p", "none", innerOff(d))); return; }
  const t = (d && d.clock) || nowSec(), m = now.mood, box = el("div", "now-mood"), orb = el("span", "orb");
  orb.style.setProperty("--mc", MOOD_COLOR[m.level] || "var(--rice)");
  const lv = el("div", "now-level", m.level);
  if (m.since) lv.append(el("small", "", `这样有 ${span(t - m.since)}了`));
  box.append(orb, lv, el("div", "now-text", m.text || "（没说为什么）")); body.append(box);
  const dl = el("dl", "now-rows");
  const row = (k, v) => { const dd = el("dd"); if (typeof v === "string") dd.textContent = v; else dd.append(v); dl.append(el("dt", "", k), dd); };
  const e = now.energy;
  if (e && e.score != null) {
    const bar = el("div", "ebar"), tr = el("span", "track"), fill = el("i");
    fill.style.width = Math.max(0, Math.min(100, e.score)) + "%"; tr.style.setProperty("--ec", ENERGY_COLOR[e.level] || "var(--matcha)");
    tr.append(fill); bar.append(el("b", "", e.level), tr, el("span", "note", String(e.score))); row("精力", bar);
    if (e.note && e.note !== e.level) row("", el("span", "note", e.note));
  } else row("精力", el("span", "note", "算不出来"));
  const g = now.grudge;
  row("别扭", g ? el("span", "tag warn", `跟${g.who}${g.why ? "（" + g.why + "）" : ""} · 还有 ${span(g.until - t)}消气`) : el("span", "note", "没有"));
  const soft = now.soft || [];
  if (soft.length) { const tags = el("div", "tags"); for (const s of soft) tags.append(el("span", "tag soft", `${s.who}「${s.text}」· 还剩 ${span(s.until - t)}`)); row("收着点", tags); }
  else row("收着点", el("span", "note", "没有"));
  const wants = now.wants || [];
  if (wants.length) {
    const tags = el("div", "tags");
    for (const w of wants) { const x = el("span", "tag"); x.append(el("em", "", (WANT_LABEL[w.kind] || w.kind) + (w.who ? " " + w.who : "")), w.text); tags.append(x); }
    row("心愿", tags);
  } else row("心愿", el("span", "note", "没有"));
  body.append(dl);
}

/* 2. 曲线：精力折线 + 心情色带，反思点点了跳到时间线 */
function moodSpans(log, from, to) {
  const out = []; let cur = null;
  for (const r of log) {
    if (r.kind !== "reflect" || !r.mood) continue;
    if (cur) out.push([cur.t, r.t, cur.level]); cur = {t: r.t, level: r.mood.level};
  }
  if (cur) out.push([cur.t, to, cur.level]);
  return out.map(([a, b, l]) => [Math.max(a, from), Math.min(b, to), l]).filter(([a, b]) => b > a);
}
function renderCurve(container, log, d) {
  container.textContent = "";
  if (d && d.source === "live" && !(d.now && d.now.mood) && !log.length) { container.append(el("p", "none", "（没开反思）")); return; }
  const to = nowSec(), from = to - IN.hours * 3600, rows = log.filter(r => r.t >= from && r.t <= to);
  const energy = rows.filter(r => r.kind === "energy" && r.score != null), reflects = rows.filter(r => r.kind === "reflect");
  if (!energy.length && !reflects.length) { container.append(el("p", "none", "（这段时间还没有记录）")); return; }
  const W = Math.max(320, container.clientWidth || 600), H = 190, L = 6, R = 46, T = 8, B = 24, iw = W - L - R, ih = H - T - B;
  const x = t => L + (t - from) / (to - from) * iw, y = s => T + (1 - s / 100) * ih;
  const s = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "精力和心情曲线"});
  for (const [a, b, l] of moodSpans(log, from, to)) {
    const r = svg("rect", {x: x(a), y: T, width: Math.max(1, x(b) - x(a)), height: ih});
    r.style.fill = MOOD_COLOR[l] || "var(--rice)"; r.style.opacity = ".13"; s.append(r);
  }
  for (const [v, name] of [[70, "精神"], [50, "还行"], [30, "有点累"]]) {
    s.append(svg("line", {class: "grid", x1: L, x2: L + iw, y1: y(v), y2: y(v)}));
    const tx = svg("text", {class: "axis", x: L + iw + 6, y: y(v) + 4}); tx.textContent = name; s.append(tx);
  }
  const tc = svg("text", {class: "axis", x: L + iw + 6, y: y(10) + 4}); tc.textContent = "困"; s.append(tc);
  const step = IN.hours <= 24 ? 6 * 3600 : 86400, first = Math.ceil(from / step) * step;
  for (let t = first; t <= to; t += step) {
    const d0 = new Date(t * 1000), tx = svg("text", {class: "axis", x: x(t), y: H - 6, "text-anchor": "middle"});
    tx.textContent = IN.hours <= 24 ? hhmm(t) : `${d0.getMonth() + 1}/${d0.getDate()}`; s.append(tx);
  }
  if (energy.length) {
    const pts = energy.map(r => `${x(r.t).toFixed(1)},${y(r.score).toFixed(1)}`).join(" ");
    const pl = svg("polyline", {points: pts, fill: "none", "stroke-width": "2", "stroke-linejoin": "round", "stroke-linecap": "round"});
    pl.style.stroke = "var(--matcha)"; s.append(pl);
    const last = energy[energy.length - 1], c = svg("circle", {cx: x(last.t), cy: y(last.score), r: 3}); c.style.fill = "var(--matcha)"; s.append(c);
  }
  for (const r of reflects) {
    const c = svg("circle", {class: "dot", cx: x(r.t), cy: T + ih - 7, r: r.final ? 7 : 5});
    c.style.fill = MOOD_COLOR[r.mood && r.mood.level] || "var(--rice)";
    const tip = svg("title"); tip.textContent = `${dayTime(r.t)} 反思${r.final ? "（下线）" : ""}：${(r.changes || []).length} 处改动`; c.append(tip);
    c.onclick = () => jumpToLog(r.t); s.append(c);
  }
  container.append(s);
  const lg = el("div", "legend");
  for (const [k, v] of Object.entries(MOOD_COLOR)) { const x0 = el("span", "", k); x0.style.setProperty("--lc", v); lg.append(x0); }
  const e0 = el("span", "", "精力（线）"); e0.style.setProperty("--lc", "var(--matcha)"); lg.append(e0, el("span", "note", "圆点 = 一次反思，点它看改了什么"));
  container.append(lg);
}
function jumpToLog(t) {
  let li = document.querySelector(`#inner-log li[data-t="${t}"]`);
  if (!li && $("inner-changed").checked) { $("inner-changed").checked = false; renderLog($("inner-log"), (IN.data && IN.data.log) || []); li = document.querySelector(`#inner-log li[data-t="${t}"]`); }
  if (!li) return; li.scrollIntoView({block: "nearest", behavior: "smooth"}); li.classList.remove("flash"); void li.offsetWidth; li.classList.add("flash");
}

/* 3. 反思记录（新的在上） */
function renderLog(container, log) {
  container.textContent = "";
  const only = $("inner-changed").checked;
  const items = log.filter(r => r.kind === "reflect" || r.kind === "forget").filter(r => !only || r.kind === "forget" || (r.changes || []).length).reverse();
  if (!items.length) { container.append(el("p", "none", only ? "（最近没有改动）" : "（还没有反思过）")); return; }
  const ul = el("ul", "tl");
  for (const r of items) {
    const li = el("li"); li.dataset.t = String(r.t); const when = el("div", "when", dayTime(r.t));
    if (r.kind === "forget") {
      when.append(el("span", "forgot", "网页上删的"));
      li.append(when, el("div", "head", `删了${KIND_LABEL[r.what] || r.what || "条目"}：${r.topic ? r.topic + "——" : r.who ? r.who + "——" : ""}${r.text || ""}`)); ul.append(li); continue;
    }
    if (r.final) when.append(el("span", "final", "下线"));
    const m = r.mood || {}; li.style.setProperty("--mc", MOOD_COLOR[m.level] || "var(--ash)");
    const head = el("div", "head"); head.append(el("b", "", m.level || "？"), el("span", "", m.text || "")); li.append(when, head);
    const ch = r.changes || [];
    if (ch.length) { const u = el("ul"); for (const c of ch) u.append(el("li", "", c)); li.append(u); }
    else li.append(el("div", "quiet", "没改什么"));
    const dr = r.dropped || [];
    if (dr.length) {
      const det = el("details"); det.append(el("summary", "", `没收下的 ${dr.length} 条`));
      const u = el("ul"); for (const c of dr) u.append(el("li", "", c)); det.append(u); li.append(det);
    }
    ul.append(li);
  }
  container.append(ul);
}

/* 4. 性格档案：能删 */
function traitRow(kind, t, label) {
  const row = el("div", "trait"), body = el("div", "body"), line = el("div", "t");
  if (label) line.append(el("i", "", label + "——"));
  line.append(t.text || t.stance || "");
  const now = nowSec(), used = t.last_used ? `上次 ${span(now - t.last_used)}前` : t.since ? `记下 ${span(now - t.since)}了` : "";
  body.append(line, el("div", "meta", `用过 ${t.hits || 0} 次${used ? " · " + used : ""}`));
  const b = el("button", "btn sm halt", "删"); b.type = "button";
  if (innerBusy()) { b.disabled = true; b.title = "团子正在启动 / 停止，稍等再删"; }
  b.onclick = () => forgetTrait(kind, t, b);
  row.append(body, b); return row;
}
async function forgetTrait(kind, t, btn) {
  const what = kind === "opinion" ? `${t.topic}——${t.stance}` : kind === "joke" ? `${t.who}——${t.text}` : t.text;
  if (!(await ask(`删掉这条${KIND_LABEL[kind]}？\n${what}\n\n删了团子就不会再用它（不能撤销）。`, {ok: "删", danger: true}))) return;
  btn.disabled = true;
  const body = {kind, text: kind === "opinion" ? t.stance : t.text, who: t.who || "", topic: t.topic || ""};
  if (IN.source === "sandbox") body.source = "sandbox";
  const r = await post("api/inner/forget", body);
  if (!r.data.ok) toast(r.data.error || r.data.text || "没删成", "bad");
  loadInner(IN.source);
}
function renderPersona(container, p, d) {
  container.textContent = "";
  if (!p) { container.append(el("p", "none", d && d.source === "live" ? "（没开反思 / 性格档案）" : "（还没有）")); return; }
  const groups = [["口头禅", "catchphrase", p.catchphrases || [], null], ["看法", "opinion", p.opinions || [], t => t.topic]];
  let any = false;
  for (const [title, kind, list, label] of groups) {
    if (!list.length) continue; any = true; const g = el("div", "pgroup"); g.append(el("h4", "", title));
    for (const t of list) g.append(traitRow(kind, t, label && label(t))); container.append(g);
  }
  const jokes = p.jokes || [], who = [...new Set(jokes.map(j => j.who))];
  for (const w of who) {
    any = true; const g = el("div", "pgroup"); g.append(el("h4", "", `和${w}的老梗`));
    for (const t of jokes.filter(j => j.who === w)) g.append(traitRow("joke", t, null)); container.append(g);
  }
  if (!any) container.append(el("p", "none", "（还是空的：反思时慢慢攒）"));
  if (innerBusy()) container.append(el("p", "note", "团子正在启动 / 停止，稍等再删"));
}

/* 5. 关系卡 */
function renderCards(container, cards) {
  container.textContent = "";
  if (!cards.length) { container.append(el("p", "none", "（还没有）")); return; }
  const wrap = el("div", "tbl-wrap"), tb = el("table", "tbl"), head = el("tr"), now = nowSec();
  for (const h of ["名字", "认识", "一起玩过", "见过", "上次在身边", "说过 / 跟你", "装扮"]) head.append(el("th", "", h));
  tb.append(head);
  for (const c of cards) {
    const tr = el("tr", c.friend ? "" : "stranger");
    const who = el("td", "", c.name); if (!c.friend) who.append(el("div", "note", "不在好友名单里"));
    tr.append(who, el("td", "", `${Math.max(1, Math.ceil((now - c.first_met) / 86400))} 天`),
      el("td", "", `${c.days} 天`), el("td", "", `${c.visits} 次`), el("td", "", c.last_seen ? `${span(now - c.last_seen)}前` : "还没见过"),
      el("td", "", `${c.said} / ${c.to_me}`));
    const fit = el("td", "");  // 最近几套装扮（新的在上），来自 main 的认装扮
    for (const o of (c.outfits || []).slice().reverse())
      fit.append(el("div", "note", `${o.desc || "（还没描述）"}（${o.first === o.last ? o.first : o.first + " ~ " + o.last}）`));
    tr.append(fit); tb.append(tr);
  }
  wrap.append(tb); container.append(wrap);
}

/* 6. 日子和日记 */
function renderDays(container, days, diaries) {
  container.textContent = "";
  container.append(el("h4", "note", "最近 10 次上线"));
  if (days.length) { const ul = el("ul", "days"); for (const x of days) ul.append(el("li", "", x)); container.append(ul); }
  else container.append(el("p", "none", "（还没有）"));
  container.append(el("h4", "note", "日记"));
  if (!diaries.length) { container.append(el("p", "none", "（还没有）")); return; }
  for (const x of diaries) {
    const i = x.indexOf("："), p = el("p", "diary");
    if (i > 0 && i < 8) { p.append(el("b", "", x.slice(0, i)), x.slice(i + 1)); } else p.textContent = x;
    container.append(p);
  }
}

function pageActive() { return !$("page-inner").hidden; }
// 纯判断：状态变了要重读；5 秒的定时器在团子（这页在看的那个）醒着时也重读。onState 每秒都来，不带 periodic，所以只在变化时读。
function innerShouldReload(prevState, state, periodic, mine) { return state !== prevState || (!!periodic && state === "running" && !!mine); }
function pullInner(state, periodic) {
  const r = state && state.run, st = r ? r.state : null, prev = IN.lastState;
  IN.lastState = st;
  if (!pageActive() || !r) return;
  if (innerShouldReload(prev, st, periodic, innerMine(r))) loadInner(IN.source);
}
let ticker = null;
function bind() {
  $("inner-refresh").onclick = () => loadInner(IN.source);
  $("inner-changed").onchange = () => renderLog($("inner-log"), (IN.data && IN.data.log) || []);
  for (const b of $("inner-range").querySelectorAll("button")) b.onclick = () => {
    IN.hours = Number(b.dataset.h);
    for (const x of $("inner-range").querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
    if (IN.data) renderCurve($("inner-curve"), IN.data.log || [], IN.data);
  };
  window.addEventListener("resize", () => { if (IN.data && pageActive()) renderCurve($("inner-curve"), IN.data.log || [], IN.data); });
  for (const b of $("inner-source-toggle").querySelectorAll("button")) b.onclick = () => { setSource(b.dataset.src); loadInner(IN.source); };
}

globalThis.Inner = {renderNow, setSource, innerShouldReload};  // 沙盒页右栏调 Inner.renderNow(容器, now, d)
Pages.inner = {
  init() { bind(); },
  show() {  // 可能被重复调用（点当前导航项）：只重读，定时器不重复建
    loadInner(IN.source);
    if (!ticker) ticker = setInterval(() => { if (pageActive()) pullInner(S.state, true); }, 5000);
  },
  hide() { if (ticker) { clearInterval(ticker); ticker = null; } },
};
onState(st => pullInner(st));
})();
