/* 团子跑道（真机页中栏，画面和大脑控制台之间）：三颗团子沿闭环跑道一格一格走，每一步都是团子真的做了的一件事。
 * 数据：live/flow 长轮询（brain/flow.py 的 FlowLog），一条事件 = {id, t, node, label, outcome, speaker?, value?, model?, cost?}。
 * - 团子只往前走：目标节点在身后就绕一圈；同一个节点再来一条只原地更新
 * - 事件来得比动画快就排队；积压多了加速（tempo），太多了前面的只更新贴纸、最后几条照常播（compact）
 * - 拿不准（to_me unsure）/ 被拦（can_send ask）：团子停住、头顶冒金色「？」、轨道变暗、「停下来 · …」亮起
 * - 没有新事件：团子在原地轻轻呼吸，不自己跑；页面看不见（document.hidden / 不在这一页）时停掉动画、不拉数据
 * - prefers-reduced-motion：拉丝不扰动、不弹跳、不放星星
 * 页面里不能有网址字样（离线测试），SVG 的命名空间从 HTML 里现成的 <svg> 上取。
 * 顶层不碰 document：node 里 require 它能测 NODES / stickerOf / subOf / haltText / compact / tempo / ahead（tests/test_console_page.py）。 */
(function () {
"use strict";

/* ---- 跑道几何：左边中点出发，顺时针（viewBox 1000×260） ---- */
const R = 74, CX1 = 150, CX2 = 850, CY = 130, TOP = CY - R, BOT = CY + R, ST = CX2 - CX1, QA = Math.PI * R / 2, HA = Math.PI * R;
const LENGTH = 2 * QA + 2 * ST + HA;
function arc(cx, a) { const c = Math.cos(a), s = Math.sin(a); return {x: cx + R * c, y: CY + R * s, nx: c, ny: s}; }
function at(s) {
  s = ((s % LENGTH) + LENGTH) % LENGTH;
  if (s < QA) return arc(CX1, Math.PI + s / R); s -= QA;
  if (s < ST) return {x: CX1 + s, y: TOP, nx: 0, ny: -1}; s -= ST;
  if (s < HA) return arc(CX2, -Math.PI / 2 + s / R); s -= HA;
  if (s < ST) return {x: CX2 - s, y: BOT, nx: 0, ny: 1}; s -= ST;
  return arc(CX1, Math.PI / 2 + s / R);
}
function ahead(from, to) { const d = ((to - from) % LENGTH + LENGTH) % LENGTH; return d < 0.5 ? 0 : d; }  // 往前走到 to 要多远

/* ---- 节点：act = 做一件事（药丸），gate = 做一个决定（软菱形） ---- */
const NODES = [
  {id: "look", f: 0, kind: "act", t: "看一眼", sub: "截屏 · 眼睛"},
  {id: "has_new", f: .13, kind: "gate", verb: "数一数", q: "有新消息吗？"},
  {id: "read", f: .21, kind: "act", t: "读面板", sub: "拆说话人"},
  {id: "to_me", f: .29, kind: "gate", verb: "认一认", q: "是在跟我说吗？"},
  {id: "batch", f: .37, kind: "act", t: "攒一攒", sub: "合并连发"},
  {id: "reply_or_not", f: .5, kind: "gate", verb: "想不想", q: "回不回？"},
  {id: "think", f: .63, kind: "act", t: "想一想", sub: "大脑"},
  {id: "say_or_act", f: .71, kind: "gate", verb: "选一选", q: "说话还是动作？"},
  {id: "can_send", f: .79, kind: "gate", verb: "能不能", q: "能发吗？"},
  {id: "send", f: .87, kind: "act", t: "发出去", sub: "限速"},
].map(n => ({...n, s: Math.round(n.f * LENGTH * 10) / 10}));
const INDEX = Object.fromEntries(NODES.map((n, i) => [n.id, i]));

/* ---- 纯函数（node 里测） ---- */
const TONE = {ok: "ok", skip: "skip", unsure: "halt", ask: "halt", sent: "ok"};
function short(text, n) { text = String(text || ""); return text.length > n ? text.slice(0, n) + "…" : text; }
function fmtCost(c) { return c == null ? "" : "¥" + (c < 0.01 ? c.toFixed(4) : c.toFixed(c < 1 ? 3 : 2)); }
function stickerOf(e) { return {text: short(e.label, 14), tone: TONE[e.outcome] || "ok"}; }
function subOf(e) { return [e.label, fmtCost(e.cost)].filter(Boolean).join(" · "); }
function haltText(e) {
  if (e.outcome === "unsure") return "停下来 · 拿不准";
  if (e.outcome === "ask") return "停下来 · 被拦了";
  return null;
}
const KEEP = 4, SKIP_OVER = 20;
function compact(queue) {  // 积压超过 SKIP_OVER 条：前面的只更新贴纸（silent），最后 KEEP 条照常播
  if (queue.length <= SKIP_OVER) return {silent: [], play: queue.slice()};
  return {silent: queue.slice(0, -KEEP), play: queue.slice(-KEEP)};
}
function tempo(backlog) { return backlog > 12 ? 4 : backlog > 4 ? 2.5 : 1; }

/* ---- 画 ---- */
const PAUSE = {base: 380, halt: 1800, send: 1300};
const STRANDS = 9, SAMPLES = 48, STRAND_MS = 50;

function mount(box, url, opts) {
  opts = opts || {};
  const visible = opts.visible || (() => true);
  const svg = box.querySelector("svg"), NS = svg.namespaceURI;
  const reduce = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const $ = sel => svg.querySelector(sel);
  const T = {stopped: false, v: 0, boot: "", queue: [], cur: 0, s: 0, phase: "idle", move: null, until: 0, arrived: 0,
    bubbleUntil: 0, haltUntil: 0, lastStrand: 0, raf: 0, stars: [], ctrl: null, empty: true};

  /* 底子：色带、拉丝、虚线车道 */
  const loop = `M${CX1 - R},${CY} A${R},${R} 0 0 1 ${CX1},${TOP} H${CX2} A${R},${R} 0 0 1 ${CX2},${BOT} H${CX1} A${R},${R} 0 0 1 ${CX1 - R},${CY} Z`;
  svg.textContent = "";
  el("path", {class: "tk-band", d: loop}, svg);
  const strandG = el("g", {class: "tk-strands"}, svg), strands = [];
  const tones = ["tk-s-pink", "tk-s-rice", "tk-s-matcha"];
  for (let k = 0; k < STRANDS; k++) {
    const r1 = Math.sin(k * 12.9898) * 43758.5453, a = r1 - Math.floor(r1), r2 = Math.sin(k * 78.233) * 12345.678, b = r2 - Math.floor(r2);
    strands.push({e: el("path", {class: tones[k % 3], "stroke-opacity": (.22 + a * .18).toFixed(2)}, strandG),
      off: (a - .5) * 22, amp: 2 + b * 4, m: 2 + (k % 4), w: .5 + a * .8, p: b * 6.28});
  }
  el("path", {class: "tk-lane", d: loop}, svg);

  /* 中间：最后一步 + 停下来的牌子 */
  const lastT = el("text", {class: "tk-last", x: 500, y: CY - 12}, svg); lastT.textContent = "等团子的第一步…";
  const halt = el("g", {class: "tk-halt", transform: `translate(500,${CY + 12})`}, svg);
  el("rect", {x: -78, y: -14, width: 156, height: 28, rx: 14}, halt);
  const haltT = el("text", {y: 5}, halt); haltT.textContent = "停下来 · 拿不准";

  /* 节点 */
  const layer = el("g", {}, svg);
  for (const n of NODES) {
    const p = at(n.s), g = el("g", {class: "tk-node " + n.kind, transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`}, layer);
    n.g = g; n.tip = el("title", {}, g);
    const up = p.y < CY - 5, mid = Math.abs(p.y - CY) < 5;
    if (n.kind === "act") {
      el("rect", {class: "tk-pill", x: -70, y: -22, width: 140, height: 44, rx: 22}, g);
      el("text", {class: "tk-nt", y: -2}, g).textContent = n.t;
      n.subEl = el("text", {class: "tk-ns", y: 14}, g); n.subEl.textContent = n.sub;
    } else {
      el("rect", {class: "tk-dia", x: -19, y: -19, width: 38, height: 38, rx: 11, transform: "rotate(45)"}, g);
      el("text", {class: "tk-nv", y: 5}, g).textContent = n.verb;
      const qx = mid ? -44 : 0, qy = mid ? 5 : up ? 44 : -36, sy = up ? -40 : mid ? -42 : 44;
      el("text", {class: "tk-nq" + (mid ? " end" : ""), x: qx, y: qy}, g).textContent = n.q;
      const o = el("g", {transform: `translate(0,${sy})`}, g), stk = el("g", {class: "tk-stk tone-skip"}, o);
      const r = el("g", {transform: `rotate(${((n.f * 97) % 7 - 3.5).toFixed(1)})`}, stk);
      n.stkRect = el("rect", {x: -18, y: -11, width: 36, height: 22, rx: 7}, r);
      n.stkText = el("text", {y: 4}, r); n.stkText.textContent = "…"; n.stk = stk;
    }
  }

  /* 团子串 + 气泡 + 星星 */
  const starG = el("g", {}, svg), dG = el("g", {class: "tk-dango"}, svg);
  const tail = el("circle", {r: 10, class: "d-matcha"}, dG), midD = el("circle", {r: 10, class: "d-rice"}, dG), lead = el("g", {}, dG);
  el("circle", {r: 11, class: "d-pink"}, lead);
  el("circle", {cx: -3.6, cy: -2, r: 1.6, class: "eye"}, lead); el("circle", {cx: 3.6, cy: -2, r: 1.6, class: "eye"}, lead);
  el("ellipse", {cx: -6.5, cy: 3, rx: 2.4, ry: 1.5, class: "blush"}, lead); el("ellipse", {cx: 6.5, cy: 3, rx: 2.4, ry: 1.5, class: "blush"}, lead);
  const bub = el("g", {class: "tk-bub", visibility: "hidden"}, svg), bubR = el("rect", {x: -18, y: -13, width: 36, height: 26, rx: 13}, bub);
  const bubT = el("text", {y: 5}, bub);

  function setSticker(n, e, quiet) {
    if (n.kind === "act") { n.subEl.textContent = short(subOf(e), 18); n.tip.textContent = subOf(e); return; }
    const st = stickerOf(e); n.stkText.textContent = st.text; n.tip.textContent = e.label + (e.speaker ? `（${e.speaker}）` : "");
    let w = 36; try { w = Math.max(36, n.stkText.getComputedTextLength() + 18); } catch (_) {}
    n.stkRect.setAttribute("x", -w / 2); n.stkRect.setAttribute("width", w);
    n.stk.setAttribute("class", "tk-stk tone-" + st.tone);
    if (!quiet && !reduce) { void n.stk.getBoundingClientRect(); n.stk.classList.add("pop"); }
  }
  function bubble(text, kind, ms) {
    bubT.textContent = text; bub.setAttribute("class", "tk-bub" + (kind ? " " + kind : ""));
    let w = 32; try { w = Math.max(32, bubT.getComputedTextLength() + 20); } catch (_) {}
    bubR.setAttribute("x", -w / 2); bubR.setAttribute("width", w);
    bub.setAttribute("visibility", "visible"); T.bubbleUntil = performance.now() + ms;
  }
  function burst() {
    if (reduce) return;
    const p = at(T.s);
    for (let k = 0; k < 6; k++) {
      const a = k / 6 * Math.PI * 2 + Math.random(), e = el("text", {class: "tk-star"}, starG); e.textContent = "✦";
      T.stars.push({e, x: p.x, y: p.y, vx: Math.cos(a) * 50, vy: Math.sin(a) * 50 - 28, life: 0});
    }
  }
  function lastLine(e) {
    const t = e.t ? new Date(e.t * 1000).toTimeString().slice(0, 8) : "";
    lastT.textContent = `${t} · ${short(e.label, 26)}`;
  }
  function highlight(i) { NODES.forEach((m, j) => m.g.classList.toggle("on", j === i)); }
  function clearHalt() { svg.classList.remove("halt"); halt.classList.remove("lit"); NODES.forEach(m => m.g.classList.remove("halt")); }

  /* 到一个节点：返回停多久（毫秒，已按积压加速） */
  function arrive(e) {
    const i = INDEX[e.node], n = NODES[i], speed = tempo(T.queue.length), now = performance.now();
    T.cur = i; T.s = n.s; T.arrived = now; T.empty = false;
    setSticker(n, e, false); highlight(i); lastLine(e);
    if (T.haltUntil <= now) clearHalt();
    let pause = PAUSE.base;
    const h = haltText(e);
    if (h) {
      pause = PAUSE.halt; haltT.textContent = h; svg.classList.add("halt"); halt.classList.add("lit"); n.g.classList.add("halt");
      T.haltUntil = now + pause / speed; bubble("？", "ask", pause / speed);
    } else if (e.node === "send" && e.outcome === "sent") {
      pause = PAUSE.send; bubble(short(e.label, 16), "say", pause / speed); burst();
    } else if (e.node === "think" && e.label === "在想…") {
      bubble("…", "", 1e9);  // 一直冒着，直到下一步
    } else T.bubbleUntil = 0;
    return pause / speed;
  }
  function next(now) {
    if (!T.queue.length) return;
    const c = compact(T.queue);
    for (const e of c.silent) if (e.node in INDEX) { setSticker(NODES[INDEX[e.node]], e, true); lastLine(e); }
    T.queue = c.play;
    const e = T.queue.shift(); if (!(e.node in INDEX)) return;
    const dist = ahead(T.s, NODES[INDEX[e.node]].s);
    if (!dist) { T.until = now + arrive(e); T.phase = "pause"; return; }
    const speed = tempo(T.queue.length);
    T.move = {e, from: T.s, dist, start: now, dur: Math.min(1800, Math.max(320, dist / 0.42)) / speed};
    T.phase = "move"; highlight(-1);
  }
  const ease = t => t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

  function drawStrands(t) {
    for (const s of strands) {
      let d = "";
      for (let i = 0; i <= SAMPLES; i++) {
        const ss = i / SAMPLES * LENGTH, q = at(ss);
        const off = s.off + (reduce ? 0 : s.amp * Math.sin(ss / LENGTH * Math.PI * 2 * s.m + t * s.w + s.p));
        d += (i ? "L" : "M") + (q.x + q.nx * off).toFixed(1) + "," + (q.y + q.ny * off).toFixed(1);
      }
      s.e.setAttribute("d", d + "Z");
    }
  }
  function placeDango(now) {
    const a = at(T.s), b = at(T.s - 20), c = at(T.s - 40);
    tail.setAttribute("cx", c.x.toFixed(1)); tail.setAttribute("cy", c.y.toFixed(1));
    midD.setAttribute("cx", b.x.toFixed(1)); midD.setAttribute("cy", b.y.toFixed(1));
    let sx = 1, sy = 1;
    if (!reduce && T.phase === "pause") { const t = now - T.arrived, q = .2 * Math.exp(-t / 140) * Math.cos(t / 38); sx = 1 + q; sy = 1 - q; }
    else if (!reduce && T.phase === "idle") { const q = .035 * Math.sin(now / 520); sx = 1 + q; sy = 1 - q; }  // 原地呼吸
    lead.setAttribute("transform", `translate(${a.x.toFixed(1)},${a.y.toFixed(1)}) scale(${sx.toFixed(3)},${sy.toFixed(3)})`);
    if (now > T.bubbleUntil) bub.setAttribute("visibility", "hidden");
    else bub.setAttribute("transform", `translate(${a.x.toFixed(1)},${(a.y < CY - 5 ? a.y + 30 : a.y - 28).toFixed(1)})`);  // 上排：贴纸在外侧，气泡放内侧
  }
  let last = 0;
  function frame(now) {
    T.raf = 0;
    if (T.stopped || !running()) return;
    const calm = T.phase === "idle" && !T.queue.length && !T.stars.length;
    if (calm && reduce && now > T.bubbleUntil) { placeDango(now); return; }  // 不动的时候不占帧；新事件来了 wake()
    if (calm && now - last < STRAND_MS) { T.raf = requestAnimationFrame(frame); return; }  // 只是在呼吸：约 20 帧/秒就够
    const dt = Math.min(64, now - (last || now)); last = now;
    if (T.phase === "idle") next(now);
    if (T.phase === "move") {
      const m = T.move, p = Math.min(1, (now - m.start) / m.dur);
      T.s = m.from + m.dist * ease(p);
      if (p >= 1) { T.until = now + arrive(m.e); T.phase = "pause"; }
    } else if (T.phase === "pause" && now >= T.until) T.phase = "idle";
    if (T.haltUntil && now >= T.haltUntil) { T.haltUntil = 0; clearHalt(); }
    placeDango(now);
    T.stars = T.stars.filter(s => {
      s.life += dt / 1000; s.x += s.vx * dt / 1000; s.y += s.vy * dt / 1000; s.vy += 60 * dt / 1000;
      s.e.setAttribute("x", s.x.toFixed(1)); s.e.setAttribute("y", s.y.toFixed(1)); s.e.setAttribute("opacity", Math.max(0, 1 - s.life / .9).toFixed(2));
      if (s.life > .9) { s.e.remove(); return false; } return true;
    });
    if (!reduce && now - T.lastStrand > STRAND_MS) { T.lastStrand = now; drawStrands(now / 1000); }
    T.raf = requestAnimationFrame(frame);
  }
  function running() { return !(typeof document !== "undefined" && document.hidden) && visible(); }
  function wake() { if (!T.raf && !T.stopped && running()) { last = 0; T.raf = requestAnimationFrame(frame); } }
  const onVis = () => wake();
  document.addEventListener("visibilitychange", onVis);

  /* 第一次拿到（接上正在跑的团子）：不回放，每个节点贴上最后一条，团子直接站到最后那一步 */
  function jump(events) {
    for (const e of events) if (e.node in INDEX) setSticker(NODES[INDEX[e.node]], e, true);
    const e = events[events.length - 1];
    if (e && e.node in INDEX) { T.cur = INDEX[e.node]; T.s = NODES[T.cur].s; highlight(T.cur); lastLine(e); T.empty = false;
      if (e.node === "think" && e.label === "在想…") bubble("…", "", 1e9); }
  }
  function reset() {
    T.queue = []; T.phase = "idle"; T.s = 0; T.cur = 0; T.bubbleUntil = 0; T.haltUntil = 0; clearHalt(); highlight(0);
    for (const n of NODES) { if (n.kind === "act") { n.subEl.textContent = n.sub; n.tip.textContent = ""; } else { n.stkText.textContent = "…"; n.stk.setAttribute("class", "tk-stk tone-skip"); } }
    lastT.textContent = "等团子的第一步…";
  }
  drawStrands(0); highlight(0); placeDango(0);

  (async function poll() {
    let first = true;
    while (!T.stopped) {
      if (!running()) { await new Promise(ok => setTimeout(ok, 1000)); continue; }
      try {
        T.ctrl = new AbortController();
        const r = await fetch(`${url}?after=${T.v}&wait=2`, {cache: "no-store", signal: T.ctrl.signal});
        if (T.stopped) return;
        if (r.status === 404) { await new Promise(ok => setTimeout(ok, 3000)); continue; }  // 身体还没建好
        if (!r.ok) throw new Error(r.status);
        const d = await r.json(); if (T.stopped) return;
        if (d.boot !== T.boot) {  // 换了一个团子进程：清空，从头拉一次（这一份是按旧版本号给的）
          const again = !!T.boot && T.v !== 0;
          if (T.boot) reset();
          T.boot = d.boot; first = true;
          if (again) { T.v = 0; continue; }
        }
        const evs = d.events || [];
        if (first) { jump(evs); first = false; } else T.queue.push(...evs);
        T.v = d.v; wake();
      } catch (e) {
        if (T.stopped) return;
        await new Promise(ok => setTimeout(ok, 1000));
      }
    }
  })();
  wake();
  return {
    stop() {  // 停下：画面留着最后的样子
      T.stopped = true; if (T.ctrl) T.ctrl.abort(); if (T.raf) cancelAnimationFrame(T.raf); T.raf = 0;
      document.removeEventListener("visibilitychange", onVis);
    },
    wake,
  };
}

const api = {NODES, LENGTH, at, ahead, stickerOf, subOf, haltText, fmtCost, compact, tempo, mount};
if (typeof module !== "undefined" && module.exports) module.exports = api;
globalThis.DangoTrack = api;
})();
