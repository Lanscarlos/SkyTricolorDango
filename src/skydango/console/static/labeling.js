/* 标注页（spec 2026-10-01-gesture-labeling-training §3）：左 片段列表（筛选 + 各类计数），中 16 帧动图（canvas 放大 3 倍、按 fps 循环），
 * 右 Claude 的猜测 + 按钮。确认 / 改类别 / 丢弃 / 撤销走 api/gesture/*，后端把片段文件夹挪到对应目录。
 * 键盘（焦点不在输入框、没开对话框时）：1~5 = 挥手 / 鞠躬 / 欢呼 / 害羞 / 都不是、0 不要、Z 撤销、空格 暂停、← → 逐帧（暂停时）。
 * 动作页没有「同意 Claude」（10-03 去掉：Claude 初分认不出动作，直接人工标）；有旧的猜测照样显示、按钮上标出来。
 * 页顶「动作 / 外形 / 动作名」三个标签：外形页标人物裁图（api/form/*，Enter / 1~6 = 外形类别 / 0 不要 / Z 撤销，只在外形页开着时有效）；
 * 动作名页给动作图标起名（emotenames.js，这里只管切过去和刷新，开着时这里的按键都不生效）。
 * 模型写的理由一律 textContent。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const FRAMES = 16, FPS = 8, PX = 336, AHEAD = 3, KEEP = 12;  // PX = 112 × 3；AHEAD = 预加载后几段；KEEP = 最多缓存几段的帧
const UNL = "_unlabeled", DIS = "_discard";
const LB = {data: null, filter: "todo", list: [], cur: null, i: 0, playing: true, speed: 1, timer: null,
  busy: false, active: false, loading: false, cache: new Map(), keyBound: false, tab: "gesture"};
/* 外形页（感知层第二层的人物裁图）：条目是单张 jpg，类别 = FORMS（后端给），接口 api/form/*。按键 Enter / 1~6 / 0 / Z 只在外形页开着时有效。 */
const FM = {data: null, filter: "todo", list: [], cur: null, busy: false, loading: false};
const FORM_NAMES = {not_person: "不是人", lit: "点亮的人", unlit: "黑影", spirit: "先祖", shared: "共享空间", morph: "变身"};

/* ---- 名字 ---- */
function nameOf(l) {
  if (l === "none") return "都不是";
  if (l === "discard" || l === DIS) return "不要";
  if (l === "unsure") return "看不清";
  if (l === UNL) return "待确认";
  return (LB.data && LB.data.names && LB.data.names[l]) || l;
}
function whereText(w) { return w === UNL ? "待确认" : w === DIS ? "丢弃了" : `已确认：${nameOf(w)}`; }
function shortName(clip) { const i = clip.indexOf("__"); return i < 0 ? clip : clip.slice(i + 2); }
function keyOrder() {  // 按键 1~n 对应的类别：先四个动作，最后"都不是"
  const labels = (LB.data && LB.data.labels) || [];
  const acts = labels.filter(l => l !== "none");
  if (labels.includes("none")) acts.push("none");
  return acts;
}
function pct(c) { return Math.round((Number(c) || 0) * 100) + "%"; }

/* ---- 列表 / 标注 / 撤销 / 读数据：动作页和外形页共用一份，差别都写在页面描述 P 里 ----
 * P = {s 状态对象, tab, itemsKey 数据里条目数组的字段, key 条目名字段, post 标注请求里名字的字段, api 接口前缀, ids 元素 id 前缀, noun,
 *      labels() 类别（按键顺序）, name(l), whereText(w), short(name), agreeable(guess), select(name), renderSide(), row(c) 列表行的内容,
 *      rowClass, onCounts(n), onError()} */
function guessKey(c) { return c.guess ? c.guess.label : "-"; }
function matches(c, f) {
  if (f === "todo") return c.where === UNL;
  if (f === "discard") return c.where === DIS;
  if (f.startsWith("guess:")) return c.where === UNL && guessKey(c) === f.slice(6);
  if (f.startsWith("done:")) return c.where === f.slice(5);
  return false;
}
function rank(c) { return !c.guess ? 1 : c.guess.label === "unsure" ? 0 : 2; }  // 看不清最前，没猜的其次，其余按把握从低到高
function pItems(P) { return (P.s.data && P.s.data[P.itemsKey]) || []; }
function pBy(P, name) { return name == null ? null : pItems(P).find(c => c[P.key] === name) || null; }
function pBuild(P) {
  const f = P.s.filter, k = P.key, list = pItems(P).filter(c => matches(c, f));
  if (f === "todo" || f.startsWith("guess:"))
    list.sort((a, b) => rank(a) - rank(b) || (a.guess ? a.guess.confidence : 0) - (b.guess ? b.guess.confidence : 0) || (a[k] < b[k] ? -1 : a[k] > b[k] ? 1 : 0));
  P.s.list = list;
}
function pCounts(P) {
  const n = {};
  for (const c of pItems(P)) n[c.where] = (n[c.where] || 0) + 1;
  return n;
}
function pValid(P, f) {
  const labels = P.labels();
  if (f === "todo" || f === "discard") return true;
  if (f.startsWith("done:")) return labels.includes(f.slice(5));
  if (f.startsWith("guess:")) return [...labels, "unsure", "-"].includes(f.slice(6));
  return false;
}
function pRenderFilter(P) {
  const sel = $(P.ids + "-filter"), n = pCounts(P), pending = pItems(P).filter(c => c.where === UNL);
  const gn = {}; for (const c of pending) gn[guessKey(c)] = (gn[guessKey(c)] || 0) + 1;
  sel.textContent = "";
  const opt = (v, t) => { const o = el("option", "", t); o.value = v; return o; };
  sel.append(opt("todo", `待确认（${n[UNL] || 0}）`));
  const g1 = el("optgroup"); g1.label = "Claude 猜的（待确认里）";
  for (const g of [...P.labels(), "unsure", "-"]) {
    const v = "guess:" + g;
    if ((gn[g] || 0) || P.s.filter === v) g1.append(opt(v, `${g === "-" ? "还没猜" : "猜" + P.name(g)}（${gn[g] || 0}）`));
  }
  if (g1.children.length) sel.append(g1);
  const g2 = el("optgroup"); g2.label = "已确认";
  for (const l of P.labels()) g2.append(opt("done:" + l, `${P.name(l)}（${n[l] || 0}）`));
  sel.append(g2);
  sel.append(opt("discard", `丢弃的（${n[DIS] || 0}）`));
  sel.value = P.s.filter;
}
function pRenderCounts(P) {
  const box = $(P.ids + "-counts"), n = pCounts(P);
  box.textContent = "";
  const item = (f, label, k) => {
    const b = el("button", "lb-count"); b.type = "button";
    b.setAttribute("aria-pressed", String(P.s.filter === f));
    b.append(label, el("b", "", String(n[k] || 0)));
    b.onclick = () => pSetFilter(P, f);
    box.append(b);
  };
  item("todo", "待确认", UNL);
  for (const l of P.labels()) item("done:" + l, P.name(l), l);
  item("discard", "丢弃", DIS);
  P.onCounts(n);
}
function guessTag(P, c) {
  if (!c.guess) return el("span", "tag", "没猜");
  const g = c.guess;
  return el("span", "tag " + (g.label === "unsure" ? "warn" : "sakura"), g.label === "unsure" ? "看不清" : `${P.name(g.label)} ${pct(g.confidence)}`);
}
function pRenderList(P) {
  const box = $(P.ids + "-list");
  box.textContent = "";
  if (!P.s.list.length) {
    box.append(el("p", "none lb-empty", P.s.filter === "todo" ? (pItems(P).length ? "都标完了。" : `还没有${P.noun}。`) : `这里没有${P.noun}。`));
    return;
  }
  const frag = document.createDocumentFragment();
  for (const c of P.s.list) {
    const b = el("button", P.rowClass); b.type = "button"; b.dataset.k = c[P.key];
    b.setAttribute("aria-current", String(c[P.key] === P.s.cur));
    b.title = c[P.key];
    b.append(...P.row(c));
    b.onclick = () => P.select(c[P.key]);
    frag.append(b);
  }
  box.append(frag);
}
function pMarkCurrent(P) {
  let hit = null;
  for (const b of $(P.ids + "-list").querySelectorAll(".lb-item")) {
    const on = b.dataset.k === P.s.cur;
    b.setAttribute("aria-current", String(on));
    if (on) hit = b;
  }
  if (hit) hit.scrollIntoView({block: "nearest"});
}
function pRerender(P) { pBuild(P); pRenderFilter(P); pRenderCounts(P); pRenderList(P); }
function pSetFilter(P, f) {
  if (!pValid(P, f)) f = "todo";
  P.s.filter = f;
  pRerender(P);
  const keep = P.s.list.some(c => c[P.key] === P.s.cur);
  P.select(keep ? P.s.cur : (P.s.list[0] ? P.s.list[0][P.key] : null));
}
function pApply(P, item) {
  const c = pBy(P, item[P.key]);
  if (c) Object.assign(c, item);
  else pItems(P).push(item);
}
function pAdvance(P, old, idx) {  // 选完跳下一条：原列表里排在它后面、还留在新列表里的第一条；后面没了往前找
  const k = P.key;
  pRerender(P);
  const left = new Set(P.s.list.map(c => c[k]));
  for (let j = idx + 1; j < old.length; j++) if (old[j][k] !== P.s.cur && left.has(old[j][k])) return P.select(old[j][k]);
  for (let j = idx - 1; j >= 0; j--) if (old[j][k] !== P.s.cur && left.has(old[j][k])) return P.select(old[j][k]);
  P.select(left.has(P.s.cur) ? P.s.cur : (P.s.list[0] ? P.s.list[0][k] : null));
}
async function pLabel(P, to) {
  const c = pBy(P, P.s.cur);
  if (!c || P.s.busy) return;
  const old = P.s.list.slice(), idx = old.findIndex(x => x[P.key] === c[P.key]);
  if (c.where === (to === "discard" ? DIS : to)) { pAdvance(P, old, idx); return; }  // 已经是这一类：当作确认过，跳下一条
  P.s.busy = true; P.renderSide();
  let r;
  try { r = await post(P.api + "/label", {[P.post]: c[P.key], to}); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  P.s.busy = false;
  if (!r.data.ok) {
    toast(r.data.text || "没标上", "bad");
    if (r.status === 409) pLoad(P); else P.renderSide();
    return;
  }
  pApply(P, r.data);
  pAdvance(P, old, idx);
}
function pAgree(P) {
  const c = pBy(P, P.s.cur);
  if (c && c.guess && P.agreeable(c.guess)) pLabel(P, c.guess.label);  // 看不清 / 没猜：不动
}
async function pUndo(P) {
  if (P.s.busy) return;
  P.s.busy = true; P.renderSide();
  let r;
  try { r = await post(P.api + "/undo"); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  P.s.busy = false;
  if (!r.data.ok) { toast(r.data.text || "没撤销成", r.status === 409 ? "warn" : "bad"); P.renderSide(); return; }
  pApply(P, r.data);
  pRerender(P); P.select(r.data[P.key]);
  toast(`撤销了：${P.short(r.data[P.key])} 回到「${P.whereText(r.data.where)}」`, "ok");
}
async function pLoad(P) {
  const s = P.s;
  if (s.loading) return;
  s.loading = true;
  let d;
  try { d = await getJSON(P.api + "/state"); }
  catch (e) { d = {ok: false, text: `读不到${P.noun}（面板停了？）`}; }
  finally { s.loading = false; }
  const banner = $(P.ids + "-empty");
  if (!d || !d.ok) {
    s.data = null; s.list = []; s.cur = null;
    banner.textContent = ""; banner.append(el("span", "", (d && d.text) || `读不到${P.noun}`));
    banner.hidden = LB.tab !== P.tab; $(P.ids + "-cols").hidden = true;
    if (P.onError) P.onError();
    return;
  }
  s.data = d; banner.hidden = true; $(P.ids + "-cols").hidden = LB.tab !== P.tab;
  if (!pValid(P, s.filter)) s.filter = "todo";
  pRerender(P);
  const keep = pBy(P, s.cur);
  P.select(keep ? s.cur : (s.list[0] ? s.list[0][P.key] : null));
}



/* ---- 帧：每段 16 张 Image 预加载，canvas 画（接口是 no-store，换 img.src 会重新下载） ---- */
function frameUrl(clip, i) { return `api/gesture/frame?clip=${encodeURIComponent(clip)}&i=${i}`; }
function frames(clip) {
  let e = LB.cache.get(clip);
  if (e) { LB.cache.delete(clip); LB.cache.set(clip, e); return e; }  // 最近用过的挪到最后
  e = {imgs: [], bad: new Set()};
  for (let i = 0; i < FRAMES; i++) {
    const img = new Image();
    img.onload = () => { if (LB.cur === clip && LB.i === i) draw(); };
    img.onerror = () => { e.bad.add(i); if (LB.cur === clip) draw(); };
    img.src = frameUrl(clip, i);
    e.imgs.push(img);
  }
  LB.cache.set(clip, e);
  for (const [k, v] of LB.cache) {  // 超出就扔最久没用的
    if (LB.cache.size <= KEEP) break;
    if (k === clip) continue;
    for (const img of v.imgs) { img.onload = img.onerror = null; }
    LB.cache.delete(k);
  }
  return e;
}
function preload() {
  if (!LB.cur) return;
  frames(LB.cur);
  const at = LB.list.findIndex(c => c.clip === LB.cur);
  for (let j = at + 1; j <= at + AHEAD && j < LB.list.length; j++) frames(LB.list[j].clip);
  frames(LB.cur);  // 当前这段最后碰：缓存满了也不会被挤掉
}
function draw() {
  const cv = $("lb-canvas"), ctx = cv.getContext("2d"), note = $("lb-stage-note");
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1)), w = Math.round(PX * dpr);
  if (cv.width !== w) { cv.width = w; cv.height = w; }
  ctx.clearRect(0, 0, w, w);
  let text = "";
  if (!LB.cur) text = LB.data ? (LB.list.length ? "从列表里选一段" : "这个筛选里没有片段") : "";
  else {
    const e = LB.cache.get(LB.cur) || frames(LB.cur), img = e.imgs[LB.i];
    if (img && img.complete && img.naturalWidth) { ctx.imageSmoothingEnabled = true; ctx.drawImage(img, 0, 0, w, w); }
    else text = e.bad.has(LB.i) ? "这一帧读不到" : "载入中…";
    const ticks = $("lb-ticks").children;
    for (let i = 0; i < ticks.length; i++) {
      ticks[i].classList.toggle("on", i === LB.i);
      ticks[i].classList.toggle("bad", e.bad.has(i));
    }
  }
  note.textContent = text; note.hidden = !text;
  $("lb-frame").textContent = LB.cur ? `第 ${LB.i + 1} / ${FRAMES} 帧` : "";
}

/* ---- 播放 ---- */
function stopTimer() { if (LB.timer) { clearInterval(LB.timer); LB.timer = null; } }
function startTimer() {
  stopTimer();
  if (!LB.active || !LB.playing || LB.tab !== "gesture") return;
  LB.timer = setInterval(() => { if (!LB.cur) return; LB.i = (LB.i + 1) % FRAMES; draw(); }, 1000 / (FPS * LB.speed));
}
function setPlaying(p) {
  LB.playing = !!p;
  const b = $("lb-play"); b.textContent = ""; b.append(LB.playing ? "暂停" : "播放", " ", el("kbd", "k", "空格"));
  $("lb-prev").disabled = $("lb-next").disabled = LB.playing;
  startTimer();
}
function setSpeed(s) {
  LB.speed = s === 0.5 ? 0.5 : 1;
  for (const b of $("lb-speed").querySelectorAll("button")) b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === LB.speed));
  startTimer();
}
function step(d) {
  if (LB.playing || !LB.cur) return;
  LB.i = (LB.i + d + FRAMES) % FRAMES;
  draw();
}
function seek(i) {
  if (!LB.cur) return;
  if (LB.playing) setPlaying(false);
  LB.i = i; draw();
}

/* ---- 选中一段 ---- */
function select(clip) {
  if (clip !== LB.cur) LB.i = 0;
  LB.cur = clip || null;
  preload(); pMarkCurrent(G); renderHead(); renderSide(); draw();
}
function renderHead() {
  const at = LB.list.findIndex(c => c.clip === LB.cur);
  $("lb-title").textContent = LB.cur ? shortName(LB.cur) : "—";
  $("lb-title").title = LB.cur || "";
  $("lb-pos").textContent = !LB.cur ? "" : at < 0 ? "（不在当前筛选里）" : `第 ${at + 1} / ${LB.list.length} 段`;
}
function kbdBtn(cls, text, key, onClick) {
  const b = el("button", cls); b.type = "button";
  b.append(el("kbd", "k", key), el("span", "t", text));
  b.onclick = onClick;
  return b;
}
function renderSide() {
  const c = byName(LB.cur);
  // 猜测
  const gbox = $("lb-guess"); gbox.textContent = "";
  if (!c) gbox.append(el("p", "none", "—"));
  else if (!c.guess) gbox.append(el("p", "none", "没有 Claude 的猜测（它认不准动作，直接按键标）"));
  else {
    const g = c.guess, main = el("div", "lb-guess-main" + (g.label === "unsure" ? " unsure" : ""));
    main.append(el("b", "", nameOf(g.label)), el("span", "lb-conf", `把握 ${pct(g.confidence)}`));
    const bar = el("div", "lb-bar"), fill = el("i"); fill.style.width = pct(g.confidence); bar.append(fill);
    gbox.append(main, bar);
    if (g.reason) gbox.append(el("p", "lb-reason", g.reason));  // 模型写的：只当纯文本
  }
  // 这一段
  const meta = $("lb-meta"); meta.textContent = "";
  if (c) {
    const row = (k, v) => { meta.append(el("dt", "", k)); const dd = el("dd"); dd.append(v); meta.append(dd); };
    row("现在", el("span", "tag " + (c.where === UNL ? "" : c.where === DIS ? "bad" : "ok"), whereText(c.where)));
    row("录像", c.recording || "（没有录像名）");
    row("片段", el("code", "", c.clip));
  }
  // 按钮
  const acts = $("lb-actions"); acts.textContent = "";
  const off = !c || LB.busy, g = c && c.guess;
  const grid = el("div", "lb-grid");
  keyOrder().forEach((l, k) => {
    const b = kbdBtn("btn lb-act", nameOf(l), String(k + 1), () => label(l));
    if (c && c.where === l) { b.classList.add("on"); b.title = "现在就在这一类"; }
    if (g && g.label === l) b.classList.add("guess");
    b.disabled = off; grid.append(b);
  });
  const dis = kbdBtn("btn halt lb-act", "不要", "0", () => label("discard"));
  if (c && c.where === DIS) dis.classList.add("on");
  dis.disabled = off; grid.append(dis);
  acts.append(grid);
  const undoBtn = kbdBtn("btn sm lb-undo", "撤销上一步", "Z", undo);
  undoBtn.disabled = LB.busy;
  acts.append(undoBtn);
}

function fmName(l) { return l === "discard" || l === DIS ? "不要" : l === UNL ? "待确认" : l === "unsure" ? "看不清" : FORM_NAMES[l] || l; }
function fmWhereText(w) { return w === UNL ? "待确认" : w === DIS ? "丢弃了" : `已确认：${fmName(w)}`; }

function cropUrl(name) { return `api/form/crop?name=${encodeURIComponent(name)}`; }
function fmShowImages() {
  const im = $("fm-crop"), cx = $("fm-ctx"), n1 = $("fm-crop-note"), n2 = $("fm-ctx-note");
  n1.hidden = n2.hidden = true;
  if (!FM.cur) { im.removeAttribute("src"); im.hidden = true; cx.removeAttribute("src"); cx.hidden = true; n1.textContent = FM.data ? (FM.list.length ? "从列表里选一张" : "这个筛选里没有裁图") : ""; n1.hidden = !n1.textContent; return; }
  im.hidden = cx.hidden = false;
  im.onerror = () => { im.hidden = true; n1.textContent = "这张图读不到"; n1.hidden = false; };
  cx.onerror = () => { cx.hidden = true; n2.hidden = false; };
  im.src = cropUrl(FM.cur);
  cx.src = `api/form/context?name=${encodeURIComponent(FM.cur)}`;
}
function fmSelect(name) {
  FM.cur = name || null;
  const at = FM.list.findIndex(c => c.crop === FM.cur);
  for (let j = at + 1; at >= 0 && j <= at + AHEAD && j < FM.list.length; j++) new Image().src = cropUrl(FM.list[j].crop);  // 预加载后几张
  pMarkCurrent(F); fmShowImages();
  $("fm-title").textContent = FM.cur || "—"; $("fm-title").title = FM.cur || "";
  $("fm-pos").textContent = !FM.cur ? "" : at < 0 ? "（不在当前筛选里）" : `第 ${at + 1} / ${FM.list.length} 张`;
  fmRenderSide();
}
function fmRenderSide() {
  const c = fmBy(FM.cur);
  const gbox = $("fm-guess"); gbox.textContent = "";
  if (!c) gbox.append(el("p", "none", "—"));
  else if (!c.guess) gbox.append(el("p", "none", "Claude 还没猜这一张（跑 perception attrs-label）"));
  else {
    const g = c.guess, main = el("div", "lb-guess-main" + (g.label === "unsure" ? " unsure" : ""));
    main.append(el("b", "", fmName(g.label)), el("span", "lb-conf", `把握 ${pct(g.confidence)}`));
    const bar = el("div", "lb-bar"), fill = el("i"); fill.style.width = pct(g.confidence); bar.append(fill);
    gbox.append(main, bar);
    if (g.reason) gbox.append(el("p", "lb-reason", g.reason));  // 模型写的：只当纯文本
  }
  const meta = $("fm-meta"); meta.textContent = "";
  if (c) {
    const row = (k, v) => { meta.append(el("dt", "", k)); const dd = el("dd"); dd.append(v); meta.append(dd); };
    row("现在", el("span", "tag " + (c.where === UNL ? "" : c.where === DIS ? "bad" : "ok"), fmWhereText(c.where)));
    row("裁图", el("code", "", c.crop));
    if (c.image) row("原图", el("code", "", c.image));
  }
  const acts = $("fm-actions"); acts.textContent = "";
  const off = !c || FM.busy, g = c && c.guess, canAgree = !!g && fmForms().includes(g.label);
  const agree = kbdBtn("btn go lb-agree", canAgree ? `同意 Claude：${fmName(g.label)}` : "同意 Claude", "Enter", fmAgree);
  agree.disabled = off || !canAgree;
  acts.append(agree);
  const grid = el("div", "lb-grid");
  fmForms().forEach((l, k) => {
    const b = kbdBtn("btn lb-act", fmName(l), String(k + 1), () => fmLabel(l));
    if (c && c.where === l) { b.classList.add("on"); b.title = "现在就在这一类"; }
    if (g && g.label === l) b.classList.add("guess");
    b.disabled = off; grid.append(b);
  });
  const dis = kbdBtn("btn halt lb-act", "不要", "0", () => fmLabel("discard"));
  if (c && c.where === DIS) dis.classList.add("on");
  dis.disabled = off; grid.append(dis);
  acts.append(grid);
  const undoBtn = kbdBtn("btn sm lb-undo", "撤销上一步", "Z", fmUndo);
  undoBtn.disabled = FM.busy;
  acts.append(undoBtn);
}

/* ---- 动作页的描述 / 外形页的描述（共用实现在上面的「列表 / 标注 / 撤销」） ---- */
const G = {
  s: LB, tab: "gesture", itemsKey: "clips", key: "clip", post: "clip", api: "api/gesture", ids: "lb", noun: "片段",
  rowClass: "lb-item",
  labels: keyOrder, name: nameOf, whereText, short: shortName,
  select: clip => select(clip), renderSide: () => renderSide(),
  row: c => [el("span", "n", shortName(c.clip)), guessTag(G, c), el("span", "r", c.recording || "（没有录像名）")],
  onCounts: n => { const m = $("mark-labeling"); if (m) m.textContent = n[UNL] ? `${n[UNL]} 待确认` : ""; },
  onError: () => { const m = $("mark-labeling"); if (m) m.textContent = ""; },
};
const F = {
  s: FM, tab: "form", itemsKey: "items", key: "crop", post: "name", api: "api/form", ids: "fm", noun: "裁图",
  rowClass: "lb-item fm-item",
  labels: () => (FM.data && FM.data.forms) || [], name: fmName, whereText: fmWhereText, short: n => n,
  agreeable: g => F.labels().includes(g.label),
  select: name => fmSelect(name), renderSide: () => fmRenderSide(),
  row: c => {
    const im = el("img"); im.loading = "lazy"; im.decoding = "async"; im.alt = ""; im.src = cropUrl(c.crop);
    return [im, el("span", "n", c.crop), guessTag(F, c), el("span", "r", fmWhereText(c.where))];
  },
  onCounts: n => { const t = document.querySelector('#lb-tabs [data-tab="form"]'); if (t) t.textContent = n[UNL] ? `外形（${n[UNL]}）` : "外形"; },
};
function clips() { return pItems(G); }
function byName(clip) { return pBy(G, clip); }
function label(to) { return pLabel(G, to); }
function undo() { return pUndo(G); }
function load() { return pLoad(G); }
function setFilter(f) { return pSetFilter(G, f); }
function fmBy(name) { return pBy(F, name); }
function fmForms() { return F.labels(); }
function fmLabel(to) { return pLabel(F, to); }
function fmAgree() { return pAgree(F); }
function fmUndo() { return pUndo(F); }
function fmLoad() { return pLoad(F); }
function fmSetFilter(f) { return pSetFilter(F, f); }

const NOTES = {
  gesture: "确认的片段挪进 datasets/gesture/<动作>/，不要的进 _discard/，每一步记在 _labels.jsonl",
  form: "确认的裁图挪进 datasets/attrs/form/<类别>/，不要的进 _discard/，每一步记在 _labels.jsonl",
  names: "只起名，团子能做哪些动作照旧由轮盘 / 白名单决定；团子在跑也能起，下次启动才生效",
};
function setTab(tab) {
  LB.tab = tab === "form" || tab === "names" ? tab : "gesture";
  window.LabelingTab = LB.tab;
  for (const b of $("lb-tabs").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.tab === LB.tab));
  const form = LB.tab === "form", names = LB.tab === "names", gesture = LB.tab === "gesture";
  $("lb-note").textContent = NOTES[LB.tab];
  $("lb-cols").hidden = !gesture || !LB.data; $("lb-empty").hidden = !gesture || !!LB.data;
  $("fm-cols").hidden = !form || !FM.data; $("fm-empty").hidden = !form || !!FM.data;
  const en = window.EmoteNamesTab, has = !!(en && en.hasData());
  $("en-cols").hidden = !names || !has; $("en-empty").hidden = !names || has;
  startTimer();  // 动作页之外不放动图
  reload();
}
function reload() {
  if (LB.tab === "form") fmLoad();
  else if (LB.tab === "names") { if (window.EmoteNamesTab) window.EmoteNamesTab.load(); }
  else load();
}

/* ---- 键盘 ---- */
function onKey(e) {
  if (!LB.active || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  const dlg = $("dialog"); if (dlg && !dlg.hidden) return;
  const t = e.target;
  if (t && t.closest && t.closest("input,textarea,select,[contenteditable]:not([contenteditable=false])")) return;
  const k = e.key;
  if ((k === "Enter" || k === " ") && t && t.closest && t.closest("button,a,summary")) return;  // 键盘移到按钮上时回车 / 空格交给按钮
  let act = null;
  if (LB.tab === "names") return;  // 动作名页：输入框里回车就是保存，别的键不管
  if (LB.tab === "form") {
    if (k === "Enter") act = fmAgree;
    else if (k === "z" || k === "Z") act = fmUndo;
    else if (k === "0") act = () => fmLabel("discard");
    else if (/^[1-9]$/.test(k)) { const l = fmForms()[Number(k) - 1]; if (l) act = () => fmLabel(l); }
  }
  else if (k === " " || k === "Spacebar") act = () => setPlaying(!LB.playing);
  else if (k === "ArrowLeft") act = () => step(-1);
  else if (k === "ArrowRight") act = () => step(1);
  else if (k === "z" || k === "Z") act = undo;
  else if (k === "0") act = () => label("discard");
  else if (/^[1-9]$/.test(k)) { const l = keyOrder()[Number(k) - 1]; if (l) act = () => label(l); }
  if (!act) return;
  e.preventDefault();
  if (e.repeat && k !== "ArrowLeft" && k !== "ArrowRight") return;  // 按住不放不连着标
  act();
}

Pages.labeling = {
  init() {
    const ticks = $("lb-ticks");
    for (let i = 0; i < FRAMES; i++) {
      const b = el("button"); b.type = "button"; b.title = `第 ${i + 1} 帧`; b.setAttribute("aria-label", `第 ${i + 1} 帧`);
      b.onclick = () => seek(i); ticks.append(b);
    }
    $("lb-play").onclick = () => setPlaying(!LB.playing);
    $("lb-prev").onclick = () => step(-1);
    $("lb-next").onclick = () => step(1);
    for (const b of $("lb-speed").querySelectorAll("button")) b.onclick = () => setSpeed(Number(b.dataset.speed));
    $("lb-filter").onchange = e => { setFilter(e.currentTarget.value); e.currentTarget.blur(); };  // 选完把焦点还给页面，按键才有用
    $("lb-refresh").onclick = reload;
    $("fm-filter").onchange = e => { fmSetFilter(e.currentTarget.value); e.currentTarget.blur(); };
    for (const b of $("lb-tabs").querySelectorAll("button")) b.onclick = () => setTab(b.dataset.tab);
    // 鼠标点按钮不抢焦点：点完接着按键就行（Tab 过去的按钮照样能用回车 / 空格）
    $("page-labeling").addEventListener("mousedown", e => { if (e.target.closest && e.target.closest("button")) e.preventDefault(); });
    setPlaying(true); setSpeed(1); draw();
  },
  show() {  // 可能被重复调用（点当前导航项）：只重读，键盘不重复绑
    LB.active = true;
    if (!LB.keyBound) { document.addEventListener("keydown", onKey); LB.keyBound = true; }
    startTimer();
    reload();
  },
  hide() {
    LB.active = false;
    stopTimer();
    if (LB.keyBound) { document.removeEventListener("keydown", onKey); LB.keyBound = false; }
  },
};
})();
