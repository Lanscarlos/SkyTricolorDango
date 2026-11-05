/* 标注页（spec 2026-10-01-gesture-labeling-training §3）：左 片段列表（筛选 + 各类计数），中 16 帧动图（canvas 放大 3 倍、按 fps 循环），
 * 右 Claude 的猜测 + 按钮。确认 / 改类别 / 丢弃 / 撤销走 api/gesture/*，后端把片段文件夹挪到对应目录。
 * 键盘（焦点不在输入框、没开对话框时）：1~5 = 挥手 / 鞠躬 / 欢呼 / 害羞 / 都不是、0 不要、Z 撤销、空格 暂停、← → 逐帧（暂停时）。
 * 动作页没有「同意 Claude」（10-03 去掉：Claude 初分认不出动作，直接人工标）；有旧的猜测照样显示、按钮上标出来。
 * 页顶「动作 / 外形 / 动作名」三个标签：外形页标人物裁图（api/form/*，Enter / 1~6 = 外形类别 / 0 不要 / Z 撤销，只在外形页开着时有效）；
 * 动作名页给动作图标起名（emotenames.js，这里只管切过去和刷新，开着时这里的按键都不生效）。
 * 整帧页过目 / 编辑收件箱里整理好的帧（frames.js，这里只管切过去、刷新，开着时按键转给 FramesTab.key）。
 * 外形页的「来自整理」筛选（难例收件箱，后端 inbox）：整理时外形头拿不准、交给人判的裁图都在这里。
 * 外形页的「导入未确认」筛选（10-04）：datasets/sky 直接导进 form/ 的裁图没人看过（后端 confirmed = false）；在原类别上按一下 = 原地确认。
 * 其中「回放用」（后端 replay）是数据集验证集帧里的框 = attrs-train 整帧回放的标准答案，先过它们回放才算得准。
 * 外形页「按轨迹」（默认开，页顶复选框，记在 localStorage）：后端把同一个人连着几帧的裁图分成一组（vision/cropgroups.py），列表一行一组，
 * 当前组上方一排缩略图（点一下换大图，Shift + 点 / 点 × = 拆出去不参与这一下），按键整组判（api/form/label 的 names，每张仍是一条记账，同一个 batch），
 * Z 整组撤销；纯逻辑在 formgroup.js。外形页「重点核对」筛选：perception crops-audit 写的 _focus.json 名单。
 * 模型写的理由一律 textContent。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const FRAMES = 16, FPS = 8, PX = 336, AHEAD = 3, KEEP = 12;  // PX = 112 × 3；AHEAD = 预加载后几段；KEEP = 最多缓存几段的帧
const UNL = "_unlabeled", DIS = "_discard";
const LB = {data: null, filter: "todo", list: [], cur: null, i: 0, playing: true, speed: 1, timer: null,
  busy: false, active: false, loading: false, cache: new Map(), keyBound: false, tab: "gesture"};
/* 外形页（感知层第二层的人物裁图）：条目是单张 jpg，类别 = FORMS（后端给），接口 api/form/*。按键 Enter / 1~4（后端的 buttons）/ 0 / Z 只在外形页开着时有效。 */
const FM = {data: null, filter: "todo", list: [], cur: null, busy: false, loading: false,
  byTrack: true, view: null, excluded: new Set(), gmap: new Map()};  // 按轨迹：view = 大图显示的那张，excluded = 拆出去的，gmap = 列表行名 → 组
const FOCUS = new Set();  // 重点核对名单（state.focus）
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
 *      rowClass, onCounts(n), onError(), confirmable 条目有 confirmed（外形页：多「导入未确认」筛选、原类别上再标 = 原地确认）} */
function guessKey(c) { return c.guess ? c.guess.label : "-"; }
function matches(c, f) {
  if (f === "todo") return c.where === UNL;
  if (f === "discard") return c.where === DIS;
  if (f.startsWith("guess:")) return c.where === UNL && guessKey(c) === f.slice(6);
  if (f.startsWith("done:")) return c.where === f.slice(5);
  if (f === "unconfirmed") return unconfirmed(c);
  if (f === "replay") return unconfirmed(c) && c.replay === true;
  if (f === "inbox") return c.inbox === true;
  if (f === "focus") return window.FormGroup.focusMatch(c, FOCUS, (FM.data && FM.data.forms) || []);
  if (f.startsWith("unconfirmed:")) return unconfirmed(c) && c.where === f.slice(12);
  return false;
}
function unconfirmed(c) { return c.confirmed === false && c.where !== UNL && c.where !== DIS; }
function rank(c) { return !c.guess ? 1 : c.guess.label === "unsure" ? 0 : 2; }  // 看不清最前，没猜的其次，其余按把握从低到高
function pItems(P) { return (P.s.data && P.s.data[P.itemsKey]) || []; }
function pBy(P, name) { return name == null ? null : pItems(P).find(c => c[P.key] === name) || null; }
function pBuild(P) {
  if (P.build && P.build()) return;  // 外形页按轨迹：列表一行一组
  const f = P.s.filter, k = P.key, list = pItems(P).filter(c => matches(c, f));
  if (f === "inbox") list.sort((a, b) => (a.where === UNL ? 0 : 1) - (b.where === UNL ? 0 : 1) || rank(a) - rank(b) || (a[k] < b[k] ? -1 : a[k] > b[k] ? 1 : 0));  // 没判的在前
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
  if (P.confirmable && (f === "unconfirmed" || f === "replay" || f === "inbox")) return true;
  if (P.confirmable && f === "focus") return FOCUS.size > 0;
  if (P.confirmable && f.startsWith("unconfirmed:")) return labels.includes(f.slice(12));
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
  if (P.confirmable) {
    const un = pItems(P).filter(unconfirmed), un1 = {};
    for (const c of un) un1[c.where] = (un1[c.where] || 0) + 1;
    const g3 = el("optgroup"); g3.label = "导入未确认（没人看过）";
    g3.append(opt("unconfirmed", `全部（${un.length}）`));
    g3.append(opt("replay", `回放用的（${un.filter(c => c.replay === true).length}）`));
    for (const l of P.labels()) if (un1[l] || P.s.filter === "unconfirmed:" + l) g3.append(opt("unconfirmed:" + l, `${P.name(l)}（${un1[l] || 0}）`));
    sel.append(g3);
    const ib = pItems(P).filter(c => c.inbox === true);
    if (ib.length || P.s.filter === "inbox") sel.append(opt("inbox", `来自整理（${ib.filter(c => c.where === UNL).length} 待判 / ${ib.length}）`));
    if (FOCUS.size) sel.append(opt("focus", `重点核对（${pItems(P).filter(c => matches(c, "focus")).length} / ${FOCUS.size}）`));
  }
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
  if (P.confirmable) {
    const un = pItems(P).filter(unconfirmed);
    n.unconfirmed = un.length; n.replay = un.filter(c => c.replay === true).length;
    item("unconfirmed", "导入未确认", "unconfirmed");
    if (n.replay || P.s.filter === "replay") item("replay", "其中回放用", "replay");
    n.inbox = pItems(P).filter(c => c.inbox === true).length;
    if (n.inbox || P.s.filter === "inbox") item("inbox", "来自整理", "inbox");
    if (FOCUS.size) { n.focus = pItems(P).filter(c => matches(c, "focus")).length; item("focus", "重点核对", "focus"); }
  }
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
    box.append(el("p", "none lb-empty", P.s.filter === "todo" ? (pItems(P).length ? "都标完了。" : `还没有${P.noun}。`)
      : P.s.filter.startsWith("unconfirmed") || P.s.filter === "replay" ? "导入的都确认过了。"
      : P.s.filter === "inbox" ? "还没有来自整理的裁图（先跑整理）。"
      : P.s.filter === "focus" ? "重点核对的都看过了。" : `这里没有${P.noun}。`));
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
  // 已经是这一类：当作确认过，跳下一条；外形页导入未确认的要发给后端记一笔原地确认
  if (c.where === (to === "discard" ? DIS : to) && !(P.confirmable && unconfirmed(c))) { pAdvance(P, old, idx); return; }
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
  if (Array.isArray(r.data.items)) {  // 外形页整组撤销（成功，或撤了一部分停在被手动挪过的那张）
    r.data.items.forEach(it => pApply(P, it));
    pRerender(P);
    if (r.data.items.length) P.select(P.pick ? P.pick(r.data.items) : r.data.items[0][P.key]); else P.renderSide();
    if (!r.data.ok) toast(r.data.text || "没撤销完", "warn");
    else toast(`撤销了这一组：${r.data.items.length} 张`, "ok");
    return;
  }
  if (!r.data.ok) { toast(r.data.text || "没撤销成", r.status === 409 ? "warn" : "bad"); P.renderSide(); return; }
  pApply(P, r.data);
  pRerender(P); P.select(r.data[P.key]);
  toast(`撤销了：${P.short(r.data[P.key])} 回到「${P.itemWhere ? P.itemWhere(r.data) : P.whereText(r.data.where)}」`, "ok");
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
  if (P.onData) P.onData(d);
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
function fmRowWhere(c) { return unconfirmed(c) ? `导入未确认${c.replay ? "（回放用）" : ""}：${fmName(c.where)}` : fmWhereText(c.where); }

function cropUrl(name) { return `api/form/crop?name=${encodeURIComponent(name)}`; }
function fmShowImages() {
  const im = $("fm-crop"), cx = $("fm-ctx"), n1 = $("fm-crop-note"), n2 = $("fm-ctx-note");
  n1.hidden = n2.hidden = true;
  const shown = FM.view || FM.cur;
  if (!shown) { im.removeAttribute("src"); im.hidden = true; cx.removeAttribute("src"); cx.hidden = true; n1.textContent = FM.data ? (FM.list.length ? "从列表里选一张" : "这个筛选里没有裁图") : ""; n1.hidden = !n1.textContent; return; }
  im.hidden = cx.hidden = false;
  im.onerror = () => { im.hidden = true; n1.textContent = "这张图读不到"; n1.hidden = false; };
  cx.onerror = () => { cx.hidden = true; n2.hidden = false; };
  im.src = cropUrl(shown);
  cx.src = `api/form/context?name=${encodeURIComponent(shown)}`;
}
function fmGroup(name) { return FM.byTrack && name != null ? FM.gmap.get(name) || null : null; }
function fmSelect(name) {
  if ((name || null) !== FM.cur) FM.excluded = new Set();  // 拆出去的只在当前组里算数
  FM.cur = name || null;
  const g = fmGroup(FM.cur);
  FM.view = g ? g.shown[window.FormGroup.centerIndex(g.shown.length)].crop : FM.cur;
  const at = FM.list.findIndex(c => c.crop === FM.cur);
  for (let j = at + 1; at >= 0 && j <= at + AHEAD && j < FM.list.length; j++) new Image().src = cropUrl(FM.list[j].crop);  // 预加载后几张
  pMarkCurrent(F); fmShowImages();
  fmShowTitle();
  fmRenderSide();
}
function fmShowTitle() {
  const v = FM.view || FM.cur, g = fmGroup(FM.cur), at = FM.list.findIndex(c => c.crop === FM.cur);
  $("fm-title").textContent = v || "—"; $("fm-title").title = v || "";
  $("fm-pos").textContent = !FM.cur ? "" : at < 0 ? "（不在当前筛选里）"
    : g ? `第 ${at + 1} / ${FM.list.length} 组（本组 ${window.FormGroup.targets(g, FM.excluded).length} / ${g.shown.length} 张）` : `第 ${at + 1} / ${FM.list.length} 张`;
  fmRenderStrip();
}
/* 当前组的缩略图条：点 = 大图换成这张；Shift + 点 / 点 × = 拆出去（这一下不标它） */
function fmRenderStrip() {
  const box = $("fm-strip"), g = fmGroup(FM.cur);
  box.textContent = "";
  box.hidden = !g || g.shown.length < 2;
  if (box.hidden) return;
  for (const c of g.shown) {
    const out = FM.excluded.has(c.crop), cell = el("span", "fm-thumb" + (c.crop === FM.view ? " on" : "") + (out ? " out" : ""));
    const b = el("button"); b.type = "button"; b.title = `${c.crop}（Shift + 点 = ${out ? "放回来" : "拆出去"}）`;
    const im = el("img"); im.alt = ""; im.loading = "lazy"; im.decoding = "async"; im.src = cropUrl(c.crop);
    b.append(im);
    b.onclick = e => { if (e.shiftKey) fmToggleOut(c.crop); else fmView(c.crop); };
    const x = el("button", "fm-x", out ? "+" : "×"); x.type = "button"; x.title = out ? "放回这一组" : "拆出去，这一下不标它";
    x.setAttribute("aria-label", x.title);
    x.onclick = () => fmToggleOut(c.crop);
    cell.append(b, x);
    box.append(cell);
  }
}
function fmView(name) { FM.view = name; fmShowImages(); fmShowTitle(); fmRenderSide(); }
function fmToggleOut(name) {
  if (FM.excluded.has(name)) FM.excluded.delete(name); else FM.excluded.add(name);
  fmShowTitle(); fmRenderSide();
}
function fmRenderSide() {
  const c = fmBy(FM.view || FM.cur);
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
    row("现在", unconfirmed(c) ? el("span", "tag warn", fmRowWhere(c)) : el("span", "tag " + (c.where === UNL ? "" : c.where === DIS ? "bad" : "ok"), fmWhereText(c.where)));
    row("裁图", el("code", "", c.crop));
    if (c.inbox) row("来源", "难例收件箱（整理时外形头拿不准）");
    if (c.image) row("原图", el("code", "", c.image));
  }
  const acts = $("fm-actions"); acts.textContent = "";
  const grp = fmGroup(FM.cur);
  if (grp && grp.shown.length > 1) acts.append(el("p", "note fm-batch", `按键标整组：${window.FormGroup.targets(grp, FM.excluded).length} 张`));
  const off = !c || FM.busy, g = c && c.guess, canAgree = !!g && fmForms().includes(g.label);
  const agree = kbdBtn("btn go lb-agree", canAgree ? `同意 Claude：${fmName(g.label)}` : "同意 Claude", "Enter", fmAgree);
  agree.disabled = off || !canAgree;
  acts.append(agree);
  const grid = el("div", "lb-grid");
  fmForms().forEach((l, k) => {
    const b = kbdBtn("btn lb-act", fmName(l), String(k + 1), () => fmLabel(l));
    if (c && c.where === l) { b.classList.add("on"); b.title = unconfirmed(c) ? "导入时就在这一类：按一下 = 确认" : "现在就在这一类"; }
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
  rowClass: "lb-item fm-item", confirmable: true, itemWhere: c => fmRowWhere(c),
  labels: () => (FM.data && FM.data.forms) || [], name: fmName, whereText: fmWhereText, short: n => n,
  agreeable: g => fmForms().includes(g.label),
  select: name => fmSelect(name), renderSide: () => fmRenderSide(),
  build: () => fmBuild(),
  pick: items => { for (const g of FM.gmap.values()) if (g.shown.some(c => c.crop === items[0].crop)) return g.shown[0].crop; return items[0].crop; },
  onData: d => { FOCUS.clear(); (d.focus || []).forEach(n => FOCUS.add(n)); },
  row: c => {
    const im = el("img"); im.loading = "lazy"; im.decoding = "async"; im.alt = ""; im.src = cropUrl(c.crop);
    const g = fmGroup(c.crop);
    return [im, el("span", "n", c.crop), guessTag(F, c), el("span", "r", (g && g.shown.length > 1 ? `${g.shown.length} 张 · ` : "") + fmRowWhere(c))];
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
function fmForms() { return (FM.data && FM.data.buttons) || F.labels(); }  // 出按钮 / 数字键 / 能同意的类别；筛选照旧用全部
/* 按轨迹：列表一行一组（行 = 组里第一张显示的裁图）。返回 false = 没开按轨迹，走原来的逐张列表 */
function fmBuild() {
  FM.gmap = new Map();
  if (!FM.byTrack) return false;
  const f = FM.filter, groups = window.FormGroup.buildGroups(pItems(F), (FM.data && FM.data.groups) || [], c => matches(c, f));
  FM.list = groups.map(g => { FM.gmap.set(g.shown[0].crop, g); return g.shown[0]; });
  return true;
}
async function fmLabel(to) {
  const grp = fmGroup(FM.cur);
  if (!grp) return pLabel(F, to);  // 没开按轨迹：逐字照旧
  if (FM.busy) return;
  const dst = to === "discard" ? DIS : to, old = FM.list.slice(), idx = old.findIndex(x => x.crop === FM.cur);
  // 已经在这一类且确认过的不用再发（和单张一样：按一下 = 当作看过，跳下一组）
  const names = window.FormGroup.targets(grp, FM.excluded).filter(n => { const c = fmBy(n); return !c || c.where !== dst || unconfirmed(c); });
  if (!names.length) { pAdvance(F, old, idx); return; }
  FM.busy = true; fmRenderSide();
  let r;
  try { r = await post("api/form/label", {name: names[0], names, to}); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  FM.busy = false;
  const d = r.data || {};
  if (!d.ok) {
    toast(d.text || "没标上", "bad");
    if (r.status === 409) fmLoad(); else fmRenderSide();
    return;
  }
  (d.items || []).forEach(it => pApply(F, it));
  if (d.skipped && d.skipped.length) toast(`跳过 ${d.skipped.length} 张：${d.skipped.map(x => x.text).filter((t, i, a) => a.indexOf(t) === i).join("；")}`, "warn");
  pAdvance(F, old, idx);
}
function fmAgree() {
  const c = fmBy(FM.view || FM.cur);
  if (!fmGroup(FM.cur)) return pAgree(F);
  if (c && c.guess && F.agreeable(c.guess)) fmLabel(c.guess.label);
}
function fmUndo() { return pUndo(F); }
function fmLoad() { return pLoad(F).then(() => fgApplyVisibility()); }
function fmSetFilter(f) { return pSetFilter(F, f); }

/* ---- 外形页「网格」模式（计划 2026-11-05-form-grid-review；纯逻辑在 formgrid.js）----
 * 一页 36 张，默认是人，只点错的：单击 = 不是人、Shift + 单击 = 不要、双击 = 看整帧；Enter 整页确认（api/form/check）、Z 整页撤销。
 * 记录带 scope: person，不算外形确认；左列表和右按钮区收起，筛选沿用 FM.filter。 */
const FG = {mode: "single", page: 0, marks: new Map(), list: [], busy: false, hist: [], gp: null};
FG.gp = Object.assign({}, F, {ids: "fg"});  // 网格自己的筛选下拉（fg-filter）复用 pRenderFilter
function fgOn() { return FG.mode === "grid" && LB.tab === "form"; }
function fgPersonSet() { return new Set((FM.data && FM.data.person) || []); }
function fgApplyVisibility() {  // 单张 / 网格两套界面谁显示
  const has = !!FM.data, form = LB.tab === "form", grid = FG.mode === "grid";
  $("fm-cols").hidden = !form || !has || grid;
  $("fg-cols").hidden = !form || !has || !grid;
  if (form && has && grid) fgRender();
}
function fgSetMode(mode) {
  FG.mode = mode;
  FG.page = 0; FG.marks = new Map();
  let store = null;
  try { store = window.localStorage; } catch (e) { /* 读不到存储 = 不记 */ }
  window.FormGrid.writeMode(store, mode);
  fgApplyVisibility();
}
function fgBuild() {
  const f = FM.filter, person = fgPersonSet(), scores = (FM.data && FM.data.scores) || {};
  FG.list = window.FormGrid.gridItems(pItems(F), c => matches(c, f), person, scores);
  const pages = window.FormGrid.pageCount(FG.list.length);
  if (FG.page >= pages) FG.page = Math.max(0, pages - 1);
}
function fgRender() {
  fgBuild();
  pRenderFilter(FG.gp);
  const G = window.FormGrid, items = G.page(FG.list, FG.page), scores = (FM.data && FM.data.scores) || {};
  const all = pItems(F).filter(c => matches(c, FM.filter)), person = fgPersonSet();
  $("fg-progress").textContent = `核过是人 ${all.filter(c => person.has(c.crop)).length} / 这个筛选共 ${all.length}`;
  const pages = G.pageCount(FG.list.length);
  const marked = items.filter(c => FG.marks.has(c.crop)).length;
  $("fg-pos").textContent = items.length ? `第 ${FG.page + 1} / ${pages} 页 · 本页 ${items.length} 张，点出 ${marked} 张` : "";
  const box = $("fg-grid");
  box.textContent = "";
  $("fg-ctx").hidden = true;
  if (!items.length) {
    box.append(el("p", "none lb-empty", FM.filter === "focus" ? "重点核对的都看过了。" : FG.list.length ? "" : "这个筛选里没有要核的裁图（核过是人的不再出现）。"));
  }
  for (const c of items) {
    const kind = FG.marks.get(c.crop), p = G.personProb(scores[c.crop]);
    const cell = el("button", "fg-cell" + (kind ? " " + kind : "")); cell.type = "button"; cell.dataset.k = c.crop;
    cell.title = `${c.crop}\n单击 = 不是人，Shift + 单击 = 不要，双击 = 看整帧`;
    const im = el("img"); im.loading = "lazy"; im.decoding = "async"; im.alt = ""; im.src = cropUrl(c.crop);
    const tag = el("span", "fg-tag", `${fmName(c.where)}${p === null ? "" : " " + Math.round(p * 100) + "%"}`);
    cell.append(im, tag);
    if (kind) cell.append(el("span", "fg-mark", kind === "discard" ? "不要" : "×"));
    cell.onclick = e => { FG.marks = G.toggle(FG.marks, c.crop, e.shiftKey ? "discard" : "not_person"); fgMarkCell(cell, c.crop); fgCount(items); };
    cell.ondblclick = () => fgShowCtx(c.crop);
    box.append(cell);
  }
  $("fg-confirm").disabled = FG.busy || !items.length;
  $("fg-undo").disabled = FG.busy;
}
function fgMarkCell(cell, crop) {  // 点一下只改这一格，不重画整页
  const kind = FG.marks.get(crop);
  cell.classList.remove("not_person", "discard");
  const old = cell.querySelector(".fg-mark"); if (old) old.remove();
  if (kind) { cell.classList.add(kind); cell.append(el("span", "fg-mark", kind === "discard" ? "不要" : "×")); }
}
function fgCount(items) {
  const marked = items.filter(c => FG.marks.has(c.crop)).length, pages = window.FormGrid.pageCount(FG.list.length);
  $("fg-pos").textContent = `第 ${FG.page + 1} / ${pages} 页 · 本页 ${items.length} 张，点出 ${marked} 张`;
}
function fgShowCtx(crop) {
  const box = $("fg-ctx"), im = $("fg-ctx-img");
  im.onerror = () => { box.hidden = true; toast("没有原图（来源画面读不到）", "warn"); };
  im.src = `api/form/context?name=${encodeURIComponent(crop)}`;
  box.hidden = false;
}
async function fgFetch() {
  let d;
  try { d = await getJSON("api/form/state"); } catch (e) { d = null; }
  if (!d || !d.ok) { toast((d && d.text) || "读不到裁图（面板停了？）", "bad"); return false; }
  FM.data = d; FOCUS.clear(); (d.focus || []).forEach(n => FOCUS.add(n));
  return true;
}
async function fgConfirm() {
  if (!fgOn() || FG.busy) return;
  const items = window.FormGrid.page(FG.list, FG.page);
  if (!items.length) return;
  const sub = window.FormGrid.submission(items, FG.marks), marks = new Map(FG.marks);
  FG.busy = true; fgRender();
  let r;
  try { r = await post("api/form/check", sub); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  FG.busy = false;
  const d = r.data || {};
  if (!d.ok) {
    toast(d.text || "没确认上", "bad");
    if (r.status === 409) { await fgFetch(); }
    fgRender();
    return;
  }
  FG.hist.push({batch: d.batch, marks, page: FG.page});
  if (FG.hist.length > 50) FG.hist.shift();
  for (const n of Object.keys(sub).flatMap(k => sub[k])) FG.marks.delete(n);
  if (d.skipped && d.skipped.length) toast(`跳过 ${d.skipped.length} 张：${d.skipped.map(x => x.text).filter((t, i, a) => a.indexOf(t) === i).join("；")}`, "warn");
  await fgFetch();  // 核过的从列表里退出，原来第二页的补到第一页 = 进下一页
  fgRender();
}
async function fgUndo() {
  if (!fgOn() || FG.busy) return;
  FG.busy = true; fgRender();
  let r;
  try { r = await post("api/form/undo"); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  FG.busy = false;
  const d = r.data || {};
  const undone = Array.isArray(d.items) ? d.items.map(x => x.crop) : d.crop ? [d.crop] : [];
  if (!d.ok && !undone.length) { toast(d.text || "没撤销成", r.status === 409 ? "warn" : "bad"); fgRender(); return; }
  await fgFetch();
  const hi = FG.hist.findIndex(h => h.batch === d.batch);
  FG.marks = new Map();
  if (hi >= 0) {  // 是网格自己确认的那一页：标记恢复成确认前的样子（不是人 / 不要的重新标上）
    FG.marks = new Map(FG.hist[hi].marks);
    FG.hist.splice(hi, 1);
  }
  fgBuild();
  const at = FG.list.findIndex(c => undone.includes(c.crop));
  if (at >= 0) FG.page = Math.floor(at / window.FormGrid.PAGE);
  fgRender();
  if (!d.ok) toast(d.text || "没撤销完", "warn");
  else toast(`撤销了：${undone.length} 张`, "ok");
}

const NOTES = {
  identity: "答案写回 tmp/identity-truth/<录像>/truth-draft.toml，标过的记在 reviewed.json；写完去命令行 --apply",
  gesture: "确认的片段挪进 datasets/gesture/<动作>/，不要的进 _discard/，每一步记在 _labels.jsonl",
  form: "确认的裁图挪进 datasets/attrs/form/<类别>/，不要的进 _discard/，每一步记在 _labels.jsonl",
  names: "只起名，团子能做哪些动作照旧由轮盘 / 白名单决定；团子在跑也能起，下次启动才生效",
  frames: "通过 = 原图和标注复制进 datasets/sky/images|labels/<分边>/；不要的留在收件箱；决定记在收件箱各次运行的 frames.json",
};
function setTab(tab) {
  if (LB.tab === "identity" && tab !== "identity" && window.IdentityTab) window.IdentityTab.hide();
  if (tab === "frames" && LB.framesOn === false) tab = "gesture";  // 难例收件箱关着：没有整帧页
  LB.tab = ["form", "names", "frames", "identity"].includes(tab) ? tab : "gesture";
  window.LabelingTab = LB.tab;
  for (const b of $("lb-tabs").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.tab === LB.tab));
  const form = LB.tab === "form", names = LB.tab === "names", gesture = LB.tab === "gesture";
  $("lb-note").textContent = NOTES[LB.tab];
  $("lb-cols").hidden = !gesture || !LB.data; $("lb-empty").hidden = !gesture || !!LB.data;
  $("fm-cols").hidden = !form || !FM.data; $("fm-empty").hidden = !form || !!FM.data;
  fgApplyVisibility();
  const en = window.EmoteNamesTab, has = !!(en && en.hasData());
  $("en-cols").hidden = !names || !has; $("en-empty").hidden = !names || has;
  const fr = window.FramesTab, frTab = LB.tab === "frames", fhas = !!(fr && fr.hasData());
  $("fr-train").hidden = !frTab; if (frTab && window.RetrainBox) window.RetrainBox.load();
  $("fr-cols").hidden = !frTab || !fhas; $("fr-empty").hidden = !frTab || fhas;
  const idTab = LB.tab === "identity", identity = window.IdentityTab, idHas = !!(identity && identity.hasData());
  $("id-cols").hidden = !idTab || !idHas; $("id-empty").hidden = !idTab || idHas;
  startTimer();  // 动作页之外不放动图
  reload();
}
/* 难例收件箱关着（[inbox] enabled = false → /api/state 的 inbox 为 null，spec §8）：不出「整帧」标签，正开着就切回动作页 */
function syncFramesTab(st) {
  const on = !!(st && st.inbox);
  if (on === LB.framesOn) return;
  LB.framesOn = on;
  const b = document.querySelector('#lb-tabs [data-tab="frames"]');
  if (b) b.hidden = !on;
  if (!on && LB.tab === "frames") setTab("gesture");
}
function reload() {
  if (LB.tab === "identity") { if (window.IdentityTab) window.IdentityTab.load(); }
  else if (LB.tab === "form") fmLoad();
  else if (LB.tab === "names") { if (window.EmoteNamesTab) window.EmoteNamesTab.load(); }
  else if (LB.tab === "frames") { if (window.FramesTab) window.FramesTab.load(); }
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
  if (LB.tab === "identity") { if (window.IdentityTab) window.IdentityTab.key(e); return; }
  if (LB.tab === "frames") { if (window.FramesTab) window.FramesTab.key(e); return; }  // 整帧页自己管按键（过目 / 编辑两套）
  if (LB.tab === "form" && FG.mode === "grid") {
    if (k === "Enter") act = fgConfirm;
    else if (k === "z" || k === "Z") act = fgUndo;
  }
  else if (LB.tab === "form") {
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
    let store = null;
    try { store = window.localStorage; } catch (e) { /* 读不到存储 = 默认开 */ }
    FM.byTrack = window.FormGroup.readByTrack(store);
    const bt = $("fm-bytrack");
    bt.checked = FM.byTrack;
    bt.onchange = e => {
      FM.byTrack = e.currentTarget.checked;
      window.FormGroup.writeByTrack(store, FM.byTrack);
      e.currentTarget.blur();
      if (FM.data) pSetFilter(F, FM.filter);
    };
    FG.mode = window.FormGrid.readMode(store);
    $("fm-gridmode").onclick = () => fgSetMode("grid");
    $("fg-single").onclick = () => { fgSetMode("single"); if (FM.data) pSetFilter(F, FM.filter); };
    $("fg-filter").onchange = e => {
      const f = e.currentTarget.value;
      FM.filter = pValid(F, f) ? f : "todo"; FG.page = 0; FG.marks = new Map();
      fgRender(); e.currentTarget.blur();
    };
    $("fg-confirm").onclick = fgConfirm;
    $("fg-undo").onclick = fgUndo;
    $("fg-cols").addEventListener("click", () => { $("fg-ctx").hidden = true; });  // 点别处收起整帧
    for (const b of $("lb-tabs").querySelectorAll("button")) b.onclick = () => setTab(b.dataset.tab);
    // 鼠标点按钮不抢焦点：点完接着按键就行（Tab 过去的按钮照样能用回车 / 空格）
    $("page-labeling").addEventListener("mousedown", e => { if (e.target.closest && e.target.closest("button")) e.preventDefault(); });
    setPlaying(true); setSpeed(1); draw();
    onState(syncFramesTab);
  },
  show(arg) {  // 可能被重复调用（点当前导航项）：只重读，键盘不重复绑；arg = "frames" 直接切到整帧页（侧栏提示点过来）
    LB.active = true;
    if (arg === "frames" && LB.tab !== "frames") setTab("frames");
    if (!LB.keyBound) { document.addEventListener("keydown", onKey); LB.keyBound = true; }
    startTimer();
    reload();
    if (LB.tab !== "frames" && LB.framesOn !== false && window.FramesTab) window.FramesTab.peek();  // 标签上的「整帧（N）」不用先点进去
  },
  hide() {
    LB.active = false;
    if (window.IdentityTab) window.IdentityTab.hide();
    stopTimer();
    if (window.FramesTab) window.FramesTab.hide();  // 松开空格、丢掉拖到一半的框
    if (LB.keyBound) { document.removeEventListener("keydown", onKey); LB.keyBound = false; }
  },
};
})();
