/* 标注页（spec 2026-10-01-gesture-labeling-training §3）：左 片段列表（筛选 + 各类计数），中 16 帧动图（canvas 放大 3 倍、按 fps 循环），
 * 右 Claude 的猜测 + 按钮。确认 / 改类别 / 丢弃 / 撤销走 api/gesture/*，后端把片段文件夹挪到对应目录。
 * 键盘（焦点不在输入框、没开对话框时）：Enter 同意 Claude、1~5 = 挥手 / 鞠躬 / 欢呼 / 害羞 / 都不是、0 不要、Z 撤销、空格 暂停、← → 逐帧（暂停时）。
 * 模型写的理由一律 textContent。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const FRAMES = 16, FPS = 8, PX = 336, AHEAD = 3, KEEP = 12;  // PX = 112 × 3；AHEAD = 预加载后几段；KEEP = 最多缓存几段的帧
const UNL = "_unlabeled", DIS = "_discard";
const LB = {data: null, filter: "todo", list: [], cur: null, i: 0, playing: true, speed: 1, timer: null,
  busy: false, active: false, loading: false, cache: new Map(), keyBound: false};

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

/* ---- 列表 ---- */
function clips() { return (LB.data && LB.data.clips) || []; }
function byName(clip) { return clip == null ? null : clips().find(c => c.clip === clip) || null; }
function guessKey(c) { return c.guess ? c.guess.label : "-"; }
function matches(c, f) {
  if (f === "todo") return c.where === UNL;
  if (f === "discard") return c.where === DIS;
  if (f.startsWith("guess:")) return c.where === UNL && guessKey(c) === f.slice(6);
  if (f.startsWith("done:")) return c.where === f.slice(5);
  return false;
}
function rank(c) { return !c.guess ? 1 : c.guess.label === "unsure" ? 0 : 2; }  // 看不清最前，没猜的其次，其余按把握从低到高
function buildList() {
  const f = LB.filter, list = clips().filter(c => matches(c, f));
  if (f === "todo" || f.startsWith("guess:"))
    list.sort((a, b) => rank(a) - rank(b) || (a.guess ? a.guess.confidence : 0) - (b.guess ? b.guess.confidence : 0) || (a.clip < b.clip ? -1 : a.clip > b.clip ? 1 : 0));
  LB.list = list;
}
function counts() {
  const n = {};
  for (const c of clips()) n[c.where] = (n[c.where] || 0) + 1;
  return n;
}
function validFilter(f) {
  const labels = (LB.data && LB.data.labels) || [];
  if (f === "todo" || f === "discard") return true;
  if (f.startsWith("done:")) return labels.includes(f.slice(5));
  if (f.startsWith("guess:")) return [...labels, "unsure", "-"].includes(f.slice(6));
  return false;
}

function renderFilter() {
  const sel = $("lb-filter"), n = counts(), pending = clips().filter(c => c.where === UNL);
  const gn = {}; for (const c of pending) gn[guessKey(c)] = (gn[guessKey(c)] || 0) + 1;
  sel.textContent = "";
  const opt = (v, t) => { const o = el("option", "", t); o.value = v; return o; };
  sel.append(opt("todo", `待确认（${n[UNL] || 0}）`));
  const g1 = el("optgroup"); g1.label = "Claude 猜的（待确认里）";
  for (const g of [...keyOrder(), "unsure", "-"]) {
    const v = "guess:" + g;
    if ((gn[g] || 0) || LB.filter === v) g1.append(opt(v, `${g === "-" ? "还没猜" : "猜" + nameOf(g)}（${gn[g] || 0}）`));
  }
  if (g1.children.length) sel.append(g1);
  const g2 = el("optgroup"); g2.label = "已确认";
  for (const l of keyOrder()) g2.append(opt("done:" + l, `${nameOf(l)}（${n[l] || 0}）`));
  sel.append(g2);
  sel.append(opt("discard", `丢弃的（${n[DIS] || 0}）`));
  sel.value = LB.filter;
}
function renderCounts() {
  const box = $("lb-counts"), n = counts();
  box.textContent = "";
  const item = (f, label, k) => {
    const b = el("button", "lb-count"); b.type = "button";
    b.setAttribute("aria-pressed", String(LB.filter === f));
    b.append(label, el("b", "", String(n[k] || 0)));
    b.onclick = () => setFilter(f);
    box.append(b);
  };
  item("todo", "待确认", UNL);
  for (const l of keyOrder()) item("done:" + l, nameOf(l), l);
  item("discard", "丢弃", DIS);
  const m = $("mark-labeling"); if (m) m.textContent = n[UNL] ? `${n[UNL]} 待确认` : "";
}
function guessTag(c) {
  if (!c.guess) return el("span", "tag", "没猜");
  const g = c.guess;
  return el("span", "tag " + (g.label === "unsure" ? "warn" : "sakura"), g.label === "unsure" ? "看不清" : `${nameOf(g.label)} ${pct(g.confidence)}`);
}
function renderList() {
  const box = $("lb-list");
  box.textContent = "";
  if (!LB.list.length) {
    box.append(el("p", "none lb-empty", LB.filter === "todo" ? (clips().length ? "都标完了。" : "还没有片段。") : "这里没有片段。"));
    return;
  }
  const frag = document.createDocumentFragment();
  for (const c of LB.list) {
    const b = el("button", "lb-item"); b.type = "button"; b.dataset.clip = c.clip;
    b.setAttribute("aria-current", String(c.clip === LB.cur));
    b.title = c.clip;
    b.append(el("span", "n", shortName(c.clip)), guessTag(c), el("span", "r", c.recording || "（没有录像名）"));
    b.onclick = () => select(c.clip);
    frag.append(b);
  }
  box.append(frag);
}
function markCurrent() {
  let hit = null;
  for (const b of $("lb-list").querySelectorAll(".lb-item")) {
    const on = b.dataset.clip === LB.cur;
    b.setAttribute("aria-current", String(on));
    if (on) hit = b;
  }
  if (hit) hit.scrollIntoView({block: "nearest"});
}
function setFilter(f) {
  if (!validFilter(f)) f = "todo";
  LB.filter = f;
  buildList(); renderFilter(); renderCounts(); renderList();
  const keep = LB.list.some(c => c.clip === LB.cur);
  select(keep ? LB.cur : (LB.list[0] ? LB.list[0].clip : null));
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
  if (!LB.active || !LB.playing) return;
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
  preload(); markCurrent(); renderHead(); renderSide(); draw();
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
  else if (!c.guess) gbox.append(el("p", "none", "Claude 还没猜这一段（跑 perception gesture-label）"));
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
  const off = !c || LB.busy, g = c && c.guess, canAgree = !!g && g.label !== "unsure";
  const agree = kbdBtn("btn go lb-agree", canAgree ? `同意 Claude：${nameOf(g.label)}` : "同意 Claude", "Enter", agreeClaude);
  agree.disabled = off || !canAgree;
  acts.append(agree);
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

/* ---- 标注 / 撤销 ---- */
function apply(item) {
  const c = byName(item.clip);
  if (c) { c.where = item.where; c.guess = item.guess; c.recording = item.recording; }
  else clips().push({clip: item.clip, recording: item.recording, where: item.where, guess: item.guess});
}
function rerender() { buildList(); renderFilter(); renderCounts(); renderList(); }
function advance(old, idx) {  // 选完跳下一段：原列表里排在它后面、还留在新列表里的第一段；后面没了往前找
  rerender();
  const left = new Set(LB.list.map(c => c.clip));
  for (let j = idx + 1; j < old.length; j++) if (old[j].clip !== LB.cur && left.has(old[j].clip)) return select(old[j].clip);
  for (let j = idx - 1; j >= 0; j--) if (old[j].clip !== LB.cur && left.has(old[j].clip)) return select(old[j].clip);
  select(left.has(LB.cur) ? LB.cur : (LB.list[0] ? LB.list[0].clip : null));
}
async function label(to) {
  const c = byName(LB.cur);
  if (!c || LB.busy) return;
  const old = LB.list.slice(), idx = old.findIndex(x => x.clip === c.clip);
  if (c.where === (to === "discard" ? DIS : to)) { advance(old, idx); return; }  // 已经是这一类：当作确认过，跳下一段
  LB.busy = true; renderSide();
  let r;
  try { r = await post("api/gesture/label", {clip: c.clip, to}); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  LB.busy = false;
  if (!r.data.ok) {
    toast(r.data.text || "没标上", "bad");
    if (r.status === 409) load(); else renderSide();
    return;
  }
  apply(r.data);
  advance(old, idx);
}
function agreeClaude() {
  const c = byName(LB.cur);
  if (c && c.guess && c.guess.label !== "unsure") label(c.guess.label);
}
async function undo() {
  if (LB.busy) return;
  LB.busy = true; renderSide();
  let r;
  try { r = await post("api/gesture/undo"); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  LB.busy = false;
  if (!r.data.ok) { toast(r.data.text || "没撤销成", r.status === 409 ? "warn" : "bad"); renderSide(); return; }
  apply(r.data);
  rerender(); select(r.data.clip);
  toast(`撤销了：${shortName(r.data.clip)} 回到「${whereText(r.data.where)}」`, "ok");
}

/* ---- 读数据 ---- */
async function load() {
  if (LB.loading) return;
  LB.loading = true;
  let d;
  try { d = await getJSON("api/gesture/state"); }
  catch (e) { d = {ok: false, text: "读不到片段（面板停了？）"}; }
  finally { LB.loading = false; }
  const banner = $("lb-empty");
  if (!d || !d.ok) {
    LB.data = null; LB.list = []; LB.cur = null;
    banner.textContent = ""; banner.append(el("span", "", (d && d.text) || "读不到片段"));
    banner.hidden = false; $("lb-cols").hidden = true;
    const m = $("mark-labeling"); if (m) m.textContent = "";
    return;
  }
  LB.data = d; banner.hidden = true; $("lb-cols").hidden = false;
  if (!validFilter(LB.filter)) LB.filter = "todo";
  rerender();
  const keep = byName(LB.cur);
  select(keep ? LB.cur : (LB.list[0] ? LB.list[0].clip : null));
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
  if (k === " " || k === "Spacebar") act = () => setPlaying(!LB.playing);
  else if (k === "Enter") act = agreeClaude;
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
    $("lb-refresh").onclick = () => load();
    // 鼠标点按钮不抢焦点：点完接着按键就行（Tab 过去的按钮照样能用回车 / 空格）
    $("page-labeling").addEventListener("mousedown", e => { if (e.target.closest && e.target.closest("button")) e.preventDefault(); });
    setPlaying(true); setSpeed(1); draw();
  },
  show() {  // 可能被重复调用（点当前导航项）：只重读，键盘不重复绑
    LB.active = true;
    if (!LB.keyBound) { document.addEventListener("keydown", onKey); LB.keyBound = true; }
    startTimer();
    load();
  },
  hide() {
    LB.active = false;
    stopTimer();
    if (LB.keyBound) { document.removeEventListener("keydown", onKey); LB.keyBound = false; }
  },
};
})();
