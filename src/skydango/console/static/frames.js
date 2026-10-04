/* 标注页第四个标签「整帧」（spec 2026-10-04-hardcase-inbox §5.2）：难例收件箱里整理好的帧，整张截图 + 全部类别的框。
 * 左 帧列表（运行 / 状态筛选、各状态计数），中 画布（滚轮以光标为中心缩放、拖动 / 按住空格拖动画面），右 这一帧、操作、类别、按键。
 * 两种模式，画面上方一直标着：
 *   过目（默认）：回车 通过（「要编辑」的帧只提示先按 E）、E 编辑、0 整帧不要、Z 撤销上一步、← → 翻帧、H / 按住空格 隐藏框；通过 / 不要后自动跳下一帧
 *   编辑中：空白处拖 = 画新框、点框 = 选中、拖框内 = 移动、拖边角 = 调大小、Delete = 删、数字键 = 类别（1 点过火的人 … 0 乐器）、
 *          Z = 撤一步、回车 = 保存并通过、Esc = 放弃修改（空格只用来拖画面）
 * 接口 api/frames/state | image | act（后端 console/frames.py），框的类别用 [perception] classes 的编号。
 * 帧名、原因都是数据：只用 textContent / fillText。labeling.js 切到这个标签时调 FramesTab.load()，按键转给 FramesTab.key()。
 * 顶层不碰 document / window：node 里 require 它能测坐标换算、归一、点中、拖框、按键（tests/test_console_page.py）。 */
(function () {
"use strict";

/* ---- 纯计算（node 里测） ---- */
// 编辑模式数字键 → 类别编号（[perception] classes：player 0、name_tag 1、social_ring 2、self 3、player_unlit 4、typing 5、bench 6、bonfire 7、instrument 8、spirit 9）
const KEY_CLASS = {"1": 0, "2": 4, "3": 3, "4": 9, "5": 1, "6": 2, "7": 5, "8": 6, "9": 7, "0": 8};
const CLASS_NAMES = ["点过火的人", "名字标签", "圆圈", "团子", "黑影", "气泡", "座位", "篝火", "乐器", "先祖"];
const CLASS_KIND = ["player", "tag", "ring", "self", "unlit", "typing", "bench", "bonfire", "instrument", "spirit"];  // 配色借 stage.js 的哪一种
const MIN_SIDE = 4;  // 同后端 frames.MIN_SIDE

function keyClass(key) { return Object.prototype.hasOwnProperty.call(KEY_CLASS, key) ? KEY_CLASS[key] : null; }
/* 屏幕 = 图像 × scale + 偏移（view = {scale, ox, oy}，屏幕是画布的 CSS 像素） */
function toScreen(p, v) { return [p[0] * v.scale + v.ox, p[1] * v.scale + v.oy]; }
function toImage(p, v) { return [(p[0] - v.ox) / v.scale, (p[1] - v.oy) / v.scale]; }
/* 以屏幕点 p 为中心缩放 f 倍：p 下面的图像点不动 */
function zoomAt(v, p, f) {
  const s = v.scale * f;
  return {scale: s, ox: p[0] - (p[0] - v.ox) * f, oy: p[1] - (p[1] - v.oy) * f};
}
/* 整张图放进 cw × ch，居中 */
function fitView(w, h, cw, ch) {
  const s = Math.min(cw / w, ch / h) || 1;
  return {scale: s, ox: (cw - w * s) / 2, oy: (ch - h * s) / 2};
}
function pyRound(x) {  // 同 Python 的 round：正好 .5 取偶数
  const f = Math.floor(x), d = x - f;
  return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1);
}
/* 同后端 frames.normalize_box：负宽高翻正、裁进 [0, w) × [0, h)；宽或高 < MIN_SIDE 返回 null */
function normBox(b, w, h) {
  let [x, y, bw, bh] = b.map(v => pyRound(Number(v)));
  if (![x, y, bw, bh].every(Number.isFinite)) return null;
  if (bw < 0) { x += bw; bw = -bw; }
  if (bh < 0) { y += bh; bh = -bh; }
  const x1 = Math.max(0, x), y1 = Math.max(0, y), x2 = Math.min(w, x + bw), y2 = Math.min(h, y + bh);
  if (x2 - x1 < MIN_SIDE || y2 - y1 < MIN_SIDE) return null;
  return [x1, y1, x2 - x1, y2 - y1];
}
/* 点 pt（图像坐标）落在哪个框的哪个部位：角（nw ne sw se）> 边（n s w e）> 框里（move），同一档取面积最小的；tol = 容差（图像像素） */
function hitTest(boxes, pt, tol) {
  const [px, py] = pt;
  let best = null;
  const rankOf = part => part === "move" ? 2 : part.length === 2 ? 0 : 1;
  boxes.forEach((item, i) => {
    const [x, y, w, h] = item.box || item;
    const x2 = x + w, y2 = y + h, area = Math.abs(w * h);
    const nearX = d => Math.abs(px - d) <= tol, nearY = d => Math.abs(py - d) <= tol;
    const inX = px >= Math.min(x, x2) - tol && px <= Math.max(x, x2) + tol, inY = py >= Math.min(y, y2) - tol && py <= Math.max(y, y2) + tol;
    let part = null;
    const v = nearY(y) ? "n" : nearY(y2) ? "s" : "", hz = nearX(x) ? "w" : nearX(x2) ? "e" : "";
    if (v && hz) part = v + hz;
    else if (v && inX) part = v;
    else if (hz && inY) part = hz;
    else if (px >= Math.min(x, x2) && px <= Math.max(x, x2) && py >= Math.min(y, y2) && py <= Math.max(y, y2)) part = "move";
    if (!part) return;
    const r = rankOf(part);
    if (!best || r < best.r || (r === best.r && area < best.area)) best = {i, part, r, area};
  });
  return best && {i: best.i, part: best.part};
}
/* 拖动部位 part 移动 (dx, dy)：move 平移，边角改对应的边；可能出现负宽高，松手时 normBox 翻正 */
function resizeBox(b, part, dx, dy) {
  let [x, y, w, h] = b;
  if (part === "move") return [x + dx, y + dy, w, h];
  if (part.includes("n")) { y += dy; h -= dy; }
  if (part.includes("s")) h += dy;
  if (part.includes("w")) { x += dx; w -= dx; }
  if (part.includes("e")) w += dx;
  return [x, y, w, h];
}

/* 一次只跑一个：跑着时再调就排一次（只排一次；带了 pick 的留最新的 pick，不带的不盖掉），返回的 Promise 等排着的也跑完 */
function serial(fn) {
  let running = null, queued = null;
  return function run(pick) {
    if (running) { queued = {pick: pick || (queued && queued.pick) || null}; return running; }
    running = (async () => {
      let next = {pick: pick || null};
      try {
        while (next) {
          queued = null;
          try { await fn(next.pick); } catch (e) { console.error("整帧页：读收件箱出错", e); }
          next = queued;
        }
      } finally { running = null; }
    })();
    return running;
  };
}
/* 标注页顶上「整帧（N）」：所有运行里待过目 + 要编辑的帧 */
function badgeText(list) {
  const n = (list || []).filter(f => f.state === "glance" || f.state === "edit").length;
  return n ? `整帧（${n}）` : "整帧";
}

/* ---- 页面 ---- */
const STATES = [["glance", "待过目"], ["edit", "要编辑"], ["crops", "等判裁图"], ["done", "已通过"], ["discarded", "不要了"], ["error", "整理出错"]];
const STATE_NAME = Object.fromEntries(STATES);
const STATE_TAG = {glance: "sakura", edit: "warn", crops: "", done: "ok", discarded: "bad", error: "bad"};
const SRC = {auto: "自动", judged: "你判的", yolo: "YOLO 预标", drawn: "你画的", edited: "你改的", final: "通过时的"};  // final：没编辑直接通过时写进去的框
const REASONS = {
  attrs_reject: "外形头撤下了 YOLO 的高分框", attrs_disagree: "YOLO 和外形头对点没点火意见相反", low_conf: "置信度低",
  flicker: "框一闪一闪", unlit_vs_player: "点没点火来回变", ocr_only: "整图 OCR 读到好友、YOLO 没认出", appearance: "按外观认的人和名字标签对不上",
};
const KEYS = {
  glance: [["Enter", "通过"], ["E", "编辑"], ["0", "整帧不要"], ["Z", "撤销上一步"], ["← →", "翻帧"], ["H / 按住空格", "隐藏框"],
    ["滚轮", "以光标为中心缩放"], ["拖动", "拖画面（双击复原）"]],
  edit: [["拖空白处", "画新框"], ["点框", "选中"], ["拖框里 / 边角", "移动 / 调大小"], ["Delete", "删选中的框"],
    ["1~9 0", "类别（选中的框或下一个新框）"], ["Z", "撤一步"], ["H", "隐藏框"], ["空格 + 拖", "拖画面"], ["滚轮", "缩放"],
    ["Enter", "保存并通过"], ["Esc", "放弃修改"]],
};
const ZOOM_MAX = 16, HANDLE = 4;
const FR = {data: null, run: "", filter: "glance", list: [], cur: null, mode: "glance",
  boxes: [], sel: -1, cls: 0, steps: [], dirty: false,  // 编辑中：框的副本、选中、新框类别、撤一步用的快照
  hide: false, space: false, drag: null, view: null, fitted: true, img: null, imgFor: null, imgBad: false,
  done: [], busy: false, bound: false};

function colorOf(cls) { const C = (globalThis.Stage && globalThis.Stage.COLORS) || {}; return C[CLASS_KIND[cls]] || "#ffffff"; }
function className(cls) { return CLASS_NAMES[cls] || `类别 ${cls}`; }
function reasonText(r) { return r ? (REASONS[r] || r) : "（没记原因）"; }
function frames() { return (FR.data && FR.data.frames) || []; }
function byName(name) { return name == null ? null : frames().find(f => f.frame === name) || null; }
function inRun(f) { return !FR.run || f.run === FR.run; }
function matches(f) { return inRun(f) && (FR.filter === "all" || f.state === FR.filter); }
function counts() {
  const n = {all: 0};
  for (const f of frames()) if (inRun(f)) { n[f.state] = (n[f.state] || 0) + 1; n.all++; }
  return n;
}
function editing() { return FR.mode === "edit"; }
function hidden() { return FR.hide || (FR.space && !editing()); }
function imageUrl(name) { return `api/frames/image?frame=${encodeURIComponent(name)}`; }
function size(f) {
  const w = (f && f.w) || (FR.img && FR.img.naturalWidth) || 0, h = (f && f.h) || (FR.img && FR.img.naturalHeight) || 0;
  return [w, h];
}

/* ---- 列表 ---- */
function build() {
  FR.list = frames().filter(matches).sort((a, b) => (a.frame < b.frame ? -1 : a.frame > b.frame ? 1 : 0));
}
function renderFilters() {
  const n = counts(), all = frames();
  const opt = (v, t) => { const o = el("option", "", t); o.value = v; return o; };
  const run = $("fr-run"); run.textContent = "";
  run.append(opt("", `全部运行（${all.length}）`));
  const runs = {};
  for (const f of all) runs[f.run] = (runs[f.run] || 0) + 1;
  for (const r of Object.keys(runs).sort().reverse()) run.append(opt(r, `${r}（${runs[r]}）`));
  run.value = FR.run;
  const sel = $("fr-filter"); sel.textContent = "";
  for (const [k, t] of STATES) if (k !== "error" || n.error || FR.filter === "error") sel.append(opt(k, `${t}（${n[k] || 0}）`));
  sel.append(opt("all", `全部（${n.all}）`));
  sel.value = FR.filter;
}
function renderCounts() {
  const box = $("fr-counts"), n = counts();
  box.textContent = "";
  for (const [k, t] of [...STATES, ["all", "全部"]]) {
    if (k === "error" && !n.error && FR.filter !== "error") continue;
    const b = el("button", "lb-count"); b.type = "button";
    b.setAttribute("aria-pressed", String(FR.filter === k));
    b.append(t, el("b", "", String(n[k] || 0)));
    b.onclick = () => setFilter(k);
    box.append(b);
  }
  setBadge(frames());
}
function renderList() {
  const box = $("fr-list");
  box.textContent = "";
  if (!FR.list.length) {
    box.append(el("p", "none lb-empty", !frames().length ? "收件箱里还没有整理好的帧（先跑整理）。" : FR.filter === "glance" ? "都过目完了。" : "这里没有帧。"));
    return;
  }
  const frag = document.createDocumentFragment();
  for (const f of FR.list) {
    const b = el("button", "lb-item"); b.type = "button"; b.dataset.k = f.frame;
    b.setAttribute("aria-current", String(f.frame === FR.cur));
    b.title = f.frame;
    b.append(el("span", "n", f.frame), el("span", "tag " + (STATE_TAG[f.state] || ""), STATE_NAME[f.state] || f.state), el("span", "r", reasonText(f.reason)));
    b.onclick = () => select(f.frame);
    frag.append(b);
  }
  box.append(frag);
}
function markCurrent() {
  let hit = null;
  for (const b of $("fr-list").querySelectorAll(".lb-item")) {
    const on = b.dataset.k === FR.cur;
    b.setAttribute("aria-current", String(on));
    if (on) hit = b;
  }
  if (hit) hit.scrollIntoView({block: "nearest"});
}
function rerender() { build(); renderFilters(); renderCounts(); renderList(); }
function setFilter(f) {
  if (editing()) { toast("编辑中：先保存（回车）或放弃（Esc）", "warn"); renderFilters(); return; }
  FR.filter = f === "all" || STATE_NAME[f] ? f : "glance";
  rerender();
  select(FR.list.some(x => x.frame === FR.cur) ? FR.cur : (FR.list[0] ? FR.list[0].frame : null));
}
function setRun(r) {
  if (editing()) { toast("编辑中：先保存（回车）或放弃（Esc）", "warn"); renderFilters(); return; }
  FR.run = r || "";
  rerender();
  select(FR.list.some(x => x.frame === FR.cur) ? FR.cur : (FR.list[0] ? FR.list[0].frame : null));
}

/* ---- 选中一帧 ---- */
function select(name) {
  if (editing() && name !== FR.cur) { toast("编辑中：先保存（回车）或放弃（Esc）", "warn"); return; }
  const changed = name !== FR.cur;
  FR.cur = name || null;
  if (changed) {
    FR.img = null; FR.imgBad = false; FR.imgFor = FR.cur; FR.fitted = true; FR.view = null;
    if (FR.cur) {
      const img = new Image(), want = FR.cur;
      img.onload = () => { if (FR.cur === want) { FR.img = img; FR.fitted = true; draw(); } };
      img.onerror = () => { if (FR.cur === want) { FR.imgBad = true; draw(); } };
      img.src = imageUrl(want);
      const at = FR.list.findIndex(f => f.frame === want);  // 预加载后两帧
      for (let j = at + 1; at >= 0 && j <= at + 2 && j < FR.list.length; j++) new Image().src = imageUrl(FR.list[j].frame);
    }
  }
  markCurrent(); renderHead(); renderSide(); draw();
}
function step(d) {
  if (editing() || !FR.list.length) return;
  const at = FR.list.findIndex(f => f.frame === FR.cur);
  const j = at < 0 ? 0 : Math.min(FR.list.length - 1, Math.max(0, at + d));
  select(FR.list[j].frame);
}
function renderHead() {
  const f = byName(FR.cur), at = FR.list.findIndex(x => x.frame === FR.cur);
  const mode = $("fr-mode");
  mode.dataset.mode = FR.mode;
  mode.textContent = editing() ? `编辑中 · 新框：${className(FR.cls)}` : "过目";
  $("fr-stage").classList.toggle("edit", editing());
  $("fr-title").textContent = FR.cur || "—"; $("fr-title").title = FR.cur || "";
  $("fr-pos").textContent = !FR.cur ? "" : at < 0 ? "（不在当前筛选里）" : `第 ${at + 1} / ${FR.list.length} 帧`;
  $("fr-reason").textContent = f ? `当初为什么存：${reasonText(f.reason)}` : "";
}
function kbdBtn(cls, text, key, onClick) {
  const b = el("button", cls); b.type = "button";
  b.append(el("kbd", "k", key), el("span", "t", text));
  b.onclick = onClick;
  return b;
}
function renderSide() {
  const f = byName(FR.cur);
  const meta = $("fr-meta"); meta.textContent = "";
  if (f) {
    const row = (k, v) => { meta.append(el("dt", "", k)); const dd = el("dd"); dd.append(v); meta.append(dd); };
    row("现在", el("span", "tag " + (STATE_TAG[f.state] || ""), STATE_NAME[f.state] || f.state));
    row("运行", el("code", "", f.run));
    row("分边", f.split === "val" ? "验证集" : f.split === "train" ? "训练集" : (f.split || "—"));
    const boxes = editing() ? FR.boxes : f.boxes.filter(b => !b.dropped), dropped = f.boxes.filter(b => b.dropped).length;
    row("框", `${boxes.length} 个${dropped ? `（另有 ${dropped} 个自动丢掉的，淡虚线）` : ""}`);
    if (f.state === "crops") row("提示", "还有裁图没在「外形」页判，判完才能直接通过；也可以按 E 自己改框");
    if (f.state === "edit") row("提示", "有裁图被判了不要：按 E 改好框、回车保存才能通过");
  }
  const acts = $("fr-actions"); acts.textContent = "";
  const off = !f || FR.busy;
  if (editing()) {
    const save = kbdBtn("btn go lb-agree", "保存并通过", "Enter", saveEdit); save.disabled = off;
    const grid = el("div", "lb-grid");
    const del = kbdBtn("btn lb-act", "删选中的框", "Del", deleteSel); del.disabled = off || FR.sel < 0;
    const back = kbdBtn("btn lb-act", "撤一步", "Z", undoStep); back.disabled = off || !FR.steps.length;
    const quit = kbdBtn("btn halt lb-act", "放弃修改", "Esc", cancelEdit); quit.disabled = FR.busy;
    grid.append(del, back, quit);
    acts.append(save, grid);
  } else {
    const pass = kbdBtn("btn go lb-agree", "通过", "Enter", passFrame); pass.disabled = off || f.state === "done";
    const grid = el("div", "lb-grid");
    const edit = kbdBtn("btn lb-act", "编辑", "E", startEdit); edit.disabled = off || f.state === "done";
    const dis = kbdBtn("btn halt lb-act", "整帧不要", "0", discardFrame); dis.disabled = off || f.state === "done" || f.state === "discarded";
    grid.append(edit, dis);
    const undo = kbdBtn("btn sm lb-undo", "撤销上一步", "Z", undoAct); undo.disabled = FR.busy;
    acts.append(pass, grid, undo);
  }
  const cbox = $("fr-classes"); cbox.textContent = "";
  const now = FR.sel >= 0 && FR.boxes[FR.sel] ? FR.boxes[FR.sel].cls : FR.cls;
  for (const [key, cls] of Object.entries(KEY_CLASS).sort((a, b) => (a[0] === "0") - (b[0] === "0") || a[0] - b[0])) {
    const b = kbdBtn("btn fr-cls", className(cls), key, () => setClass(cls));
    const sw = el("i"); sw.style.background = colorOf(cls); b.prepend(sw);
    if (editing() && cls === now) b.classList.add("on");
    b.disabled = !editing() || FR.busy;
    cbox.append(b);
  }
  const keys = $("fr-keys"); keys.textContent = "";
  for (const [k, t] of KEYS[FR.mode]) { const dt = el("dt"); dt.append(el("kbd", "k", k)); keys.append(dt, el("dd", "", t)); }
}

/* ---- 画 ---- */
function shownBoxes() {  // [{cls, box, src, dropped, i}]：编辑中是可改的副本（i = 下标）+ 只读的自动丢掉的框
  const f = byName(FR.cur);
  if (!f) return [];
  if (!editing()) return f.boxes.map(b => ({...b, i: -1}));
  return [...f.boxes.filter(b => b.dropped).map(b => ({...b, i: -1})), ...FR.boxes.map((b, i) => ({...b, i}))];
}
function draw() {
  const cv = $("fr-canvas"), stage = $("fr-stage"), note = $("fr-stage-note");
  const r = stage.getBoundingClientRect(), cw = Math.round(r.width), ch = Math.round(r.height);
  if (!cw || !ch) return;  // 标签没开着
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
  if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const f = byName(FR.cur);
  let text = "";
  if (!FR.cur) text = FR.data ? (FR.list.length ? "从列表里选一帧" : "这个筛选里没有帧") : "";
  else if (FR.imgBad) text = "这一帧的图读不到";
  else if (!FR.img) text = "载入中…";
  note.textContent = text; note.hidden = !text;
  if (!FR.img || !f) return;
  const [w, h] = size(f);
  if (!FR.view || FR.fitted) FR.view = fitView(w, h, cw, ch);
  const v = FR.view;
  ctx.imageSmoothingEnabled = v.scale < 2;
  ctx.drawImage(FR.img, v.ox, v.oy, w * v.scale, h * v.scale);
  if (hidden()) { label(ctx, "框藏起来了", 8, 8, "rgba(11,13,18,.75)", "#e6e8ee"); return; }
  const fs = 12;
  ctx.font = `${fs}px system-ui,"Microsoft YaHei",sans-serif`; ctx.textBaseline = "middle";
  for (const b of shownBoxes()) {
    const col = colorOf(b.cls), [x, y] = toScreen(b.box, v), bw = b.box[2] * v.scale, bh = b.box[3] * v.scale;
    const sel = editing() && b.i >= 0 && b.i === FR.sel;
    ctx.save();
    if (b.dropped) { ctx.globalAlpha = 0.55; ctx.setLineDash([5, 5]); }
    ctx.strokeStyle = col; ctx.lineWidth = sel ? 3 : 2; ctx.strokeRect(x, y, bw, bh);
    ctx.setLineDash([]);
    if (sel) {
      ctx.fillStyle = col;
      for (const [hx, hy] of [[x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]]) ctx.fillRect(hx - HANDLE, hy - HANDLE, HANDLE * 2, HANDLE * 2);
    }
    const t = `${className(b.cls)} · ${b.dropped ? "自动丢掉" : SRC[b.src] || b.src || ""}`;
    const th = fs + 6, ty = y - th < 0 ? y + bh : y - th;
    label(ctx, t, x, ty, col, "#0b0d12");
    ctx.restore();
  }
  const d = FR.drag;
  if (d && d.kind === "new" && d.cur) {
    const b = normBox([d.start[0], d.start[1], d.cur[0] - d.start[0], d.cur[1] - d.start[1]], w, h);
    if (b) {
      const [x, y] = toScreen(b, v);
      ctx.save(); ctx.strokeStyle = colorOf(FR.cls); ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, b[2] * v.scale, b[3] * v.scale); ctx.restore();
    }
  }
}
function label(ctx, t, x, y, bg, fg) {
  const fs = 12, th = fs + 6;
  ctx.font = `${fs}px system-ui,"Microsoft YaHei",sans-serif`; ctx.textBaseline = "middle";
  const tw = ctx.measureText(t).width + 8;
  ctx.fillStyle = bg; ctx.fillRect(x, y, tw, th);
  ctx.fillStyle = fg; ctx.fillText(t, x + 4, y + th / 2);
}

/* ---- 鼠标 ---- */
function local(e) { const r = $("fr-canvas").getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
const CURSOR = {move: "move", nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", w: "ew-resize", e: "ew-resize"};
function onWheel(e) {
  if (!FR.img || !FR.view) return;
  e.preventDefault();
  const f0 = byName(FR.cur), [w, h] = size(f0), r = $("fr-stage").getBoundingClientRect();
  const fit = fitView(w, h, r.width, r.height).scale;
  const want = Math.min(fit * ZOOM_MAX, Math.max(fit * 0.5, FR.view.scale * Math.exp(-e.deltaY * 0.0015)));
  FR.view = zoomAt(FR.view, local(e), want / FR.view.scale);
  FR.fitted = false;
  draw();
}
function onDown(e) {
  if (!FR.img || !FR.view || FR.busy) return;
  if (e.button !== 0 && e.button !== 1) return;
  e.preventDefault();
  const p = local(e);
  if (FR.space || e.button === 1 || !editing() || FR.hide) {  // 拖画面：过目时直接拖，编辑时按住空格 / 中键（框藏着时也只拖画面）
    FR.drag = {kind: "pan", start: p, view: {...FR.view}};
  } else {
    const pt = toImage(p, FR.view), hit = hidden() ? null : hitTest(FR.boxes, pt, 6 / FR.view.scale);
    if (hit) {
      FR.sel = hit.i;
      FR.drag = {kind: "box", i: hit.i, part: hit.part, start: pt, box0: FR.boxes[hit.i].box.slice(), snap: snapshot(), moved: false};
    } else {
      FR.sel = -1;
      FR.drag = {kind: "new", start: pt, cur: null};
    }
    renderSide();
  }
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  draw();
}
function onMove(e) {
  const d = FR.drag; if (!d) return;
  const p = local(e);
  if (d.kind === "pan") {
    FR.view = {scale: d.view.scale, ox: d.view.ox + p[0] - d.start[0], oy: d.view.oy + p[1] - d.start[1]};
    FR.fitted = false;
  } else {
    const pt = toImage(p, FR.view);
    if (d.kind === "new") d.cur = pt;
    else {
      const dx = Math.round(pt[0] - d.start[0]), dy = Math.round(pt[1] - d.start[1]);
      if (dx || dy) d.moved = true;
      FR.boxes[d.i].box = resizeBox(d.box0, d.part, dx, dy);
    }
  }
  draw();
}
function onUp() {
  window.removeEventListener("mousemove", onMove);
  window.removeEventListener("mouseup", onUp);
  const d = FR.drag; FR.drag = null;
  if (!d || d.kind === "pan") { draw(); return; }
  const [w, h] = size(byName(FR.cur));
  if (d.kind === "new") {
    const b = d.cur && normBox([d.start[0], d.start[1], d.cur[0] - d.start[0], d.cur[1] - d.start[1]], w, h);
    if (b) { remember(); FR.boxes.push({cls: FR.cls, box: b, src: "drawn"}); FR.sel = FR.boxes.length - 1; }
  } else if (d.moved) {
    const b = normBox(FR.boxes[d.i].box, w, h);
    if (b) {
      FR.steps.push(d.snap); FR.dirty = true;
      FR.boxes[d.i].box = b;
      if (FR.boxes[d.i].src !== "drawn") FR.boxes[d.i].src = "edited";
    } else FR.boxes[d.i].box = d.box0;  // 拖得太小 / 拖出画面：放回原样
  }
  renderSide(); draw();
}
function cancelDrag() {  // Esc / 切走：拖到一半的不算，框放回原样
  const d = FR.drag; if (!d) return;
  window.removeEventListener("mousemove", onMove);
  window.removeEventListener("mouseup", onUp);
  FR.drag = null;
  if (d.kind === "box") FR.boxes[d.i].box = d.box0;
  draw();
}
function onHover(e) {
  if (FR.drag) return;
  const cv = $("fr-canvas");
  if (FR.space) { cv.style.cursor = "grab"; return; }
  if (!editing() || !FR.view || hidden()) { cv.style.cursor = ""; return; }
  const hit = hitTest(FR.boxes, toImage(local(e), FR.view), 6 / FR.view.scale);
  cv.style.cursor = hit ? CURSOR[hit.part] : "";
}

/* ---- 编辑 ---- */
function snapshot() { return FR.boxes.map(b => ({...b, box: b.box.slice()})); }
function remember() { FR.steps.push(snapshot()); FR.dirty = true; }
function setClass(cls) {
  if (!editing()) return;
  FR.cls = cls;
  const b = FR.boxes[FR.sel];
  if (b && b.cls !== cls) { remember(); b.cls = cls; if (b.src !== "drawn") b.src = "edited"; }
  renderHead(); renderSide(); draw();
}
function deleteSel() {
  if (!editing() || FR.sel < 0 || !FR.boxes[FR.sel]) return;
  remember(); FR.boxes.splice(FR.sel, 1); FR.sel = -1;
  renderSide(); draw();
}
function undoStep() {
  if (!editing() || !FR.steps.length) return;
  FR.boxes = FR.steps.pop(); FR.sel = -1;
  renderSide(); draw();
}
async function act(body) {
  FR.busy = true; renderSide();
  let r;
  try { r = await post("api/frames/act", body); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  FR.busy = false;
  if (!r.data.ok) toast(r.data.text || "没做成", r.status === 409 ? "warn" : "bad");
  renderSide();
  return r.data.ok ? r.data : null;
}
async function startEdit() {
  const f = byName(FR.cur);
  if (!f || editing() || FR.busy) return;
  if (!(await act({frame: f.frame, do: "edit"}))) return;
  FR.mode = "edit"; FR.boxes = f.boxes.filter(b => !b.dropped).map(b => ({cls: b.cls, box: b.box.slice(), src: b.src}));
  FR.sel = -1; FR.steps = []; FR.dirty = false; FR.space = false;
  renderHead(); renderSide(); draw();
  load();  // 状态变成「要编辑」
}
function leaveEdit() { FR.mode = "glance"; FR.boxes = []; FR.sel = -1; FR.steps = []; FR.dirty = false; FR.drag = null; }
async function cancelEdit() {
  if (!editing() || FR.busy) return;
  if (FR.dirty && !(await ask("放弃这一帧的修改？画的框都不要了。", {ok: "放弃", cancel: "接着改", danger: true}))) return;
  const name = FR.cur;
  if (!(await act({frame: name, do: "cancel_edit"}))) return;
  leaveEdit();
  renderHead(); renderSide(); draw();
  load();
}
async function saveEdit() {
  const f = byName(FR.cur);
  if (!f || !editing() || FR.busy) return;
  const [w, h] = size(f);
  const boxes = FR.boxes.map(b => ({cls: b.cls, box: normBox(b.box, w, h)})).filter(b => b.box);
  const old = FR.list.slice(), name = f.frame;
  const ok = await act({frame: name, do: "pass", boxes});
  if (!ok) return;
  leaveEdit();
  FR.done.push(name);
  toast(`通过了（改过框）：${name}`, "ok");
  await load(() => nextAfter(old, name));
}
/* 过目 */
async function passFrame() {
  const f = byName(FR.cur);
  if (!f || editing() || FR.busy) return;
  if (f.state === "edit") { toast("这一帧要先编辑：有裁图被判了不要，按 E 改好框再回车保存", "warn"); return; }  // 直接通过会把那个人悄悄去掉（后端也回 409）
  const old = FR.list.slice(), name = f.frame;
  if (!(await act({frame: name, do: "pass"}))) return;
  FR.done.push(name);
  await load(() => nextAfter(old, name));
}
async function discardFrame() {
  const f = byName(FR.cur);
  if (!f || editing() || FR.busy) return;
  const old = FR.list.slice(), name = f.frame;
  if (!(await act({frame: name, do: "discard"}))) return;
  FR.done.push(name);
  await load(() => nextAfter(old, name));
}
async function undoAct() {  // 撤销上一步：这次打开页面以来最后通过 / 不要的那一帧；没有就撤当前帧的决定
  if (editing() || FR.busy) return;
  const cur = byName(FR.cur), name = FR.done.length ? FR.done[FR.done.length - 1]
    : cur && (cur.state === "done" || cur.state === "discarded") ? cur.frame : null;
  if (!name) { toast("没有可以撤销的", "warn"); return; }
  if (!(await act({frame: name, do: "undo"}))) return;
  if (FR.done.length && FR.done[FR.done.length - 1] === name) FR.done.pop();
  await load(() => name);
  const f = byName(name);
  toast(`撤销了：${name} 回到「${f ? STATE_NAME[f.state] || f.state : "—"}」`, "ok");
}
function nextAfter(old, name) {  // 原列表里排在它后面、还留在新列表里的第一帧；后面没了往前找
  const left = new Set(FR.list.map(f => f.frame)), idx = old.findIndex(f => f.frame === name);
  for (let j = idx + 1; j < old.length; j++) if (old[j].frame !== name && left.has(old[j].frame)) return old[j].frame;
  for (let j = idx - 1; j >= 0; j--) if (old[j].frame !== name && left.has(old[j].frame)) return old[j].frame;
  return FR.list[0] ? FR.list[0].frame : null;
}

/* ---- 读数据 ---- */
function isOpen() { return window.LabelingTab === "frames"; }
function setBadge(list) { const t = document.querySelector('#lb-tabs [data-tab="frames"]'); if (t) t.textContent = badgeText(list); }
async function peek() {  // 标注页打开时只为标签上的数读一次（整帧页没开过）；开过的用手上的数据
  if (FR.data) { setBadge(frames()); return; }
  try { const d = await getJSON("api/frames/state"); if (d && d.ok && !FR.data) setBadge(d.frames); }
  catch (e) { /* 读不到就不标数 */ }
}
const load = serial(loadOnce);  // startEdit 不等的 load 还在跑时，保存后的 load(跳下一帧) 排在它后面、不丢
async function loadOnce(pick) {
  let d;
  try { d = await getJSON("api/frames/state"); }
  catch (e) { d = {ok: false, text: "读不到收件箱（面板停了？）"}; }
  const banner = $("fr-empty");
  if (!d || !d.ok) {
    if (editing()) { toast((d && d.text) || "读不到收件箱", "bad"); return; }  // 编辑中别把改了一半的框扔掉
    FR.data = null; FR.list = []; FR.cur = null; FR.img = null;
    banner.textContent = (d && d.text) || "读不到收件箱";
    banner.hidden = !isOpen(); $("fr-cols").hidden = true;
    return;
  }
  FR.data = d; banner.hidden = true; $("fr-cols").hidden = !isOpen();
  if (FR.run && !frames().some(f => f.run === FR.run)) FR.run = "";
  rerender();
  if (editing()) {  // 编辑中只刷新列表，不换帧
    if (byName(FR.cur)) { markCurrent(); renderHead(); renderSide(); draw(); return; }
    leaveEdit();  // 这一帧没了（被别处删了）
  }
  const want = pick ? pick() : FR.cur;
  select(byName(want) ? want : (FR.list[0] ? FR.list[0].frame : null));
}

/* ---- 键盘（labeling.js 的 onKey 已经滤掉输入框、对话框、组合键） ---- */
function key(e) {
  const k = e.key;
  let fn = null;
  if (k === " " || k === "Spacebar") {
    e.preventDefault();
    if (!e.repeat && !FR.space) { FR.space = true; $("fr-canvas").style.cursor = "grab"; draw(); }
    return;
  }
  if (k === "h" || k === "H") fn = () => { FR.hide = !FR.hide; draw(); };
  else if (editing()) {
    if (k === "Enter") fn = saveEdit;
    else if (k === "Escape") fn = () => { if (FR.drag) cancelDrag(); else cancelEdit(); };
    else if (k === "Delete" || k === "Backspace") fn = deleteSel;
    else if (k === "z" || k === "Z") fn = undoStep;
    else if (keyClass(k) !== null) fn = () => setClass(keyClass(k));
  } else {
    if (k === "Enter") fn = passFrame;
    else if (k === "e" || k === "E") fn = startEdit;
    else if (k === "0") fn = discardFrame;
    else if (k === "z" || k === "Z") fn = undoAct;
    else if (k === "ArrowLeft") fn = () => step(-1);
    else if (k === "ArrowRight") fn = () => step(1);
  }
  if (!fn) return;
  e.preventDefault();
  if (e.repeat && k !== "ArrowLeft" && k !== "ArrowRight") return;  // 按住不放不连着做
  fn();
}
function release() { if (FR.space) { FR.space = false; $("fr-canvas").style.cursor = ""; draw(); } }

function bind() {
  if (FR.bound) return;
  FR.bound = true;
  const cv = $("fr-canvas");
  cv.addEventListener("wheel", onWheel, {passive: false});
  cv.addEventListener("mousedown", onDown);
  cv.addEventListener("mousemove", onHover);
  cv.addEventListener("dblclick", () => { if (!editing()) { FR.fitted = true; draw(); } });
  cv.addEventListener("contextmenu", e => e.preventDefault());
  document.addEventListener("keyup", e => { if (e.key === " " || e.key === "Spacebar") release(); });
  window.addEventListener("blur", release);
  $("fr-filter").onchange = e => { setFilter(e.currentTarget.value); e.currentTarget.blur(); };  // 选完把焦点还给页面，按键才有用
  $("fr-run").onchange = e => { setRun(e.currentTarget.value); e.currentTarget.blur(); };
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe($("fr-stage"));
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  window.FramesTab = {
    load() { bind(); return load(); },
    hasData() { return !!FR.data; },
    peek,
    key,
    hide() { release(); cancelDrag(); },
  };
}
if (typeof module !== "undefined" && module.exports) module.exports = {toImage, toScreen, normBox, hitTest, resizeBox, keyClass, KEY_CLASS, zoomAt, serial, badgeText};
})();
