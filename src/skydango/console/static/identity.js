/* 认人：按帧切段、标身份、撤销。纯计算可由 node 直接载入。 */
(function () {
"use strict";
const clone = segs => segs.map(s => s.slice());
function segAt(segs, t) {
  let i = segs.length - 1;
  while (i > 0 && segs[i][0] > t) i--;
  return i;
}
function splitAt(segs, t) {
  if (t <= 0 || segs.some(s => s[0] === t)) return clone(segs);
  const out = clone(segs), i = segAt(segs, t);
  out.splice(i + 1, 0, [t, segs[i][1]]);
  return out;
}
function merge(segs) { return segs.filter((s, i) => !i || s[1] !== segs[i - 1][1]); }
function removeCut(segs, t) {
  const out = clone(segs);
  if (out.length < 2) return out;
  let nearest = 1;
  for (let i = 2; i < out.length; i++) if (Math.abs(out[i][0] - t) < Math.abs(out[nearest][0] - t)) nearest = i;
  out.splice(nearest, 1);
  return merge(out);
}
function relabel(segs, t, who) {
  const out = clone(segs);
  out[segAt(segs, t)][1] = who;
  return merge(out);
}
function sampleIdx(n, k = 12) {
  const count = Math.max(0, Math.min(n, k));
  return count < 2 ? (count ? [0] : []) : Array.from({length: count}, (_, i) => Math.round(i * (n - 1) / (count - 1)));
}
function keyLabel(key, friends) {
  if (/^[1-9]$/.test(key)) return friends[Number(key) - 1] || null;
  return ({d: "团子", s: "陌生人", x: "不要", "0": "?"})[key.toLowerCase()] || null;
}
function reviewed(t) { return t.reviewed || JSON.stringify(t.segments) !== JSON.stringify([[0, t.guess]]); }
function coverage(tracks) {
  let done = 0, total = 0;
  for (const t of tracks) {
    total += t.frames.length;
    if (reviewed(t)) done += t.frames.filter(f => t.segments[segAt(t.segments, f.t)][1] !== "?").length;
  }
  return {done, total};
}
if (typeof module !== "undefined") module.exports = {splitAt, removeCut, relabel, segAt, sampleIdx, keyLabel, coverage};
if (typeof document === "undefined") return;

const ID = {data: null, run: "", cur: null, i: 0, filter: "all", expanded: false, stack: [], busy: false,
  active: false, epoch: 0, imageEpoch: 0, imageURL: null};
const current = () => ID.data && ID.data.tracks.find(t => t.id === ID.cur);
const uri = (kind, values) => "api/identity/" + kind + "?" + new URLSearchParams({run: ID.run, ...values});
function button(text, fn, cls = "btn sm") {
  const b = el("button", cls, text); b.type = "button"; b.onclick = fn; return b;
}
function color(who) {
  const special = {"团子": "dango", "陌生人": "stranger", "不要": "skip", "?": "unsure"};
  const index = ID.data ? ID.data.friends.indexOf(who) : -1;
  return "var(--id-" + (special[who] || (index < 0 ? "unsure" : "c" + (index % 6 + 1))) + ")";
}
function notice(text) { $("id-empty").textContent = text; $("id-empty").hidden = !text || !ID.active; }
function matches(t) {
  return ID.filter === "all" || (ID.filter === "todo" && !reviewed(t)) ||
    (ID.filter === "two" && t.two_names) || (ID.filter === "unsure" && t.segments.some(s => s[1] === "?"));
}
function progress() {
  const c = coverage(ID.data.tracks), percent = c.total ? Math.round(c.done * 100 / c.total) : 0;
  $("id-progress").textContent = `已标 ${c.done} / ${c.total} 框（${percent}%）`;
  const ready = c.total > 0 && c.done / c.total >= .9;
  $("id-progress").dataset.done = ready ? "1" : "0";
  $("id-ready").textContent = ready ? `可以去命令行跑 perception identity-truth ${ID.data.source} --apply，再跑 identity-eval` : "";
}
function renderList() {
  const list = $("id-list"); list.replaceChildren();
  for (const t of ID.data.tracks.filter(matches)) {
    const f = t.frames, range = f.length ? `${f[0].t.toFixed(1)}~${f[f.length - 1].t.toFixed(1)} s` : "无帧";
    const b = button(`t${t.id} · ${f.length} 帧 · ${range}`, () => select(t.id), "lb-item id-item");
    b.dataset.selected = String(t.id === ID.cur);
    b.setAttribute("aria-pressed", String(t.id === ID.cur));
    b.append(el("span", "note", `初判 ${t.guess} → 现在：${t.segments.map(s => s[1]).join(" | ")}`));
    if (t.two_names) b.append(el("span", "id-warning", "挂过两个名字"));
    if (reviewed(t)) b.append(el("span", "note", "已看过"));
    list.append(b);
  }
  if (!list.children.length) list.append(el("p", "note", "这个筛选下没有轨迹"));
  progress();
}
function select(id, index = 0) {
  if (ID.busy) return;
  ID.cur = id; ID.i = index; ID.expanded = false;
  renderList(); renderTrack();
}
function stepTrack(delta) {
  const list = ID.data.tracks.filter(matches), at = list.findIndex(t => t.id === ID.cur);
  const t = list[Math.max(0, Math.min(list.length - 1, at + delta))];
  if (t) select(t.id);
}
function seek(i) {
  const t = current(); if (!t || ID.busy) return;
  ID.i = Math.max(0, Math.min(t.frames.length - 1, i));
  renderTrack();
}
function revoke() {
  if (ID.imageURL) URL.revokeObjectURL(ID.imageURL);
  ID.imageURL = null;
}
async function renderFrame(t, f) {
  const epoch = ++ID.imageEpoch, img = $("id-frame-image"), canvas = $("id-frame-boxes");
  img.hidden = true; canvas.hidden = true;
  try {
    const response = await fetch(uri("frame", {frame: f.name}), {cache: "no-store"});
    if (!response.ok) { const data = await response.json(); throw new Error(data.text || "读不到整帧"); }
    const w = Number(response.headers.get("X-Image-Width")), h = Number(response.headers.get("X-Image-Height"));
    if (!w || !h) throw new Error("整帧缺少原始尺寸，刷新一下");
    const blob = await response.blob();
    if (epoch !== ID.imageEpoch || !ID.active) return;
    revoke(); ID.imageURL = URL.createObjectURL(blob);
    img.onload = () => {
      if (epoch !== ID.imageEpoch || !ID.active) return;
      img.hidden = false; canvas.hidden = false;
      canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d"), sx = canvas.width / w, sy = canvas.height / h;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const ink = getComputedStyle(document.documentElement).getPropertyValue("--ink");
      for (const other of ID.data.tracks) {
        const frame = other.frames.find(row => row.name === f.name); if (!frame) continue;
        ctx.strokeStyle = ink; ctx.globalAlpha = other.id === t.id ? 1 : .3;
        ctx.lineWidth = other.id === t.id ? 4 : 1;
        const [x, y, bw, bh] = frame.box; ctx.strokeRect(x * sx, y * sy, bw * sx, bh * sy);
      }
      ctx.globalAlpha = 1;
    };
    img.src = ID.imageURL;
  } catch (err) { if (epoch === ID.imageEpoch && ID.active) notice(err.message); }
}
function renderTrack() {
  const t = current(), strip = $("id-strip"), tags = $("id-tagline"), bar = $("id-segbar");
  strip.replaceChildren(); tags.replaceChildren(); bar.replaceChildren();
  if (!t || !t.frames.length) {
    ID.imageEpoch++; $("id-frame-image").hidden = true; $("id-frame-boxes").hidden = true;
    $("id-title").textContent = "选一条轨迹"; return;
  }
  ID.i = Math.max(0, Math.min(t.frames.length - 1, ID.i));
  $("id-title").textContent = `t${t.id} · 第 ${ID.i + 1} / ${t.frames.length} 帧`;
  $("id-expand").textContent = ID.expanded ? "收起" : "展开全部帧";
  t.frames.forEach((f, i) => {
    const b = button("", () => seek(i), "id-tick");
    b.style.background = color(f.tag); b.title = `${f.t.toFixed(2)} s · ${f.tag || "没挂名字"}`;
    b.setAttribute("aria-label", b.title); b.dataset.selected = String(i === ID.i); tags.append(b);
  });
  let indices = ID.expanded ? t.frames.map((_, i) => i) : sampleIdx(t.frames.length);
  indices = [...new Set([...indices, ID.i])].sort((a, b) => a - b);
  for (const i of indices) {
    const f = t.frames[i], b = button("", () => seek(i), "id-crop");
    b.dataset.selected = String(i === ID.i);
    const img = el("img"); img.src = uri("crop", {track: t.id, frame: f.name}); img.loading = "lazy";
    img.alt = `${f.t.toFixed(2)} 秒的人物裁图`;
    b.append(img, el("span", "", `${f.t.toFixed(2)} s`), el("span", "note", f.tag || "没挂名字")); strip.append(b);
  }
  t.segments.forEach((s, i) => {
    const count = t.frames.filter(f => segAt(t.segments, f.t) === i).length;
    const b = button(`${s[0]} s · ${s[1]}`, () => { const idx = t.frames.findIndex(f => f.t >= s[0]); seek(Math.max(0, idx)); }, "id-segment");
    b.style.background = color(s[1]); b.style.flexGrow = Math.max(count, 1); bar.append(b);
  });
  renderFrame(t, t.frames[ID.i]);
}
function renderKeys() {
  const keys = $("id-keys"); keys.replaceChildren();
  ID.data.friends.forEach((who, i) => keys.append(button(`${i < 9 ? i + 1 + " · " : ""}${who}`, () => mark(who))));
  for (const [key, who] of [["D", "团子"], ["S", "陌生人"], ["X", "不要"], ["0", "?"]]) keys.append(button(`${key} · ${who}`, () => mark(who)));
  for (const [text, key] of [["回车 · 采纳初判", "Enter"], ["/ · 在此切开", "/"], ["⌫ · 删最近切点", "Backspace"], ["Z · 撤销", "z"], ["J · 下一条", "j"], ["K · 上一条", "k"]])
    keys.append(button(text, () => onKey({key, preventDefault() {}})));
  $("id-note").textContent = ID.data.friends_note || "1~9 选好友；← → 逐帧；J / K 换轨迹。";
}
async function readState(run, epoch) {
  const response = await fetch(uri("state", {run}), {cache: "no-store"}), data = await response.json();
  if (epoch !== ID.epoch || !ID.active) return;
  if (!response.ok || !data.ok) throw new Error(data.text || "读不到草稿");
  ID.data = data; ID.run = run;
  if (!data.tracks.some(t => t.id === ID.cur)) ID.cur = data.tracks[0]?.id ?? null;
  $("id-cols").hidden = false;
  notice(data.source_missing ? `录像目录不在了：${data.source}` : "");
  renderList(); renderKeys(); renderTrack();
}
async function load() {
  if (ID.busy) return;
  ID.active = true; const epoch = ++ID.epoch;
  ID.stack = []; ID.data = null;
  $("id-cols").hidden = true; notice("正在读身份草稿…");
  try {
    const response = await fetch("api/identity/runs", {cache: "no-store"}), data = await response.json();
    if (epoch !== ID.epoch || !ID.active) return;
    if (!response.ok) throw new Error(data.text || "读不到录像列表");
    const runs = data.runs || [], select = $("id-run"); select.replaceChildren();
    for (const run of runs) { const option = el("option", "", run); option.value = run; select.append(option); }
    if (!runs.length) { notice("还没有身份草稿。先在命令行跑 perception identity-truth <录像目录>，再刷新这里。"); return; }
    ID.run = runs.includes(ID.run) ? ID.run : runs[0]; select.value = ID.run;
    select.onchange = async () => {
      if (ID.busy) { select.value = ID.run; return; }
      ID.run = select.value; ID.cur = null; ID.stack = []; ID.imageEpoch++; ID.data = null;
      select.blur(); $("id-cols").hidden = true;
      try { await readState(ID.run, ++ID.epoch); } catch (err) { notice(err.message); }
    };
    $("id-filter").onchange = e => { ID.filter = e.target.value; e.target.blur(); renderList(); };
    $("id-expand").onclick = () => { ID.expanded = !ID.expanded; renderTrack(); };
    await readState(ID.run, epoch);
  } catch (err) { if (epoch === ID.epoch && ID.active) notice(err.message); }
}
async function save(t, segments, remember = true, advance = false) {
  if (ID.busy || !ID.data) return false;
  ID.busy = true; $("id-run").disabled = true;
  const before = clone(t.segments), state = ID.data;
  try {
    const {status, data} = await post("api/identity/label", {run: ID.run, track: t.id, segments, mtime: state.mtime});
    if (status !== 200 || !data.ok) throw new Error(data.text || "保存失败");
    state.mtime = data.mtime; t.segments = clone(segments); t.reviewed = true;
    if (remember) ID.stack.push({track: t.id, segments: before});
    if (advance && !segments.some(s => s[1] === "?")) {
      const tracks = state.tracks, at = tracks.indexOf(t);
      const next = [...tracks.slice(at + 1), ...tracks.slice(0, at)].find(x => !reviewed(x) && matches(x));
      if (next) { ID.cur = next.id; ID.i = 0; ID.expanded = false; }
    }
    if (ID.active) { notice(""); renderList(); renderTrack(); }
    return true;
  } catch (err) { toast(err.message, "bad"); if (ID.active) notice(err.message); return false; }
  finally { ID.busy = false; $("id-run").disabled = false; }
}
function mark(who) {
  const t = current(); if (!t || !t.frames.length) return;
  return save(t, relabel(t.segments, t.frames[ID.i].t, who), true, true);
}
async function undo() {
  if (ID.busy || !ID.data) return;
  const item = ID.stack[ID.stack.length - 1];
  if (item) {
    const t = ID.data.tracks.find(t => t.id === item.track);
    if (await save(t, item.segments, false)) { ID.stack.pop(); ID.cur = t.id; ID.i = 0; if (ID.active) { renderList(); renderTrack(); } }
    return;
  }
  ID.busy = true;
  try {
    const {status, data} = await post("api/identity/undo", {run: ID.run, mtime: ID.data.mtime});
    if (status !== 200 || !data.ok) throw new Error(data.text || "撤销失败");
    if (ID.active) await readState(ID.run, ++ID.epoch);
  } catch (err) { toast(err.message, "bad"); if (ID.active) notice(err.message); }
  finally { ID.busy = false; }
}
function onKey(e) {
  if (!ID.active || !ID.data || ID.busy || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  if (e.target?.closest?.("input,textarea,select,[contenteditable=true]")) return;
  if (e.repeat && !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
  const t = current(), k = e.key; let act = null;
  if (k.toLowerCase() === "z") act = undo;
  else if (k.toLowerCase() === "j") act = () => stepTrack(1);
  else if (k.toLowerCase() === "k") act = () => stepTrack(-1);
  else if (t && t.frames.length) {
    const who = keyLabel(k, ID.data.friends), at = t.frames[ID.i].t;
    if (who !== null) act = () => mark(who);
    else if (k === "Enter") act = () => save(t, [[0, t.guess]], true, true);
    else if (k === "/") act = () => save(t, splitAt(t.segments, at));
    else if (k === "Backspace") act = () => save(t, removeCut(t.segments, at));
    else if (k === "ArrowLeft") act = () => seek(ID.i - 1);
    else if (k === "ArrowRight") act = () => seek(ID.i + 1);
  }
  if (act) { e.preventDefault(); act(); }
}
window.IdentityTab = {load, hasData: () => !!ID.data, key: onKey, hide() {
  ID.active = false; ID.epoch++; ID.imageEpoch++; revoke();
  $("id-cols").hidden = true; $("id-empty").hidden = true;
}};
})();
