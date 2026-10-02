/* 标注页第三个标签「动作名」（设计见 docs/progress/2026-10-02-tonight.md「动作命名页」）：
 * 左 网格列出 emotes/scan/ 的全部图标（编号 + 名字，「全部 / 只看没名字」、进度），右 大图 + 名字输入框。
 * 回车保存并跳到下一个没名字的；已命名的可以改名 / 清除名字（清除 = 挪进 emotes/_removed/），不做撤销。接口 api/emotes/*。
 * 只起名：团子能做哪些动作照旧由轮盘 / 白名单决定；改名时旧名字还被配置引用，只提示、不自动改。
 * labeling.js 切到这个标签时调 EmoteNamesTab.load()；这个标签开着时 labeling.js 的数字键 / 回车不生效。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const EN = {data: null, filter: "all", cur: null, busy: false, loading: false, bound: false};

function items() { return (EN.data && EN.data.items) || []; }
function byId(id) { return items().find(x => x.id === id) || null; }
function shown() { return EN.filter === "todo" ? items().filter(x => !x.name || x.id === EN.cur) : items(); }
function iconUrl(id) { return `api/emotes/icon?id=${encodeURIComponent(id)}`; }
function nextUnnamed(after) {  // 排在 after 后面的第一个没名字的；后面没了从头找
  const all = items(), at = all.findIndex(x => x.id === after);
  for (let j = at + 1; j < all.length; j++) if (!all[j].name) return all[j].id;
  for (let j = 0; j < Math.max(at, 0); j++) if (!all[j].name) return all[j].id;
  return null;
}

function renderHead() {
  const n = EN.data ? EN.data.named : 0, total = EN.data ? EN.data.total : 0;
  $("en-count").textContent = `已起名 ${n} / ${total}`;
  $("en-bar").style.width = total ? `${Math.round(n * 100 / total)}%` : "0";
  for (const b of $("en-filter").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.f === EN.filter));
  const t = document.querySelector('#lb-tabs [data-tab="names"]');
  if (t) t.textContent = EN.data && total - n ? `动作名（${total - n}）` : "动作名";
  const orph = $("en-orphans"), o = (EN.data && EN.data.orphans) || [];
  orph.hidden = !o.length;
  orph.textContent = o.length ? `图标库里有 ${o.length} 张图没对上任何扫描图标（可能是重扫前的旧截图）：${o.join("、")}` : "";
}
function renderGrid() {
  const box = $("en-grid"), list = shown();
  box.textContent = "";
  if (!list.length) { box.append(el("p", "none lb-empty", items().length ? "都起好名了。" : "还没有扫描图标。")); return; }
  const frag = document.createDocumentFragment();
  for (const x of list) {
    const b = el("button", "en-cell" + (x.name ? " named" : "")); b.type = "button"; b.dataset.k = x.id;
    b.setAttribute("aria-current", String(x.id === EN.cur));
    b.title = x.name ? `${x.id}：${x.name}` : `${x.id}（还没名字）`;
    const im = el("img"); im.loading = "lazy"; im.decoding = "async"; im.alt = ""; im.src = iconUrl(x.id);
    b.append(im, el("span", "id", x.id), el("span", "nm", x.name || "—"));
    b.onclick = () => select(x.id, true);
    frag.append(b);
  }
  box.append(frag);
}
function markCurrent() {
  let hit = null;
  for (const b of $("en-grid").querySelectorAll(".en-cell")) {
    const on = b.dataset.k === EN.cur;
    b.setAttribute("aria-current", String(on));
    if (on) hit = b;
  }
  if (hit) hit.scrollIntoView({block: "nearest"});
}
function renderSide() {
  const x = byId(EN.cur), im = $("en-big"), note = $("en-big-note"), input = $("en-name");
  $("en-title").textContent = x ? `${x.id}${x.name ? "：" + x.name : "（还没名字）"}` : "—";
  note.hidden = true;
  if (!x) {
    im.hidden = true; im.removeAttribute("src");
    note.textContent = EN.data ? "从左边选一个图标" : ""; note.hidden = !note.textContent;
  } else {
    im.hidden = false;
    im.onerror = () => { im.hidden = true; note.textContent = "这张图读不到"; note.hidden = false; };
    im.src = iconUrl(x.id);
  }
  input.disabled = !x || EN.busy;
  $("en-save").disabled = !x || EN.busy;
  $("en-clear").disabled = !x || !x.name || EN.busy;
  $("en-save-t").textContent = x && x.name ? "改名" : "保存";
}
function select(id, focus) {
  EN.cur = id || null;
  const x = byId(EN.cur);
  $("en-name").value = x && x.name ? x.name : "";
  $("en-refs").hidden = true;
  markCurrent(); renderSide();
  if (focus && x) { const i = $("en-name"); i.focus(); i.select(); }
}
function rerender() { renderHead(); renderGrid(); markCurrent(); }

function showRefs(old, refs, verb) {
  const box = $("en-refs");
  box.textContent = "";
  if (!refs || !refs.length) { box.hidden = true; return; }
  box.append(el("span", "", `${verb}了，但配置里还在用旧名字「${old}」：`));
  refs.forEach((r, k) => { if (k) box.append("、"); box.append(el("code", "", r)); });
  box.append(el("span", "", "。要自己去改（设置页或 config.toml），不然那些地方找不到这个动作。"));
  box.hidden = false;
}
async function call(path, body) {
  EN.busy = true; renderSide();
  let r;
  try { r = await post(path, body); }
  catch (e) { r = {status: 0, data: {ok: false, text: "面板没回应"}}; }
  EN.busy = false;
  if (!r.data.ok) {
    toast(r.data.text || "没改成", r.status === 400 || r.status === 409 ? "warn" : "bad");
    if (r.status === 404) load(); else renderSide();
    return null;
  }
  return r.data;
}
async function save() {
  const x = byId(EN.cur);
  if (!x || EN.busy) return;
  const name = $("en-name").value.trim();
  if (!name) { toast("名字是空的（要去掉名字点「清除名字」）", "warn"); return; }
  if (name === x.name) { const n = nextUnnamed(x.id); if (n) select(n, true); return; }  // 没改：当作确认，跳下一个
  const d = await call("api/emotes/name", {id: x.id, name});
  if (!d) return;
  const was = x.name;
  x.name = d.name;
  if (!was) EN.data.named += 1;
  if (EN.data.orphans && d.old) EN.data.orphans = EN.data.orphans.filter(o => o !== d.old);
  rerender();
  if (was) {  // 改名：留在这一格，好看清引用提示
    select(x.id, false);
    toast(`「${was}」改成了「${d.name}」`, "ok");
    showRefs(d.old, d.refs, "改名");
    return;
  }
  const n = nextUnnamed(x.id);
  if (n) select(n, true);
  else { select(x.id, false); toast("全部都起好名了", "ok"); }
}
async function clearName() {
  const x = byId(EN.cur);
  if (!x || !x.name || EN.busy) return;
  const d = await call("api/emotes/clear", {id: x.id});
  if (!d) return;
  x.name = null; EN.data.named -= 1;
  rerender(); select(x.id, true);
  toast(`去掉了名字「${d.old}」（图挪进了 _removed/）`, "ok");
  showRefs(d.old, d.refs, "去掉名字");
}
async function load() {
  if (EN.loading) return;
  EN.loading = true;
  let d;
  try { d = await getJSON("api/emotes/state"); }
  catch (e) { d = {ok: false, text: "读不到动作图标（面板停了？）"}; }
  finally { EN.loading = false; }
  const banner = $("en-empty"), open = window.LabelingTab === "names";
  if (!d || !d.ok) {
    EN.data = null; EN.cur = null;
    banner.textContent = (d && d.text) || "读不到动作图标";
    banner.hidden = !open; $("en-cols").hidden = true;
    renderHead();
    return;
  }
  EN.data = d; banner.hidden = true; $("en-cols").hidden = !open;
  $("en-lib").textContent = d.library || "emotes";
  rerender();
  const keep = byId(EN.cur);
  select(keep ? EN.cur : (nextUnnamed(null) || (items()[0] ? items()[0].id : null)), false);
}
function setFilter(f) {
  EN.filter = f === "todo" ? "todo" : "all";
  rerender();
}
function bind() {
  if (EN.bound) return;
  EN.bound = true;
  $("en-form").addEventListener("submit", e => { e.preventDefault(); save(); });
  $("en-clear").onclick = clearName;
  for (const b of $("en-filter").querySelectorAll("button")) b.onclick = () => setFilter(b.dataset.f);
}

window.EmoteNamesTab = {
  load() { bind(); return load(); },
  hasData() { return !!EN.data; },
};
})();
