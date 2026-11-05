/* 外形页「网格」的纯逻辑（计划 2026-11-05-form-grid-review）：一页 36 张、默认是人、只点错的、整页确认。
 * 排序（按「是人的概率」从低到高）、分页、点击标记的切换、整页提交的三份名单、模式开关记在 localStorage 的哪个键。不碰 document，node 里能 require。 */
(function () {
"use strict";
const PAGE = 36;
const MODE_KEY = "skydango.form.mode";
const PERSON_CLASSES = ["lit", "unlit", "spirit", "shared", "morph"];

/* 「是人的概率」= lit + unlit + spirit + shared + morph（缺的按 0）；没有分数 = null。 */
function personProb(scores) {
  if (!scores || typeof scores !== "object") return null;
  let t = 0, any = false;
  for (const k of PERSON_CLASSES) {
    const v = Number(scores[k]);
    if (Number.isFinite(v)) { t += v; any = true; }
  }
  return any ? t : null;
}

/* 网格里要显示的：match 的、没核过是人的，按是人的概率从低到高（最可疑在前），没分数的排最后，同分按裁图名。 */
function gridItems(items, match, person, scores) {
  const prob = c => personProb(scores && scores[c.crop]);
  return items.filter(c => match(c) && !person.has(c.crop)).sort((a, b) => {
    const pa = prob(a), pb = prob(b);
    if (pa === null && pb !== null) return 1;
    if (pa !== null && pb === null) return -1;
    if (pa !== null && pa !== pb) return pa - pb;
    return a.crop < b.crop ? -1 : a.crop > b.crop ? 1 : 0;
  });
}

function page(list, i) { return list.slice(i * PAGE, (i + 1) * PAGE); }
function pageCount(n) { return Math.ceil(n / PAGE); }

/* 点一下：同一种再点 = 去掉，另一种 = 改成这一种。kind = "not_person"（不是人）/ "discard"（不要）。返回新的 Map，不改传进来的。 */
function toggle(marks, name, kind) {
  const m = new Map(marks);
  if (m.get(name) === kind) m.delete(name); else m.set(name, kind);
  return m;
}

/* 整页提交：没标的进 person。 */
function submission(pageItems, marks) {
  const out = {person: [], not_person: [], discard: []};
  for (const c of pageItems) {
    const k = marks.get(c.crop);
    (k === "not_person" ? out.not_person : k === "discard" ? out.discard : out.person).push(c.crop);
  }
  return out;
}

function readMode(storage) {
  try { return storage && storage.getItem(MODE_KEY) === "grid" ? "grid" : "single"; } catch (e) { return "single"; }
}
function writeMode(storage, mode) {
  try { if (storage) storage.setItem(MODE_KEY, mode === "grid" ? "grid" : "single"); } catch (e) { /* 读不到 / 写不了就当没存 */ }
}

const api = {PAGE, personProb, gridItems, page, pageCount, toggle, submission, readMode, writeMode};
if (typeof window !== "undefined") window.FormGrid = api;
if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
