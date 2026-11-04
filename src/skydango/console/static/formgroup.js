/* 外形页「按轨迹」的纯逻辑（计划 2026-11-05-form-track-labeling）：把后端给的分组（api/form/state 的 groups）和当前筛选合成列表里的一行一组，
 * 算整组标注要发哪几张、「重点核对」筛选怎么匹配、开关记在 localStorage 的哪个键。不碰 document，node 里能 require。 */
(function () {
"use strict";
const BY_TRACK_KEY = "skydango.form.byTrack";

/* items：state 的条目（有 crop）；groups：名字列表的列表；match(item) → 是否在当前筛选里。
 * 返回 [{key, members, shown}]：members = 组里现在还存在的条目（组里的顺序），shown = 其中 match 的；shown 为空的组不返回；
 * 没进任何组的条目各自一组（排在最后）；key = 组里第一张的名字。 */
function buildGroups(items, groups, match) {
  const by = new Map();
  for (const c of items) by.set(c.crop, c);
  const used = new Set(), out = [];
  const add = members => {
    const shown = members.filter(match);
    if (shown.length) out.push({key: members[0].crop, members, shown});
  };
  for (const names of groups || []) {
    const members = [];
    for (const n of names) {
      const c = by.get(n);
      if (c && !used.has(n)) { used.add(n); members.push(c); }
    }
    if (members.length) add(members);
  }
  for (const c of items) if (!used.has(c.crop)) add([c]);
  return out;
}

/* 整组标注要发的名字：shown 里没被「拆出去」的。 */
function targets(group, excluded) {
  return group.shown.map(c => c.crop).filter(n => !excluded.has(n));
}

function centerIndex(n) { return Math.floor((n - 1) / 2); }

/* 「重点核对」：在名单里，且（还没判，或在某个外形类别里但没人确认过）。 */
function focusMatch(item, focus, forms) {
  if (!focus.has(item.crop)) return false;
  return item.where === "_unlabeled" || (forms.includes(item.where) && item.confirmed === false);
}

function readByTrack(storage) {
  try {
    const v = storage && storage.getItem(BY_TRACK_KEY);
    return v !== "0";
  } catch (e) { return true; }
}
function writeByTrack(storage, on) {
  try { if (storage) storage.setItem(BY_TRACK_KEY, on ? "1" : "0"); } catch (e) { /* 读不到 / 写不了就当没存 */ }
}

const api = {buildGroups, targets, centerIndex, focusMatch, readByTrack, writeByTrack};
if (typeof window !== "undefined") window.FormGroup = api;
if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
