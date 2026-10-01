/* 聊天记录的一行（沙盒页和真机页共用，spec 2026-10-01-console-live-page §2.4）：别人说的在右、团子说的在左（樱花底）、
 * 动作是旁白、事件是分隔线、反思是虚线框、被拦的话删除线 + 原因。行的格式见 brain/transcript.py：{seq, t, kind, who, text, why?}。
 * 样式类名沿用沙盒的 sb-*（console.css）。只用 textContent 放数据。顶层不碰 document：node 里 require 能测 lineKind。 */
(function () {
"use strict";
function lineKind(r) {  // 聊天行用哪种样式
  if (r.kind === "event") return /^──\s*反思/.test(r.text || "") ? "refl" : "ev";
  if (r.kind === "act") return "act";
  if (r.kind === "said") return "msg me";
  if (r.kind === "blocked") return r.who === "团子" ? "msg me blocked" : "msg blocked";
  return "msg";
}
function chatLine(r) {
  const k = lineKind(r), time = el("time", "", hhmm(r.t));
  if (k === "ev") { const d = el("div", "sb-ev"); d.append(time, " ", (r.text || "").replace(/^──\s*|\s*──$/g, "")); return d; }
  if (k === "refl") { const d = el("div", "sb-refl"); d.append(time, " ", r.text || ""); return d; }
  if (k === "act") { const d = el("div", "sb-act", r.text || ""); d.title = dayTime(r.t); return d; }
  const d = el("div", "sb-" + k.split(" ").join(" sb-")), who = el("div", "who");
  who.append(r.kind === "heard" ? r.who || "（不知道是谁）" : r.kind === "said" ? "团子" : r.who === "团子" ? "团子（没说出去）" : "（被拦下）", " ", time);
  d.append(who, el("div", "b", r.text || ""));
  if (r.kind === "blocked" && r.why) d.append(el("div", "why", r.why));
  return d;
}
function appendChat(box, rows) {  // 原本在底部才跟到底；占位的 .none 第一次有内容时去掉；最多留 500 行
  if (!rows.length) return;
  const follow = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
  for (const p of box.querySelectorAll(":scope > .none")) p.remove();
  for (const r of rows) box.append(chatLine(r));
  while (box.childNodes.length > 500) box.firstChild.remove();
  if (follow) box.scrollTop = box.scrollHeight;
}
globalThis.Chat = {lineKind, line: chatLine, append: appendChat};
if (typeof module !== "undefined" && module.exports) module.exports = {lineKind};
})();
