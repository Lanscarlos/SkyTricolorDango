/* 剧本和报告页的极简 markdown 渲染（spec 2026-10-01-console-redesign §4.4）。
 * 认：# ~ ### 标题、- 列表、1. 有序列表、> 引用、``` 代码块、行内 `code` 和 **粗**、段落（单换行 = <br>）。
 * 报告里有聊天原话（可能是 <script> 之类），所以每一行先整体转义，再加自己的标签。顶层不碰 document，node 里能 require。 */
(function () {
"use strict";
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
function inline(s) {  // s 已转义；代码里不再做加粗
  return s.split(/(`[^`\n]+`)/).map((p, i) => i % 2 ? "<code>" + p.slice(1, -1) + "</code>"
    : p.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")).join("");
}
function renderMarkdown(text) {
  const lines = String(text == null ? "" : text).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let i = 0;
  const special = l => /^(#{1,3}\s|[-*]\s|\d+\.\s|>\s?|```)/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      const code = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++;  // 收尾的 ```（没有就到头了）
      out.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>");
      continue;
    }
    let m = /^(#{1,3})\s+(.*)$/.exec(line);
    if (m) { out.push(`<h${m[1].length}>${inline(esc(m[2]))}</h${m[1].length}>`); i++; continue; }
    m = /^([-*]|\d+\.)\s+/.exec(line);
    if (m) {
      const ordered = /\d/.test(m[1]), re = ordered ? /^\d+\.\s+(.*)$/ : /^[-*]\s+(.*)$/, items = [];
      while (i < lines.length && re.test(lines[i])) items.push("<li>" + inline(esc(re.exec(lines[i++])[1])) + "</li>");
      out.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    if (/^>\s?/.test(line)) {
      const q = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) q.push(inline(esc(lines[i++].replace(/^>\s?/, ""))));
      out.push("<blockquote>" + q.join("<br>") + "</blockquote>");
      continue;
    }
    const p = [];
    while (i < lines.length && lines[i].trim() && (!p.length || !special(lines[i]))) p.push(inline(esc(lines[i++])));
    out.push("<p>" + p.join("<br>") + "</p>");
  }
  return out.join("\n");
}
globalThis.renderMarkdown = renderMarkdown;
if (typeof module !== "undefined") module.exports = {renderMarkdown};
})();
