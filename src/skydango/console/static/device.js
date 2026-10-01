/* 设备页（spec §4.6）：只读检测模拟器连接；团子 / 沙盒在跑时置灰（设备归它们独占）。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const MARKS = {ok: "✓", fail: "✗", warn: "!", skip: "–"};
let checking = false;

function sync(st) {
  const busy = !!st && BUSY.includes(st.run.state);
  $("check").disabled = busy || checking;
  $("device-note").textContent = busy ? "团子（或沙盒）在跑，设备归它独占，先停下再检测。" : "检测只读状态，不会往游戏里发任何输入。";
}

async function check() {
  const btn = $("check"), ul = $("checks"); checking = true; btn.disabled = true; btn.textContent = "检测中…";
  let r;
  try { r = await post("api/device"); } finally { checking = false; btn.textContent = "再检测一次"; sync(S.state); }
  ul.textContent = ""; ul.hidden = false;
  if (!r.data.ok) { const li = el("li"); li.append(el("span", "chk-mark fail", MARKS.fail), el("span", "detail", r.data.text)); ul.append(li); return; }
  for (const c of r.data.checks) {
    const li = el("li"), body = el("div");
    body.append(el("b", "", c.label));
    if (c.detail) body.append(el("div", "detail", c.detail));
    if (c.hint) body.append(el("div", "hint", c.hint));
    if (c.data && c.data.devices && c.data.devices.length) {
      const list = el("div", "devlist");
      for (const d of c.data.devices) {
        const b = el("button", "btn sm", `用 ${d}`); b.type = "button";
        b.onclick = async () => { await Pages.settings.ensureLoaded(); Pages.settings.markDirty("device.serial", d); go("settings", "device.serial"); };
        list.append(b);
      }
      body.append(list);
    }
    if (c.data && c.data.thumb) { const img = el("img"); img.alt = "模拟器截图"; img.src = "data:image/jpeg;base64," + c.data.thumb; body.append(img); }
    li.append(el("span", "chk-mark " + c.status, MARKS[c.status] || "?"), body); ul.append(li);
  }
}

Pages.device = {
  init() { $("check").onclick = check; onState(sync); },
};
})();
