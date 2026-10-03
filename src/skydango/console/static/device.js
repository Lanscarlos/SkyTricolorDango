/* 设备页（spec §4.6）：只读检测模拟器连接 + 手动切输入法；团子 / 沙盒在跑时置灰（设备归它们独占）。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const MARKS = {ok: "✓", fail: "✗", warn: "!", skip: "–"};
let checking = false, imeBusy = false;

function sync(st) {
  const busy = !!st && BUSY.includes(st.run.state);
  $("check").disabled = busy || checking;
  $("device-note").textContent = busy ? "团子（或沙盒）在跑，设备归它独占，先停下再检测。" : "检测只读状态，不会往游戏里发任何输入。";
  $("ime-read").disabled = busy || imeBusy;
  for (const b of document.querySelectorAll("#imes button")) b.disabled = busy || imeBusy;
}

/* 输入法：{} 读，{set: id} 切；切完列表照新的当前输入法重画 */
async function ime(body) {
  imeBusy = true; sync(S.state); $("ime-msg").textContent = body.set ? "切换中…" : "读取中…";
  let r;
  try { r = await post("api/device/ime", body); } finally { imeBusy = false; }
  const ul = $("imes"), d = r.data;
  if (!d.ok) { $("ime-msg").textContent = d.text || "出错了"; sync(S.state); return; }
  ul.textContent = ""; ul.hidden = false;
  for (const m of d.imes) {
    const li = el("li"), name = el("div");
    name.append(el("b", "", m.label));
    if (m.label !== m.id) name.append(el("span", "id", m.id));
    li.append(name);
    if (m.id === d.current) li.append(el("span", "now", "当前"));
    else { const b = el("button", "btn sm", "切到这个"); b.type = "button"; b.onclick = () => ime({set: m.id}); li.append(b); }
    ul.append(li);
  }
  const cur = d.imes.find(m => m.id === d.current);
  $("ime-msg").textContent = body.set ? `已切到 ${cur ? cur.label : d.current}` : (d.imes.some(m => m.adb) ? "" : "模拟器里没装 ADBKeyboard，团子打不了中文");
  $("ime-read").textContent = "重新读取";
  sync(S.state);
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
  init() { $("check").onclick = check; $("ime-read").onclick = () => ime({}); onState(sync); },
};
})();
