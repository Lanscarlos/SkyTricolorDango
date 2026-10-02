/* 真机团子页左栏的手动控制（spec 2026-10-01-console-live-page §2.2）：照 viewer 的手动控制栏（vision/viewer.py PAGE 的 control 段）重排，
 * 接口 live/control/options（404 = 身体还没建好，3 秒后再试）、POST live/control；在画面上选人走 live.js 的 LiveView；
 * 「看人」点了之后用 ask() 确认（内嵌浏览器里原生弹窗用不了）。总是真执行，照样过身体的护栏。
 * 顶层不碰 document：node 里 require 它能测 controlLine（tests/test_console_page.py）。 */
(function () {
"use strict";
const CAM = {left: "左转", right: "右转", up: "抬头", down: "低头", zoom_in: "拉近", zoom_out: "拉远"};
function controlLine(action, args, res) {
  const a = args || {}; let what;
  if (action === "say") what = `说「${a.text}」`;
  else if (action === "emote") what = `动作「${a.name}」`;
  else if (action === "camera") what = `${CAM[a.action] || a.action} ×${a.steps}`;
  else if (action === "camera_reset") what = "复位";
  else if (action === "look_around") what = "环视一圈";
  else if (action === "panel_read") what = "读面板";
  else if (action === "panel_close") what = "关面板";
  else if (action === "check_friend") what = `看人 (${a.x}, ${a.y})`;
  else if (action === "track") what = `盯着${a.name}（${a.seconds} 秒）`;
  else if (action === "stop_task") what = "停下";
  else if (action === "call") what = "喊一声";
  else what = action;
  return `${what} → ${res.text}`;
}
if (typeof module !== "undefined" && module.exports) module.exports = {controlLine};
if (typeof document === "undefined") return;

const K = {opts: null, busy: false, picking: false, trackPick: false, on: false, timer: 0, retry: 0};

function lock() {
  const o = K.opts, b = K.busy || !o, opts = o || {emotes: [], camera: []};
  $("lc-none").hidden = !!o;
  $("lc-warn").hidden = !(o && o.dry_run);
  $("lc-say-text").disabled = b; $("lc-say-go").disabled = b || !$("lc-say-text").value.trim();
  $("lc-emote-name").disabled = $("lc-emote-go").disabled = b || !opts.emotes.length;
  for (const x of document.querySelectorAll("#lc-camera button,#lc-steps,#lc-reset,#lc-around")) x.disabled = b || !(opts.camera || []).length;
  const pick = $("lc-pick");
  pick.disabled = b || !opts.friend_check; pick.classList.toggle("on", K.picking);
  $("lc-pick-tip").textContent = !o ? "" : !opts.friend_check ? "没开（[friend_check] enabled = false）" : K.picking ? "点一下画面上的人" : "";
  $("lc-call").disabled = b || !opts.call;  // 按 Q 喊一声：要开 [call] 和感知层
  $("lc-call-tip").textContent = o && !opts.call ? "要开 [call] 和感知层（[perception]）" : "";
  const tr = !!opts.track;
  $("lc-track-name").disabled = $("lc-track-sec").disabled = $("lc-track-pick").disabled = b || !tr;
  $("lc-track-go").disabled = b || !tr || !$("lc-track-name").value.trim(); $("lc-stop").disabled = b;
  $("lc-track-sec").max = opts.max_track_seconds || 60; $("lc-track-pick").classList.toggle("on", K.trackPick);
  if (o && !tr) $("lc-track-tip").textContent = "要开感知层（[perception]）和镜头";
  else if (K.trackPick) $("lc-track-tip").textContent = "点一下画面上的好友";
  $("lc-panels").hidden = !opts.panels; $("lc-panel-read").disabled = $("lc-panel-close").disabled = b || !opts.panels;
}
function apply() {
  const o = K.opts; if (!o) { lock(); return; }
  const sel = $("lc-emote-name"), keep = sel.value;
  if ([...sel.options].map(x => x.value).join("|") !== o.emotes.join("|")) {  // 列表变了才重建：定时刷新时别把打开的下拉框关掉
    sel.textContent = "";
    for (const n of o.emotes) { const op = el("option", "", n); op.value = n; sel.append(op); }
    if (o.emotes.includes(keep)) sel.value = keep;
  }
  $("lc-steps").max = o.max_steps; count();
}
function count() {
  const n = [...$("lc-say-text").value.trim()].length, max = K.opts ? K.opts.max_chars : 0;
  $("lc-count").textContent = K.opts ? `${n} / ${max}` : ""; lock();
}
function logLine(action, args, res) {
  const li = el("li");
  li.append(el("span", "t", new Date().toTimeString().slice(0, 8) + " "), el("span", res.ok ? "ok" : "bad", controlLine(action, args, res)));
  const ul = $("lc-log"); ul.prepend(li); while (ul.children.length > 10) ul.lastChild.remove();
}
async function options(retry) {
  try {
    const r = await fetch("live/control/options", {cache: "no-store"});
    if (!K.on) return;
    if (r.status === 404) {  // 身体还没建好：3 秒后再试
      K.opts = null; $("lc-none").textContent = "身体还没准备好…"; lock();
      if (retry) K.retry = setTimeout(() => options(true), 3000);
      return;
    }
    if (!r.ok) throw new Error(r.status);
    K.opts = await r.json(); apply();
  } catch (e) { if (retry && K.on) K.retry = setTimeout(() => options(true), 3000); }
}
async function send(action, args) {
  if (K.busy) return null;
  K.busy = true; lock();
  $("lc-busy").textContent = `正在做：${controlLine(action, args, {text: "…"}).split(" → ")[0]}…`;
  const r = await post("live/control", {action, args}), res = r.data && r.data.ok !== undefined ? r.data : {ok: false, text: `HTTP ${r.status}`};
  logLine(action, args, res); $("lc-busy").textContent = ""; K.busy = false; lock();
  await options(false);
  return res;
}

function bind() {
  $("lc-say-text").oninput = count;
  $("lc-say-text").onkeydown = e => { if (e.key === "Enter" && !e.isComposing) $("lc-say-go").click(); };
  $("lc-say-go").onclick = async () => {
    const t = $("lc-say-text").value.trim(); if (!t) return;
    const res = await send("say", {text: t}); if (res && res.ok) { $("lc-say-text").value = ""; count(); }
  };
  $("lc-emote-go").onclick = () => send("emote", {name: $("lc-emote-name").value});
  for (const x of document.querySelectorAll("#lc-camera button[data-cam]")) x.onclick = () => {
    const max = K.opts ? K.opts.max_steps : 4, n = Math.min(max, Math.max(1, parseInt($("lc-steps").value, 10) || 1));
    $("lc-steps").value = n; send("camera", {action: x.dataset.cam, steps: n});
  };
  $("lc-reset").onclick = () => send("camera_reset", {});
  $("lc-around").onclick = () => send("look_around", {});
  $("lc-panel-read").onclick = () => send("panel_read", {});
  $("lc-panel-close").onclick = () => send("panel_close", {});
  $("lc-stop").onclick = () => send("stop_task", {});
  $("lc-call").onclick = () => send("call", {});  // 身体按完键后要等约 6 秒的呼喊窗口才回结果
  $("lc-track-name").oninput = lock;
  $("lc-track-go").onclick = () => {
    const name = $("lc-track-name").value.trim(); if (!name) return;
    const max = (K.opts && K.opts.max_track_seconds) || 60, n = Math.min(max, Math.max(1, parseInt($("lc-track-sec").value, 10) || 30));
    $("lc-track-sec").value = n; send("track", {name, seconds: n});
  };
  $("lc-pick").onclick = () => {  // 看人：点画面 → 画十字 → ask() 确认 → check_friend
    K.trackPick = false; K.picking = !K.picking; lock();
    if (!K.picking) { LiveView.pick(null); return; }
    LiveView.pick(async (x, y) => {
      LiveView.mark([x, y]);
      const go = await ask(`点 (${x}, ${y}) 这个人？`, {ok: "点"});
      K.picking = false; lock();
      if (go) await send("check_friend", {x, y});
      LiveView.mark(null);
    });
  };
  $("lc-track-pick").onclick = () => {  // 盯人：按快照里的框认出点的是谁，填进名字
    K.picking = false; K.trackPick = !K.trackPick; $("lc-track-tip").textContent = ""; lock();
    if (!K.trackPick) { LiveView.pick(null); return; }
    LiveView.pick((x, y, s) => {
      const name = Stage.nameAt(s.boxes || [], x, y);
      K.trackPick = false;
      $("lc-track-tip").textContent = name ? "" : "那里没认出好友的名字，换个地方点，或者直接输入";
      if (name) $("lc-track-name").value = name;
      lock();
    });
  };
}

globalThis.LiveCtl = {
  start(brain) {  // brain = false：普通 Agent 不挂手动控制（viewer 只在大脑模式挂 control）
    if (brain === false) { $("lc-none").textContent = "普通 Agent 没有手动控制（只有统管大脑有）"; return; }
    if (K.on) return;
    K.on = true; $("lc-none").textContent = "连接身体…"; options(true);
    K.timer = setInterval(() => { if (K.on && K.opts && !K.busy) options(false); }, 5000);  // 动作冷却后列表会变：定时刷新
  },
  stop() {
    K.on = false; clearInterval(K.timer); clearTimeout(K.retry); K.opts = null; K.picking = K.trackPick = false;
    $("lc-none").textContent = "叫醒之后能在这里手动让团子说话、做动作、转视角"; lock();
  },
};
function ready() { bind(); lock(); }
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready); else setTimeout(ready, 0);  // 同 common.js 末尾
})();
