/* 整帧页顶上的重训区（spec 2026-10-04-hardcase-inbox §7.1 / §7.3）：上次训练以来通过的帧数、「重训」按钮、
 * 「最近一次报告」（renderMarkdown 渲染）、「换上新 YOLO / 外形头」和各自的「回退」。
 * 接口 api/retrain/latest | adopt | rollback；任务槽 api/jobs/start {job:"retrain"}；计数和任务状态来自 S.state（inbox / job）。
 * 报告里可能有聊天原话：renderMarkdown 先转义；路径、名字等都用 textContent。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const R = {latest: null, open: false, busy: false, bound: false, jobKey: ""};
const WHAT = [["yolo", "YOLO"], ["attrs", "外形头"]];

async function load() {
  try { R.latest = await getJSON("api/retrain/latest"); } catch (e) { R.latest = null; }
  render();
}
function state() { return S.state || {}; }
function running() { const j = state().job || {}; return j.state === "running"; }

function render() {
  const box = $("fr-train");
  if (!box) return;
  const ib = state().inbox || {}, job = state().job || {}, n = ib.passed_since_train || 0, min = ib.retrain_min || 0;
  const count = $("fr-train-count"), btn = $("fr-train-go");
  count.textContent = `上次训练以来通过：${n} 帧` + (min ? `（攒到 ${min} 帧提示重训）` : "");
  btn.classList.toggle("go", min > 0 && n >= min);
  btn.disabled = R.busy || running();
  btn.textContent = running() && job.job === "retrain" ? "训练中…" : "重训";
  $("fr-train-prog").textContent = job.job === "retrain" && job.state === "running" ? (job.progress || "") : job.job === "retrain" && job.state === "failed" ? "上次重训失败了，看报告或日志" : "";
  const L = R.latest, rep = $("fr-train-report");
  $("fr-train-toggle").textContent = (R.open ? "收起" : "展开") + "最近一次报告" + (L && L.dir ? `（${L.dir}）` : "");
  $("fr-train-toggle").disabled = !L || !L.dir;
  rep.hidden = !R.open || !L || !L.dir;
  if (!rep.hidden) {
    $("fr-train-md").innerHTML = renderMarkdown(L.report || "（这次没有报告）");  // 转义过再加自己的标签
    renderAdopt(L);
  }
  if (L && !L.dir) $("fr-train-toggle").textContent = "还没训练过";
}
function renderAdopt(L) {
  const box = $("fr-train-adopt");
  box.textContent = "";
  const done = L.result && L.result.ok, adopted = L.adopted || {};
  if (!done) { box.append(el("p", "note", "这次重训没有做完，不能换上新模型。")); return; }
  for (const [what, name] of WHAT) {
    const row = el("div", "fr-train-row");
    const rec = adopted[what];
    row.append(el("span", "path", `新${name}：${L.result[what] || "—"}`));
    const go = el("button", "btn sm", `换上新${name}`);
    go.disabled = R.busy || !!rec;
    go.onclick = () => act("adopt", what, name);
    row.append(go);
    if (rec) {
      const back = el("button", "btn sm halt", "回退");
      back.disabled = R.busy;
      back.onclick = () => act("rollback", what, name);
      row.append(back, el("span", "note", `已换上（${rec.t || ""}），之前是 ${rec.old || "—"}`));
    }
    box.append(row);
  }
}
async function act(kind, what, name) {
  if (R.busy) return;
  R.busy = true; render();
  try {
    const r = await post(`api/retrain/${kind}`, {what});
    if (r.data && r.data.ok) toast((kind === "adopt" ? `已换上新${name}` : `已回退${name}`) + (r.data.note ? `（${r.data.note}）` : ""), "ok");
    else toast((r.data && r.data.text) || "没成功", "bad");
  } catch (e) { toast("请求失败了", "bad"); }
  R.busy = false;
  await load();
}
async function start() {
  const ib = state().inbox || {}, n = ib.passed_since_train || 0, min = ib.retrain_min || 0;
  if (min && n < min && !await ask(`上次训练以来只通过了 ${n} 帧（提示线是 ${min}），数量偏少，训出来的提升可能不明显。还是要重训吗？`, {ok: "还是重训", cancel: "再攒攒"})) return;
  if (!await ask("重训要占显卡很久（YOLO 训练 + 外形头 + 回放对比），这期间不能叫醒团子，也不能起沙盒。现在开始吗？", {ok: "开始重训", cancel: "再等等"})) return;
  R.busy = true; render();
  const r = await post("api/jobs/start", {job: "retrain"});
  if (!r.data || !r.data.ok) toast((r.data && (r.data.text || r.data.error)) || "没起来", "bad");
  R.busy = false;
  if (typeof refresh === "function") refresh();
  render();
}
function bind() {
  if (R.bound || !$("fr-train")) return;
  R.bound = true;
  $("fr-train-go").onclick = start;
  $("fr-train-toggle").onclick = () => { R.open = !R.open; render(); };
  onState(() => {
    const j = state().job || {}, key = `${j.job}|${j.state}`;
    if (key !== R.jobKey) { R.jobKey = key; if (j.job === "retrain" && j.state !== "running") load(); }  // 重训刚结束：读新报告
    if (!$("fr-train").hidden) render();
  });
}
window.RetrainBox = {load() { bind(); return load(); }};
})();
