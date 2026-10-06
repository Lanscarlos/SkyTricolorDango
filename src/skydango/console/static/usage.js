/* 「模型用量」卡片（spec 2026-10-06-model-usage §6.2）：真机页、沙盒页左栏各一块，模型页顶上一行。
 * 数据都从 api/usage 来：团子 / 沙盒在跑时面板转发子进程的 /usage，停着读账本的今天 + 面板自己查的余额。
 * usageModel(data) 是纯计算（node 测试用），renderUsage 画 DOM，mountUsage 每 5 秒拉一次、拉不到留着上一次。 */
(function () {
"use strict";
const g = typeof globalThis !== "undefined" ? globalThis : window;

function trim(text) { return text.replace(/\.?0+$/, ""); }
function fmtTokens(n) {
  n = Number(n) || 0;
  if (n < 1000) return String(n);
  if (n < 1e6) return trim((n / 1000).toFixed(1)) + "k";
  return trim((n / 1e6).toFixed(2)) + "M";
}
function fmtYuan(x) {
  if (x == null) return "—";
  if (x === 0) return "¥0";
  return "¥" + (x < 0.1 ? x.toFixed(3) : x.toFixed(2));  // 几分钱的时候多一位
}
function clock(sec) {
  const d = new Date(sec * 1000), now = new Date(), p = n => String(n).padStart(2, "0");
  const hm = `${p(d.getHours())}:${p(d.getMinutes())}`;
  return d.toDateString() === now.toDateString() ? hm : `${d.getMonth() + 1}月${d.getDate()}日 ${hm}`;
}
function ago(sec) {
  const m = Math.max(0, Math.round((Date.now() / 1000 - sec) / 60));
  return m < 1 ? "刚查的" : `${m} 分钟前查的`;
}

const WINDOWS = {five_hour: "5 小时", seven_day: "7 天", seven_day_overage_included: "7 天（含超额）"};
const STATUS = {allowed: "额度正常", allowed_warning: "额度快用完", rejected: "额度用完"};

function quotaOf(rate) {
  if (!rate) return null;
  const parts = [];
  let frac = null;
  const windows = Object.assign({}, rate.windows || {});
  if (rate.utilization != null && rate.type && !windows[rate.type]) windows[rate.type] = {utilization: rate.utilization, resets_at: rate.resets_at};
  for (const name of Object.keys(WINDOWS)) {
    const w = windows[name];
    if (!w || w.utilization == null) continue;
    if (frac == null) frac = w.utilization;
    let text = `${WINDOWS[name]} ${Math.round(w.utilization * 100)}%`;
    if (w.resets_at) text += `，${clock(w.resets_at)} 重置`;
    parts.push(text);
  }
  if (!parts.length) {
    let text = STATUS[rate.status] || rate.status || "";
    if (text && rate.resets_at) text += `，${clock(rate.resets_at)} 重置`;
    return text ? {text, frac: null} : null;
  }
  return {text: parts.join("；"), frac};
}

function moneyOf(balance) {
  if (!balance) return {money: "", moneyTitle: ""};
  const item = (balance.items || [])[0];
  if (!item) return {money: balance.error ? "查不到余额" : "", moneyTitle: balance.error || ""};
  const sign = item.currency === "USD" ? "$" : "¥";
  const yuan = x => sign + Number(x || 0).toFixed(2);
  let title = `赠送 ${yuan(item.granted)}，充值 ${yuan(item.topped_up)}；${ago(balance.at)}`;
  if (balance.error) title += `（这次没查到：${balance.error}）`;
  if (balance.available === false) title += "；余额不够用了";
  return {money: `余额 ${yuan(item.total)}`, moneyTitle: title};
}

function usageModel(data) {
  const kinds = {};
  const providers = (data.providers || []).map(p => {
    kinds[p.id] = p.kind;
    return {id: p.id, state: p.closed ? `关了：${p.closed}` : "开着", bad: !!p.closed, ...moneyOf(p.balance), quota: quotaOf(p.rate)};
  });
  const part = data.run || data.today || {rows: []};
  const rows = (part.rows || []).map(r => {
    const [pid, ...rest] = r.model.split("/");
    const cost = r.cost != null ? fmtYuan(r.cost) : (kinds[pid] === "claude-code" ? "订阅" : "没填单价");
    let tokens = `入 ${fmtTokens(r.input)}`;
    if (r.cache_read) tokens += `（命中 ${fmtTokens(r.cache_read)}）`;
    tokens += ` 出 ${fmtTokens(r.output)}`;
    return {title: r.label, model: rest.join("/") || r.model, provider: pid, backup: !!r.backup,
            calls: `${r.calls} 次` + (r.fails ? `（失败 ${r.fails}）` : ""), failed: !!r.fails, tokens, cost, est: !!r.est};
  });
  const notes = [];
  for (const u of data.uses || []) {
    if (u.disabled) notes.push(`${u.label}停用：${u.disabled}`);
    else if (u.backup && u.current) notes.push(`${u.label}现在走备用 ${u.current}`);
  }
  const scope = data.run ? (data.source === "sandbox" ? "这次沙盒" : "这次运行") : "今天";
  const est = !!((data.run && data.run.est) || (data.today && data.today.est));
  return {providers, rows, notes, scope,
          totals: {run: data.run ? fmtYuan(data.run.cost) : null, today: data.today ? fmtYuan(data.today.cost) : null, est}};
}

const EST_TIP = "按 DeepSeek 价格页的单价折算（工作日 9–12、14–18 点按高峰价，不认节假日；deepseek-chat 按 Flash 价估），以控制台扣费为准";

function renderUsage(box, data, opts) {
  const m = usageModel(data), E = g.el;
  box.textContent = "";
  const totals = E("p", "us-totals");
  if (m.totals.run != null) totals.append(E("span", "", `${m.scope} `), E("b", "", m.totals.run));
  if (m.totals.today != null) totals.append(E("span", "", `${m.totals.run != null ? "　" : ""}今天 `), E("b", "", m.totals.today));
  if (m.totals.est) { const e = E("span", "us-est", "估"); e.title = EST_TIP; totals.append(e); }
  box.append(totals);
  const provs = E("ul", "us-provs");
  for (const p of m.providers) {
    const li = E("li", p.bad ? "bad" : "");
    li.append(E("b", "", p.id), E("span", "us-gate", p.state));
    if (p.money) { const s = E("span", "us-money", p.money); s.title = p.moneyTitle; li.append(s); }
    if (p.quota) {
      const q = E("span", "us-quota");
      if (p.quota.frac != null) {
        const bar = E("i", "us-bar"), fill = E("i", p.quota.frac >= 0.9 ? "hot" : p.quota.frac >= 0.7 ? "warm" : "");
        fill.style.width = Math.min(100, Math.round(p.quota.frac * 100)) + "%"; bar.append(fill); q.append(bar);
      }
      q.append(E("span", "", p.quota.text)); li.append(q);
    }
    provs.append(li);
  }
  box.append(provs);
  if (opts && opts.compact) return;
  for (const n of m.notes) box.append(E("p", "us-note", n));
  if (!m.rows.length) { box.append(E("p", "none", m.scope === "今天" ? "今天还没调过模型" : "这次还没调过模型")); return; }
  const list = E("ul", "us-rows");
  for (const r of m.rows) {
    const li = E("li"), head = E("div", "us-head");
    head.append(E("b", "", r.title), E("span", "us-model", r.model));
    if (r.backup) head.append(E("span", "us-tag", "备"));
    head.append(E("span", "grow"), E("span", "us-cost", r.cost));
    const sub = E("div", "us-sub");
    sub.append(E("span", r.failed ? "us-fail" : "", r.calls), E("span", "", r.tokens));
    li.append(head, sub); list.append(li);
  }
  box.append(list);
}

function mountUsage(box, url, opts) {
  opts = opts || {};
  const getJSON = opts.getJSON || g.getJSON, render = opts.render || ((c, d) => renderUsage(c, d, opts));
  const visible = opts.visible || (() => true);
  let timer = null;
  async function tick() {
    if (typeof document !== "undefined" && document.hidden) return;
    if (!visible()) return;
    let data;
    try { data = await getJSON(url); } catch (e) { return; }  // 面板 / 子进程刚好不在：留着上一次
    if (!data || data.ok === false) return;
    try { render(box, data); } catch (e) { console.error("画模型用量出错", e); }
  }
  if (opts.every !== 0) timer = setInterval(tick, opts.every || 5000);
  return {tick, stop() { if (timer) clearInterval(timer); timer = null; }};
}

const api = {fmtTokens, fmtYuan, usageModel, renderUsage, mountUsage};
if (typeof module !== "undefined" && module.exports) module.exports = api;
g.Usage = api;
})();
