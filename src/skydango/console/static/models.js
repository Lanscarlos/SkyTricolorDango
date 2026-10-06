/* 「模型」页（spec 2026-10-05-model-providers §3）：上半供应商卡片（接入方式、地址 / 路径、Key、模型列表和「能看图」、测试、删除），
 * 下半每个用处选主 / 备模型；整页一个「保存」。未保存的修改放在这个文件自己的状态里（同设置页），切页再回来不丢。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const KIND_NAMES = {"claude-code": "Claude Code", openai: "OpenAI 兼容"};
const SOURCES = {default: "默认", config: "config.toml", console: "面板", legacy: "旧配置换算"};
const SECRET_SOURCES = {secrets: "secrets.toml", env: "环境变量", none: "没设置"};
const UI_ONLY = ["secret", "secret_source", "source", "used_by", "isNew"];
const ID_RE = /^[a-z0-9_-]+$/;
const ST = {view: null, providers: [], uses: {}, secrets: {}, dirty: false, results: {}, loading: null, flashTimer: 0};

const useRowId = name => "mu-" + name;

function reset(view) {
  ST.view = view;
  ST.providers = view.providers.map(p => ({...p, models: p.models.map(m => ({...m})),
    prices: Object.fromEntries(Object.entries(p.prices || {}).map(([k, v]) => [k, [...v]]))}));
  ST.uses = Object.fromEntries(view.uses.map(u => [u.name, {main: u.main, backup: u.backup}]));
  ST.secrets = {}; ST.dirty = false; ST.results = {};
}
async function load() {
  if (!ST.loading) ST.loading = getJSON("api/models").then(v => { reset(v); render(); }).finally(() => { ST.loading = null; });
  return ST.loading;
}
function touch() { ST.dirty = true; $("models-save-msg").textContent = ""; render(); }

/* ---- 小控件 ---- */
function textInput(value, onchange, opts) {
  const i = el("input"); i.type = (opts && opts.type) || "text"; i.value = value == null ? "" : value;
  if (opts && opts.placeholder) i.placeholder = opts.placeholder;
  if (opts && opts.label) i.setAttribute("aria-label", opts.label);
  if (i.type === "password") i.autocomplete = "off";
  i.onchange = () => onchange(i.value); return i;
}
function field(label, input, help) {
  const row = el("div", "mp-field"), lab = el("span", "mp-label", label);
  row.append(lab, input); if (help) row.append(el("span", "note", help)); return row;
}
function usersOf(id) {
  return Object.entries(ST.uses).filter(([, sel]) => [sel.main, sel.backup].some(v => (v || "").split("/")[0] === id)).map(([n]) => n);
}

/* ---- 供应商卡片 ---- */
function providerCard(p) {
  const card = el("div", "mp-card");
  const head = el("div", "mp-head");
  head.append(el("b", "", p.id), el("span", "src", KIND_NAMES[p.kind] || p.kind));
  if (p.source === "config") head.append(el("span", "src", "在 config.toml 里，面板不能删"));
  else if (p.isNew) head.append(el("span", "src console", "新加的，还没保存"));
  const users = usersOf(p.id);
  head.append(el("span", "note", users.length ? "在用：" + users.join("、") : "没有用处在用"));
  const del = el("button", "btn sm", "删除"); del.type = "button"; del.onclick = () => removeProvider(p);
  head.append(del); card.append(head);

  const set = (key, cast) => v => { p[key] = cast ? cast(v) : v.trim(); touch(); };
  if (p.kind === "claude-code") {
    card.append(field("claude 路径", textInput(p.path, set("path"), {label: "claude 路径"}), "命令行里 claude 能用就不用改"));
    card.append(field("配置目录", textInput(p.config_dir, set("config_dir"), {label: "配置目录"}), "和你自己的 Claude Code 隔开"));
    card.append(field("令牌变量名", textInput(p.token_env, set("token_env"), {label: "令牌变量名"})));
  } else {
    card.append(field("接口地址", textInput(p.base_url, set("base_url"), {label: "接口地址", placeholder: "比如 DeepSeek 的 api.deepseek.com（带上协议头）"})));
    card.append(field("Key 变量名", textInput(p.key_env, set("key_env"), {label: "Key 变量名"})));
    card.append(field("高峰倍数", textInput(p.peak == null ? 1 : p.peak, v => { p.peak = Number(v) || 1; touch(); }, {type: "number", label: "高峰倍数"}),
      "高峰价 = 空闲价 × 这个数（DeepSeek 是 2，北京时间工作日 9–12、14–18 点）；1 = 不分时段"));
  }
  const secretLabel = p.kind === "claude-code" ? "令牌" : "Key";
  const pending = p.id in ST.secrets;
  const secret = textInput(pending ? ST.secrets[p.id] : "", v => { ST.secrets[p.id] = v.trim(); touch(); },
    {type: "password", label: secretLabel, placeholder: p.secret || "没设置"});
  const line = el("span", "mp-line"); line.append(secret);
  const srcText = pending ? (ST.secrets[p.id] ? "将保存" : "将清除") : SECRET_SOURCES[p.secret_source] || "";
  if (srcText) line.append(el("span", "src" + (pending || p.secret_source === "secrets" ? " console" : ""), srcText));
  if (p.secret_source === "secrets" && !pending) {
    const clr = el("button", "linkish", "清除"); clr.type = "button"; clr.onclick = () => { ST.secrets[p.id] = ""; touch(); }; line.append(clr);
  }
  card.append(field(secretLabel, line, p.kind === "claude-code" ? "运行一次 claude setup-token 生成" : "明文写在本机 secrets.toml"));

  const list = el("div", "mp-models");
  list.append(el("span", "mp-label", "模型"));
  const rows = el("div", "mp-model-rows");
  p.models.forEach((m, i) => {
    const r = el("div", "mp-model");
    const name = textInput(m.name, v => {
      const old = m.name; m.name = v.trim();
      if (p.prices && old in p.prices) { p.prices[m.name] = p.prices[old]; delete p.prices[old]; }
      touch();
    }, {label: "模型名"});
    const vis = el("label", "mp-vision"), box = el("input"); box.type = "checkbox"; box.checked = !!m.vision;
    box.onchange = () => { m.vision = box.checked; touch(); }; vis.append(box, el("span", "", "能看图"));
    const rm = el("button", "linkish", "去掉"); rm.type = "button"; rm.onclick = () => { p.models.splice(i, 1); touch(); };
    r.append(name, vis);
    if (p.kind === "openai") r.append(priceInputs(p, m));
    r.append(rm); rows.append(r);
  });
  const add = el("button", "linkish", "+ 模型"); add.type = "button";
  add.onclick = () => { p.models.push({name: "", vision: false}); touch(); };
  rows.append(add); list.append(rows); card.append(list);

  const probe = el("div", "probe"), btn = el("button", "btn sm", "测试"), res = ST.results[p.id];
  const msg = el("span", "msg" + (res ? " " + res[0] : ""), res ? res[1] : "");
  btn.type = "button"; btn.onclick = () => runTest(p, btn, msg);
  probe.append(btn, msg, el("span", "note", "发一句「只回复 ok」，勾了能看图的再问一张红图；会花一点额度，用页面上现在填的值"));
  card.append(probe);
  return card;
}

/* 单价（spec 2026-10-06-model-usage §3.1）：元 / 百万 token 的空闲价，三个都空 = 不填单价 */
const PRICE_PARTS = ["命中缓存", "没命中", "输出"];
function priceInputs(p, m) {
  const box = el("span", "mp-price"), cur = (p.prices || {})[m.name];
  PRICE_PARTS.forEach((label, k) => {
    const i = textInput(cur ? cur[k] : "", v => {
      p.prices = p.prices || {};
      const row = p.prices[m.name] ? [...p.prices[m.name]] : [null, null, null];
      row[k] = v === "" ? null : Number(v);
      if (row.every(x => x == null)) delete p.prices[m.name]; else p.prices[m.name] = row;
      touch();
    }, {type: "number", label: `${m.name || "模型"} ${label}单价`, placeholder: label});
    i.min = "0"; i.step = "any"; box.append(i);
  });
  box.append(el("span", "note", "命中缓存 / 没命中 / 输出，元/百万 token（三个都填才算）"));
  return box;
}

async function runTest(p, btn, msg) {
  btn.disabled = true; msg.className = "msg"; msg.textContent = "测试中…";
  const r = await post("api/models/test", {provider: clean(p), secret: ST.secrets[p.id] || ""});
  ST.results[p.id] = [r.data.ok ? "ok" : "bad", r.data.text]; btn.disabled = false; render();
}

async function removeProvider(p) {
  if (p.source === "config") { toast(`${p.id} 写在 config.toml 里，面板不能删`, "warn"); return; }
  const users = usersOf(p.id);
  if (users.length) { toast(`还有 ${users.join("、")} 在用 ${p.id}，先给它们换个模型`, "warn"); return; }
  if (!await ask(`删掉供应商 ${p.id}？保存后才生效。`, {ok: "删掉", danger: true})) return;
  ST.providers = ST.providers.filter(x => x !== p); delete ST.secrets[p.id]; touch();
}

async function addProvider() {
  const pick = $("mp-template").value, view = ST.view;
  const t = pick.startsWith("blank:") ? {id: "", kind: pick.slice(6), models: []} : view.templates[Number(pick)];
  const got = await ask("新供应商的 id（小写字母、数字、- 和 _，模型写成「id/模型名」）", {ok: "添加", fields: [{name: "id", label: "id", value: t.id || ""}]});
  if (!got) return;
  const id = (got.id || "").trim();
  if (!ID_RE.test(id) || id === "echo") { toast("id 只能用小写字母、数字、- 和 _（echo 是内部保留的）", "bad"); return; }
  if (ST.providers.some(x => x.id === id)) { toast(`已经有 ${id} 了`, "bad"); return; }
  const p = {id, kind: t.kind, isNew: true, path: t.path || "claude", token_env: t.token_env || "SKYDANGO_CLAUDE_TOKEN",
    config_dir: t.config_dir || ".brain-claude", base_url: t.base_url || "", key_env: t.key_env || "", secret: "没设置", secret_source: "none",
    prices: {}, peak: 1, peak_hours: [],
    models: (t.models || []).map(([name, vision]) => ({name, vision}))};
  ST.providers.push(p); touch();
}

function clean(p) {
  const out = {};
  for (const k in p) if (!UI_ONLY.includes(k)) out[k] = p[k];
  out.models = p.models.filter(m => m.name).map(m => ({name: m.name, vision: !!m.vision}));
  const names = new Set(out.models.map(m => m.name));
  out.prices = Object.fromEntries(Object.entries(p.prices || {}).filter(([k, v]) => names.has(k) && v.every(x => x != null)));
  return out;
}

/* ---- 用处 ---- */
function options(vision) {
  const out = [];
  for (const p of ST.providers) for (const m of p.models) if (m.name && (!vision || m.vision)) out.push(`${p.id}/${m.name}`);
  return out;
}
function modelSelect(use, which, allowNone) {
  const s = el("select"), cur = ST.uses[use.name][which] || "", opts = options(use.vision);
  s.setAttribute("aria-label", (which === "main" ? "主模型：" : "备用：") + use.label);
  if (allowNone) { const o = el("option", "", "无"); o.value = ""; s.append(o); }
  else if (!cur) { const o = el("option", "", "（没选）"); o.value = ""; s.append(o); }
  for (const v of opts) { const o = el("option", "", v); o.value = v; s.append(o); }
  if (cur && !opts.includes(cur)) {  // 供应商删了、模型去掉了、看图的用处选了看不了图的
    const o = el("option", "", "（已失效）" + cur); o.value = cur; s.append(o); s.classList.add("bad");
  }
  s.value = cur;
  s.onchange = () => { ST.uses[use.name][which] = s.value; touch(); };
  return s;
}

function useRow(use) {
  const row = el("div", "row"); row.id = useRowId(use.name);
  const sel = ST.uses[use.name], orig = ST.view.uses.find(u => u.name === use.name);
  const changed = sel.main !== orig.main || sel.backup !== orig.backup;
  if (changed) row.classList.add("dirty");
  const left = el("div");
  left.append(el("label", "", use.label));
  if (use.vision) left.append(el("span", "src", "看图"));
  left.append(el("span", "src " + use.source, changed ? "改了，还没保存" : SOURCES[use.source] || use.source));
  left.append(el("div", "help", use.help));
  const ctl = el("div", "control");
  const main = el("div", "line"); main.append(el("span", "mp-label", "主"), modelSelect(use, "main", false));
  const backup = el("div", "line"); backup.append(el("span", "mp-label", "备"), modelSelect(use, "backup", true));
  ctl.append(main, backup);
  if (use.problem && !changed) ctl.append(el("div", "msg bad", use.problem));
  if (sel.main !== use.default_main || sel.backup !== use.default_backup) {
    const b = el("button", "linkish", "恢复默认"); b.type = "button";
    b.onclick = () => { ST.uses[use.name] = {main: use.default_main, backup: use.default_backup}; touch(); };
    ctl.append(b);
  }
  row.append(left, ctl);
  return row;
}

/* ---- 整页 ---- */
function render() {
  const v = ST.view; if (!v) return;
  $("models-error").hidden = !v.error;
  $("models-error").textContent = v.error ? `${v.error}（先改好这个文件，否则保存不了）` : "";
  const tpl = $("mp-template");
  if (!tpl.options.length) {
    v.templates.forEach((t, i) => { const o = el("option", "", t.label); o.value = String(i); tpl.append(o); });
    for (const k of v.kinds) { const o = el("option", "", "空白：" + (KIND_NAMES[k] || k)); o.value = "blank:" + k; tpl.append(o); }
  }
  const box = $("providers"); box.textContent = "";
  for (const p of ST.providers) box.append(providerCard(p));
  if (!ST.providers.length) box.append(el("p", "note", "还没有供应商：从下面的模板加一个"));
  const uses = $("uses"); uses.textContent = "";
  for (const u of v.uses) uses.append(useRow(u));
  $("models-savebar").hidden = !ST.dirty && !$("models-save-msg").textContent;
  $("models-save").disabled = !ST.dirty || !!v.error;
}

function flash(name) {
  const row = $(useRowId(name)); if (!row) return;
  row.scrollIntoView({block: "center", behavior: "smooth"});
  row.classList.remove("flash"); void row.offsetWidth; row.classList.add("flash");
  clearTimeout(ST.flashTimer); ST.flashTimer = setTimeout(() => row.classList.remove("flash"), 2000);
}

async function save() {
  const btn = $("models-save"); btn.disabled = true;
  const r = await post("api/models", {providers: ST.providers.map(clean), uses: ST.uses, secrets: ST.secrets});
  if (!r.data.ok) {
    $("models-save-msg").className = "msg bad"; $("models-save-msg").textContent = r.data.text || "保存不了";
    btn.disabled = false; return;
  }
  const text = r.data.restart ? "已保存，下次启动团子 / 沙盒生效" : "已保存";
  toast(text, "ok");
  await load();
  $("models-save-msg").className = "msg ok"; $("models-save-msg").textContent = text; render();
  refresh();
}

let usage = null;  // 顶上一行：今天合计 + 余额（usage.js）
Pages.models = {
  init() {
    usage = Usage.mountUsage($("models-usage"), "api/usage", {compact: true, every: 30000,
      visible: () => !$("page-models").hidden});
    $("models-save").onclick = save;
    $("models-discard").onclick = () => { reset(ST.view); $("models-save-msg").textContent = ""; render(); };
    $("mp-add").onclick = addProvider;
  },
  async show(arg) {
    usage.tick();
    if (!ST.view) { try { await load(); } catch (e) { toast("读取模型设置失败：" + e, "bad"); return; } }
    if (arg) flash(arg);
  },
};
})();
