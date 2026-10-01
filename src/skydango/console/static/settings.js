/* 设置页（spec 2026-10-01-console-redesign §4.5）：左侧粘性分组目录、每项一个锚点 id、#settings/<key> 跳转并高亮。
 * 未保存的修改（dirty / revert）放在这个文件自己的状态里；show() 可以反复调用，只重新滚动和高亮，不会丢掉没保存的修改。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const GROUPS = [["connect", "连接"], ["llm", "普通模式的大模型"], ["brain", "统管大脑"], ["features", "功能开关"], ["identity", "身份"]];
const SOURCES = {default: "默认", config: "config.toml", console: "面板", secrets: "secrets.toml", env: "环境变量", none: "没设置"};
const PROBES = {llm: ["llm", "测试大模型"], brain: ["claude", "测试 Claude 令牌"]};
const ST = {view: null, dirty: {}, revert: new Set(), results: {}, loading: null, flashTimer: 0};

const rowId = key => "set-" + key.replaceAll(".", "-");

async function loadSettings() {
  if (!ST.loading) ST.loading = getJSON("api/settings").then(v => { ST.view = v; render(); }).finally(() => { ST.loading = null; });
  return ST.loading;
}
function currentValue(f) { return f.key in ST.dirty ? ST.dirty[f.key] : f.value; }
function markDirty(key, value) { ST.dirty[key] = value; ST.revert.delete(key); render(); }

function inputFor(f) {
  const val = currentValue(f);
  if (f.kind === "bool") {
    const w = el("span", "toggle"), i = el("input"); i.type = "checkbox"; i.checked = !!val; i.id = "f-" + f.key;
    i.onchange = () => markDirty(f.key, i.checked); w.append(i, el("span")); return w;
  }
  if (f.kind === "choice") {
    const s = el("select"); s.id = "f-" + f.key;
    for (const c of f.choices) { const o = el("option", "", c); o.value = c; s.append(o); }
    s.value = val; s.onchange = () => markDirty(f.key, s.value); return s;
  }
  const i = el("input"); i.id = "f-" + f.key;
  if (f.kind === "secret") { i.type = "password"; i.autocomplete = "off"; i.placeholder = f.value; i.value = f.key in ST.dirty ? ST.dirty[f.key] : ""; }
  else if (f.kind === "float" || f.kind === "int") { i.type = "number"; i.value = val; }
  else { i.type = "text"; i.value = val; }
  i.onchange = () => markDirty(f.key, f.kind === "float" || f.kind === "int" ? Number(i.value) : i.value);
  return i;
}

function render() {
  const v = ST.view; if (!v) return;
  $("settings-error").hidden = !v.error;
  $("settings-error").textContent = v.error ? `${v.error}（先改好这个文件，否则保存不了）` : "";
  const act = document.activeElement, focusId = act && act.id && $("groups").contains(act) ? act.id : "";
  const toc = $("settings-toc"); toc.textContent = "";
  const box = $("groups"); box.textContent = "";
  for (const [g, title] of GROUPS) {
    const a = el("a", "", title); a.href = "#settings-group-" + g;
    a.onclick = e => { e.preventDefault(); const t = $("settings-group-" + g); if (t) t.scrollIntoView({block: "start", behavior: "smooth"}); };
    toc.append(a);
    const sec = el("section", "group"); sec.id = "settings-group-" + g; sec.append(el("h2", "", title));
    for (const f of v.fields.filter(x => x.group === g)) {
      const row = el("div", "row"); row.id = rowId(f.key);
      if (f.key in ST.dirty || ST.revert.has(f.key)) row.classList.add("dirty");
      const left = el("div"), lab = el("label", "", f.label); lab.htmlFor = "f-" + f.key;
      const src = ST.revert.has(f.key) ? (f.kind === "secret" ? "将清除" : "将用回 config.toml") : SOURCES[f.source] || f.source;
      left.append(lab, el("span", "src " + f.source, src), el("div", "help", f.help));
      const ctl = el("div", "control"), line = el("div", "line"); line.append(inputFor(f));
      if (f.kind === "secret" && f.source === "secrets") {
        const b = el("button", "btn sm", "清除"); b.type = "button";
        b.onclick = () => { delete ST.dirty[f.key]; ST.revert.add(f.key); render(); }; line.append(b);
      }
      ctl.append(line);
      const res = ST.results[f.key];
      if (res) ctl.append(el("div", "msg " + res[0], res[1]));
      else if (f.warning) ctl.append(el("div", "msg warn", f.warning));
      if (f.kind !== "secret" && f.source === "console" && !ST.revert.has(f.key)) {
        const b = el("button", "linkish", "用回 config.toml 的值"); b.type = "button";
        b.onclick = () => { delete ST.dirty[f.key]; ST.revert.add(f.key); render(); }; ctl.append(b);
      }
      row.append(left, ctl); sec.append(row);
    }
    if (PROBES[g]) {
      const [what, label] = PROBES[g], p = el("div", "probe"), b = el("button", "btn sm", label), r = ST.results["probe-" + what];
      const m = el("span", "msg", r ? r[1] : ""); if (r) m.classList.add(r[0]);
      b.type = "button"; b.onclick = () => runProbe(what, b, m);
      p.append(b, m, el("span", "note", "会花一点额度；用的是页面上现在填的值")); sec.append(p);
    }
    box.append(sec);
  }
  const changed = Object.keys(ST.dirty).length + ST.revert.size;
  $("savebar").hidden = !changed && !$("save-msg").textContent;
  $("save").disabled = !changed || !!v.error;
  if (focusId && $(focusId)) $(focusId).focus({preventScroll: true});
}

async function runProbe(what, btn, msg) {
  btn.disabled = true; msg.className = "msg"; msg.textContent = "测试中…";
  const r = await post("api/settings/test", {what, values: ST.dirty});
  ST.results["probe-" + what] = [r.data.ok ? "ok" : "bad", r.data.text]; btn.disabled = false; render();
}

function flash(key) {
  const row = $(rowId(key)); if (!row) return;
  row.scrollIntoView({block: "center", behavior: "smooth"});
  row.classList.remove("flash"); void row.offsetWidth; row.classList.add("flash");
  clearTimeout(ST.flashTimer); ST.flashTimer = setTimeout(() => row.classList.remove("flash"), 2000);
}

Pages.settings = {
  init() {
    $("save").onclick = async () => {
      const r = await post("api/settings", {values: ST.dirty, revert: [...ST.revert]});
      ST.results = {};
      if (!r.data.ok) {
        const errs = r.data.errors || {}; for (const k in errs) ST.results[k] = ["bad", errs[k]];
        $("save-msg").className = "msg bad"; $("save-msg").textContent = errs._file || r.data.text || "有几项不对，看标红的地方"; render(); return;
      }
      for (const k in r.data.warnings || {}) ST.results[k] = ["warn", r.data.warnings[k]];
      ST.dirty = {}; ST.revert.clear();
      $("save-msg").className = "msg ok"; $("save-msg").textContent = r.data.restart ? "已保存，重启团子后生效" : "已保存";
      await loadSettings(); refresh();
    };
    $("discard").onclick = () => { ST.dirty = {}; ST.revert.clear(); ST.results = {}; $("save-msg").textContent = ""; render(); };
  },
  async show(arg) {
    if (!ST.view) { try { await loadSettings(); } catch (e) { toast("读取设置失败：" + e, "bad"); return; } }
    if (arg) flash(arg);
  },
  markDirty,
  ensureLoaded: loadSettings,
};
})();
