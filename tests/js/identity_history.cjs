/* DOM 只代替渲染宿主；前端历史和 HTTP 读写都运行真实实现。 */
const assert = require('node:assert/strict');
class Element {
  constructor() { this.children = []; this.style = {}; this.dataset = {}; this.hidden = true; this.disabled = false; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  setAttribute() {}
  blur() {}
}
const elements = new Map();
global.document = {documentElement: {}};
global.window = {};
global.$ = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
global.el = () => new Element();
global.toast = () => {};
const base = process.argv[3], rawFetch = global.fetch;
global.fetch = (url, options) => rawFetch(new URL(url, base), options);
let pending = 0;
global.post = async (url, body) => {
  pending++;
  try {
    const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Skydango': '1'}, body: JSON.stringify(body)});
    return {status: response.status, data: await response.json()};
  } finally { pending--; }
};
require(process.argv[2]);
const state = async () => (await fetch('api/identity/state?run=r1')).json();
async function key(key) {
  window.IdentityTab.key({key, preventDefault() {}});
  const deadline = Date.now() + 5000;
  do {
    await new Promise(resolve => setTimeout(resolve, 10));
    if (Date.now() > deadline) throw new Error('保存超时');
  } while (pending || $('id-run').disabled);
}
(async () => {
  await window.IdentityTab.load();
  await key('Enter');
  assert.equal((await state()).tracks[0].reviewed, true);
  await key('z');
  assert.equal((await state()).tracks[0].reviewed, false, '撤销采纳初判须恢复未看过');
  assert.equal((await state()).coverage.done, 0);
  await key('z');
  assert.equal((await state()).tracks[0].reviewed, false, '历史耗尽后不能重新做刚撤销的操作');
  await key('s');
  await key('d');
  await key('z');
  window.IdentityTab.hide();
  await window.IdentityTab.load();
  await key('z');
  assert.equal((await state()).coverage.done, 0, '撤销一步后刷新，仍应能撤销前一个操作');
  await key('s');
  window.IdentityTab.hide();
  await window.IdentityTab.load();
  await key('z');
  assert.equal((await state()).coverage.done, 0, '刷新后撤销须恢复后端审核状态');
})().catch(error => { console.error(error); process.exitCode = 1; });
