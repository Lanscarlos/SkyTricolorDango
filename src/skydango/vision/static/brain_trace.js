/* 大脑时间线（识别可视化 viewer 和管理面板共用，见 docs/superpowers/specs/2026-09-28-viewer-brain-trace-design.md）。
 * mountBrainTrace(root, url)：在 root 里画头部状态、"只看做了事的轮次"、轮次列表，对 url + "?after=N" 长轮询；返回 {stop()}。
 * 第一次拿到数据才把 root.hidden 设成 false；第一次就 404（没有大脑）就不再拉。
 * 只用 textContent 放数据（大脑的话、工具返回都可能带尖括号），不拼 HTML 字符串。
 * 包在函数里：只往全局放 mountBrainTrace，不和页面自己的 $ / el 撞名。 */
(function(){
"use strict";
const REASONS={events:"新消息 / 事件",background:"周围的变化",heartbeat:"心跳",farewell:"退出前总结",outside:"轮外"};
const ICONS={thinking:"💭 思考",text:"💬 说",tool:"🔧 调用",result:"↩ 返回"};
const FOLD=10;
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function kilo(n){return n==null?"—":n>=1000?(n/1000).toFixed(1)+"k":String(n)}
function hms(t){return t?new Date(t*1000).toTimeString().slice(0,8):"--:--:--"}
// 点过的按用户的来；没点过的：进行中的和最新一轮展开（收尾时不自己收起），下一轮开始后收起
function wantOpen(t,want,newestId){return want!==undefined?want:(t.end===null||t.id===newestId)}
function acted(t){return t.end===null||!!t.error||t.steps.some(s=>s.kind==="tool"||s.kind==="text")}
function toolCounts(t){const c={};for(const n of t.tools)c[n]=(c[n]||0)+1;
  const s=Object.entries(c).map(([n,k])=>`${n}×${k}`).join(" ");return s||(t.steps.some(x=>x.kind==="text")?"（只说了话）":"（什么都没做）")}
function tokens(r){if(!r)return "";const k=r.tokens||{};return `in ${kilo(k.input)} · out ${kilo(k.output)}`+(k.thinking?` · 思考 ${kilo(k.thinking)}`:"")}
function stepTitle(s){return s.kind==="tool"?`${ICONS.tool} ${s.name}`:ICONS[s.kind]+(s.kind==="result"&&s.error?"（出错 / 被拒绝）":"")}
function stepText(s){return s.kind==="tool"?JSON.stringify(s.input,null,2):(s.text||"")}
function endText(t){const r=t.result,k=(r&&r.tokens)||{},parts=[];
  if(r)parts.push(`${r.subtype??"—"} · ${r.num_turns??"—"} 步`);
  if(t.seconds!=null)parts.push(`${t.seconds.toFixed(1)}s`);
  if(r)parts.push(`输入 ${kilo(k.input)} · 输出 ${kilo(k.output)} · 缓存读 ${kilo(k.cache_read)} · 缓存写 ${kilo(k.cache_write)} · 思考 ${kilo(k.thinking)}`);
  if(r&&r.cost!=null)parts.push(`参考 $${r.cost.toFixed(4)}`);
  return (t.error?`失败：${t.error}\n`:"")+parts.join(" · ")}
function copyText(t){const lines=[`[${hms(t.start)}] ${REASONS[t.reason]||t.reason}`,"== 收到 ==",t.prompt];
  for(const s of t.steps)lines.push(`== ${stepTitle(s)} ==`,stepText(s));
  if(t.end!==null)lines.push("== 结果 ==",endText(t));return lines.join("\n")}
async function copy(t,btn){const text=copyText(t);
  try{await navigator.clipboard.writeText(text)}catch(e){const a=el("textarea");a.value=text;document.body.append(a);a.select();document.execCommand("copy");a.remove()}
  btn.textContent="已复制";setTimeout(()=>btn.textContent="复制这一轮",1200)}
function block(kind,title,text,error){const d=el("div",`step k-${kind}${error?" error":""}`);d.append(el("h4","",title));
  const lines=text.split("\n"),pre=el("pre");d.append(pre);
  if(lines.length<=FOLD){pre.textContent=text;return d}
  let full=false;const btn=el("button","");
  const show=()=>{pre.textContent=full?text:lines.slice(0,FOLD).join("\n")+"\n…";btn.textContent=full?"收起":`展开全部（${lines.length} 行）`};
  btn.onclick=()=>{full=!full;show()};show();d.append(btn);return d}

function mountBrainTrace(root,url){
  const B={turns:new Map(),els:new Map(),version:0,boot:null,want:new Map(),acted:false,stopped:false,ctrl:null};
  root.classList.add("brain-trace");root.textContent="";
  const head=el("div","bt-head"),stateEl=el("span","brain-state","连接中…"),label=el("label"),check=el("input","brain-acted");
  check.type="checkbox";label.append(check,"只看做了事的轮次");head.append(el("b","","大脑"),stateEl,label);
  const box=el("div","brain-turns");root.append(head,box);
  function newest(){return Math.max(0,...B.turns.keys())}
  function turnEl(t){const live=t.end===null,d=el("details",`turn${t.error?" err":""}${live?" live":""}`);
    d.open=wantOpen(t,B.want.get(t.id),newest());
    const sm=el("summary");sm.append(el("span","t",hms(t.start)),el("span","",REASONS[t.reason]||t.reason));
    if(t.error)sm.append(el("span","",`失败：${t.error}`));
    else if(live)sm.append(el("span","",`进行中…${t.tools.length?" "+toolCounts(t):""}`));
    else sm.append(el("span","",toolCounts(t)),el("span","n",t.seconds!=null?`${t.seconds.toFixed(1)}s`:""),el("span","n",tokens(t.result)));
    sm.onclick=()=>B.want.set(t.id,!d.open);d.append(sm);
    const body=el("div","tbody");
    if(t.reason!=="outside")body.append(block("prompt","收到",t.prompt||""));
    for(const s of t.steps)body.append(block(s.kind,stepTitle(s),stepText(s),s.kind==="result"&&s.error));
    if(!live)body.append(block("end","结果",endText(t),!!t.error));
    const btn=el("button","","复制这一轮");btn.onclick=()=>copy(t,btn);body.append(btn);
    d.append(body);d.hidden=B.acted&&!acted(t);return d}
  function brainState(st){let text=`${st.model??"?"} / ${st.effort??"?"} · 已醒 ${st.turns??0} 轮 · `,cls="";
    if(st.offline){text+="已转备用回复（DeepSeek）";cls="bad"}
    else if(st.retry_in!=null){text+=`连续失败 ${st.failures} 次，${Math.ceil(st.retry_in)} 秒后重试`;cls="warn"}
    else text+="在线";stateEl.textContent=text;stateEl.className="brain-state"+(cls?" "+cls:"")}
  function brainRender(changed){
    for(const t of changed){const old=B.els.get(t.id),neu=turnEl(t);if(old)old.replaceWith(neu);B.els.set(t.id,neu)}
    const top=newest();for(const [id,e] of B.els)if(!B.want.has(id))e.open=wantOpen(B.turns.get(id),undefined,top);
    const ids=[...B.turns.keys()].sort((a,b)=>(a===0)-(b===0)||b-a);
    ids.forEach((id,i)=>{const e=B.els.get(id);if(box.children[i]!==e)box.insertBefore(e,box.children[i]||null)})}
  function brainMerge(d){
    if((B.boot&&d.boot!==B.boot)||d.version<B.version){  // 程序重启过：这次的增量不可信，清空后从头拉
      B.turns.clear();for(const e of B.els.values())e.remove();B.els.clear();B.want.clear();B.boot=d.boot;B.version=0;return}
    B.boot=d.boot;B.version=d.version;for(const t of d.turns)B.turns.set(t.id,t);
    if(d.oldest!=null)for(const id of [...B.turns.keys()])if(id!==0&&id<d.oldest){B.turns.delete(id);B.els.get(id)?.remove();B.els.delete(id)}
    brainState(d.state||{});brainRender(d.turns.filter(t=>B.turns.has(t.id)))}
  check.onchange=e=>{B.acted=e.target.checked;for(const [id,x] of B.els)x.hidden=B.acted&&!acted(B.turns.get(id))};
  async function brainLoop(){let seen=false;
    while(!B.stopped){
      try{B.ctrl=new AbortController();
        const r=await fetch(`${url}?after=${B.version}`,{cache:"no-store",signal:B.ctrl.signal});
        if(r.status===404&&!seen)return;if(!r.ok)throw new Error(r.status);
        const d=await r.json();if(B.stopped)return;seen=true;root.hidden=false;brainMerge(d);
      }catch(e){if(B.stopped)return;if(seen){stateEl.textContent="连不上（程序停了？）";stateEl.className="brain-state bad"}await new Promise(r=>setTimeout(r,1000))}
    }
  }
  brainLoop();
  return {stop(){B.stopped=true;if(B.ctrl)B.ctrl.abort()}};
}
window.mountBrainTrace=mountBrainTrace;
})();
