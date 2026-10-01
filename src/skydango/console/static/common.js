/* 管理面板公共脚本（spec 2026-10-01-console-redesign §2 §5）。各页文件用这里挂在全局的东西：
 *   $(id) / el(tag, cls?, text?) / getJSON(url) / post(url, body?) -> {status, data}
 *   pad2 / hhmm / dayTime / span / fmtUptime（时间文字）
 *   ask(text, {ok, cancel, danger, fields}) -> Promise：页面内确认框（内嵌浏览器里原生对话框会被直接取消）
 *   toast(text, "ok"|"warn"|"bad")：右下角提示条；problemList(容器, [{text, setting}])：预检问题 + 「去设置 →」
 *   Pages.<名字> = {init, show(arg), hide}；go(page, arg?)；parseHash(hash) -> {page, arg}
 *   S.state = 最近一次 /api/state；onState(fn) 每次拉到新状态都调；refresh() 立刻拉一次；BUSY
 * 顶层不碰 document / window：node 里 require 它能测 parseHash（tests/test_console_page.py）。 */
(function(g){
"use strict";
const HAS_DOM=typeof document!=="undefined";
const PAGES=["sandbox","live","inner","scenarios","labeling","settings","device"];
const TITLES={sandbox:"沙盒",live:"真机团子",inner:"内心",scenarios:"剧本和报告",labeling:"标注",settings:"设置",device:"设备"};
const ALIAS={overview:"live"};  // 旧书签
const BUSY=["starting","running","stopping"];
const S={state:null,offline:false};
const Pages={};
const listeners=[];

/* ---- 小工具（照搬旧页面） ---- */
const $=id=>document.getElementById(id);
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
async function getJSON(url){const r=await fetch(url,{cache:"no-store"});return r.json()}
async function post(url,body){
  const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json","X-Skydango":"1"},body:JSON.stringify(body||{})});
  let data=null;try{data=await r.json()}catch(e){}
  return {status:r.status,data:data||{ok:false,text:`面板没回应（${r.status}）`}}
}
function pad2(n){return String(n).padStart(2,"0")}
function hhmm(t){const d=new Date(t*1000);return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`}
function dayTime(t){const d=new Date(t*1000);return `${d.getMonth()+1}月${d.getDate()}日 ${hhmm(t)}`}
function span(sec){sec=Math.max(0,sec);if(sec<90)return"不到 1 分钟";const m=Math.round(sec/60);if(m<60)return`${m} 分钟`;
  const h=Math.round(sec/3600);if(h<48)return`${h} 小时`;return`${Math.round(sec/86400)} 天`}
function fmtUptime(s){if(s==null)return"";s=Math.floor(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
  return h?`${h} 小时 ${m} 分`:m?`${m} 分钟`:`${s} 秒`}

/* ---- 路由：#page 或 #page/arg ---- */
function parseHash(hash){
  const h=String(hash||"").replace(/^#/,""),i=h.indexOf("/");
  let page=i<0?h:h.slice(0,i),arg=i<0?"":h.slice(i+1);
  page=ALIAS[page]||page;
  if(!PAGES.includes(page))return {page:"sandbox",arg:null};
  try{arg=decodeURIComponent(arg)}catch(e){}
  return {page,arg:arg===""?null:arg};
}
function hashOf(page,arg){return "#"+page+(arg==null||arg===""?"":"/"+arg)}
function sameHash(a,b){const d=x=>{try{return decodeURIComponent(x||"")}catch(e){return x||""}};return d(a)===d(b)}  // 中文参数在 location.hash 里是编码过的
let current=null;
function hook(name,fn,...args){const p=Pages[name];if(!p||typeof p[fn]!=="function")return;
  try{p[fn](...args)}catch(e){console.error(`Pages.${name}.${fn}`,e)}}
function route(){
  const {page,arg}=parseHash(location.hash),want=hashOf(page,arg);
  if(!sameHash(location.hash,want))history.replaceState(null,"",want);  // #overview → #live；空的、不认识的 → #sandbox
  for(const p of PAGES){const sec=$("page-"+p);if(sec)sec.hidden=p!==page}
  for(const a of document.querySelectorAll("#nav [data-page]")){
    if(a.dataset.page===page)a.setAttribute("aria-current","page");else a.removeAttribute("aria-current")}
  document.title=`${TITLES[page]} · 团子管理面板`;
  if(current&&current!==page)hook(current,"hide");
  current=page;hook(page,"show",arg);
}
function go(page,arg){
  const want=hashOf(page,arg);
  if(sameHash(location.hash,want))route();  // 同一个 hash 不会触发 hashchange（比如再点一次「去设置 →」）
  else location.hash=want;
}

/* ---- 页面内确认框 ---- */
let closeDialog=null;
function ask(text,opts){
  opts=opts||{};
  if(closeDialog)closeDialog(false);  // 同时只开一个：旧的当取消
  return new Promise(resolve=>{
    const root=$("dialog"),back=document.activeElement,fields=Array.isArray(opts.fields)&&opts.fields.length?opts.fields:null;
    const card=el("form","dlg"+(opts.danger?" danger":""));
    card.setAttribute("role","dialog");card.setAttribute("aria-modal","true");card.noValidate=true;
    const msg=el("p","dlg-text",text);msg.id="dlg-text";card.setAttribute("aria-labelledby","dlg-text");card.append(msg);
    const inputs=[];
    for(const f of fields||[]){const lab=el("label","dlg-field"),inp=el("input");
      inp.name=f.name;inp.type="text";inp.autocomplete="off";
      if(f.placeholder)inp.placeholder=f.placeholder;if(f.value!=null)inp.value=f.value;
      lab.append(el("span","",f.label||f.name),inp);card.append(lab);inputs.push(inp)}
    const row=el("div","dlg-actions"),no=el("button","btn",opts.cancel||"取消"),yes=el("button",opts.danger?"btn danger":"btn go",opts.ok||"确定");
    no.type="button";yes.type="submit";row.append(no,yes);card.append(row);
    let done=false;
    function finish(ok){
      if(done)return;done=true;closeDialog=null;
      document.removeEventListener("keydown",onKey,true);
      const values={};for(const i of inputs)values[i.name]=i.value;
      root.hidden=true;root.textContent="";root.onmousedown=null;
      if(back&&typeof back.focus==="function"&&document.contains(back))back.focus();
      resolve(fields?(ok?values:null):!!ok);
    }
    function onKey(e){
      if(e.key==="Escape"){e.preventDefault();e.stopPropagation();finish(false);return}
      if(e.key==="Enter"&&!e.isComposing&&!(e.target&&e.target.tagName==="BUTTON")){e.preventDefault();finish(true);return}
      if(e.key==="Tab"){  // 焦点留在框里
        const items=[...inputs,no,yes],n=items.length,i=items.indexOf(document.activeElement);
        e.preventDefault();items[i<0?0:(i+(e.shiftKey?-1:1)+n)%n].focus();
      }
    }
    card.onsubmit=e=>{e.preventDefault();finish(true)};
    no.onclick=()=>finish(false);
    root.onmousedown=e=>{if(e.target===root)finish(false)};  // 点遮罩 = 取消
    document.addEventListener("keydown",onKey,true);
    closeDialog=finish;
    root.textContent="";root.append(card);root.hidden=false;
    (inputs[0]||yes).focus();
  });
}

/* ---- 提示条 ---- */
function toast(text,kind){
  kind=kind==="warn"||kind==="bad"?kind:"ok";
  const box=$("toasts");if(!box)return null;
  const t=el("div","toast "+kind);t.setAttribute("role",kind==="bad"?"alert":"status");
  t.append(el("span","toast-text",String(text)));
  if(kind==="bad"){const x=el("button","toast-x","×");x.type="button";x.title="关掉";x.setAttribute("aria-label","关掉");x.onclick=()=>t.remove();t.append(x)}
  else setTimeout(()=>t.remove(),4000);
  box.append(t);
  const all=box.querySelectorAll(".toast:not(.bad)");if(all.length>4)all[0].remove();  // 别堆太多
  return t;
}

/* ---- 预检问题 ---- */
function problemList(container,problems){
  if(!container)return;
  const list=(problems||[]).map(p=>typeof p==="string"?{text:p,setting:null}:p);
  container.textContent="";container.hidden=!list.length;
  if(!list.length)return;
  const ul=el("ul","problems");
  for(const p of list){const li=el("li","",p.text);
    if(p.setting){const a=el("a","to-setting","去设置 →");a.href=hashOf("settings",p.setting);li.append(" ",a)}
    ul.append(li)}
  container.append(ul);
}

/* ---- 左栏：正在跑卡片、导航标记 ---- */
function runKind(run){return run&&run.kind==="sandbox"?"sandbox":"dango"}
function renderRunCard(st){
  const card=$("runcard");if(!card)return;
  const run=(st&&st.run)||{},s=run.state,isSb=runKind(run)==="sandbox",who=isSb?"沙盒":"真机团子";
  let cs="idle",title="都没在跑",sub="在「沙盒」或「真机团子」页启动";
  if(s==="starting"){cs="busy";title=`${who} 正在启动…`;sub=run.slow_start?"已超过 60 秒，看看日志":""}
  else if(s==="running"){cs="busy";title=`${who} 运行中`;
    if(isSb){let t="";try{t=typeof g.sandboxSummary==="function"?g.sandboxSummary()||"":""}catch(e){}sub=t}
    else{const o=run.options||{};sub=[o.live?"真的发送":"只打印",run.uptime!=null?`已运行 ${fmtUptime(run.uptime)}`:""].filter(Boolean).join(" · ")}}
  else if(s==="stopping"){cs="stopping";title=`${who} 在收尾…`;sub=isSb?"最终反思、写日记":"恢复轮盘、复原镜头"}
  else if(s==="crashed"){cs="crashed";title=`${who} 出错停下了`;sub=run.error||(run.exit_code!=null?`退出码 ${run.exit_code}`:"")}
  card.dataset.state=cs;
  card.classList.toggle("go",cs!=="idle");
  card.title=cs!=="idle"?`去「${isSb?"沙盒":"真机团子"}」页`:"";
  $("rc-title").textContent=title;$("rc-sub").textContent=sub;
  const log=$("rc-log");
  if(cs==="crashed")log.hidden=false;
  else if(!log.hidden){log.hidden=true;log.open=false;$("rc-log-text").textContent="读取中…"}
  const stop=$("rc-stop"),canStop=s==="starting"||s==="running";
  stop.hidden=!canStop;
  if(canStop){stop.textContent=isSb?"下线（写日记）":"停止";stop.dataset.kind=runKind(run)}
  else stop.disabled=false;
}
function renderMarks(st){
  const run=(st&&st.run)||{},dangoBusy=BUSY.includes(run.state)&&runKind(run)==="dango";
  const live=$("mark-live");if(live){live.textContent=dangoBusy?"在跑":"没在跑";live.classList.toggle("on",dangoBusy)}
  const set=$("mark-settings");if(set){const bad=((st&&st.problems)||[]).some(p=>p&&p.setting);
    set.textContent=bad?"!":"";set.classList.toggle("bad",bad);set.title=bad?"有设置要补":""}
}
function renderOffline(){
  const card=$("runcard");if(!card)return;
  card.dataset.state="crashed";card.classList.remove("go");card.title="";
  $("rc-title").textContent="连不上面板（程序停了？）";$("rc-sub").textContent="面板的终端关了的话，重新运行 python -m skydango console";
  $("rc-stop").hidden=true;$("rc-log").hidden=true;
  document.body.dataset.state="crashed";
}
async function loadCrashLog(){
  const pre=$("rc-log-text");pre.textContent="读取中…";
  try{const r=await getJSON("api/logs?after=0");const lines=(r.lines||[]).slice(-20);pre.textContent=lines.length?lines.join("\n"):"（没有日志）"}
  catch(e){pre.textContent="读不到日志（面板停了？）"}
}
async function stopRun(btn){
  const sb=btn.dataset.kind==="sandbox";
  const ok=await ask(sb?"让沙盒下线？会先做最终反思、写日记，再停下。":"让团子停下？会先恢复轮盘、复原镜头，再退出。",{ok:sb?"下线":"停止",danger:true});
  if(!ok)return;
  btn.disabled=true;
  const r=await post(sb?"api/sandbox/stop":"api/run/stop");
  if(!r.data.ok)toast(r.data.text||"没停成","bad");
  btn.disabled=false;refresh();
}

/* ---- 轮询 /api/state ---- */
function onState(fn){listeners.push(fn);if(S.state)try{fn(S.state)}catch(e){console.error(e)}}
async function refresh(){
  let st;
  try{st=await getJSON("api/state")}catch(e){S.offline=true;renderOffline();return}
  S.state=st;S.offline=false;
  document.body.dataset.state=(st.run&&st.run.state)||"idle";
  renderRunCard(st);renderMarks(st);
  for(const fn of listeners){try{fn(st)}catch(e){console.error(e)}}
}
async function tick(){await refresh();setTimeout(tick,1000)}

function start(){
  for(const name of Object.keys(Pages))hook(name,"init");
  $("rc-stop").onclick=e=>{e.stopPropagation();stopRun(e.currentTarget)};
  $("rc-log").addEventListener("toggle",e=>{if(e.currentTarget.open)loadCrashLog()});
  $("runcard").addEventListener("click",e=>{
    if(e.target.closest("button,a,details,input"))return;
    const run=S.state&&S.state.run;if(!run||!$("runcard").classList.contains("go"))return;
    go(runKind(run)==="sandbox"?"sandbox":"live");
  });
  document.addEventListener("click",e=>{  // 点当前页的链接（hash 没变）也重新路由，比如再点一次同一个「去设置 →」
    const a=e.target.closest&&e.target.closest('a[href^="#"]');
    if(a&&sameHash(a.getAttribute("href"),location.hash)){e.preventDefault();route()}
  });
  g.addEventListener("hashchange",route);
  route();tick();
}

Object.assign(g,{$,el,getJSON,post,pad2,hhmm,dayTime,span,fmtUptime,ask,toast,problemList,Pages,go,parseHash,S,onState,refresh,BUSY});
if(HAS_DOM){if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",start);else setTimeout(start,0)}
if(typeof module!=="undefined"&&module.exports)module.exports={parseHash};
})(globalThis);
