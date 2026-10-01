/* 识别画面：画图 + 识别框 + 标签 + 十字 + 悬停描述（viewer 网页和管理面板真机页共用，spec 2026-10-01-console-live-page §3.1）。
 * 快照格式见 vision/viewer.py 的 _render：{seq, width, height, image, boxes:[{x,y,w,h,kind,label?,score?,desc?}], info}。
 * 标签、描述来自识别 / 模型：只用 fillText 画。顶层不碰 document：node 里 require 它能测 nameAt / toFrame（tests/test_viewer.py）。 */
(function () {
"use strict";
const COLORS={friend:"#3ddc84",name:"#3ddc84",tag:"#facc15",stranger:"#ff9f43",unlit:"#a78bfa",player:"#60a5fa",self:"#cbd5e1",maybe:"#86efac",
ring:"#22d3ee",request:"#f43f5e",panel:"#6b7280",message:"#f472b6",typing:"#e879f9",
bench:"#1d4ed8",bonfire:"#ea580c",instrument:"#fda4af",spirit:"#ffffff",
panel_ok:"#3b82f6",panel_new:"#facc15",panel_unknown:"#ef4444",button_ok:"#22c55e",button_ask:"#9ca3af",button_never:"#dc2626"};
const NAMES={friend:"好友",tag:"没认出的名字",stranger:"陌生人",unlit:"没点火",player:"没判定的人",self:"团子",maybe:"按外观认的好友",ring:"互动圆圈",
request:"互动请求",panel:"聊天面板",message:"新消息",typing:"正在输入",
bench:"座位",bonfire:"篝火",instrument:"乐器",spirit:"先祖",
panel_ok:"面板（已核对）",panel_new:"面板（未核对）",panel_unknown:"不认识的面板",button_ok:"能按",button_ask:"要放行",button_never:"不能按"};

/* 在画面上点人：好友框，或名字标签往下一块（宽 3 倍、连标签 7 倍高，同身体 _below_tag）；重叠时取面积最小的 */
function nameAt(boxes,x,y){let best=null,area=Infinity;for(const b of boxes){if(!b.label||(b.kind!=="friend"&&b.kind!=="name"))continue;const tag=b.kind==="name",x1=tag?b.x-b.w:b.x,w=tag?b.w*3:b.w,h=tag?b.h*7:b.h;if(x<x1||x>=x1+w||y<b.y||y>=b.y+h)continue;if(w*h<area){best=b.label;area=w*h}}return best}
/* 画布被 CSS 缩放显示：按显示尺寸换算回原图像素 */
function toFrame(clientX,clientY,rect,width,height){return [Math.round((clientX-rect.left)*width/rect.width),Math.round((clientY-rect.top)*height/rect.height)]}

/* o.boxes = false 只画图（十字照画）；o.mark = 原图坐标的十字；o.hover = 画布像素坐标，落在有 desc 的框里时在框下方写一行 */
function draw(canvas,img,s,o){
  o=o||{};const ctx=canvas.getContext("2d");
  canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);
  const k=canvas.width/s.width,fs=Math.max(12,Math.round(canvas.width/80));
  if(o.mark){const x=o.mark[0]*k,y=o.mark[1]*k,r=Math.max(12,canvas.width/60);
    ctx.strokeStyle="#f472b6";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-r,y);ctx.lineTo(x+r,y);ctx.moveTo(x,y-r);ctx.lineTo(x,y+r);ctx.stroke()}
  if(o.boxes===false)return;
  ctx.font=`${fs}px system-ui,"Microsoft YaHei",sans-serif`;ctx.textBaseline="middle";
  const boxes=s.boxes||[];
  for(const b of boxes){const col=COLORS[b.kind]||"#fff",x=b.x*k,y=b.y*k,w=b.w*k,h=b.h*k;
    ctx.strokeStyle=col;ctx.lineWidth=b.kind==="request"?4:2;ctx.setLineDash(b.kind.startsWith("panel")?[8,5]:b.kind==="maybe"?[6,4]:[]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
    const t=(b.label||"")+(b.score!==undefined?` ${b.score.toFixed(2)}`:"");if(!t)continue;
    const below=b.kind==="ring"||b.kind==="request",tw=ctx.measureText(t).width+8,th=fs+6,ty=(below||y-th<0)?y+h:y-th;
    ctx.fillStyle=col;ctx.fillRect(x,ty,tw,th);ctx.fillStyle="#0b0d12";ctx.fillText(t,x+4,ty+th/2);}
  if(!o.hover)return;
  let hit=null;
  for(const b of boxes){if(!b.desc)continue;const x=b.x*k,y=b.y*k;if(o.hover[0]>=x&&o.hover[0]<=x+b.w*k&&o.hover[1]>=y&&o.hover[1]<=y+b.h*k)hit=b}
  if(!hit)return;const t=String(hit.desc),tw=ctx.measureText(t).width+8,th=fs+6,x=Math.min(hit.x*k,canvas.width-tw),y=Math.min((hit.y+hit.h)*k,canvas.height-th);
  ctx.fillStyle="rgba(11,13,18,.85)";ctx.fillRect(x,y,tw,th);ctx.fillStyle="#e6e8ee";ctx.fillText(t,x+4,y+th/2)}

const Stage={COLORS,NAMES,nameAt,toFrame,draw};
globalThis.Stage=Stage;
if(typeof module!=="undefined"&&module.exports)module.exports=Stage;
})();
