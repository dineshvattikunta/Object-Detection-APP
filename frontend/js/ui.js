/* VisionAI v4 — ui.js */
(function(){
  if (!document.getElementById('toast-container')) {
    const el=document.createElement('div'); el.id='toast-container';
    document.body.appendChild(el);
  }
})();

function showToast(msg, type='info', dur=3500) {
  const c=document.getElementById('toast-container');
  const t=document.createElement('div');
  t.className=`toast ${type}`;
  t.innerHTML=`<span class="toast-icon"></span><span>${msg}</span>`;
  c.appendChild(t);
  const dismiss=()=>{t.classList.add('dismissing');t.addEventListener('animationend',()=>t.remove(),{once:true});};
  setTimeout(dismiss,dur);
  t.addEventListener('click',dismiss);
}

function showLoader(id) {
  const el=document.getElementById(id); if(!el) return;
  el.dataset.orig=el.innerHTML;
  el.innerHTML=`<div class="skeleton" style="height:200px;border-radius:14px;margin-bottom:1rem;"></div><div class="skeleton" style="height:14px;width:55%;margin-bottom:0.5rem;"></div><div class="skeleton" style="height:14px;width:38%;"></div>`;
}
function hideLoader(id,html=null) {
  const el=document.getElementById(id); if(!el) return;
  el.innerHTML=html!==null?html:(el.dataset.orig||'');
}

function fmtConf(v){return `${(v*100).toFixed(1)}%`;}
function fmtMs(ms){return `${Math.round(ms)}ms`;}

function classColor(name) {
  let h=5381;
  for(let i=0;i<name.length;i++){h=((h<<5)+h)+name.charCodeAt(i);h|=0;}
  return hsl2hex(Math.abs(h)%360,78,62);
}
function hsl2hex(h,s,l){
  s/=100;l/=100;
  const k=n=>(n+h/30)%12,a=s*Math.min(l,1-l);
  const f=n=>l-a*Math.max(-1,Math.min(k(n)-3,Math.min(9-k(n),1)));
  const x=v=>Math.round(v*255).toString(16).padStart(2,'0');
  return `#${x(f(0))}${x(f(8))}${x(f(4))}`;
}

function animBar(el,pct){
  el.style.width='0%';
  requestAnimationFrame(()=>requestAnimationFrame(()=>{el.style.width=`${pct}%`;}));
}

function updateStats(s){
  _anim('stat-total',s.total_detections);
  _anim('stat-frames',s.total_frames_processed);
  _set('stat-class',s.most_detected_class||'—');
  _set('stat-conf',`${s.average_confidence}%`);
  _set('sb-total',(s.total_detections||0).toLocaleString());
  _set('sb-frames',(s.total_frames_processed||0).toLocaleString());
}
function _anim(id,end){
  const el=document.getElementById(id);if(!el)return;
  const start=parseInt(el.textContent.replace(/,/g,''))||0;
  if(start===end)return;
  const t0=performance.now();
  const step=now=>{
    const p=Math.min((now-t0)/700,1),e=1-Math.pow(1-p,3);
    el.textContent=Math.round(start+(end-start)*e).toLocaleString();
    if(p<1)requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
function _set(id,v){const el=document.getElementById(id);if(el)el.textContent=v;}

function initTheme(){
  const btn=document.getElementById('theme-btn');if(!btn)return;
  btn.addEventListener('click',()=>{
    document.body.classList.toggle('light');
    btn.textContent=document.body.classList.contains('light')?'🌙':'☀️';
  });
}

function initNav(){
  const cur=location.pathname.split('/').pop()||'index.html';
  document.querySelectorAll('.nav-link').forEach(a=>{
    if((a.getAttribute('href')||'')===cur) a.classList.add('active');
  });
}

async function initGpuBadge(){
  const el=document.getElementById('gpu-badge');if(!el)return;
  try{
    const h=await API.checkHealth(3);
    el.innerHTML='';
    const dot=document.createElement('span');dot.className='g-dot';
    el.appendChild(dot);
    el.appendChild(document.createTextNode(h.gpu_available?(h.gpu_name||'GPU Active'):'CPU Mode'));
    el.className=h.gpu_available?'gpu-pill on':'gpu-pill';
  }catch{
    el.innerHTML='<span class="g-dot"></span>Offline';
    el.className='gpu-pill';
  }
}

async function initSidebarStats(){
  try{
    const s=await API.getStats();
    _set('sb-total',(s.total_detections||0).toLocaleString());
    _set('sb-frames',(s.total_frames_processed||0).toLocaleString());
  }catch{}
}

window.UI={showToast,showLoader,hideLoader,fmtConf,fmtMs,classColor,animBar,updateStats,initTheme,initNav,initGpuBadge,initSidebarStats};