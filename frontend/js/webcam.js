/* VisionAI v4 — webcam.js */
(function(){
  let ws=null,stream=null,loop=null,running=false;
  let fts=[],history=[],MAX_H=30;

  const permBox =document.getElementById('perm-box');
  const vidLay  =document.getElementById('vid-layout');
  const video   =document.getElementById('wc-video');
  const overlay =document.getElementById('wc-overlay');
  const offscreen=document.createElement('canvas');
  const ctx     =overlay.getContext('2d');
  const ssBtn   =document.getElementById('ss-btn');
  const mdSel   =document.getElementById('model-select');
  const confSl  =document.getElementById('conf-slider');
  const confVl  =document.getElementById('conf-val');
  const fpsBadge=document.getElementById('fps-badge');
  const msBadge =document.getElementById('ms-badge');
  const sideList=document.getElementById('side-list');
  const histBody=document.getElementById('hist-body');
  const camBtn  =document.getElementById('cam-btn');

  async function init(){
    UI.initNav();UI.initTheme();UI.initGpuBadge();UI.initSidebarStats();
    await loadModels();restore();bind();requestCam();
  }

  async function loadModels(){
    try{
      const d=await API.getModels();
      mdSel.innerHTML='';
      d.models.forEach(m=>{
        const o=document.createElement('option');
        o.value=m.name;o.textContent=m.name;
        if(m.name==='yolov8s') o.selected=true;
        mdSel.appendChild(o);
      });
    }catch{}
  }

  function restore(){
    const c=localStorage.getItem('vai-conf');
    if(c){confSl.value=c;confVl.textContent=`${Math.round(c*100)}%`;}
  }

  function bind(){
    confSl.addEventListener('input',()=>{confVl.textContent=`${Math.round(confSl.value*100)}%`;localStorage.setItem('vai-conf',confSl.value);});
    camBtn&&camBtn.addEventListener('click',requestCam);
    ssBtn.addEventListener('click',toggle);
    mdSel.addEventListener('change',()=>{if(running)UI.showToast(`Switched to ${mdSel.value}`,'info');});
    document.addEventListener('keydown',e=>{
      if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
      if(e.key==='s'||e.key==='S') toggle();
      if(e.key==='u'||e.key==='U') location.href='image.html';
    });
    window.addEventListener('beforeunload',cleanup);
  }

  async function requestCam(){
    try{
      stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
      video.srcObject=stream;await video.play();
      permBox.classList.add('hidden');
      vidLay.classList.remove('hidden');
      ssBtn.disabled=false;
      UI.showToast('Camera ready — press Start','success');
    }catch(e){UI.showToast(`Camera denied: ${e.message}`,'error');}
  }

  function toggle(){running?stop():start();}

  function start(){
    if(!stream){UI.showToast('Camera not available','warning');return;}
    ws=API.createWebSocket(onMsg,onErr,onClose);
    ws.addEventListener('open',()=>{
      running=true;
      ssBtn.textContent='⏹ Stop';ssBtn.className='btn btn-warm';
      fts=[];
      loop=setInterval(capture,100);
      UI.showToast('Live detection started 🚀','success');
    });
  }

  function stop(){
    running=false;clearInterval(loop);loop=null;
    if(ws){ws.close();ws=null;}
    ctx.clearRect(0,0,overlay.width,overlay.height);
    ssBtn.textContent='▶ Start';ssBtn.className='btn btn-green';
    if(fpsBadge) fpsBadge.textContent='FPS: —';
    if(msBadge)  msBadge.textContent='—ms';
    UI.showToast('Detection stopped','info');
  }

  function cleanup(){if(running)stop();if(stream)stream.getTracks().forEach(t=>t.stop());}

  function capture(){
    if(!ws||ws.readyState!==WebSocket.OPEN||!video.videoWidth) return;
    if(offscreen.width!==video.videoWidth){offscreen.width=video.videoWidth;offscreen.height=video.videoHeight;}
    offscreen.getContext('2d').drawImage(video,0,0);
    const b64=offscreen.toDataURL('image/jpeg',0.8).split(',')[1];
    API.sendFrame(ws,b64,parseFloat(confSl.value),0.40,mdSel.value);
  }

  function onMsg(data){
    if(data.type==='ping'||!data.success) return;
    overlay.width=video.videoWidth||overlay.offsetWidth;
    overlay.height=video.videoHeight||overlay.offsetHeight;
    if(data.annotated_frame){
      const img=new Image();
      img.onload=()=>{ctx.clearRect(0,0,overlay.width,overlay.height);ctx.drawImage(img,0,0,overlay.width,overlay.height);};
      img.src=`data:image/jpeg;base64,${data.annotated_frame}`;
    }
    const now=performance.now()/1000;fts.push(now);if(fts.length>12)fts.shift();
    if(fts.length>=2&&fpsBadge){
      fpsBadge.textContent=`${((fts.length-1)/(fts.at(-1)-fts[0])).toFixed(1)} FPS`;
    }
    if(msBadge) msBadge.textContent=UI.fmtMs(data.inference_ms);
    updateSide(data.detections);
    addHistory(data.detections);
  }

  function onErr(){UI.showToast('WebSocket error','error');stop();}
  function onClose(){if(running){running=false;UI.showToast('Connection closed','warning');ssBtn.textContent='▶ Start';ssBtn.className='btn btn-green';}}

  function updateSide(dets){
    if(!sideList) return;
    sideList.innerHTML='';
    if(!dets.length){sideList.innerHTML='<p class="t2 text-sm">No objects detected</p>';return;}
    dets.forEach(d=>{
      const col=d.color_hex||UI.classColor(d.class_name);
      const row=document.createElement('div');
      row.className='det-row';row.style.marginBottom='0.35rem';
      row.innerHTML=`<span class="det-badge" style="background:${col}20;color:${col};border:1px solid ${col}40;">${d.class_name}</span><span class="det-pct" style="margin-left:auto;">${UI.fmtConf(d.confidence)}</span>`;
      sideList.appendChild(row);
    });
  }

  function addHistory(dets){
    if(!histBody||!dets.length) return;
    const t=new Date().toLocaleTimeString(),m=mdSel.value;
    dets.forEach(d=>history.unshift({t,cls:d.class_name,conf:d.confidence,m}));
    if(history.length>MAX_H) history.length=MAX_H;
    histBody.innerHTML=history.map(r=>`<tr><td>${r.t}</td><td>${r.cls}</td><td>${UI.fmtConf(r.conf)}</td><td>${r.m}</td></tr>`).join('');
  }

  document.readyState==='loading'?document.addEventListener('DOMContentLoaded',init):init();
})(); 