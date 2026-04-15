/* VisionAI v4 — image.js — FIXED */
(function(){
  let file=null, dets=[], b64='';

  const dz     =document.getElementById('drop-zone');
  const fi     =document.getElementById('file-input');
  const dBtn   =document.getElementById('detect-btn');
  const resSec =document.getElementById('results-section');
  const imgO   =document.getElementById('img-original');
  const imgA   =document.getElementById('img-annotated');
  const detList=document.getElementById('det-list');
  const sumTxt =document.getElementById('sum-text');
  const mdSel  =document.getElementById('model-select');
  const confSl =document.getElementById('conf-slider');
  const confVl =document.getElementById('conf-val');
  const iouSl  =document.getElementById('iou-slider');
  const iouVl  =document.getElementById('iou-val');
  const dlBtn  =document.getElementById('dl-btn');
  const csvBtn =document.getElementById('csv-btn');
  const jsonBtn=document.getElementById('json-btn');

  async function init(){
    UI.initNav();UI.initTheme();UI.initGpuBadge();UI.initSidebarStats();
    await loadModels();restore();bind();
  }

  async function loadModels(){
    try{
      const d=await API.getModels();
      mdSel.innerHTML='';
      d.models.forEach(m=>{
        const o=document.createElement('option');
        o.value=m.name;
        o.textContent=`${m.name} — ${m.description}`;
        if(m.name==='yolov8x') o.selected=true;
        mdSel.appendChild(o);
      });
    }catch{UI.showToast('Cannot load models — is backend running?','error');}
  }

  function restore(){
    const c=localStorage.getItem('vai-conf'),i=localStorage.getItem('vai-iou'),m=localStorage.getItem('vai-model');
    if(c){confSl.value=c;confVl.textContent=`${Math.round(c*100)}%`;}
    if(i){iouSl.value=i;iouVl.textContent=`${Math.round(i*100)}%`;}
    if(m) mdSel.value=m;
  }
  function save(){
    localStorage.setItem('vai-conf',confSl.value);
    localStorage.setItem('vai-iou',iouSl.value);
    localStorage.setItem('vai-model',mdSel.value);
  }

  function bind(){
    confSl.addEventListener('input',()=>{confVl.textContent=`${Math.round(confSl.value*100)}%`;save();});
    iouSl.addEventListener('input',()=>{iouVl.textContent=`${Math.round(iouSl.value*100)}%`;save();});
    mdSel.addEventListener('change',save);
    dz.addEventListener('click',()=>fi.click());
    fi.addEventListener('change',()=>fi.files[0]&&pick(fi.files[0]));
    dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover');});
    dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
    dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('dragover');e.dataTransfer.files[0]&&pick(e.dataTransfer.files[0]);});
    dBtn.addEventListener('click',run);
    dlBtn&&dlBtn.addEventListener('click',dlImage);
    csvBtn&&csvBtn.addEventListener('click',exportCsv);
    jsonBtn&&jsonBtn.addEventListener('click',exportJson);
  }

  function pick(f){
    if(!['image/jpeg','image/png','image/webp'].includes(f.type)){UI.showToast('Use JPG, PNG or WEBP','error');return;}
    if(f.size>10*1024*1024){UI.showToast('Max 10 MB','error');return;}
    file=f;dz.classList.add('has-file');dBtn.disabled=false;
    const r=new FileReader();
    r.onload=e=>{
      dz.innerHTML=`<img src="${e.target.result}" style="max-height:180px;max-width:100%;border-radius:10px;object-fit:contain;display:block;margin:0 auto;position:relative;z-index:1;"/><p style="margin-top:0.6rem;font-size:0.75rem;color:var(--t3);position:relative;z-index:1;">${f.name} · ${(f.size/1024).toFixed(0)} KB</p>`;
    };
    r.readAsDataURL(f);
  }

  async function run(){
    if(!file){UI.showToast('Select an image first','warning');return;}
    dBtn.disabled=true;
    dBtn.innerHTML='<span class="spinner"></span> Detecting…';
    resSec.classList.add('hidden');
    try{
      const data=await API.detectImage(file,parseFloat(confSl.value),parseFloat(iouSl.value),mdSel.value);
      if(!data.success) throw new Error(data.error||'Detection failed');
      dets=data.detections; b64=data.annotated_image;
      render(data);
      UI.showToast(`${data.total_count} object${data.total_count!==1?'s':''} detected!`,'success');
    }catch(e){
      UI.showToast(e.message,'error');
    }finally{
      dBtn.disabled=false;dBtn.textContent='Detect Objects';
    }
  }

  function render(data){
    resSec.classList.remove('hidden');

    // Original image from file
    const r=new FileReader();
    r.onload=e=>{ imgO.src=e.target.result; };
    r.readAsDataURL(file);

    // Annotated image from backend base64
    imgA.src=`data:image/jpeg;base64,${data.annotated_image}`;

    sumTxt.innerHTML=`<strong>${data.total_count}</strong> object${data.total_count!==1?'s':''} · <strong>${UI.fmtMs(data.inference_ms)}</strong> · <strong>${data.model_used}</strong> · ${data.image_width}×${data.image_height}px`;

    detList.innerHTML='';
    if(!data.detections.length){
      detList.innerHTML='<p class="t2 text-sm" style="padding:0.5rem 0;">No objects found. Try lowering confidence to 20–30%.</p>';
      return;
    }
    data.detections.forEach((det,i)=>{
      const col=det.color_hex||UI.classColor(det.class_name);
      const pct=Math.round(det.confidence*100);
      const row=document.createElement('div');
      row.className='det-row';row.style.animationDelay=`${i*0.04}s`;
      row.innerHTML=`
        <span class="det-num">${String(i+1).padStart(2,'0')}</span>
        <span class="det-badge" style="background:${col}20;color:${col};border:1px solid ${col}40;">${det.class_name}</span>
        <div class="det-bg"><div class="det-bar" style="background:linear-gradient(90deg,${col},${col}aa);"></div></div>
        <span class="det-pct">${pct}%</span>`;
      detList.appendChild(row);
      requestAnimationFrame(()=>UI.animBar(row.querySelector('.det-bar'),pct));
    });
  }

  function dlImage(){
    if(!b64){UI.showToast('No result yet','warning');return;}
    const a=document.createElement('a');
    a.href=`data:image/jpeg;base64,${b64}`;a.download='visionai_result.jpg';a.click();
  }
  function exportCsv(){
    if(!dets.length){UI.showToast('No detections','warning');return;}
    const rows=['class,confidence,x1,y1,x2,y2',...dets.map(d=>`${d.class_name},${d.confidence},${d.x1},${d.y1},${d.x2},${d.y2}`)].join('\n');
    _dl(new Blob([rows],{type:'text/csv'}),'detections.csv');
    UI.showToast('CSV exported!','success');
  }
  function exportJson(){
    if(!dets.length){UI.showToast('No detections','warning');return;}
    _dl(new Blob([JSON.stringify(dets,null,2)],{type:'application/json'}),'detections.json');
    UI.showToast('JSON exported!','success');
  }
  function _dl(blob,name){
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download=name;
    document.body.appendChild(a);a.click();document.body.removeChild(a);
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }

  document.readyState==='loading'?document.addEventListener('DOMContentLoaded',init):init();
})();