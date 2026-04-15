/* VisionAI v4 — api.js */
const BASE    = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

async function detectImage(file, conf=0.25, iou=0.40, model='yolov8x') {
  const form = new FormData();
  form.append('file', file);
  const url = new URL(`${BASE}/detect/image`);
  url.searchParams.set('confidence', conf);
  url.searchParams.set('iou', iou);
  url.searchParams.set('model', model);
  const res = await fetch(url.toString(), {method:'POST', body:form});
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.error||`HTTP ${res.status}`); }
  return res.json();
}

async function getModels() {
  const res = await fetch(`${BASE}/models`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function getStats() {
  const res = await fetch(`${BASE}/stats`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function checkHealth(retries=3) {
  for (let i=1; i<=retries; i++) {
    try {
      const res = await fetch(`${BASE}/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    } catch(e) {
      if (i===retries) throw new Error(`Offline: ${e.message}`);
      await new Promise(r=>setTimeout(r,1000));
    }
  }
}

function createWebSocket(onMsg, onErr, onClose) {
  const ws = new WebSocket(`${WS_BASE}/detect/webcam`);
  ws.addEventListener('open',    ()=>console.info('[VisionAI] WS open'));
  ws.addEventListener('message', e=>{ try{onMsg(JSON.parse(e.data));}catch(_){} });
  ws.addEventListener('error',   ()=>{ if(onErr) onErr(new Error('WS error')); });
  ws.addEventListener('close',   e=>{ if(onClose) onClose(e); });
  return ws;
}

function sendFrame(ws, b64, conf, iou, model) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({frame:b64, confidence:conf, iou, model}));
}

window.API = {detectImage, getModels, getStats, checkHealth, createWebSocket, sendFrame};