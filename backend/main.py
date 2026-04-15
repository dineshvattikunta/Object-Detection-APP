"""
main.py — VisionAI FastAPI Backend v4
YOLO-World powered — detects ANY object, not limited to 80 classes.
"""

import asyncio
import logging
import time
from collections import Counter
from datetime import datetime
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detector import DetectorEngine, MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionai.main")

app = FastAPI(title="VisionAI — YOLO-World", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

detector: Optional[DetectorEngine] = None

SESSION = {
    "total_detections":  0,
    "total_frames":      0,
    "class_counter":     Counter(),
    "confidence_sum":    0.0,
    "confidence_count":  0,
    "session_start":     datetime.utcnow().isoformat(),
}

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES     = 10 * 1024 * 1024


@app.on_event("startup")
async def startup():
    global detector
    logger.info("Loading YOLO-World — universal open-vocabulary detector …")
    detector = DetectorEngine("yolov8x-world")
    logger.info("YOLO-World ready on %s", detector.device)


@app.get("/health")
async def health():
    cuda = torch.cuda.is_available()
    return {
        "status":        "ok",
        "model_loaded":  detector is not None,
        "current_model": detector.model_name if detector else None,
        "gpu_available": cuda,
        "device":        detector.device if detector else "unknown",
        "cuda_version":  detector.cuda_version if detector else None,
        "gpu_name":      torch.cuda.get_device_name(0) if cuda else None,
    }


@app.get("/models")
async def models():
    return {
        "models":        DetectorEngine.available_models(),
        "current_model": detector.model_name if detector else None,
    }


@app.get("/stats")
async def stats():
    s   = SESSION
    top = s["class_counter"].most_common(1)[0][0] if s["class_counter"] else "—"
    avg = round(s["confidence_sum"] / s["confidence_count"] * 100, 1) if s["confidence_count"] else 0.0
    return {
        "total_detections":       s["total_detections"],
        "total_frames_processed": s["total_frames"],
        "most_detected_class":    top,
        "average_confidence":     avg,
        "session_start_time":     s["session_start"],
    }


@app.post("/detect/image")
async def detect_image(
    file:       UploadFile = File(...),
    confidence: float      = Query(0.20, ge=0.01, le=1.0),
    iou:        float      = Query(0.40, ge=0.01, le=1.0),
    model:      str        = Query("yolov8x-world"),
):
    if file.content_type not in ALLOWED_TYPES:
        return JSONResponse(status_code=400,
            content={"success": False,
                     "error": f"Unsupported type: {file.content_type}. Use JPG, PNG or WEBP."})

    data = await file.read()
    if len(data) > MAX_BYTES:
        return JSONResponse(status_code=413,
            content={"success": False, "error": "File too large — max 10 MB"})

    if not detector:
        return JSONResponse(status_code=503,
            content={"success": False, "error": "Detector not ready"})

    result = detector.detect_image(data, confidence, iou, model)
    if result["success"]:
        _update_stats(result["detections"])
    return result


@app.websocket("/detect/webcam")
async def webcam_ws(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected: %s", ws.client)
    ftimes: list[float] = []

    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
                continue

            b64   = data.get("frame", "")
            conf  = float(data.get("confidence", 0.20))
            iou   = float(data.get("iou", 0.40))
            model = data.get("model", "yolov8s-world")

            if not b64 or not detector:
                await ws.send_json({"error": "No frame or detector not ready"})
                continue

            result = detector.detect_frame(b64, conf, iou, model)

            now = time.perf_counter()
            ftimes.append(now)
            if len(ftimes) > 15: ftimes.pop(0)
            fps = 0.0
            if len(ftimes) >= 2:
                fps = round((len(ftimes)-1) / (ftimes[-1]-ftimes[0]), 1)
            result["fps"] = fps

            if result["success"]:
                SESSION["total_frames"] += 1
                _update_stats(result["detections"])

            await ws.send_json(result)

    except WebSocketDisconnect:
        logger.info("WS disconnected: %s", ws.client)
    except Exception as e:
        logger.exception("WS error: %s", e)
        try: await ws.send_json({"error": str(e)})
        except: pass
    

def _update_stats(dets: list[dict]) -> None:
    SESSION["total_detections"] += len(dets)
    for d in dets:
        SESSION["class_counter"][d["class_name"]] += 1
        SESSION["confidence_sum"]   += d["confidence"]
        SESSION["confidence_count"] += 1


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)