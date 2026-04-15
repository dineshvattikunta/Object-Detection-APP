# VisionAI 🎯

Real-time object detection powered by **YOLOv8** + **CUDA**. Detect 80 COCO object classes in images and live webcam streams, with a GPU-accelerated FastAPI backend and a pure HTML/CSS/JS frontend — no Node, no npm, no build steps.

---

## Features

- **YOLOv8 Large/Medium/Small** — switchable at runtime
- **NVIDIA GPU auto-detection** — fp16 inference, falls back to CPU
- **Image detection** — drag & drop JPG/PNG/WEBP, side-by-side annotated result
- **Live webcam** — WebSocket streaming at ~10 fps, canvas overlay
- **Export** — annotated JPEG download, CSV, JSON
- **Session stats** — live counters on the homepage
- **Dark / light mode** — persisted to localStorage
- **Keyboard shortcuts** — `S` start/stop webcam · `U` go to image page
- Zero external frontend dependencies — opens directly from the filesystem

---

## Requirements

- Windows 10/11
- Python 3.11
- NVIDIA GPU with CUDA 12.1 drivers installed
- Chrome or Edge (for `getUserMedia` webcam access)

---

## Installation

### 1. Clone / extract the project

```
VisionAI/
├── backend/
└── frontend/
```

### 2. Install PyTorch with CUDA

```cmd
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining backend dependencies

```cmd
cd VisionAI\backend
pip install -r requirements.txt
```

---

## Running the Backend

```cmd
cd VisionAI\backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On first run, Ultralytics will automatically download `yolov8l.pt` (~87 MB) from the internet and cache it. Subsequent starts are instant.

Verify the backend is running:
```
http://localhost:8000/health
```

---

## Opening the Frontend

No server needed. Just open any HTML file directly in Chrome:

```
VisionAI\frontend\index.html     → Homepage + stats
VisionAI\frontend\image.html     → Image detection
VisionAI\frontend\webcam.html    → Live webcam
VisionAI\frontend\about.html     → Info & model details
```

Double-click `index.html` or drag it into Chrome.

---

## Keyboard Shortcuts (Webcam page)

| Key | Action |
|-----|--------|
| `S` | Start / Stop detection |
| `U` | Go to Image detection page |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Backend status, GPU info |
| `GET`  | `/models` | Available models list |
| `GET`  | `/stats`  | Session statistics |
| `POST` | `/detect/image` | Image detection (multipart) |
| `WS`   | `/detect/webcam` | Live frame detection |

### POST /detect/image — query params

| Param | Default | Description |
|-------|---------|-------------|
| `confidence` | `0.50` | Detection threshold (0.01–0.99) |
| `iou` | `0.45` | NMS IOU threshold |
| `model` | `yolov8l` | Model name |

---

## Folder Structure

```
VisionAI/
├── backend/
│   ├── main.py           FastAPI app (REST + WebSocket)
│   ├── detector.py       DetectorEngine (YOLOv8 wrapper)
│   ├── requirements.txt
│   └── models/           (auto-populated by Ultralytics)
├── frontend/
│   ├── index.html        Homepage + stats
│   ├── image.html        Image detection
│   ├── webcam.html       Live webcam
│   ├── about.html        Info page
│   ├── css/
│   │   ├── style.css     Full design system
│   │   └── animations.css  All keyframes
│   └── js/
│       ├── api.js        API + WebSocket module
│       ├── ui.js         Shared UI utilities
│       ├── image.js      Image page logic
│       └── webcam.js     Webcam page logic
└── README.md
```

---

## Model Accuracy Notes

| Model | mAP COCO val | Params | Recommended for |
|-------|-------------|--------|-----------------|
| yolov8l | 52.9% | 43.7M | Static images (default) |
| yolov8m | 50.2% | 25.9M | Balanced |
| yolov8s | 44.9% | 11.2M | Live webcam |

- `augment=True` is enabled for image inference (test-time augmentation) for maximum accuracy
- `augment=False` for webcam frames to maintain real-time speed
- `agnostic_nms=True` reduces duplicate detections across classes
- `max_det=100` caps detections per frame

---

## Known Limitations

- WebSocket requires backend on `localhost:8000` — cannot use across different machines without changing `BASE_URL` in `api.js`
- Chrome blocks `getUserMedia` on `file://` in some configurations — if webcam doesn't work, serve frontend with `python -m http.server 3000` from the `frontend/` folder and open `http://localhost:3000`
- First model load downloads weights from Ultralytics CDN — requires internet on first run
- fp16 inference requires a CUDA-capable NVIDIA GPU; CPU fallback is significantly slower (~5–15× depending on image size)
