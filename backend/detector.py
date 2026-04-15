"""
detector.py — VisionAI YOLO-World Engine
Uses YOLO-Worldv2 + CLIP for open-vocabulary detection.
Detects ANY object — lion, deer, pen, spectacles, anything.
Not limited to 80 COCO classes.
"""

import time
import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("visionai.detector")

# ---------------------------------------------------------------------------
# Comprehensive class list — 500+ real world objects
# YOLO-World uses these as text prompts via CLIP
# Add or remove anything here — model will detect it
# ---------------------------------------------------------------------------
WORLD_CLASSES = [
    # People
    "person", "man", "woman", "child", "baby",
    # Animals - wild
    "lion", "tiger", "leopard", "cheetah", "jaguar",
    "elephant", "giraffe", "zebra", "rhinoceros", "hippopotamus",
    "bear", "wolf", "fox", "deer", "antelope", "gazelle",
    "monkey", "gorilla", "chimpanzee", "panda", "koala",
    "kangaroo", "crocodile", "alligator", "snake", "lizard",
    "eagle", "owl", "parrot", "flamingo", "penguin", "peacock",
    "dolphin", "shark", "whale", "octopus", "jellyfish",
    "camel", "moose", "bison", "hyena", "lynx", "cougar",
    "raccoon", "squirrel", "rabbit", "bat", "frog",
    # Animals - domestic
    "dog", "cat", "horse", "cow", "sheep", "pig", "goat",
    "chicken", "duck", "fish", "turtle", "hamster",
    # Vehicles
    "car", "truck", "bus", "motorcycle", "bicycle", "scooter",
    "van", "taxi", "ambulance", "police car", "fire truck",
    "tractor", "train", "subway", "airplane", "helicopter",
    "drone", "ship", "boat", "sailboat", "hot air balloon",
    "rocket", "forklift", "golf cart", "tram",
    # Electronics
    "mobile phone", "smartphone", "laptop", "computer monitor",
    "keyboard", "mouse", "tablet", "smartwatch", "camera",
    "headphones", "earbuds", "speaker", "microphone",
    "remote control", "television", "printer", "router",
    "USB drive", "charger", "calculator", "game controller",
    "projector", "VR headset",
    # Accessories & clothing
    "spectacles", "glasses", "sunglasses", "watch", "ring",
    "necklace", "bracelet", "earring", "hat", "cap", "helmet",
    "shoes", "boots", "sneakers", "sandals",
    "shirt", "pants", "dress", "jacket", "coat",
    "bag", "backpack", "purse", "wallet", "umbrella",
    "tie", "scarf", "gloves", "belt", "socks",
    # Furniture
    "chair", "sofa", "couch", "bed", "table", "desk",
    "shelf", "bookshelf", "cabinet", "wardrobe", "drawer",
    "mirror", "lamp", "fan", "air conditioner",
    # Kitchen & appliances
    "refrigerator", "washing machine", "microwave", "oven",
    "stove", "dishwasher", "toaster", "kettle", "blender",
    "coffee maker", "sink", "bathtub", "toilet",
    # Kitchen items
    "cup", "mug", "glass", "bottle", "bowl", "plate",
    "fork", "knife", "spoon", "chopsticks", "pot", "pan",
    "cutting board",
    # Office & stationery
    "pen", "pencil", "marker", "highlighter", "ruler",
    "scissors", "stapler", "tape", "eraser", "notebook",
    "book", "paper", "folder", "calculator", "clock",
    "whiteboard", "desk lamp", "calendar",
    # Food
    "apple", "banana", "orange", "grapes", "strawberry",
    "watermelon", "mango", "pineapple", "pizza", "burger",
    "sandwich", "hot dog", "taco", "sushi", "rice", "noodles",
    "bread", "cake", "cookie", "donut", "ice cream",
    "chocolate", "egg", "cheese", "milk", "coffee", "tea",
    "beer", "wine", "salad", "soup", "carrot", "broccoli",
    "tomato", "potato", "onion", "corn", "mushroom",
    # Sports
    "football", "basketball", "soccer ball", "tennis ball",
    "baseball", "volleyball", "golf ball", "tennis racket",
    "cricket bat", "skateboard", "surfboard", "snowboard",
    "bicycle helmet", "dumbbell", "yoga mat",
    # Tools
    "hammer", "screwdriver", "wrench", "drill", "saw",
    "toolbox", "ladder", "rope", "flashlight", "key", "lock",
    # Medical
    "stethoscope", "syringe", "medicine", "bandage",
    "wheelchair", "mask", "thermometer",
    # Nature
    "tree", "flower", "grass", "rock", "mountain",
    "cloud", "fire", "smoke", "plant", "cactus",
    # Buildings & places
    "house", "building", "bridge", "road", "door", "window",
    "stairs", "traffic light", "stop sign", "fire hydrant",
    "bench", "street light", "trash can", "mailbox",
    # Musical instruments
    "guitar", "piano", "violin", "drum", "trumpet",
    # Misc
    "box", "bag", "balloon", "candle", "clock",
    "painting", "trophy", "coin", "newspaper", "magazine",
    "toy", "doll", "puzzle", "globe", "compass",
    "binoculars", "telescope", "microscope",
]

MODEL_REGISTRY = {
    "yolov8x-world": {
        "filename": "yolov8x-worldv2.pt",
        "description": "YOLO-World XL — detects ANY object (recommended)",
        "speed_rating": 2,
        "accuracy_rating": 5,
    },
    "yolov8s-world": {
        "filename": "yolov8s-worldv2.pt",
        "description": "YOLO-World Small — fast, any object, best for webcam",
        "speed_rating": 5,
        "accuracy_rating": 3,
    },
    "yolov8x": {
        "filename": "yolov8x.pt",
        "description": "YOLOv8 XL — standard 80 class detection",
        "speed_rating": 2,
        "accuracy_rating": 4,
    },
    "yolov8l": {
        "filename": "yolov8l.pt",
        "description": "YOLOv8 Large — standard 80 class detection",
        "speed_rating": 3,
        "accuracy_rating": 3,
    },
}

DEFAULT_MODEL = "yolov8x-world"
BOX_ALPHA    = 0.35
BORDER_THICK = 2
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.55
FONT_THICK   = 1
LABEL_PAD    = 5


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _class_color(name: str) -> tuple[int, int, int]:
    import hashlib
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = h % 180
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def _hex(name: str) -> str:
    b, g, r = _class_color(name)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# DetectorEngine
# ---------------------------------------------------------------------------
class DetectorEngine:

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = ""
        self._model      = None
        self._is_world   = False
        self._device     = self._resolve_device()
        self._half       = self._device != "cpu"
        self._load_model(model_name)

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("GPU: %s — CUDA active", torch.cuda.get_device_name(0))
                return "cuda"
        except Exception as e:
            logger.warning("CUDA check failed: %s", e)
        logger.info("Running on CPU")
        return "cpu"

    def _load_model(self, model_name: str) -> None:
        key = model_name.lower().strip()
        if key not in MODEL_REGISTRY:
            logger.warning("Unknown model '%s', using '%s'", key, DEFAULT_MODEL)
            key = DEFAULT_MODEL

        info     = MODEL_REGISTRY[key]
        filename = info["filename"]
        path     = Path(__file__).parent / "models" / filename
        src      = str(path) if path.exists() else filename

        self._is_world = "world" in key

        if self._is_world:
            from ultralytics import YOLOWorld
            logger.info("Loading YOLO-World model: %s …", key)
            self._model = YOLOWorld(src)
            # Set our comprehensive class list as detection targets
            self._model.set_classes(WORLD_CLASSES)
            logger.info("YOLO-World ready with %d classes", len(WORLD_CLASSES))
        else:
            from ultralytics import YOLO
            logger.info("Loading YOLO model: %s …", key)
            self._model = YOLO(src)

        self._model.to(self._device)
        self._model_name = key
        logger.info("Model '%s' ready on %s", key, self._device)

    def switch_model(self, name: str) -> None:
        if name != self._model_name:
            self._load_model(name)

    # ── inference ──────────────────────────────────────────
    def _infer(self, img: np.ndarray, conf: float, iou: float, augment: bool):
        kwargs = dict(
            source       = img,
            conf         = conf,
            iou          = iou,
            agnostic_nms = True,
            max_det      = 100,
            verbose      = False,
            device       = self._device,
        )
        # YOLO-World doesn't support half or augment
        if not self._is_world:
            kwargs["half"]    = self._half
            kwargs["augment"] = augment

        return self._model.predict(**kwargs)[0]

    # ── annotation ─────────────────────────────────────────
    def _annotate(self, img: np.ndarray, dets: list[dict]) -> np.ndarray:
        out     = img.copy()
        overlay = img.copy()
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            col   = _class_color(d["class_name"])
            label = f"{d['class_name']} {d['confidence']*100:.1f}%"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(out,     (x1, y1), (x2, y2), col, BORDER_THICK)

            (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
            lx1 = x1
            ly1 = max(y1 - th - 2*LABEL_PAD, 0)
            lx2 = x1 + tw + 2*LABEL_PAD
            ly2 = max(y1, th + 2*LABEL_PAD)
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), col, -1)
            cv2.putText(out, label,
                        (lx1+LABEL_PAD, ly2-LABEL_PAD-bl//2),
                        FONT, FONT_SCALE, (255,255,255), FONT_THICK, cv2.LINE_AA)

        cv2.addWeighted(overlay, BOX_ALPHA, out, 1-BOX_ALPHA, 0, out)
        return out

    # ── public API ─────────────────────────────────────────
    def detect_image(self, image_bytes: bytes,
                     confidence: float = 0.20,
                     iou: float = 0.40,
                     model_name: Optional[str] = None) -> dict:
        try:
            if model_name: self.switch_model(model_name)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("Cannot decode image")
            h, w  = img.shape[:2]

            t0  = time.perf_counter()
            res = self._infer(img, confidence, iou, augment=True)
            ms  = (time.perf_counter() - t0) * 1000

            dets = self._parse(res)
            ann  = self._annotate(img, dets)

            return {
                "success":        True,
                "detections":     dets,
                "total_count":    len(dets),
                "inference_ms":   round(ms, 2),
                "model_used":     self._model_name,
                "image_width":    w,
                "image_height":   h,
                "annotated_image": self._b64(ann),
                "error":          None,
            }
        except Exception as e:
            logger.exception("detect_image error: %s", e)
            return {"success":False,"detections":[],"total_count":0,
                    "inference_ms":0,"model_used":self._model_name,
                    "image_width":0,"image_height":0,
                    "annotated_image":"","error":str(e)}

    def detect_frame(self, frame_b64: str,
                     confidence: float = 0.20,
                     iou: float = 0.40,
                     model_name: Optional[str] = None) -> dict:
        try:
            if model_name: self.switch_model(model_name)
            raw   = base64.b64decode(frame_b64)
            nparr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: raise ValueError("Cannot decode frame")

            t0  = time.perf_counter()
            res = self._infer(frame, confidence, iou, augment=False)
            ms  = (time.perf_counter() - t0) * 1000

            dets = self._parse(res)
            ann  = self._annotate(frame, dets)

            return {
                "success":        True,
                "detections":     dets,
                "total_count":    len(dets),
                "inference_ms":   round(ms, 2),
                "annotated_frame": self._b64(ann),
                "error":          None,
            }
        except Exception as e:
            logger.exception("detect_frame error: %s", e)
            return {"success":False,"detections":[],"total_count":0,
                    "inference_ms":0,"annotated_frame":"","error":str(e)}

    @staticmethod
    def _parse(result) -> list[dict]:
        dets = []
        if result.boxes is None: return dets
        names = result.names
        for i in range(len(result.boxes)):
            cls  = int(result.boxes.cls[i].item())
            name = names.get(cls, str(cls))
            conf = float(result.boxes.conf[i].item())
            xyxy = result.boxes.xyxy[i].cpu().numpy()
            dets.append({
                "class_name": name,
                "class_id":   cls,
                "confidence": round(conf, 4),
                "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                "x2": int(xyxy[2]), "y2": int(xyxy[3]),
                "color_hex": _hex(name),
            })
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        return dets

    @staticmethod
    def _b64(img: np.ndarray, q: int = 90) -> str:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok: raise RuntimeError("imencode failed")
        return base64.b64encode(buf.tobytes()).decode()

    # ── properties ─────────────────────────────────────────
    @property
    def model_name(self)   -> str:  return self._model_name
    @property
    def device(self)       -> str:  return self._device
    @property
    def is_gpu(self)       -> bool: return self._device != "cpu"
    @property
    def cuda_version(self) -> Optional[str]:
        try:
            import torch
            return torch.version.cuda if torch.cuda.is_available() else None
        except: return None

    @staticmethod
    def available_models() -> list[dict]:
        return [
            {"name": k, "filename": v["filename"],
             "description": v["description"],
             "speed_rating": v["speed_rating"],
             "accuracy_rating": v["accuracy_rating"]}
            for k, v in MODEL_REGISTRY.items()
        ]