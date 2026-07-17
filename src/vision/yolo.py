"""
Module 2: Optional YOLO Detection (Fast Mode backend)
NOTE: Not in the core pipeline. Qwen3-VL is the single perception backbone.
Kept for optional dual-mode (fast detection).
import numpy as np
import time
from typing import List, Dict, Optional


class YOLOWrapper:
    def __init__(self, model_path: str = "/root/.config/Ultralytics/yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.last_detections = []
        self.last_inference_time = 0.0

    def detect(self, image: np.ndarray, confidence: float = 0.3) -> List[Dict]:
        start = time.time()
        results = self.model(image, conf=confidence)
        self.last_inference_time = (time.time() - start) * 1000

        detections = []
        for det in results:
            boxes = det.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = det.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "area": (x2 - x1) * (y2 - y1),
                })

        self.last_detections = detections
        return detections

    def detect_by_class(self, image: np.ndarray, target_class: str) -> List[Dict]:
        all_dets = self.detect(image)
        return [d for d in all_dets if d["class"] == target_class]

    def get_largest(self, image: np.ndarray) -> Optional[Dict]:
        dets = self.detect(image)
        if not dets:
            return None
        return max(dets, key=lambda d: d["area"])

    def get_inference_time(self) -> float:
        return self.last_inference_time
