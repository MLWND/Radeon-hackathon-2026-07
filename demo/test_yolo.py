"""
RoboPilot YOLO Detection Test
验证 YOLO 目标检测
"""
from ultralytics import YOLO
import numpy as np

# Load model
model = YOLO("/root/.config/Ultralytics/yolov8n.pt")
print("YOLO model loaded successfully")
print(f"Classes: {list(model.names.values())[:5]}...")

# Test inference with random image
dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
results = model(dummy_img)
print(f"Inference OK - detected {len(results[0].boxes)} objects")
