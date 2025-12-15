import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model (pretrained on COCO)
model = YOLO("yolov8n.pt")

# COCO class IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck

def detect_vehicles(frame):
    start_time = time.time()
    results = model(frame, stream=True)

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))

    fps = 1.0 / (time.time() - start_time)
    return detections, fps
