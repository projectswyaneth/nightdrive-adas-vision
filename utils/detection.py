import torch
from ultralytics import YOLO

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"VehicleDetector running on: {self.device}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False, device=self.device)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in VEHICLE_CLASSES and conf >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = VEHICLE_CLASSES[cls_id]
                detections.append((x1, y1, x2, y2, class_name, conf))

        return detections


def select_nearest_vehicles(detections, n=3):
    """
    Pick the N closest vehicles -- closer to the camera means lower on
    screen (bigger y2) and generally a bigger box. Sorting by y2 (bottom
    edge) descending is a simple, reliable proxy for "closest" without
    needing lane information.
    """
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d[3], reverse=True)
    return sorted_dets[:n]