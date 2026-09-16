import cv2
from utils.lane_detection import detect_lanes
from utils.detection import VehicleDetector, select_nearest_vehicles
from utils.distance import estimate_distance
from utils.plate_detection import PlateDetector

VIDEO_PATH = "dashcam_video_3.mp4"
DETECT_WIDTH = 480
DISPLAY_WIDTH = 960
DETECT_EVERY_N_FRAMES = 3          # vehicle boxes + distance refresh cadence
PLATE_DETECT_EVERY_N_FRAMES = 15   # plate OCR is expensive -- refresh far less often
NUM_NEAREST_VEHICLES = 3


def place_label_no_overlap(placed_boxes, x, y, text_w, text_h):
    box = [x, y - text_h - 8, x + text_w + 8, y + 4]
    moved = True
    while moved:
        moved = False
        for (px1, py1, px2, py2) in placed_boxes:
            overlap_x = box[0] < px2 and box[2] > px1
            overlap_y = box[1] < py2 and box[3] > py1
            if overlap_x and overlap_y:
                shift = py2 - box[1] + 4
                box[1] += shift
                box[3] += shift
                moved = True
    placed_boxes.append(tuple(box))
    return box[0], box[3] - 4


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info -> FPS: {fps}, Resolution: {width}x{height}")

    detector = VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.4)
    plate_detector = PlateDetector()

    cv2.namedWindow("Lane + Vehicle + Distance + Plates - Press Q to Quit", cv2.WINDOW_NORMAL)

    frame_num = 0
    last_detections = []
    last_plate_boxes = []  # (x1, y1, x2, y2, text) in ORIGINAL frame coordinates

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Stopped at frame {frame_num}. ret={ret}")
            break
        frame_num += 1

        orig_h, orig_w = frame.shape[:2]

        if frame_num % DETECT_EVERY_N_FRAMES == 0:
            detect_scale = DETECT_WIDTH / orig_w
            detect_frame = cv2.resize(frame, (DETECT_WIDTH, int(orig_h * detect_scale)))
            last_detections = detector.detect(detect_frame)

        # plate OCR runs on its own, much slower cadence -- this is the main lag fix
        if frame_num % PLATE_DETECT_EVERY_N_FRAMES == 0:
            nearest_dets = select_nearest_vehicles(last_detections, n=NUM_NEAREST_VEHICLES)
            to_orig = orig_w / DETECT_WIDTH
            last_plate_boxes = []
            for (x1, y1, x2, y2, class_name, conf) in nearest_dets:
                ox1, oy1, ox2, oy2 = (int(x1 * to_orig), int(y1 * to_orig),
                                       int(x2 * to_orig), int(y2 * to_orig))
                pad = 10
                cx1, cy1 = max(0, ox1 - pad), max(0, oy1 - pad)
                cx2, cy2 = min(orig_w, ox2 + pad), min(orig_h, oy2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                plates = plate_detector.detect(crop)
                for (px1, py1, px2, py2, text) in plates:
                    last_plate_boxes.append((cx1 + px1, cy1 + py1, cx1 + px2, cy1 + py2, text))

        display_scale = DISPLAY_WIDTH / orig_w
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, int(orig_h * display_scale)))

        box_scale = DISPLAY_WIDTH / DETECT_WIDTH
        display_detections = []
        for (x1, y1, x2, y2, class_name, conf) in last_detections:
            dx1, dy1, dx2, dy2 = (int(x1 * box_scale), int(y1 * box_scale),
                                   int(x2 * box_scale), int(y2 * box_scale))
            display_detections.append((dx1, dy1, dx2, dy2, class_name, conf))

        vehicle_boxes_only = [(d[0], d[1], d[2], d[3]) for d in display_detections]
        display_frame, left_line, right_line = detect_lanes(display_frame, debug=False,
                                                              vehicle_boxes=vehicle_boxes_only)

        nearest = select_nearest_vehicles(display_detections, n=NUM_NEAREST_VEHICLES)
        nearest_ids = set(id(d) for d in nearest)
        for det in display_detections:
            if id(det) not in nearest_ids:
                x1, y1, x2, y2, class_name, conf = det
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 180, 0), 1)

        placed_label_boxes = []
        for (x1, y1, x2, y2, class_name, conf) in nearest:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            pixel_width = (x2 - x1) / box_scale
            distance = estimate_distance(pixel_width, class_name)

            label = f"{class_name}"
            if distance is not None:
                label += f" {distance}m"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            draw_x, draw_y = place_label_no_overlap(placed_label_boxes, x1, y1 - 8, tw, th)
            cv2.rectangle(display_frame, (draw_x, draw_y - th - 4), (draw_x + tw + 6, draw_y + 2), (0, 0, 0), -1)
            cv2.putText(display_frame, label, (draw_x + 3, draw_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for (ox1, oy1, ox2, oy2, text) in last_plate_boxes:
            dx1, dy1, dx2, dy2 = (int(ox1 * display_scale), int(oy1 * display_scale),
                                   int(ox2 * display_scale), int(oy2 * display_scale))
            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)
            label = text if text else "Plate"
            cv2.putText(display_frame, label, (dx1, max(0, dy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Lane + Vehicle + Distance + Plates - Press Q to Quit", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit by user.")
            break

    print(f"Total frames processed: {frame_num}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()