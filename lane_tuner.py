import cv2
import numpy as np

VIDEO_PATH = "dashcam_video_3.mp4"
DISPLAY_WIDTH = 640

WINDOW = "Lane Tuner - SPACE=pause  Q=quit"


def nothing(_):
    pass


def build_trackbars():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 520, 560)

    cv2.createTrackbar("ROI top Y%", WINDOW, 62, 100, nothing)
    cv2.createTrackbar("ROI bottom Y%", WINDOW, 90, 100, nothing)
    cv2.createTrackbar("ROI top-L X%", WINDOW, 40, 100, nothing)
    cv2.createTrackbar("ROI top-R X%", WINDOW, 62, 100, nothing)
    cv2.createTrackbar("ROI bot-L X%", WINDOW, 5, 100, nothing)
    cv2.createTrackbar("ROI bot-R X%", WINDOW, 98, 100, nothing)

    # adaptive threshold (this replaces the old fixed white-brightness slider)
    cv2.createTrackbar("CLAHE clip x10", WINDOW, 30, 100, nothing)      # clipLimit = value/10
    cv2.createTrackbar("Block size (odd)", WINDOW, 25, 99, nothing)     # neighborhood size
    cv2.createTrackbar("C (offset+50)", WINDOW, 35, 100, nothing)       # actual C = value-50

    cv2.createTrackbar("Canny low", WINDOW, 50, 255, nothing)
    cv2.createTrackbar("Canny high", WINDOW, 150, 255, nothing)
    cv2.createTrackbar("Hough thresh", WINDOW, 25, 200, nothing)
    cv2.createTrackbar("Min line len", WINDOW, 20, 200, nothing)
    cv2.createTrackbar("Max line gap", WINDOW, 150, 300, nothing)


def get_params():
    g = lambda name: cv2.getTrackbarPos(name, WINDOW)
    block = g("Block size (odd)")
    if block % 2 == 0:
        block += 1
    block = max(3, block)
    return {
        "roi_top_y": g("ROI top Y%") / 100,
        "roi_bottom_y": g("ROI bottom Y%") / 100,
        "roi_top_l_x": g("ROI top-L X%") / 100,
        "roi_top_r_x": g("ROI top-R X%") / 100,
        "roi_bot_l_x": g("ROI bot-L X%") / 100,
        "roi_bot_r_x": g("ROI bot-R X%") / 100,
        "clahe_clip": max(0.1, g("CLAHE clip x10") / 10),
        "block_size": block,
        "C": g("C (offset+50)") - 50,
        "canny_low": g("Canny low"),
        "canny_high": g("Canny high"),
        "hough_thresh": max(1, g("Hough thresh")),
        "min_line_len": max(1, g("Min line len")),
        "max_line_gap": max(1, g("Max line gap")),
    }


def make_roi_polygon(width, height, p):
    return np.array([[
        (int(width * p["roi_bot_l_x"]), int(height * p["roi_bottom_y"])),
        (int(width * p["roi_top_l_x"]), int(height * p["roi_top_y"])),
        (int(width * p["roi_top_r_x"]), int(height * p["roi_top_y"])),
        (int(width * p["roi_bot_r_x"]), int(height * p["roi_bottom_y"])),
    ]], np.int32)


def process(frame, p):
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    bright_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=p["block_size"], C=p["C"]
    )

    edges = cv2.Canny(blurred, p["canny_low"], p["canny_high"])
    edges = cv2.bitwise_and(edges, bright_mask)

    polygon = make_roi_polygon(width, height, p)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, polygon, 255)
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        roi_edges, 2, np.pi / 180, p["hough_thresh"],
        minLineLength=p["min_line_len"], maxLineGap=p["max_line_gap"]
    )

    overlay = frame.copy()
    cv2.polylines(overlay, polygon, True, (0, 0, 255), 2)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return overlay, bright_mask, edges, roi_edges


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open {VIDEO_PATH}")
        return

    build_trackbars()
    paused = False
    frame = None
    last_params = None

    while True:
        if not paused or frame is None:
            ret, raw = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            scale = DISPLAY_WIDTH / raw.shape[1]
            frame = cv2.resize(raw, (DISPLAY_WIDTH, int(raw.shape[0] * scale)))

        params = get_params()
        last_params = params
        overlay, mask, edges, roi_edges = process(frame, params)

        cv2.imshow(WINDOW, overlay)
        cv2.imshow("Bright/Adaptive Mask", mask)
        cv2.imshow("Canny Edges", edges)
        cv2.imshow("ROI Edges (what Hough actually sees)", roi_edges)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    if last_params:
        print("\n=== Copy these into utils/lane_detection.py ===")
        p = last_params
        print(f"ROI_TOP_Y = {p['roi_top_y']:.2f}")
        print(f"ROI_BOTTOM_Y = {p['roi_bottom_y']:.2f}")
        print(f"ROI_TOP_L_X = {p['roi_top_l_x']:.2f}")
        print(f"ROI_TOP_R_X = {p['roi_top_r_x']:.2f}")
        print(f"ROI_BOT_L_X = {p['roi_bot_l_x']:.2f}")
        print(f"ROI_BOT_R_X = {p['roi_bot_r_x']:.2f}")
        print(f"CLAHE clipLimit = {p['clahe_clip']:.1f}")
        print(f"adaptiveThreshold blockSize = {p['block_size']}, C = {p['C']}")
        print(f"Canny: {p['canny_low']} - {p['canny_high']}")
        print(f"Hough: thresh={p['hough_thresh']}, min_len={p['min_line_len']}, max_gap={p['max_line_gap']}")


if __name__ == "__main__":
    main()