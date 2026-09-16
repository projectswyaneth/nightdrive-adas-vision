import cv2
import numpy as np
import math

prev_left_line = None
prev_right_line = None

ROI_TOP_Y = 0.68
ROI_BOTTOM_Y = 0.97
ROI_TOP_L_X = 0.40
ROI_TOP_R_X = 0.62
ROI_BOT_L_X = 0.05
ROI_BOT_R_X = 0.98


def filter_lane_colors(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    bright_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=-15
    )
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    yellow_mask = cv2.inRange(hls, (15, 30, 100), (35, 204, 255))
    combined_mask = cv2.bitwise_or(bright_mask, yellow_mask)
    return blurred, combined_mask


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * ROI_BOT_L_X), int(height * ROI_BOTTOM_Y)),
        (int(width * ROI_TOP_L_X), int(height * ROI_TOP_Y)),
        (int(width * ROI_TOP_R_X), int(height * ROI_TOP_Y)),
        (int(width * ROI_BOT_R_X), int(height * ROI_BOTTOM_Y)),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def mask_out_vehicles(edges, vehicle_boxes):
    """
    Blank out each detected vehicle's box in the edge map so the car's own
    body (bright rear window, bumper trim, reflections) can never be
    mistaken for a lane line by Hough -- this is what was letting the
    fitted line's real-data range extend onto the car instead of stopping
    at the road in front of it.
    """
    if not vehicle_boxes:
        return edges
    for (x1, y1, x2, y2) in vehicle_boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(edges.shape[1], int(x2)), min(edges.shape[0], int(y2))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(edges, (x1, y1), (x2, y2), 0, -1)
    return edges


def make_line(fit_list, ys_seen, height, y_bottom, mid_x, is_left):
    slope, intercept = np.median(fit_list, axis=0)
    y1 = y_bottom
    roi_ceiling = int(height * ROI_TOP_Y)
    topmost_real_y = int(min(ys_seen))
    y2 = max(roi_ceiling, topmost_real_y)
    y2 = min(y2, y1 - 10)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    margin = 5
    if is_left:
        x1 = min(x1, int(mid_x) - margin)
        x2 = min(x2, int(mid_x) - margin)
    else:
        x1 = max(x1, int(mid_x) + margin)
        x2 = max(x2, int(mid_x) + margin)

    return (x1, y1, x2, y2)


def smooth_line(new_line, prev_line, alpha=0.2):
    if new_line is None:
        return prev_line
    if prev_line is None:
        return new_line
    if abs(new_line[0] - prev_line[0]) > 150:
        return prev_line
    return tuple(
        int(alpha * n + (1 - alpha) * p)
        for n, p in zip(new_line, prev_line)
    )


def average_slope_intercept(lines, width, height, mid_x):
    left_fit, left_ys = [], []
    right_fit, right_ys = [], []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4 or abs(slope) > 2.5:
            continue

        intercept = y1 - slope * x1
        avg_x = (x1 + x2) / 2

        if slope < 0 and avg_x < mid_x:
            left_fit.append((slope, intercept))
            left_ys.extend([y1, y2])
        elif slope > 0 and avg_x > mid_x:
            right_fit.append((slope, intercept))
            right_ys.extend([y1, y2])

    y_bottom = int(height * ROI_BOTTOM_Y)

    left_line = make_line(left_fit, left_ys, height, y_bottom, mid_x, is_left=True) if left_fit else None
    right_line = make_line(right_fit, right_ys, height, y_bottom, mid_x, is_left=False) if right_fit else None

    return left_line, right_line


def lane_x_at_y(line, y):
    if line is None:
        return None
    x1, y1, x2, y2 = line
    if y2 == y1:
        return x1
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def find_crossing_y(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    if ly2 == ly1 or ry2 == ry1:
        return None
    slope_l = (lx2 - lx1) / (ly2 - ly1)
    slope_r = (rx2 - rx1) / (ry2 - ry1)
    denom = slope_l - slope_r
    if abs(denom) < 1e-6:
        return None
    return (rx1 - lx1 + slope_l * ly1 - slope_r * ry1) / denom


def prevent_crossing(left_line, right_line, margin=15):
    if left_line is None or right_line is None:
        return left_line, right_line

    y_cross = find_crossing_y(left_line, right_line)
    if y_cross is None:
        return left_line, right_line

    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line

    top_y = min(ly2, ry2)
    bottom_y = max(ly1, ry1)

    if not (top_y < y_cross < bottom_y):
        return left_line, right_line

    new_top_y = int(y_cross) + margin
    new_top_y = min(new_top_y, bottom_y - 10)

    new_lx2 = int(lane_x_at_y(left_line, new_top_y))
    new_rx2 = int(lane_x_at_y(right_line, new_top_y))

    return (lx1, ly1, new_lx2, new_top_y), (rx1, ry1, new_rx2, new_top_y)


def draw_dashed_line(img, pt1, pt2, color, thickness=8, dash_len=22, gap_len=16):
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    step = dash_len + gap_len
    n_dashes = int(dist / step) + 1
    for i in range(n_dashes):
        start_t = min(1.0, (i * step) / dist)
        end_t = min(1.0, (i * step + dash_len) / dist)
        sx = int(x1 + (x2 - x1) * start_t)
        sy = int(y1 + (y2 - y1) * start_t)
        ex = int(x1 + (x2 - x1) * end_t)
        ey = int(y1 + (y2 - y1) * end_t)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def detect_lanes(frame, debug=False, vehicle_boxes=None):
    global prev_left_line, prev_right_line

    height, width = frame.shape[:2]

    if prev_left_line is not None and prev_right_line is not None:
        current_mid_x = (prev_left_line[0] + prev_right_line[0]) / 2
    else:
        current_mid_x = width / 2

    blurred_gray, color_mask = filter_lane_colors(frame)

    edges = cv2.Canny(blurred_gray, 50, 150)
    edges = cv2.bitwise_and(edges, color_mask)
    roi_edges = region_of_interest(edges)
    roi_edges = mask_out_vehicles(roi_edges, vehicle_boxes)

    lines = cv2.HoughLinesP(
        roi_edges, rho=2, theta=np.pi / 180,
        threshold=25, minLineLength=20, maxLineGap=150
    )

    raw_left, raw_right = average_slope_intercept(lines, width, height, current_mid_x)

    left_line = smooth_line(raw_left, prev_left_line)
    right_line = smooth_line(raw_right, prev_right_line)

    left_line, right_line = prevent_crossing(left_line, right_line)

    prev_left_line = left_line
    prev_right_line = right_line

    overlay = frame.copy()

    if left_line is not None and right_line is not None:
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        lane_polygon = np.array([[
            (lx1, ly1), (lx2, ly2), (rx2, ry2), (rx1, ry1)
        ]], np.int32)
        cv2.fillPoly(overlay, lane_polygon, (0, 255, 0))

    combined = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    if left_line is not None:
        draw_dashed_line(combined, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 8)
    if right_line is not None:
        draw_dashed_line(combined, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 8)

    if debug:
        cv2.imshow("DEBUG - Bright/Adaptive Mask", color_mask)
        cv2.imshow("DEBUG - Canny Edges", edges)
        cv2.imshow("DEBUG - ROI Masked Edges", roi_edges)

    return combined, left_line, right_line