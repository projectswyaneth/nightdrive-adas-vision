REAL_WIDTHS = {
    "car": 1.8,
    "truck": 2.5,
    "bus": 2.5,
    "motorcycle": 0.8,
}

# Calibrated using a real reference point from the actual video:
# a car ~10m away measured pixel_width ~40 at DETECT_WIDTH=640.
# focal_length = (true_distance * pixel_width) / real_width = (10 * 40) / 1.8
ASSUMED_FOCAL_LENGTH = 222


def estimate_distance(pixel_width, class_name):
    if pixel_width <= 0:
        return None

    real_width = REAL_WIDTHS.get(class_name, 1.8)
    distance = (real_width * ASSUMED_FOCAL_LENGTH) / pixel_width

    if distance < 1.5 or distance > 200:
        return None

    return round(distance, 1)