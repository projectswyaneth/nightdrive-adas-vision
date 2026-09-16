import cv2
import re

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    print("WARNING: pytesseract not installed.")

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

# Real Korean plate format: 2-3 digits, ONE Hangul character, 4 digits.
# Back to strict matching -- "any 3+ digits" was accepting garbage.
_PLATE_PATTERN = re.compile(r"^\d{2,3}[\uAC00-\uD7A3]?\d{4}$")

# Below this size, a crop just doesn't have enough real pixel detail for
# OCR to ever succeed -- skip it entirely rather than wasting time trying.
MIN_OCR_CROP_WIDTH = 70
MIN_OCR_CROP_HEIGHT = 25


class PlateDetector:
    def __init__(self, scale_factor=1.05, min_neighbors=4, min_size=(35, 12)):
        self.cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self.cascade.empty():
            raise IOError(f"Could not load cascade from {_CASCADE_PATH}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        raw_boxes = self.cascade.detectMultiScale(
            enhanced, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size,
        )

        results = []
        for (x, y, w, h) in raw_boxes:
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Skip OCR entirely on crops too small to ever read correctly --
            # this is the main speed AND accuracy fix. No point spending
            # 100+ ms on Tesseract for a 20px-wide smudge.
            if not _OCR_AVAILABLE or w < MIN_OCR_CROP_WIDTH or h < MIN_OCR_CROP_HEIGHT:
                results.append((x1, y1, x2, y2, None))
                continue

            text = self._read_text(crop[y1:y2, x1:x2])
            results.append((x1, y1, x2, y2, text))
        return results

    def _read_text(self, plate_crop):
        if plate_crop.size == 0:
            return None

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        scale = 2  # reduced from 4 -- still enough for OCR, much faster
        gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                           interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            raw_text = pytesseract.image_to_string(thresh, lang="kor+eng", config="--psm 7")
        except Exception:
            return None

        cleaned = re.sub(r"[^\d\uAC00-\uD7A3]", "", raw_text)

        # strict match only -- reject anything that doesn't look like a
        # real Korean plate, instead of accepting "close enough" garbage
        if _PLATE_PATTERN.match(cleaned):
            return cleaned
        return None