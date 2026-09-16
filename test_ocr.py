print("Step 1: importing pytesseract...")
try:
    import pytesseract
    print("  OK - pytesseract imported")
except ImportError as e:
    print(f"  FAILED: {e}")
    print("  FIX: run  pip install pytesseract")
    exit()

print("Step 2: pointing to Tesseract engine...")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("Step 3: checking installed languages...")
try:
    langs = pytesseract.get_languages()
    print(f"  Installed languages: {langs}")
    if "kor" not in langs:
        print("  WARNING: 'kor' not installed -- Korean plates won't read correctly")
except Exception as e:
    print(f"  FAILED: {e}")
    print("  FIX: check the tesseract_cmd path above matches where you installed it")
    exit()

print("Step 4: running OCR on a simple test image...")
import numpy as np
import cv2
img = np.ones((50, 200), dtype=np.uint8) * 255
cv2.putText(img, "1234", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,), 3)
text = pytesseract.image_to_string(img, config="--psm 7")
print(f"  OCR read: '{text.strip()}'")
if "1234" in text:
    print("  SUCCESS -- OCR is working correctly!")
else:
    print("  OCR ran but didn't read '1234' correctly -- something's off with the install")