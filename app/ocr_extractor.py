# app/ocr_extractor.py

import cv2
import numpy as np
import pytesseract
from PIL import Image


def extract_text_from_image(image_path: str) -> str:
    """
    Reads the screenshot at `image_path`,
    optionally preprocesses it, then runs Tesseract OCR to extract text.
    """
    # 1) Load image via OpenCV
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print(f"Warning: Could not read image at {image_path}")
        return ""

    # 2) (Optional) Convert to grayscale and do simple threshold
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Increase contrast or apply threshold if desired
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # 3) Convert back to PIL Image for pytesseract
    pil_img = Image.fromarray(gray)

    # 4) Extract text
    text = pytesseract.image_to_string(pil_img)
    return text.strip()
