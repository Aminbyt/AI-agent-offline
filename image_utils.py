import cv2
import numpy as np
from PIL import Image


def create_coloring_page(pil_image):
    """
    Takes a PIL Color Image, detects edges, and returns a clean
    Black & White coloring page (PIL Image).
    """
    try:
        # Convert PIL -> OpenCV (BGR)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Smooth image (reduces noise)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Edge detection
        # Thresholds: Lower=100, Upper=200 for strong lines
        edges = cv2.Canny(blur, 100, 200)

        # 4. Make lines thicker (Easier for kids to color)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 5. Invert colors (Black lines on White background)
        coloring_page = cv2.bitwise_not(edges)

        # Convert back to PIL
        return Image.fromarray(coloring_page)

    except Exception as e:
        print(f"❌ Coloring Page Error: {e}")
        return None



