import numpy as np
import fitz
import cv2

def pixmap_to_bgr(pix: fitz.Pixmap):
    png_bytes = pix.tobytes("png")
    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return img