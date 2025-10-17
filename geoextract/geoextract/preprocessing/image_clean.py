from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def preprocess_image(img: Image.Image) -> Image.Image:
    """Basic MVP preprocessing: grayscale, denoise, adaptive threshold, slight deskew heuristic."""
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold to enhance contrast
    th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)

    # Deskew via moments
    coords = np.column_stack(np.where(th > 0))
    angle = 0.0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = th.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        th = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(th)
