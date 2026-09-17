from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image_cv(path: Path) -> np.ndarray:
    """
    Read an image using OpenCV in BGR format.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR.
    """
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def save_image_cv(path: Path, image: np.ndarray) -> None:
    """
    Save an image using OpenCV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Failed to save image to {path}")


def save_image_pil(path: Path, image_rgb: np.ndarray) -> None:
    """
    Save an RGB image using PIL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(image_rgb.astype(np.uint8))
    image.save(path)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize an array to uint8 [0,255].
    """
    image_float = image.astype(np.float32)
    min_val = image_float.min()
    max_val = image_float.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(image_float, dtype=np.uint8)

    normalized = (image_float - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)