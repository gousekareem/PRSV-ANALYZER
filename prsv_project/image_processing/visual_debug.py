from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.utils.image_utils import save_image_cv, save_image_pil


def save_debug_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """
    Save RGB debug image.
    """
    save_image_pil(path, image_rgb)


def save_debug_gray(path: Path, image_gray: np.ndarray) -> None:
    """
    Save grayscale debug image.
    """
    save_image_cv(path, image_gray)


def save_debug_mask(path: Path, mask: np.ndarray) -> None:
    """
    Save binary or grayscale mask.
    """
    save_image_cv(path, mask)


def save_debug_bgr(path: Path, image_bgr: np.ndarray) -> None:
    """
    Save BGR visualization directly.
    """
    save_image_cv(path, image_bgr)