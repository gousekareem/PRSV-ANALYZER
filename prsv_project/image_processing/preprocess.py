from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.config import Settings
from app.utils.image_utils import normalize_to_uint8


@dataclass
class PreprocessResult:
    original_bgr: np.ndarray
    original_rgb: np.ndarray
    resized_rgb: np.ndarray
    resized_bgr: np.ndarray
    grayscale: np.ndarray
    hsv: np.ndarray
    normalized_rgb: np.ndarray
    denoised_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    metadata: dict[str, Any]


def apply_clahe_to_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE in LAB color space and return enhanced RGB image.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def preprocess_image(image_bgr: np.ndarray, settings: Settings) -> PreprocessResult:
    """
    Standard image preprocessing pipeline for PRSV analysis.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    original_bgr = image_bgr.copy()
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    resized_bgr = cv2.resize(
        original_bgr,
        settings.image_size,
        interpolation=cv2.INTER_AREA,
    )
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    grayscale = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2HSV)

    normalized_rgb = resized_rgb.astype(np.float32) / 255.0

    denoised_rgb = resized_rgb.copy()
    if settings.enable_denoising:
        denoised_rgb = cv2.fastNlMeansDenoisingColored(
            denoised_rgb,
            None,
            10,
            10,
            7,
            21,
        )

    enhanced_rgb = denoised_rgb.copy()
    if settings.enable_clahe:
        enhanced_rgb = apply_clahe_to_rgb(enhanced_rgb)

    metadata = {
        "original_shape": tuple(int(x) for x in original_bgr.shape),
        "resized_shape": tuple(int(x) for x in resized_rgb.shape),
        "grayscale_shape": tuple(int(x) for x in grayscale.shape),
        "hsv_shape": tuple(int(x) for x in hsv.shape),
        "normalized_min": float(normalized_rgb.min()),
        "normalized_max": float(normalized_rgb.max()),
    }

    return PreprocessResult(
        original_bgr=original_bgr,
        original_rgb=original_rgb,
        resized_rgb=resized_rgb,
        resized_bgr=resized_bgr,
        grayscale=grayscale,
        hsv=hsv,
        normalized_rgb=normalized_rgb,
        denoised_rgb=denoised_rgb,
        enhanced_rgb=enhanced_rgb,
        metadata=metadata,
    )


def symptom_ready_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """
    Build a grayscale image optimized for structural symptom analysis.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = normalize_to_uint8(gray)
    return gray