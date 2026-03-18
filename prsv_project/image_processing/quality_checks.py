from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class ImageQualityResult:
    blur_score: float
    brightness_score: float
    contrast_score: float
    width: int
    height: int
    is_small_image: bool
    warnings: List[str]
    quality_status: str
    metrics: Dict[str, float]


def assess_image_quality(image_bgr: np.ndarray) -> ImageQualityResult:
    """
    Assess image quality for analysis reliability.
    Uses simple classical image-processing heuristics.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image supplied for quality assessment.")

    height, width = image_bgr.shape[:2]
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    brightness_score = float(np.mean(grayscale))
    contrast_score = float(np.std(grayscale))

    warnings: List[str] = []

    is_small_image = width < 128 or height < 128
    if is_small_image:
        warnings.append("Image resolution is small and may reduce result reliability.")

    if blur_score < 40:
        warnings.append("Image appears blurry.")

    if brightness_score < 55:
        warnings.append("Image appears too dark.")
    elif brightness_score > 210:
        warnings.append("Image appears too bright.")

    if contrast_score < 25:
        warnings.append("Image contrast is low.")

    if len(warnings) == 0:
        quality_status = "good"
    elif len(warnings) == 1:
        quality_status = "acceptable"
    else:
        quality_status = "poor"

    return ImageQualityResult(
        blur_score=round(blur_score, 4),
        brightness_score=round(brightness_score, 4),
        contrast_score=round(contrast_score, 4),
        width=width,
        height=height,
        is_small_image=is_small_image,
        warnings=warnings,
        quality_status=quality_status,
        metrics={
            "blur_score": round(blur_score, 4),
            "brightness_score": round(brightness_score, 4),
            "contrast_score": round(contrast_score, 4),
            "width": float(width),
            "height": float(height),
        },
    )