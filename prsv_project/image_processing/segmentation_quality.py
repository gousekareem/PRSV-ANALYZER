from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class SegmentationQualityResult:
    quality_score: float
    quality_status: str
    leaf_coverage_ratio: float
    contour_count: int
    largest_contour_ratio: float
    mask_fill_ratio: float
    warnings: List[str]
    metrics: Dict[str, float]


def assess_segmentation_quality(mask: np.ndarray) -> SegmentationQualityResult:
    """
    Assess reliability of the binary leaf segmentation mask.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Invalid mask supplied for segmentation quality assessment.")

    binary_mask = (mask > 0).astype(np.uint8)
    total_pixels = float(binary_mask.size)
    leaf_pixels = float(binary_mask.sum())

    leaf_coverage_ratio = leaf_pixels / total_pixels if total_pixels > 0 else 0.0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    largest_contour_area = 0.0
    if contours:
        largest_contour_area = float(max(cv2.contourArea(c) for c in contours))

    largest_contour_ratio = largest_contour_area / total_pixels if total_pixels > 0 else 0.0
    mask_fill_ratio = leaf_pixels / total_pixels if total_pixels > 0 else 0.0

    warnings: List[str] = []

    score = 100.0

    if leaf_coverage_ratio < 0.08:
        warnings.append("Leaf coverage is very low in the segmented mask.")
        score -= 35.0
    elif leaf_coverage_ratio < 0.18:
        warnings.append("Leaf coverage is limited in the segmented mask.")
        score -= 15.0

    if contour_count > 6:
        warnings.append("Segmentation mask is fragmented into many regions.")
        score -= 20.0
    elif contour_count > 3:
        warnings.append("Segmentation mask shows moderate fragmentation.")
        score -= 10.0

    if largest_contour_ratio < 0.06:
        warnings.append("Largest segmented region is too small.")
        score -= 25.0
    elif largest_contour_ratio < 0.15:
        warnings.append("Largest segmented region is smaller than expected.")
        score -= 10.0

    if leaf_coverage_ratio > 0.85:
        warnings.append("Segmented region occupies most of the image and may include background.")
        score -= 15.0

    score = max(0.0, min(100.0, score))

    if score >= 75:
        quality_status = "good"
    elif score >= 50:
        quality_status = "acceptable"
    else:
        quality_status = "weak"

    return SegmentationQualityResult(
        quality_score=round(score, 4),
        quality_status=quality_status,
        leaf_coverage_ratio=round(leaf_coverage_ratio, 6),
        contour_count=contour_count,
        largest_contour_ratio=round(largest_contour_ratio, 6),
        mask_fill_ratio=round(mask_fill_ratio, 6),
        warnings=warnings,
        metrics={
            "quality_score": round(score, 4),
            "leaf_coverage_ratio": round(leaf_coverage_ratio, 6),
            "contour_count": float(contour_count),
            "largest_contour_ratio": round(largest_contour_ratio, 6),
            "mask_fill_ratio": round(mask_fill_ratio, 6),
        },
    )