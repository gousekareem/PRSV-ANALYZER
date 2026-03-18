from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from image_processing.constants import (
    DEFAULT_LEAF_HSV_LOWER,
    DEFAULT_LEAF_HSV_UPPER,
    MORPH_KERNEL_SIZE,
)


@dataclass
class SegmentationResult:
    mask: np.ndarray
    segmented_rgb: np.ndarray
    contour_visualization_rgb: np.ndarray
    leaf_area_ratio: float
    contour_count: int
    used_fallback: bool
    success: bool


def _largest_contour(mask: np.ndarray) -> tuple[np.ndarray | None, list[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, []
    largest = max(contours, key=cv2.contourArea)
    return largest, contours


def segment_leaf(image_rgb: np.ndarray) -> SegmentationResult:
    """
    Segment leaf region using HSV thresholding and morphological cleanup.
    Fallback keeps full image if segmentation is weak.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input RGB image is empty.")

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    lower = np.array(DEFAULT_LEAF_HSV_LOWER, dtype=np.uint8)
    upper = np.array(DEFAULT_LEAF_HSV_UPPER, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    largest, contours = _largest_contour(mask)

    used_fallback = False
    success = True

    final_mask = np.zeros_like(mask)

    if largest is not None:
        cv2.drawContours(final_mask, [largest], contourIdx=-1, color=255, thickness=cv2.FILLED)
    else:
        used_fallback = True
        success = False
        final_mask = np.full_like(mask, 255)

    leaf_pixels = int(np.count_nonzero(final_mask))
    total_pixels = int(final_mask.size)
    leaf_area_ratio = float(leaf_pixels / total_pixels) if total_pixels else 0.0

    if leaf_area_ratio < 0.08:
        used_fallback = True
        success = False
        final_mask = np.full_like(mask, 255)
        leaf_area_ratio = 1.0

    segmented_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=final_mask)

    contour_visualization_rgb = image_rgb.copy()
    if largest is not None and not used_fallback:
        cv2.drawContours(contour_visualization_rgb, [largest], -1, (255, 0, 0), 2)

    return SegmentationResult(
        mask=final_mask,
        segmented_rgb=segmented_rgb,
        contour_visualization_rgb=contour_visualization_rgb,
        leaf_area_ratio=leaf_area_ratio,
        contour_count=len(contours),
        used_fallback=used_fallback,
        success=success,
    )