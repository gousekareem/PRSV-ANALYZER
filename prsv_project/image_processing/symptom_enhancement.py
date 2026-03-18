from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.utils.image_utils import normalize_to_uint8
from image_processing.constants import CANNY_THRESHOLD_1, CANNY_THRESHOLD_2


@dataclass
class SymptomEnhancementResult:
    edge_map: np.ndarray
    gradient_magnitude: np.ndarray
    laplacian_response: np.ndarray
    chlorosis_map: np.ndarray
    abnormal_color_score: float
    symptom_mask: np.ndarray
    symptom_highlight_rgb: np.ndarray


def _compute_chlorosis_map(image_rgb: np.ndarray, leaf_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Build a chlorosis-like abnormal color map by emphasizing yellowish / low-green regions.
    """
    rgb_float = image_rgb.astype(np.float32)

    red = rgb_float[:, :, 0]
    green = rgb_float[:, :, 1]
    blue = rgb_float[:, :, 2]

    yellow_score = ((red + green) / 2.0) - blue
    green_loss = np.maximum(0.0, red - green) + np.maximum(0.0, blue - green)
    chlorosis = 0.6 * yellow_score + 0.4 * green_loss

    chlorosis = chlorosis * (leaf_mask > 0)
    chlorosis_uint8 = normalize_to_uint8(chlorosis)

    active_pixels = chlorosis_uint8[leaf_mask > 0]
    abnormal_color_score = float(active_pixels.mean() / 255.0) if active_pixels.size else 0.0
    return chlorosis_uint8, abnormal_color_score


def enhance_symptoms(image_rgb: np.ndarray, leaf_mask: np.ndarray) -> SymptomEnhancementResult:
    """
    Create image-processing evidence maps for PRSV-related visual irregularities.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input RGB image is empty.")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    edge_map = cv2.Canny(gray, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    edge_map = cv2.bitwise_and(edge_map, edge_map, mask=leaf_mask)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_magnitude = gradient_magnitude * (leaf_mask > 0)
    gradient_magnitude_uint8 = normalize_to_uint8(gradient_magnitude)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian_abs = np.abs(laplacian) * (leaf_mask > 0)
    laplacian_response = normalize_to_uint8(laplacian_abs)

    chlorosis_map, abnormal_color_score = _compute_chlorosis_map(image_rgb, leaf_mask)

    combined = (
        0.35 * gradient_magnitude_uint8.astype(np.float32)
        + 0.30 * laplacian_response.astype(np.float32)
        + 0.35 * chlorosis_map.astype(np.float32)
    )
    symptom_mask = normalize_to_uint8(combined)

    symptom_highlight_rgb = image_rgb.copy()
    red_overlay = np.zeros_like(symptom_highlight_rgb)
    red_overlay[:, :, 0] = symptom_mask

    symptom_highlight_rgb = cv2.addWeighted(
        symptom_highlight_rgb,
        0.72,
        red_overlay,
        0.45,
        0,
    )

    symptom_highlight_rgb = cv2.bitwise_and(
        symptom_highlight_rgb,
        symptom_highlight_rgb,
        mask=leaf_mask,
    )

    return SymptomEnhancementResult(
        edge_map=edge_map,
        gradient_magnitude=gradient_magnitude_uint8,
        laplacian_response=laplacian_response,
        chlorosis_map=chlorosis_map,
        abnormal_color_score=abnormal_color_score,
        symptom_mask=symptom_mask,
        symptom_highlight_rgb=symptom_highlight_rgb,
    )