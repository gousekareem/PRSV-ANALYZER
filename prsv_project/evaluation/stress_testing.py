from __future__ import annotations

"""
Stress testing (v3.1): applies deliberate perturbations (blur, brightness
shift, JPEG compression, partial occlusion) to test images and measures
prediction stability, characterizing failure modes rather than only
reporting best-case accuracy on clean images.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

import cv2
import numpy as np


@dataclass
class StressTestResult:
    perturbation_name: str
    n_images: int
    prediction_flip_rate: float
    mean_confidence_drop: float


def apply_blur(image_bgr: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    return cv2.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)


def apply_brightness_shift(image_bgr: np.ndarray, delta: int = -60) -> np.ndarray:
    return np.clip(image_bgr.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def apply_jpeg_compression(image_bgr: np.ndarray, quality: int = 15) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return image_bgr
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_partial_occlusion(image_bgr: np.ndarray, occlusion_fraction: float = 0.25) -> np.ndarray:
    occluded = image_bgr.copy()
    height, width = occluded.shape[:2]
    box_h, box_w = int(height * occlusion_fraction), int(width * occlusion_fraction)
    y0 = (height - box_h) // 2
    x0 = (width - box_w) // 2
    occluded[y0 : y0 + box_h, x0 : x0 + box_w] = 0
    return occluded


PERTURBATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gaussian_blur": apply_blur,
    "brightness_drop": apply_brightness_shift,
    "heavy_jpeg_compression": apply_jpeg_compression,
    "partial_occlusion": apply_partial_occlusion,
}


def run_stress_test(
    image_paths: List[str],
    predict_fn: Callable[[np.ndarray], tuple[str, float]],
    read_fn: Callable[[str], np.ndarray],
) -> List[StressTestResult]:
    """
    predict_fn: takes a BGR numpy image and returns (prediction_label, confidence),
    i.e. a thin wrapper around the full analysis pipeline's read->preprocess->
    infer steps, supplied by the caller so this module has no direct
    dependency on app.services.analysis_service (keeping it independently
    testable and reusable from a standalone script).
    read_fn: loads an image path into a BGR numpy array (typically
    app.utils.image_utils.read_image_cv).
    """
    results: List[StressTestResult] = []

    baseline_predictions = {}
    baseline_confidences = {}
    for path in image_paths:
        image = read_fn(path)
        pred, conf = predict_fn(image)
        baseline_predictions[path] = pred
        baseline_confidences[path] = conf

    for perturbation_name, perturbation_fn in PERTURBATIONS.items():
        flips = 0
        confidence_drops = []

        for path in image_paths:
            image = read_fn(path)
            perturbed = perturbation_fn(image)
            pred, conf = predict_fn(perturbed)

            if pred != baseline_predictions[path]:
                flips += 1
            confidence_drops.append(baseline_confidences[path] - conf)

        results.append(
            StressTestResult(
                perturbation_name=perturbation_name,
                n_images=len(image_paths),
                prediction_flip_rate=round(flips / max(1, len(image_paths)), 4),
                mean_confidence_drop=round(float(np.mean(confidence_drops)), 4) if confidence_drops else 0.0,
            )
        )

    return results
