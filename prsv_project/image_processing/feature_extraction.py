from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import shannon_entropy

from image_processing.constants import FEATURE_NAMES


@dataclass
class FeatureExtractionResult:
    feature_vector: list[float]
    feature_dict: dict[str, float]
    diagnostics: dict[str, float]


def _masked_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask > 0
    return image[valid]


def _compute_green_ratio(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if not np.any(valid):
        return 0.0

    rgb_float = image_rgb.astype(np.float32)
    total = rgb_float.sum(axis=2) + 1e-6
    green_ratio_map = rgb_float[:, :, 1] / total
    return float(green_ratio_map[valid].mean())


def extract_handcrafted_features(
    image_rgb: np.ndarray,
    grayscale: np.ndarray,
    hsv: np.ndarray,
    edge_map: np.ndarray,
    mask: np.ndarray,
) -> FeatureExtractionResult:
    """
    Extract the core 7 handcrafted features for PRSV research analysis.
    """
    valid_pixels_gray = _masked_pixels(grayscale, mask)
    valid_pixels_hsv = _masked_pixels(hsv, mask)
    valid_pixels_rgb = _masked_pixels(image_rgb, mask)
    valid_edges = _masked_pixels(edge_map, mask)

    if valid_pixels_gray.size == 0:
        raise ValueError("No valid masked region found for feature extraction.")

    brightness = float(valid_pixels_gray.mean() / 255.0)
    green_ratio = _compute_green_ratio(image_rgb, mask)

    hue_mean = float(valid_pixels_hsv[:, 0].mean() / 179.0) if valid_pixels_hsv.size else 0.0
    saturation_mean = float(valid_pixels_hsv[:, 1].mean() / 255.0) if valid_pixels_hsv.size else 0.0

    edge_density = float(np.count_nonzero(valid_edges) / valid_edges.size) if valid_edges.size else 0.0

    color_variance = float(np.var(valid_pixels_rgb.astype(np.float32)) / (255.0 ** 2))
    entropy = float(shannon_entropy(valid_pixels_gray.astype(np.uint8)) / 8.0)

    feature_dict = {
        "brightness": round(brightness, 6),
        "green_ratio": round(green_ratio, 6),
        "hue_mean": round(hue_mean, 6),
        "saturation_mean": round(saturation_mean, 6),
        "edge_density": round(edge_density, 6),
        "color_variance": round(color_variance, 6),
        "entropy": round(entropy, 6),
    }

    feature_vector = [feature_dict[name] for name in FEATURE_NAMES]

    diagnostics = {
        "masked_gray_pixels": float(valid_pixels_gray.size),
        "masked_rgb_pixels": float(valid_pixels_rgb.shape[0]),
        "nonzero_edge_pixels": float(np.count_nonzero(valid_edges)),
    }

    return FeatureExtractionResult(
        feature_vector=feature_vector,
        feature_dict=feature_dict,
        diagnostics=diagnostics,
    )