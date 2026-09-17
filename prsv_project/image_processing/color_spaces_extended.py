from __future__ import annotations

"""
Additional color spaces and vegetation indices (v3.1), additive to the
existing HSV-based features in image_processing/feature_extraction.py.
Same opt-in status as texture_extended.py: not folded into the production
12-feature vector (which the shipped model was trained on), available for
exploratory analysis and a future retraining run.
"""

from typing import Dict

import cv2
import numpy as np


def lab_color_histogram(image_rgb: np.ndarray, bins: int = 16) -> Dict[str, float]:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    features: Dict[str, float] = {}
    for i, channel_name in enumerate(("L", "a", "b")):
        channel = lab[:, :, i]
        hist, _ = np.histogram(channel, bins=bins, range=(0, 255), density=True)
        for bin_idx, value in enumerate(hist):
            features[f"lab_{channel_name}_bin{bin_idx}"] = round(float(value), 6)
        features[f"lab_{channel_name}_mean"] = round(float(np.mean(channel)), 4)
    return features


def ycbcr_color_histogram(image_rgb: np.ndarray, bins: int = 16) -> Dict[str, float]:
    ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    features: Dict[str, float] = {}
    for i, channel_name in enumerate(("Y", "Cr", "Cb")):
        channel = ycbcr[:, :, i]
        hist, _ = np.histogram(channel, bins=bins, range=(0, 255), density=True)
        for bin_idx, value in enumerate(hist):
            features[f"ycbcr_{channel_name}_bin{bin_idx}"] = round(float(value), 6)
        features[f"ycbcr_{channel_name}_mean"] = round(float(np.mean(channel)), 4)
    return features


def hsi_features(image_rgb: np.ndarray) -> Dict[str, float]:
    """
    HSI (Hue-Saturation-Intensity) decomposition - a classical alternative
    to HSV sometimes more robust for vegetation analysis since intensity is
    a simple RGB average rather than HSV's max-channel value.
    """
    rgb = image_rgb.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    intensity = (r + g + b) / 3.0
    min_channel = np.minimum(np.minimum(r, g), b)
    saturation = 1.0 - (3.0 / (r + g + b + 1e-8)) * min_channel

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-8
    theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
    hue = np.where(b <= g, theta, 2 * np.pi - theta)

    return {
        "hsi_hue_mean": round(float(np.mean(hue)), 6),
        "hsi_saturation_mean": round(float(np.mean(saturation)), 6),
        "hsi_intensity_mean": round(float(np.mean(intensity)), 6),
        "hsi_saturation_std": round(float(np.std(saturation)), 6),
    }


def vegetation_indices(image_rgb: np.ndarray) -> Dict[str, float]:
    """
    RGB-only vegetation indices borrowed from precision-agriculture remote
    sensing: Excess Green (ExG), Excess Green minus Excess Red (ExGR), and
    Green Leaf Index (GLI). These don't need NIR imagery (unlike NDVI),
    which is why they're usable from ordinary phone photos.
    """
    rgb = image_rgb.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    exg = 2 * g - r - b
    exr = 1.4 * r - g
    exgr = exg - exr
    gli = np.divide(
        (2 * g - r - b),
        (2 * g + r + b),
        out=np.zeros_like(g),
        where=(2 * g + r + b) != 0,
    )

    return {
        "veg_exg_mean": round(float(np.mean(exg)), 6),
        "veg_exgr_mean": round(float(np.mean(exgr)), 6),
        "veg_gli_mean": round(float(np.mean(gli)), 6),
    }


def color_moments(image_rgb: np.ndarray) -> Dict[str, float]:
    """
    Mean, variance, skewness per channel - a compact alternative to full
    color histograms (3 numbers per channel instead of `bins` numbers).
    """
    from scipy.stats import skew

    features: Dict[str, float] = {}
    for i, channel_name in enumerate(("r", "g", "b")):
        channel = image_rgb[:, :, i].astype(np.float64).flatten()
        features[f"moment_{channel_name}_mean"] = round(float(np.mean(channel)), 4)
        features[f"moment_{channel_name}_variance"] = round(float(np.var(channel)), 4)
        features[f"moment_{channel_name}_skewness"] = round(float(skew(channel)), 6)
    return features


def extract_all_extended_color_features(image_rgb: np.ndarray) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(lab_color_histogram(image_rgb))
    features.update(ycbcr_color_histogram(image_rgb))
    features.update(hsi_features(image_rgb))
    features.update(vegetation_indices(image_rgb))
    features.update(color_moments(image_rgb))
    return features
