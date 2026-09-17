from __future__ import annotations

"""
Illumination normalization (v3.1): single-scale Retinex decomposition,
chromaticity-based shadow detection, and gray-world white balance -
preprocessing options that separate illumination from reflectance before
feature extraction, addressing cross-device/lighting variance. Offered as
opt-in alternatives alongside the existing CLAHE-based enhancement in
image_processing/preprocess.py, not a replacement for it (CLAHE stays the
default the shipped model was trained against).
"""

import cv2
import numpy as np


def single_scale_retinex(image_rgb: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """
    Classic single-scale Retinex: reflectance = log(image) - log(gaussian_blur(image)).
    Separates illumination (the low-frequency blurred component) from
    reflectance (the high-frequency detail), then rescales to 0-255.
    """
    image_float = image_rgb.astype(np.float64) + 1.0  # avoid log(0)
    blurred = cv2.GaussianBlur(image_float, (0, 0), sigma)
    retinex = np.log10(image_float) - np.log10(blurred + 1.0)

    normalized = np.zeros_like(retinex)
    for channel in range(retinex.shape[2]):
        channel_data = retinex[:, :, channel]
        min_val, max_val = channel_data.min(), channel_data.max()
        if max_val - min_val > 1e-8:
            normalized[:, :, channel] = (channel_data - min_val) / (max_val - min_val) * 255.0
        else:
            normalized[:, :, channel] = 0.0

    return normalized.astype(np.uint8)


def multi_scale_retinex(image_rgb: np.ndarray, sigmas=(15.0, 80.0, 200.0)) -> np.ndarray:
    """
    Multi-scale Retinex: averages single-scale Retinex outputs across
    several Gaussian scales for a more stable illumination estimate than
    single-scale alone.
    """
    accumulated = np.zeros(image_rgb.shape, dtype=np.float64)
    for sigma in sigmas:
        accumulated += single_scale_retinex(image_rgb, sigma=sigma).astype(np.float64)
    return (accumulated / len(sigmas)).astype(np.uint8)


def detect_shadow_mask(image_rgb: np.ndarray, threshold_ratio: float = 0.6) -> np.ndarray:
    """
    Chromaticity-based shadow detection: shadowed regions have low
    intensity but roughly preserved hue, distinguishing them from
    genuinely dark/discolored diseased tissue. Returns a binary mask
    (255 = likely shadow, 0 = not shadow).
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    value_channel = hsv[:, :, 2].astype(np.float64)
    saturation_channel = hsv[:, :, 1].astype(np.float64)

    mean_value = np.mean(value_channel)
    low_intensity = value_channel < (threshold_ratio * mean_value)
    # Shadows tend to have moderate-to-low saturation relative to bright
    # regions, but not zero - very low saturation is more likely a
    # genuinely gray/necrotic area rather than a shadow.
    moderate_saturation = (saturation_channel > 15) & (saturation_channel < 180)

    shadow_mask = (low_intensity & moderate_saturation).astype(np.uint8) * 255
    return shadow_mask


def gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """
    Gray-world assumption: scales each channel so its mean matches the
    overall gray mean, correcting a color cast from non-neutral lighting
    (e.g. warm indoor light, overcast blue tint) before feature extraction.
    """
    image_float = image_rgb.astype(np.float64)
    channel_means = image_float.mean(axis=(0, 1))
    gray_mean = channel_means.mean()

    scale_factors = np.where(channel_means > 1e-6, gray_mean / channel_means, 1.0)
    balanced = image_float * scale_factors
    return np.clip(balanced, 0, 255).astype(np.uint8)
