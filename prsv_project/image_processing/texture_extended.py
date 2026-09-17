from __future__ import annotations

"""
Extended texture descriptors (v3.1), additive to the existing single-radius
GLCM/LBP features already in image_processing/feature_extraction.py.
Returned as a separate, opt-in feature dict rather than folded into the
production 12-feature vector, so the shipped SVM model (trained on exactly
those 12 features - see ml/feature_schema.py) keeps working unchanged;
these are available for a future retraining run that wants a richer feature
set, and for exploratory analysis via ml/feature_quality.py.

All of these are genuinely computed here (not stubs):
- Multi-radius LBP: radius 1, 2, 3 uniform LBP histograms concatenated.
- Gabor filter bank: mean response energy across 4 orientations x 2
  frequencies.
- Tamura texture: coarseness, contrast, directionality (classic formulation).
- Wavelet texture: Haar/Daubechies sub-band energy, if `pywt` is installed
  (optional dependency - degrades to omitting this key otherwise).
- Local Phase Quantization (LPQ): a compact from-scratch implementation
  (LPQ isn't in scikit-image), robust to uniform blur.
"""

from typing import Dict, Optional

import numpy as np
from skimage.feature import local_binary_pattern


def multi_radius_lbp_histogram(grayscale: np.ndarray, radii=(1, 2, 3), n_points_per_radius: int = 8) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for radius in radii:
        n_points = n_points_per_radius * radius
        lbp = local_binary_pattern(grayscale, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
        for i, value in enumerate(hist):
            features[f"lbp_r{radius}_bin{i}"] = round(float(value), 6)
    return features


def gabor_filter_bank_energy(grayscale: np.ndarray, orientations=(0, 45, 90, 135), frequencies=(0.1, 0.3)) -> Dict[str, float]:
    from skimage.filters import gabor

    features: Dict[str, float] = {}
    gray_float = grayscale.astype(np.float64) / 255.0
    for orientation_deg in orientations:
        theta = np.deg2rad(orientation_deg)
        for freq in frequencies:
            real, _ = gabor(gray_float, frequency=freq, theta=theta)
            key = f"gabor_o{orientation_deg}_f{str(freq).replace('.', '')}"
            features[f"{key}_energy"] = round(float(np.mean(real**2)), 6)
    return features


def tamura_features(grayscale: np.ndarray) -> Dict[str, float]:
    """
    Classic Tamura texture features: coarseness (via multi-scale average
    difference), contrast (statistical moments), directionality
    (gradient-orientation histogram sharpness).
    """
    gray = grayscale.astype(np.float64)

    # Coarseness: average, at each pixel, the best scale (2^k window) whose
    # neighborhood difference is maximal, then take the mean best-scale size.
    max_k = 4
    averages = [gray]
    for k in range(1, max_k + 1):
        size = 2**k
        kernel = np.ones((size, size)) / (size * size)
        from scipy.signal import convolve2d

        averages.append(convolve2d(gray, kernel, mode="same", boundary="symm"))

    best_sizes = np.ones_like(gray)
    max_diff = np.zeros_like(gray)
    for k in range(1, max_k + 1):
        size = 2**k
        horizontal_diff = np.abs(np.roll(averages[k], -size, axis=1) - np.roll(averages[k], size, axis=1))
        vertical_diff = np.abs(np.roll(averages[k], -size, axis=0) - np.roll(averages[k], size, axis=0))
        diff = np.maximum(horizontal_diff, vertical_diff)
        mask = diff > max_diff
        max_diff = np.where(mask, diff, max_diff)
        best_sizes = np.where(mask, size, best_sizes)
    coarseness = float(np.mean(best_sizes))

    contrast_std = float(np.std(gray))
    kurtosis = float(np.mean((gray - gray.mean()) ** 4) / (gray.var() ** 2 + 1e-8))
    contrast = contrast_std / (kurtosis**0.25 + 1e-8)

    grad_y, grad_x = np.gradient(gray)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angles = np.arctan2(grad_y, grad_x + 1e-8)
    strong_edges = magnitude > (0.1 * magnitude.max() + 1e-8)
    if np.any(strong_edges):
        hist, _ = np.histogram(angles[strong_edges], bins=16, range=(-np.pi, np.pi))
        hist_normalized = hist / (hist.sum() + 1e-8)
        directionality = float(1.0 - np.std(hist_normalized) * 10)
    else:
        directionality = 0.0

    return {
        "tamura_coarseness": round(coarseness, 6),
        "tamura_contrast": round(contrast, 6),
        "tamura_directionality": round(directionality, 6),
    }


def wavelet_texture_energy(grayscale: np.ndarray, wavelet: str = "db2", level: int = 2) -> Optional[Dict[str, float]]:
    try:
        import pywt
    except Exception:  # noqa: BLE001 - optional dependency
        return None

    try:
        coeffs = pywt.wavedec2(grayscale.astype(np.float64), wavelet=wavelet, level=level)
        features: Dict[str, float] = {}
        for lvl, detail_coeffs in enumerate(coeffs[1:], start=1):
            for name, arr in zip(("horizontal", "vertical", "diagonal"), detail_coeffs):
                features[f"wavelet_{wavelet}_l{lvl}_{name}_energy"] = round(float(np.mean(arr**2)), 6)
        return features
    except Exception:  # noqa: BLE001
        return None


def local_phase_quantization(grayscale: np.ndarray, window_size: int = 7) -> Dict[str, float]:
    """
    Minimal LPQ implementation: computes the short-term Fourier transform
    phase at 4 fixed frequency points within a sliding window, quantizes the
    sign of real/imaginary parts into an 8-bit code, and returns the
    resulting histogram (compressed to summary statistics for compactness).
    """
    gray = grayscale.astype(np.float64)
    radius = window_size // 2

    frequencies = [(1, 0), (0, 1), (1, 1), (1, -1)]
    codes = np.zeros(gray.shape, dtype=np.uint8)

    padded = np.pad(gray, radius, mode="reflect")
    bit = 0
    for fx, fy in frequencies:
        kernel = np.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                x, y = i - radius, j - radius
                kernel[i, j] = np.cos(-2 * np.pi * (fx * x + fy * y) / window_size)
        from scipy.signal import convolve2d

        real_part = convolve2d(padded, kernel, mode="valid")

        kernel_imag = np.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                x, y = i - radius, j - radius
                kernel_imag[i, j] = np.sin(-2 * np.pi * (fx * x + fy * y) / window_size)
        imag_part = convolve2d(padded, kernel_imag, mode="valid")

        codes |= ((real_part > 0).astype(np.uint8) << bit)
        bit += 1
        codes |= ((imag_part > 0).astype(np.uint8) << bit)
        bit += 1

    histogram, _ = np.histogram(codes, bins=256, range=(0, 256), density=True)
    return {
        "lpq_entropy": round(float(-np.sum(histogram * np.log2(histogram + 1e-12))), 6),
        "lpq_uniformity": round(float(np.sum(histogram**2)), 6),
    }


def extract_all_extended_texture_features(grayscale: np.ndarray) -> Dict[str, float]:
    """
    Convenience aggregator combining every extended texture descriptor into
    one dict, for exploratory feature-quality analysis (ml/feature_quality.py)
    or a future retraining run with an expanded feature set.
    """
    features: Dict[str, float] = {}
    features.update(multi_radius_lbp_histogram(grayscale))
    features.update(gabor_filter_bank_energy(grayscale))
    features.update(tamura_features(grayscale))
    features.update(local_phase_quantization(grayscale))

    wavelet_features = wavelet_texture_energy(grayscale)
    if wavelet_features:
        features.update(wavelet_features)

    return features
