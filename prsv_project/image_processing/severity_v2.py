from __future__ import annotations

"""
Severity estimation v2 (v3.0): lesion-area-ratio severity + conformal
prediction interval, offered alongside (not replacing) the original weighted
formula in image_processing/severity.py.

Two upgrades from the wishlist, implemented honestly within what the current
pipeline can actually support:

1. Lesion-area-ratio severity: the original formula blends five weighted
   indicators (inverse green ratio, edge density, entropy, abnormal color
   score, symptom region ratio). This module isolates just the direct
   lesion-area signal - symptom pixels / leaf pixels from the existing
   segmentation mask - as a simpler, more directly interpretable severity
   estimate: "X% of the visible leaf area shows symptom-like pixels."
   This does NOT require a trained instance-segmentation model (SAM/U-Net) -
   the wishlist's fuller version of this item explicitly depends on Section
   1's lesion segmentation work landing first. What's implemented here uses
   the classical HSV-threshold symptom mask the pipeline already computes.

2. Conformal prediction interval: rather than a bare point estimate
   ("38%"), wraps the severity score in a statistically-motivated interval
   ("38% +/- 6%, 90% coverage") using split conformal prediction - the
   nonconformity scores (|predicted - "true"| residuals) are computed on a
   held-out calibration set. Honesty note: this pipeline has no
   expert-annotated ground-truth severity labels yet (flagged in the
   wishlist as a prerequisite - "once expert-annotated severity ground
   truth exists"), so `calibrate_from_residuals()` is the real, reusable
   conformal machinery, and `bootstrap_interval()` is what's used by
   default today: a distribution-free bootstrap interval computed from
   perturbing the mask threshold, which is a defensible interim proxy for
   "how sensitive is this estimate to a small change in what counts as a
   symptom pixel," not a substitute for genuine expert-calibrated coverage.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class LesionRatioSeverity:
    lesion_area_ratio: float
    severity_score: float
    interval_low: float
    interval_high: float
    interval_method: str
    coverage: float


def compute_lesion_area_ratio(symptom_mask: np.ndarray, leaf_mask: np.ndarray, threshold: int = 160) -> float:
    leaf_pixels = max(1, int(np.count_nonzero(leaf_mask)))
    symptom_pixels = int(np.count_nonzero((symptom_mask > threshold) & (leaf_mask > 0)))
    return float(symptom_pixels / leaf_pixels)


def bootstrap_interval(
    symptom_mask: np.ndarray,
    leaf_mask: np.ndarray,
    base_threshold: int = 160,
    threshold_jitter: int = 20,
    n_bootstrap: int = 25,
    coverage: float = 0.90,
) -> tuple[float, float]:
    """
    Distribution-free interim uncertainty estimate: recompute the lesion
    ratio under small random perturbations of the symptom-pixel threshold,
    then take the empirical (1-coverage)/2 and 1-(1-coverage)/2 quantiles of
    the resulting ratios as the interval bounds. This characterizes
    sensitivity to an arbitrary thresholding choice - a real, useful
    uncertainty signal - but is NOT a calibrated conformal interval over
    ground-truth severity error, which requires labeled data (see
    calibrate_from_residuals below).
    """
    rng = np.random.default_rng(seed=42)
    ratios = []
    for _ in range(n_bootstrap):
        jitter = int(rng.integers(-threshold_jitter, threshold_jitter + 1))
        threshold = int(np.clip(base_threshold + jitter, 1, 254))
        ratios.append(compute_lesion_area_ratio(symptom_mask, leaf_mask, threshold=threshold))

    lower_q = (1 - coverage) / 2
    upper_q = 1 - lower_q
    low = float(np.quantile(ratios, lower_q))
    high = float(np.quantile(ratios, upper_q))
    return low, high


def estimate_lesion_ratio_severity(
    symptom_mask: np.ndarray,
    leaf_mask: np.ndarray,
    coverage: float = 0.90,
) -> LesionRatioSeverity:
    ratio = compute_lesion_area_ratio(symptom_mask, leaf_mask)
    low, high = bootstrap_interval(symptom_mask, leaf_mask, coverage=coverage)

    return LesionRatioSeverity(
        lesion_area_ratio=round(ratio, 6),
        severity_score=round(ratio * 100.0, 4),
        interval_low=round(max(0.0, low) * 100.0, 4),
        interval_high=round(min(1.0, high) * 100.0, 4),
        interval_method="threshold_perturbation_bootstrap",
        coverage=coverage,
    )


def calibrate_from_residuals(residuals: List[float], coverage: float = 0.90) -> float:
    """
    Genuine split-conformal calibration step: given |predicted - true|
    residuals on a held-out calibration set of *expert-labeled* severity
    scores, returns the interval half-width q such that
    [prediction - q, prediction + q] has the target marginal coverage on
    exchangeable future data. Call this once expert severity annotations
    exist (see the wishlist's Section 7 prerequisite); until then,
    estimate_lesion_ratio_severity() above uses the bootstrap proxy instead.
    """
    if not residuals:
        raise ValueError("Need at least one calibration residual.")
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(np.abs(residuals), q_level))
