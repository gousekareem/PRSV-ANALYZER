from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import Settings


@dataclass
class SeverityResult:
    infection_percentage: float
    severity_score: float
    severity_label: str
    severity_confidence: float
    symptom_region_ratio: float
    reasoning_trace: dict[str, float | str]


def _clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    return max(min_value, min(max_value, value))


def _severity_label(score: float, settings: Settings) -> str:
    for label, (low, high) in settings.severity_thresholds.items():
        if low <= score <= high:
            return label
    if score <= 0:
        return "Healthy"
    return "Severe"


def estimate_severity(
    feature_dict: dict[str, float],
    symptom_mask: np.ndarray,
    leaf_mask: np.ndarray,
    settings: Settings,
) -> SeverityResult:
    """
    Estimate disease severity using weighted image-derived indicators.
    """
    leaf_pixels = max(1, int(np.count_nonzero(leaf_mask)))
    symptom_pixels = int(np.count_nonzero(symptom_mask > 160))
    symptom_region_ratio = float(symptom_pixels / leaf_pixels)

    inverse_green_ratio = 1.0 - float(feature_dict.get("green_ratio", 0.0))
    edge_density = float(feature_dict.get("edge_density", 0.0))
    entropy = float(feature_dict.get("entropy", 0.0))
    abnormal_color_score = float(np.mean(symptom_mask[leaf_mask > 0]) / 255.0) if np.count_nonzero(leaf_mask) else 0.0

    weights = settings.severity_weights

    raw_score_0_1 = (
        weights["inverse_green_ratio"] * inverse_green_ratio
        + weights["edge_density"] * edge_density
        + weights["entropy"] * entropy
        + weights["abnormal_color_score"] * abnormal_color_score
        + weights["symptom_region_ratio"] * symptom_region_ratio
    )

    severity_score = _clamp(raw_score_0_1 * 100.0)
    infection_percentage = severity_score
    severity_label = _severity_label(severity_score, settings)

    confidence = _clamp(
        40.0
        + 25.0 * edge_density
        + 20.0 * symptom_region_ratio
        + 15.0 * abnormal_color_score
    )

    return SeverityResult(
        infection_percentage=round(infection_percentage, 4),
        severity_score=round(severity_score, 4),
        severity_label=severity_label,
        severity_confidence=round(confidence, 4),
        symptom_region_ratio=round(symptom_region_ratio, 6),
        reasoning_trace={
            "inverse_green_ratio": round(inverse_green_ratio, 6),
            "edge_density": round(edge_density, 6),
            "entropy": round(entropy, 6),
            "abnormal_color_score": round(abnormal_color_score, 6),
            "symptom_region_ratio": round(symptom_region_ratio, 6),
            "formula_mode": "weighted_image_processing_based",
        },
    )