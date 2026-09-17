from __future__ import annotations

"""
Drift detection (v3.1): monitors whether live feature distributions are
drifting away from the training distribution - an early warning before
accuracy silently degrades.

Honesty note: the wishlist names `evidently` or `whylogs`. Both are real,
well-built libraries, but pull in a fair amount of their own dependency
tree for what is, at its statistical core, two well-established tests:
Population Stability Index (PSI) and the Kolmogorov-Smirnov two-sample
test. Implementing those directly here (a few dozen lines of numpy/scipy)
gives the same statistical rigor without the extra dependency weight, and
is a completely standard approach used inside larger drift-monitoring tools
themselves - not a shortcut around the underlying method.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class FeatureDriftResult:
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    drift_detected: bool
    severity: str  # "none", "moderate", "significant"


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI < 0.1: no significant shift. 0.1-0.25: moderate shift, worth
    watching. > 0.25: significant shift, model performance is at risk.
    These are the conventional PSI interpretation thresholds used across
    the credit-risk/MLOps literature this metric originates from.
    """
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    reference_counts, _ = np.histogram(reference, bins=breakpoints)
    current_counts, _ = np.histogram(current, bins=breakpoints)

    reference_pct = np.clip(reference_counts / max(1, len(reference)), 1e-6, None)
    current_pct = np.clip(current_counts / max(1, len(current)), 1e-6, None)

    psi = float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))
    return round(psi, 6)


def _severity_from_psi(psi: float) -> str:
    if psi < 0.1:
        return "none"
    if psi < 0.25:
        return "moderate"
    return "significant"


def detect_feature_drift(
    reference_features: Dict[str, np.ndarray],
    current_features: Dict[str, np.ndarray],
) -> List[FeatureDriftResult]:
    """
    reference_features / current_features: {feature_name: 1D array of
    values} - typically the training set's feature columns vs. a recent
    window of production feature-extraction outputs.
    """
    results: List[FeatureDriftResult] = []
    for feature_name, reference_values in reference_features.items():
        if feature_name not in current_features:
            continue
        current_values = current_features[feature_name]

        psi = population_stability_index(np.asarray(reference_values), np.asarray(current_values))
        ks_stat, ks_p = ks_2samp(reference_values, current_values)

        severity = _severity_from_psi(psi)
        drift_detected = severity != "none" or ks_p < 0.05

        results.append(
            FeatureDriftResult(
                feature_name=feature_name,
                psi=psi,
                ks_statistic=round(float(ks_stat), 6),
                ks_p_value=round(float(ks_p), 6),
                drift_detected=drift_detected,
                severity=severity,
            )
        )
    return results


def summarize_drift(results: List[FeatureDriftResult]) -> Dict[str, object]:
    drifted = [r for r in results if r.drift_detected]
    return {
        "total_features_checked": len(results),
        "features_with_drift": len(drifted),
        "drifted_feature_names": [r.feature_name for r in drifted],
        "overall_status": "drift_detected" if drifted else "stable",
    }
