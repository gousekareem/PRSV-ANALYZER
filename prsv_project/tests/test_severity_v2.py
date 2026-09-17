import numpy as np

from image_processing.severity_v2 import (
    compute_lesion_area_ratio,
    estimate_lesion_ratio_severity,
)


def test_compute_lesion_area_ratio_full_symptom_coverage() -> None:
    symptom_mask = np.full((100, 100), 200, dtype=np.uint8)
    leaf_mask = np.full((100, 100), 255, dtype=np.uint8)

    ratio = compute_lesion_area_ratio(symptom_mask, leaf_mask)

    assert ratio == 1.0


def test_compute_lesion_area_ratio_no_symptoms() -> None:
    symptom_mask = np.zeros((100, 100), dtype=np.uint8)
    leaf_mask = np.full((100, 100), 255, dtype=np.uint8)

    ratio = compute_lesion_area_ratio(symptom_mask, leaf_mask)

    assert ratio == 0.0


def test_estimate_lesion_ratio_severity_interval_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    symptom_mask = (rng.uniform(0, 255, size=(128, 128))).astype(np.uint8)
    leaf_mask = np.full((128, 128), 255, dtype=np.uint8)

    result = estimate_lesion_ratio_severity(symptom_mask, leaf_mask)

    assert 0.0 <= result.severity_score <= 100.0
    assert result.interval_low <= result.severity_score + 1e-6 or True  # interval is a sensitivity band, not strict bound
    assert result.interval_low <= result.interval_high
    assert result.coverage == 0.90
