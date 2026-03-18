import numpy as np

from app.config import settings
from image_processing.severity import estimate_severity


def test_severity_output_range() -> None:
    feature_dict = {
        "brightness": 0.4,
        "green_ratio": 0.5,
        "hue_mean": 0.3,
        "saturation_mean": 0.4,
        "edge_density": 0.2,
        "color_variance": 0.1,
        "entropy": 0.5,
    }
    symptom_mask = np.full((128, 128), 180, dtype=np.uint8)
    leaf_mask = np.full((128, 128), 255, dtype=np.uint8)

    result = estimate_severity(feature_dict, symptom_mask, leaf_mask, settings)

    assert 0.0 <= result.infection_percentage <= 100.0
    assert 0.0 <= result.severity_score <= 100.0