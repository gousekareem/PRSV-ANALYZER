from pathlib import Path

import cv2
import numpy as np

from image_processing.quality_checks import assess_image_quality


def test_assess_image_quality_returns_expected_fields() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :] = [120, 120, 120]

    result = assess_image_quality(image)

    assert result.width == 256
    assert result.height == 256
    assert isinstance(result.warnings, list)
    assert result.quality_status in {"good", "acceptable", "poor"}
    assert "blur_score" in result.metrics
    assert "brightness_score" in result.metrics
    assert "contrast_score" in result.metrics