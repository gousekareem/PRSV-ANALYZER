import numpy as np

from image_processing.segmentation_quality import assess_segmentation_quality


def test_assess_segmentation_quality_returns_expected_fields() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[20:100, 30:110] = 255

    result = assess_segmentation_quality(mask)

    assert result.quality_status in {"good", "acceptable", "weak"}
    assert isinstance(result.warnings, list)
    assert "quality_score" in result.metrics
    assert "leaf_coverage_ratio" in result.metrics
    assert "contour_count" in result.metrics