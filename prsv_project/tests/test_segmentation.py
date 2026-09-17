import numpy as np

from image_processing.segmentation import segment_leaf


def test_segmentation_returns_mask() -> None:
    image_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    image_rgb[:, :] = [0, 180, 0]

    result = segment_leaf(image_rgb)

    assert result.mask.shape == (128, 128)
    assert result.segmented_rgb.shape == (128, 128, 3)