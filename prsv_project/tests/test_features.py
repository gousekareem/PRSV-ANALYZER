import numpy as np
import cv2

from image_processing.feature_extraction import extract_handcrafted_features


def test_feature_extraction_returns_seven_features() -> None:
    image_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    image_rgb[:, :] = [20, 180, 20]

    grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    edge_map = np.zeros((128, 128), dtype=np.uint8)
    mask = np.full((128, 128), 255, dtype=np.uint8)

    result = extract_handcrafted_features(
        image_rgb=image_rgb,
        grayscale=grayscale,
        hsv=hsv,
        edge_map=edge_map,
        mask=mask,
    )

    assert len(result.feature_vector) == 7
    assert "brightness" in result.feature_dict
    assert "entropy" in result.feature_dict