import numpy as np

from app.config import settings
from image_processing.preprocess import preprocess_image


def test_preprocess_output_shapes() -> None:
    image_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    result = preprocess_image(image_bgr, settings)

    assert result.resized_rgb.shape == (settings.image_height, settings.image_width, 3)
    assert result.grayscale.shape == (settings.image_height, settings.image_width)
    assert result.hsv.shape == (settings.image_height, settings.image_width, 3)