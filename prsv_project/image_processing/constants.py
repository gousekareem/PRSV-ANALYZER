from __future__ import annotations

FEATURE_NAMES: list[str] = [
    "brightness",
    "green_ratio",
    "hue_mean",
    "saturation_mean",
    "edge_density",
    "color_variance",
    "entropy",
]

DEFAULT_LEAF_HSV_LOWER: tuple[int, int, int] = (20, 20, 20)
DEFAULT_LEAF_HSV_UPPER: tuple[int, int, int] = (95, 255, 255)

CANNY_THRESHOLD_1: int = 50
CANNY_THRESHOLD_2: int = 150

MORPH_KERNEL_SIZE: int = 5