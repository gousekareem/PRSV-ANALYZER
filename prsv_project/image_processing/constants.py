from __future__ import annotations

FEATURE_NAMES: list[str] = [
    "brightness",
    "green_ratio",
    "hue_mean",
    "saturation_mean",
    "edge_density",
    "color_variance",
    "entropy",
    # v2.9: texture features, added on top of the original 7 color/edge features.
    # Ring-spot lesions have a distinct local texture (pitted, irregular) that
    # global color/edge averages miss - GLCM and LBP capture that texture
    # directly. This is a cheaper accuracy lever than a CNN.
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "lbp_uniformity",
]

DEFAULT_LEAF_HSV_LOWER: tuple[int, int, int] = (20, 20, 20)
DEFAULT_LEAF_HSV_UPPER: tuple[int, int, int] = (95, 255, 255)

CANNY_THRESHOLD_1: int = 50
CANNY_THRESHOLD_2: int = 150

MORPH_KERNEL_SIZE: int = 5

# Texture feature parameters
GLCM_DISTANCES: list[int] = [1, 2]
GLCM_ANGLES: list[float] = [0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345]  # 0, 45, 90, 135 deg
LBP_RADIUS: int = 2
LBP_N_POINTS: int = 16