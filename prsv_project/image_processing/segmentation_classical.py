from __future__ import annotations

"""
Classical segmentation alternatives (v3.1) to the existing HSV-threshold
leaf segmentation in image_processing/segmentation.py - still not a trained
model (SAM/U-Net remain blocked on labeled data, see ROADMAP_v3.md), but
more background-robust than a fixed HSV threshold under shadows, clutter, or
uneven lighting. Offered as alternative masking strategies, selectable per
image, not a silent replacement of the production path.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AlternativeSegmentationResult:
    mask: np.ndarray
    method: str
    leaf_area_ratio: float


def grabcut_segmentation(image_rgb: np.ndarray, iterations: int = 5) -> AlternativeSegmentationResult:
    """
    GrabCut: iterative graph-cut segmentation seeded with a rectangle
    covering the central ~80% of the image (a reasonable prior for a
    leaf-centered phone photo), refined into foreground/background/probable
    regions over several iterations.
    """
    height, width = image_rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    margin_x, margin_y = int(width * 0.1), int(height * 0.1)
    rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    leaf_area_ratio = float(np.count_nonzero(binary_mask) / binary_mask.size)

    return AlternativeSegmentationResult(mask=binary_mask, method="grabcut", leaf_area_ratio=round(leaf_area_ratio, 6))


def watershed_segmentation(image_rgb: np.ndarray) -> AlternativeSegmentationResult:
    """
    Marker-based watershed: Otsu-thresholds to get a rough foreground,
    computes the distance transform to find sure-foreground "peaks", then
    lets watershed flood-fill outward from those markers to the true
    boundary - typically cleaner edges under uneven lighting than a fixed
    HSV threshold alone.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(image_bgr, markers)

    binary_mask = np.where(markers > 1, 255, 0).astype(np.uint8)
    leaf_area_ratio = float(np.count_nonzero(binary_mask) / binary_mask.size)

    return AlternativeSegmentationResult(mask=binary_mask, method="watershed", leaf_area_ratio=round(leaf_area_ratio, 6))


def slic_superpixel_mask(image_rgb: np.ndarray, n_segments: int = 100, compactness: float = 10.0) -> AlternativeSegmentationResult:
    """
    SLIC superpixels as a *pre-segmentation* step: groups the image into
    ~n_segments perceptually-coherent regions, then keeps segments whose
    mean color falls in a plausible "leaf green" range as the mask. This is
    a genuinely different segmentation strategy from thresholding - it
    respects region boundaries first and classifies second, useful as
    preprocessing before a downstream refinement step (e.g. GrabCut seeded
    from the SLIC leaf segments instead of a blind rectangle).
    """
    from skimage.segmentation import slic
    from skimage.color import rgb2hsv

    segments = slic(image_rgb, n_segments=n_segments, compactness=compactness, start_label=1)
    hsv = rgb2hsv(image_rgb)

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for segment_id in np.unique(segments):
        segment_pixels = segments == segment_id
        mean_hue = np.mean(hsv[:, :, 0][segment_pixels])
        mean_sat = np.mean(hsv[:, :, 1][segment_pixels])
        # Green hue range in skimage's 0-1 hue scale (~0.19-0.44 covers
        # green-yellow through green-cyan) with a minimum saturation to
        # exclude gray/brown background clutter.
        if 0.15 <= mean_hue <= 0.45 and mean_sat > 0.15:
            mask[segment_pixels] = 255

    leaf_area_ratio = float(np.count_nonzero(mask) / mask.size)
    return AlternativeSegmentationResult(mask=mask, method="slic", leaf_area_ratio=round(leaf_area_ratio, 6))
