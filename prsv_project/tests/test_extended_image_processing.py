import numpy as np

from image_processing.color_spaces_extended import (
    color_moments,
    extract_all_extended_color_features,
    hsi_features,
    vegetation_indices,
)
from image_processing.illumination_normalization import (
    detect_shadow_mask,
    gray_world_white_balance,
    single_scale_retinex,
)
from image_processing.segmentation_classical import (
    grabcut_segmentation,
    slic_superpixel_mask,
    watershed_segmentation,
)
from image_processing.texture_extended import (
    extract_all_extended_texture_features,
    tamura_features,
)


def _sample_rgb_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((96, 96, 3)) * 255).astype(np.uint8)


def _sample_grayscale_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((96, 96)) * 255).astype(np.uint8)


def test_extended_color_features_return_finite_values() -> None:
    image = _sample_rgb_image()
    features = extract_all_extended_color_features(image)

    assert len(features) > 0
    assert all(np.isfinite(v) for v in features.values())


def test_hsi_features_ranges() -> None:
    image = _sample_rgb_image()
    features = hsi_features(image)

    assert 0.0 <= features["hsi_saturation_mean"] <= 1.0
    assert 0.0 <= features["hsi_intensity_mean"] <= 1.0


def test_vegetation_indices_computed() -> None:
    image = _sample_rgb_image()
    indices = vegetation_indices(image)
    assert set(indices.keys()) == {"veg_exg_mean", "veg_exgr_mean", "veg_gli_mean"}


def test_color_moments_three_channels() -> None:
    image = _sample_rgb_image()
    moments = color_moments(image)
    assert len(moments) == 9  # mean/variance/skewness x 3 channels


def test_extended_texture_features_return_finite_values() -> None:
    gray = _sample_grayscale_image()
    features = extract_all_extended_texture_features(gray)

    assert len(features) > 30  # multi-radius LBP alone contributes many bins
    assert all(np.isfinite(v) for v in features.values())


def test_tamura_features_keys() -> None:
    gray = _sample_grayscale_image()
    features = tamura_features(gray)
    assert set(features.keys()) == {"tamura_coarseness", "tamura_contrast", "tamura_directionality"}


def test_wavelet_texture_energy_when_pywt_available() -> None:
    from image_processing.texture_extended import wavelet_texture_energy

    gray = _sample_grayscale_image()
    result = wavelet_texture_energy(gray)

    # None (graceful skip) if pywt isn't installed; a populated dict of
    # finite sub-band energies if it is - both are valid outcomes.
    if result is not None:
        assert len(result) > 0
        assert all(np.isfinite(v) for v in result.values())


def test_illumination_normalization_preserves_shape() -> None:
    image = _sample_rgb_image()

    retinex = single_scale_retinex(image)
    shadow_mask = detect_shadow_mask(image)
    balanced = gray_world_white_balance(image)

    assert retinex.shape == image.shape
    assert shadow_mask.shape == image.shape[:2]
    assert balanced.shape == image.shape


def test_classical_segmentation_alternatives_produce_valid_masks() -> None:
    image = _sample_rgb_image()

    grabcut_result = grabcut_segmentation(image, iterations=2)
    watershed_result = watershed_segmentation(image)
    slic_result = slic_superpixel_mask(image, n_segments=20)

    for result in (grabcut_result, watershed_result, slic_result):
        assert result.mask.shape == image.shape[:2]
        assert 0.0 <= result.leaf_area_ratio <= 1.0
