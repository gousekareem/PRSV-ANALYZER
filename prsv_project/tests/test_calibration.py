import numpy as np

from ml.calibration import calibrate_model, predict_calibrated_proba


def _make_synthetic_data(n_samples: int = 120, n_features: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    # Linearly separable-ish labels so calibration has real signal to fit.
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_calibrate_model_returns_valid_probabilities() -> None:
    X, y = _make_synthetic_data()

    calibrated = calibrate_model(X, y, method="sigmoid", cv=3)
    probabilities = predict_calibrated_proba(calibrated, X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
    # Probabilities for each sample must sum to 1.
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_calibrate_model_handles_small_dataset_cv_fallback() -> None:
    X, y = _make_synthetic_data(n_samples=10)

    # cv=5 requested but too few samples per class - should not raise, should
    # fall back to a smaller effective cv internally.
    calibrated = calibrate_model(X, y, method="sigmoid", cv=5)
    probabilities = predict_calibrated_proba(calibrated, X)

    assert probabilities.shape[0] == len(X)
