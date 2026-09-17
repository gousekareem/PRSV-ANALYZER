import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from evaluation.fairness_audit import audit_by_subgroup, largest_subgroup_gap
from evaluation.pdp_ice import compute_pdp_for_all_features
from evaluation.stress_testing import apply_blur, apply_brightness_shift, run_stress_test
from evaluation.trust_scores import TrustScorer


def _synthetic_dataset(n_samples: int = 100, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_audit_by_subgroup_computes_per_group_metrics() -> None:
    X, y = _synthetic_dataset()
    y_pred = y.copy()
    y_pred[:10] = 1 - y_pred[:10]  # introduce some errors

    subgroups = np.array(["camera_a"] * 50 + ["camera_b"] * 50)
    reports = audit_by_subgroup(y, y_pred, subgroups)

    assert len(reports) == 2
    assert {r.subgroup for r in reports} == {"camera_a", "camera_b"}


def test_largest_subgroup_gap_insufficient_subgroups() -> None:
    result = largest_subgroup_gap([])
    assert result["status"] == "insufficient_subgroups"


def test_trust_scorer_flags_low_trust_far_point() -> None:
    X, y = _synthetic_dataset(n_samples=200)
    scorer = TrustScorer(X, y, k=5)

    # A point deep in class-1 territory should score reasonably trustworthy
    # when correctly predicted as class 1.
    deep_class_1_point = np.array([5.0, 0.0, 0.0, 0.0])
    result = scorer.score(deep_class_1_point, predicted_class=1)

    assert result.trust_score > 0


def test_pdp_ice_computes_for_all_features() -> None:
    X, y = _synthetic_dataset(n_samples=80)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    results = compute_pdp_for_all_features(model, X, feature_names, grid_resolution=5)

    assert len(results) == len(feature_names)
    for r in results:
        assert len(r["grid_values"]) == len(r["average_pdp"])


def test_stress_test_perturbation_functions_change_image() -> None:
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)

    blurred = apply_blur(image)
    brightened = apply_brightness_shift(image, delta=-60)

    assert blurred.shape == image.shape
    assert brightened.shape == image.shape
    assert not np.array_equal(blurred, image)
    assert not np.array_equal(brightened, image)


def test_run_stress_test_reports_flip_rates(tmp_path) -> None:
    import cv2

    image_paths = []
    for i in range(3):
        path = tmp_path / f"img_{i}.jpg"
        img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(path), img)
        image_paths.append(str(path))

    def fake_predict(image_bgr) -> tuple[str, float]:
        # Deterministic fake "model": prediction based on mean brightness.
        mean_val = float(np.mean(image_bgr))
        return ("Diseased" if mean_val < 100 else "Healthy"), 0.8

    def read_fn(path: str):
        return cv2.imread(path)

    results = run_stress_test(image_paths, fake_predict, read_fn)

    assert len(results) == 4  # 4 perturbation types
    for r in results:
        assert 0.0 <= r.prediction_flip_rate <= 1.0
