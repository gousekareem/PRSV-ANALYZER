import numpy as np
from sklearn.svm import SVC

from ml.active_learning import query_by_committee, self_training_pseudo_labels, uncertainty_sampling
from ml.feature_quality import analyze_feature_quality


def _synthetic_dataset(n_samples: int = 100, n_features: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = [f"f{i}" for i in range(n_features)]
    return X, y, feature_names


def test_analyze_feature_quality_runs_end_to_end() -> None:
    X, y, feature_names = _synthetic_dataset()
    report = analyze_feature_quality(X, y, feature_names)

    assert len(report.pca_explained_variance_ratio) == len(feature_names)
    assert report.pca_n_components_for_95pct >= 1
    assert set(report.mutual_information.keys()) == set(feature_names)
    assert len(report.rfecv_selected_features) >= 1
    assert 0.0 <= report.lda_class_separation_score <= 1.0


def test_uncertainty_sampling_returns_requested_count() -> None:
    X, y, _ = _synthetic_dataset()
    model = SVC(probability=True).fit(X, y)

    samples = uncertainty_sampling(model, X, n_samples=5)
    assert len(samples) == 5
    # Uncertainty scores should be sorted descending.
    scores = [s.uncertainty_score for s in samples]
    assert scores == sorted(scores, reverse=True)


def test_query_by_committee_returns_requested_count() -> None:
    X, y, _ = _synthetic_dataset()
    model_a = SVC(probability=True).fit(X, y)
    model_b = SVC(probability=True, kernel="linear").fit(X, y)

    samples = query_by_committee([model_a, model_b], X, n_samples=5)
    assert len(samples) == 5


def test_self_training_pseudo_labels_respects_confidence_threshold() -> None:
    X, y, _ = _synthetic_dataset()
    model = SVC(probability=True).fit(X, y)

    pseudo_X, pseudo_y, indices = self_training_pseudo_labels(model, X, confidence_threshold=0.99)
    assert len(pseudo_X) == len(pseudo_y) == len(indices)
    # High threshold should select a strict subset, not everything.
    assert len(pseudo_X) <= len(X)
