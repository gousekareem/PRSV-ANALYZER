import numpy as np
from sklearn.ensemble import RandomForestClassifier

from evaluation.shap_interactions import compute_shap_interactions


def test_compute_shap_interactions_returns_ranked_pairs() -> None:
    rng = np.random.default_rng(0)
    X = rng.random((60, 5))
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    result = compute_shap_interactions(model, X[:20], [f"f{i}" for i in range(5)], top_k_pairs=3)

    assert result["status"] == "computed"
    assert len(result["top_interaction_pairs"]) <= 3
    for pair in result["top_interaction_pairs"]:
        assert "feature_a" in pair and "feature_b" in pair and "mean_abs_interaction" in pair


def test_compute_shap_interactions_handles_gradient_boosting() -> None:
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(1)
    X = rng.random((60, 4))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)

    result = compute_shap_interactions(model, X[:15], [f"f{i}" for i in range(4)])

    assert result["status"] == "computed"
