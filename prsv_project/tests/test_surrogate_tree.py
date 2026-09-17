import numpy as np
from sklearn.svm import SVC

from xai.surrogate_tree import distill_surrogate_tree


def test_distill_surrogate_tree_produces_readable_rules() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 4))
    y = (X[:, 0] > 0).astype(int)

    base_model = SVC(kernel="rbf", probability=True, random_state=42)
    base_model.fit(X, y)

    result = distill_surrogate_tree(
        X, base_model, feature_names=["f0", "f1", "f2", "f3"], max_depth=3
    )

    assert 0.0 <= result.fidelity <= 1.0
    assert "f0" in result.rules_text or "f1" in result.rules_text
    assert set(result.feature_importances.keys()) == {"f0", "f1", "f2", "f3"}


def test_distill_surrogate_tree_high_fidelity_on_simple_boundary() -> None:
    """
    A tree distilling an SVM whose true boundary is itself axis-aligned
    (f0 > 0) should achieve high fidelity, since a shallow tree can represent
    that boundary almost exactly.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(int)

    base_model = SVC(kernel="linear", probability=True, random_state=42)
    base_model.fit(X, y)

    result = distill_surrogate_tree(X, base_model, feature_names=["f0", "f1", "f2"], max_depth=3)

    assert result.fidelity > 0.85
