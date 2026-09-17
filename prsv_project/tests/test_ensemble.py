import numpy as np

from ml.ensemble import train_stacking_ensemble


def _make_synthetic_data(n_samples: int = 150, n_features: int = 12, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] * 0.7 + X[:, 1] * 0.3 > 0).astype(int)
    return X, y


def test_train_stacking_ensemble_fits_and_predicts() -> None:
    X, y = _make_synthetic_data()

    result = train_stacking_ensemble(X, y, random_state=42)
    predictions = result.model.predict(X)
    probabilities = result.model.predict_proba(X)

    assert len(predictions) == len(y)
    assert probabilities.shape == (len(X), 2)
    assert result.metadata["model_type"] == "StackingClassifier"
    assert set(result.metadata["base_learners"]) == {"svm_rbf", "random_forest", "gradient_boosting"}


def test_train_stacking_ensemble_reasonable_accuracy_on_separable_data() -> None:
    X, y = _make_synthetic_data(n_samples=200)

    result = train_stacking_ensemble(X, y, random_state=42)
    predictions = result.model.predict(X)
    accuracy = float(np.mean(predictions == y))

    # Not a rigorous benchmark - just confirms the ensemble actually learns
    # something on clearly separable synthetic data rather than degenerating.
    assert accuracy > 0.7
