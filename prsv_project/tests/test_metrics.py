import numpy as np

from ml.metrics import compute_binary_metrics


def test_compute_binary_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_binary_metrics(y_true, y_pred, y_score, positive_label=1)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0