from __future__ import annotations

"""
Confidence calibration (v3.0).

SVC(probability=True) (used in ml/train_svm.py) derives its predict_proba
output from an internal 5-fold Platt scaling pass, but that pass is fit on
the *same* training data the SVM itself was fit on, which tends to produce
overconfident probabilities - the "deprecated" pattern flagged in the
upgrade list. CalibratedClassifierCV fixes this by fitting the calibration
map (Platt scaling or isotonic regression) on held-out cross-validation folds
instead, giving genuinely better-calibrated confidence scores.

This module intentionally does not touch ml/train_svm.py's default flow (it
stays the well-tested baseline). Instead it provides a `calibrate_model()
helper used by ml/train_ensemble.py and by evaluation/calibration_report.py,
so both the original and calibrated confidence behavior are available and
comparable via the calibration curve / Brier score report.
"""

from typing import Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


def calibrate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: Literal["sigmoid", "isotonic"] = "isotonic",
    cv: int = 5,
    base_estimator: SVC | None = None,
) -> CalibratedClassifierCV:
    """
    Fit a properly cross-validated calibration map on top of an SVM.

    method="sigmoid" is Platt scaling (better for small datasets / few
    hundred samples, less prone to overfitting the calibration map itself).
    method="isotonic" is more flexible but needs more calibration data
    (roughly 1000+ samples) to avoid overfitting - default to "isotonic" for
    reasonably sized datasets but callers with small datasets should pass
    method="sigmoid" explicitly.
    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf", class_weight="balanced", random_state=42)

    n_per_class = np.min(np.bincount(y_train)) if len(np.unique(y_train)) > 1 else len(y_train)
    effective_cv = max(2, min(cv, int(n_per_class)))

    calibrated = CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=effective_cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def predict_calibrated_proba(model: CalibratedClassifierCV, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)
