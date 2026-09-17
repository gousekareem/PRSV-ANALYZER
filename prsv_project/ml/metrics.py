from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute specificity for binary classification.
    Assumes negative class is index 0 in confusion matrix ordering.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0

    tn, fp, fn, tp = cm.ravel()
    denominator = tn + fp
    if denominator == 0:
        return 0.0
    return float(tn / denominator)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    positive_label: Any = 1,
) -> Dict[str, Any]:
    """
    Compute research-friendly binary classification metrics.
    """
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "specificity": float(compute_specificity(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics