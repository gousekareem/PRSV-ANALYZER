from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from app.utils.json_utils import save_json
from ml.metrics import compute_binary_metrics


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    output_dir: Path,
    class_labels: list[str],
) -> Dict[str, Any]:
    """
    Evaluate a binary classifier and save metrics and plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_binary_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=1)
    save_json(output_dir / "evaluation_metrics.json", metrics)

    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax_cm, colorbar=False)
    fig_cm.tight_layout()
    fig_cm.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig_cm)

    if y_score is not None and len(np.unique(y_true)) == 2:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax_roc)
        fig_roc.tight_layout()
        fig_roc.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close(fig_roc)

        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax_pr)
        fig_pr.tight_layout()
        fig_pr.savefig(output_dir / "precision_recall_curve.png", dpi=200)
        plt.close(fig_pr)

    return metrics