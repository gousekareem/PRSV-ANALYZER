from __future__ import annotations

"""
Statistical evaluation rigor (v3.0): McNemar's test for comparing two
classifiers on the same test set, plus calibration curves and Brier score
for checking whether a "90% confidence" prediction is actually right 90% of
the time - going beyond plain accuracy/F1, which the upgrade list flagged as
the single highest-value validation upgrade alongside an independent-source
holdout set (see evaluation/holdout_eval.py).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


@dataclass
class McNemarResult:
    statistic: float
    p_value: float
    contingency_table: List[List[int]]
    significant_at_0_05: bool
    interpretation: str


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> McNemarResult:
    """
    McNemar's test for two classifiers evaluated on the *same* test samples.

    Tests whether classifier A and classifier B disagree in a systematically
    different way (not whether one is more accurate overall) - the correct
    test for "is this new model meaningfully different from the old one",
    rather than eyeballing a difference in accuracy percentages.

    Uses the chi-square approximation with continuity correction for cell
    counts that support it, and the exact binomial test for small discordant
    counts (n01 + n10 < 25), matching common statistical practice and
    avoiding scipy.stats.contingency_tables' need for statsmodels.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    n_00 = int(np.sum(~a_correct & ~b_correct))
    n_01 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n_10 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    n_11 = int(np.sum(a_correct & b_correct))

    discordant = n_01 + n_10

    if discordant == 0:
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            contingency_table=[[n_11, n_10], [n_01, n_00]],
            significant_at_0_05=False,
            interpretation="Classifiers made identical correctness patterns on every test sample (no discordant pairs).",
        )

    if discordant < 25:
        from scipy.stats import binomtest

        result = binomtest(min(n_01, n_10), discordant, 0.5, alternative="two-sided")
        statistic = float(min(n_01, n_10))
        p_value = float(result.pvalue)
    else:
        statistic = float((abs(n_01 - n_10) - 1) ** 2 / discordant)  # continuity-corrected chi-square
        from scipy.stats import chi2

        p_value = float(1 - chi2.cdf(statistic, df=1))

    significant = p_value < 0.05
    interpretation = (
        "The two models' error patterns differ significantly (p < 0.05): "
        + ("model B corrected more of A's errors than it introduced." if n_01 > n_10
           else "model A corrected more of B's errors than it introduced.")
        if significant
        else "No statistically significant difference in error patterns between the two models (p >= 0.05)."
    )

    return McNemarResult(
        statistic=round(statistic, 4),
        p_value=round(p_value, 6),
        contingency_table=[[n_11, n_10], [n_01, n_00]],
        significant_at_0_05=significant,
        interpretation=interpretation,
    )


@dataclass
class CalibrationReport:
    brier_score: float
    bin_true_frequencies: List[float]
    bin_predicted_probabilities: List[float]
    plot_path: Optional[str]


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Optional[Path] = None,
    n_bins: int = 10,
    model_name: str = "model",
) -> CalibrationReport:
    """
    Reliability diagram + Brier score for a binary classifier's predicted
    probability of the positive class.
    """
    brier = float(brier_score_loss(y_true, y_prob))
    true_freq, pred_prob = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    plot_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        ax.plot(pred_prob, true_freq, marker="o", label=model_name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Calibration curve ({model_name}) - Brier score: {brier:.4f}")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"calibration_curve_{model_name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        plot_path = str(path)

    return CalibrationReport(
        brier_score=round(brier, 6),
        bin_true_frequencies=[round(float(v), 4) for v in true_freq],
        bin_predicted_probabilities=[round(float(v), 4) for v in pred_prob],
        plot_path=plot_path,
    )


def compare_models_report(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_prob_a: Optional[np.ndarray] = None,
    y_prob_b: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    name_a: str = "baseline_svm",
    name_b: str = "stacking_ensemble",
) -> Dict[str, Any]:
    """
    One-stop comparison bundling McNemar's test plus per-model calibration
    reports, used by ml/train_ensemble.py to produce a head-to-head summary
    of the new ensemble against the existing baseline SVM on the same holdout
    split.
    """
    mcnemar = mcnemar_test(y_true, y_pred_a, y_pred_b)

    result: Dict[str, Any] = {"mcnemar": mcnemar.__dict__}

    if y_prob_a is not None:
        result[f"calibration_{name_a}"] = calibration_report(
            y_true, y_prob_a, output_dir=output_dir, model_name=name_a
        ).__dict__
    if y_prob_b is not None:
        result[f"calibration_{name_b}"] = calibration_report(
            y_true, y_prob_b, output_dir=output_dir, model_name=name_b
        ).__dict__

    return result
