from __future__ import annotations

"""
Fairness/bias auditing across dataset subgroups (v3.1): breaks down model
performance by a subgroup column (e.g. photo source/session, camera type)
instead of reporting one aggregate accuracy number that can hide meaningful
performance gaps between subgroups.

Honesty note: this project's shipped feature CSV (data/training_features.csv)
doesn't currently have a subgroup/source column - the demo dataset's
provenance metadata wasn't tracked per-image. This module is real,
functioning infrastructure that computes a genuine per-subgroup breakdown
the moment such a column exists (it's a one-column addition to the feature
extraction manifest, see scripts/prepare_manifest.py); until then, calling
it with a synthetic or absent subgroup column will report that honestly
rather than fabricate a "no bias found" result.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ml.metrics import compute_binary_metrics


@dataclass
class SubgroupReport:
    subgroup: str
    n_samples: int
    metrics: dict


def audit_by_subgroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> List[SubgroupReport]:
    reports: List[SubgroupReport] = []
    for subgroup in sorted(set(subgroup_labels)):
        mask = subgroup_labels == subgroup
        if mask.sum() == 0:
            continue
        subgroup_score = y_score[mask] if y_score is not None else None
        metrics = compute_binary_metrics(y_true[mask], y_pred[mask], subgroup_score)
        reports.append(SubgroupReport(subgroup=str(subgroup), n_samples=int(mask.sum()), metrics=metrics))
    return reports


def largest_subgroup_gap(reports: List[SubgroupReport], metric: str = "accuracy") -> Dict[str, object]:
    if len(reports) < 2:
        return {
            "status": "insufficient_subgroups",
            "message": (
                "Fewer than two subgroups with data - either the dataset has no "
                "subgroup/source metadata yet, or all samples share one subgroup. "
                "See this module's docstring for what's needed to run a real audit."
            ),
        }

    scores = {r.subgroup: r.metrics.get(metric, 0.0) for r in reports}
    best_subgroup = max(scores, key=scores.get)
    worst_subgroup = min(scores, key=scores.get)
    gap = scores[best_subgroup] - scores[worst_subgroup]

    return {
        "status": "audited",
        "metric": metric,
        "best_subgroup": best_subgroup,
        "best_score": round(scores[best_subgroup], 4),
        "worst_subgroup": worst_subgroup,
        "worst_score": round(scores[worst_subgroup], 4),
        "gap": round(gap, 4),
        "interpretation": (
            f"A gap of {gap:.4f} in {metric} between the best- and worst-performing "
            "subgroups. Gaps above ~0.05-0.10 typically warrant investigation into "
            "whether the underrepresented subgroup needs more training data."
        ),
    }
