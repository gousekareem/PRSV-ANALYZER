from __future__ import annotations

"""
SHAP interaction values (v3.1): reveal which *pairs* of features jointly
drive predictions (e.g. "does high entropy only matter when green_ratio is
also low?"), rather than the per-feature-alone attribution the existing
SHAP explainer (ml/shap_explainer.py) provides.

Honesty note on scope: SHAP's exact interaction-value computation
(`shap_interaction_values`) is only implemented for `TreeExplainer` in the
`shap` library, not for the `KernelExplainer` used elsewhere in this project
for the SVM (KernelExplainer approximates Shapley values generically but
doesn't expose the interaction decomposition). This module therefore runs
interaction values against the tree-based members of the v3.1 stacking
ensemble (ml/ensemble.py's Random Forest / Gradient Boosting), which is the
correct, supported way to get this feature - not a limitation introduced by
this project, but an accurate reflection of what SHAP itself supports for a
kernel-based model versus a tree-based one.
"""

from typing import Dict, List, Tuple

import numpy as np


def compute_shap_interactions(tree_model, X: np.ndarray, feature_names: List[str], top_k_pairs: int = 10) -> Dict[str, object]:
    try:
        import shap
    except Exception:  # noqa: BLE001 - optional dependency (already required elsewhere, guarded for safety)
        return {"status": "shap_not_available"}

    try:
        explainer = shap.TreeExplainer(tree_model)
        interaction_values = explainer.shap_interaction_values(X)

        # Binary classification via TreeExplainer has returned interaction
        # values in different shapes across shap versions:
        #   - older versions: a list of (n_samples, n_features, n_features)
        #     arrays, one per class
        #   - newer versions (as installed here): a single 4D array shaped
        #     (n_samples, n_features, n_features, n_classes)
        # Normalize both to a (n_samples, n_features, n_features) array for
        # the positive/last class either way.
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[-1]
        else:
            interaction_values = np.asarray(interaction_values)
            if interaction_values.ndim == 4:
                interaction_values = interaction_values[:, :, :, -1]

        mean_abs_interactions = np.mean(np.abs(interaction_values), axis=0)
        n_features = len(feature_names)

        pairs: List[Tuple[str, str, float]] = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                pairs.append((feature_names[i], feature_names[j], float(mean_abs_interactions[i, j])))

        pairs.sort(key=lambda p: p[2], reverse=True)
        top_pairs = pairs[:top_k_pairs]

        return {
            "status": "computed",
            "top_interaction_pairs": [
                {"feature_a": a, "feature_b": b, "mean_abs_interaction": round(score, 6)} for a, b, score in top_pairs
            ],
        }
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        return {"status": "error", "message": str(exc)}
