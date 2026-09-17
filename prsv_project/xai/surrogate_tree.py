from __future__ import annotations

"""
Global surrogate model distillation (v3.0).

Trains a shallow, human-readable decision tree to approximate the SVM's
overall decision boundary - not to replace the SVM (the tree is meaningfully
less accurate), but to give a plain-language rule summary of what the SVM
has generally learned, e.g. "IF edge_density > 0.18 AND green_ratio < 0.42
THEN Diseased". Complements SHAP (per-instance feature attribution) and LIME
(per-instance local linear approximation) with a *global* view.

Fidelity (how often the tree agrees with the SVM it's distilling) is reported
alongside the tree so it's clear this is an approximation, not a substitute
- a low-fidelity surrogate should not be trusted as a rule summary.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text


@dataclass
class SurrogateTreeResult:
    tree: DecisionTreeClassifier
    fidelity: float
    rules_text: str
    feature_importances: Dict[str, float]


def distill_surrogate_tree(
    X_train_scaled: np.ndarray,
    base_model: Any,
    feature_names: List[str],
    max_depth: int = 4,
    random_state: int = 42,
) -> SurrogateTreeResult:
    """
    base_model: the already-trained classifier (e.g. the SVM) whose decision
    boundary the tree should imitate. The tree is trained on the *model's own
    predictions*, not the original labels - that's what makes it a surrogate
    of the model rather than a second, independent classifier.
    """
    surrogate_labels = base_model.predict(X_train_scaled)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(5, int(0.02 * len(X_train_scaled))),
        random_state=random_state,
    )
    tree.fit(X_train_scaled, surrogate_labels)

    tree_predictions = tree.predict(X_train_scaled)
    fidelity = float(accuracy_score(surrogate_labels, tree_predictions))

    rules_text = export_text(tree, feature_names=feature_names)

    importances = {
        name: round(float(importance), 6)
        for name, importance in zip(feature_names, tree.feature_importances_)
    }

    return SurrogateTreeResult(
        tree=tree,
        fidelity=round(fidelity, 4),
        rules_text=rules_text,
        feature_importances=importances,
    )
