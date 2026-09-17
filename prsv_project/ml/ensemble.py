from __future__ import annotations

"""
Stacked ensemble classifier (v3.0).

Upgrades the single-SVM baseline (ml/train_svm.py) to a proper stacked
generalization ensemble: calibrated SVM + Random Forest + Gradient Boosting
as base learners, with a Logistic Regression meta-learner combining their
out-of-fold predictions. This is a real StackingClassifier (scikit-learn),
not soft-voting - the meta-learner learns how much to trust each base
model's opinion rather than averaging them blindly.

Kept as a separate, additive model (models/ensemble_model.joblib) rather than
replacing the SVM pipeline, so the original model stays available for direct
comparison (see evaluation/statistical_tests.py's McNemar's test, which is
specifically designed to compare this ensemble against the baseline SVM).
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


@dataclass
class EnsembleTrainingResult:
    model: StackingClassifier
    metadata: Dict[str, Any]


def build_stacking_ensemble(random_state: int = 42, cv_folds: int = 5) -> StackingClassifier:
    base_learners = [
        (
            "svm",
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
            ),
        ),
    ]

    meta_learner = LogisticRegression(max_iter=1000, class_weight="balanced")

    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    return stacking_model


def train_stacking_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> EnsembleTrainingResult:
    min_class_count = int(np.min(np.bincount(y_train))) if len(np.unique(y_train)) > 1 else len(y_train)
    cv_folds = max(2, min(5, min_class_count))

    model = build_stacking_ensemble(random_state=random_state, cv_folds=cv_folds)
    model.fit(X_train, y_train)

    metadata = {
        "model_type": "StackingClassifier",
        "base_learners": ["svm_rbf", "random_forest", "gradient_boosting"],
        "meta_learner": "logistic_regression",
        "cv_folds": cv_folds,
        "stack_method": "predict_proba",
    }
    return EnsembleTrainingResult(model=model, metadata=metadata)
