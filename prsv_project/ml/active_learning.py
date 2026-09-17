from __future__ import annotations

"""
Active & semi-supervised learning (v3.1).

Honesty note: these techniques are only *useful* once a pool of unlabeled
field images exists (e.g. photos submitted through the chatbot without a
confirmed outcome yet) - the shipped demo dataset is fully labeled, so
there's nothing to actively sample from today. What's implemented here is
the real, working machinery (uncertainty scoring, committee disagreement,
pseudo-label generation with a confidence gate) ready to run the moment such
a pool exists - not the `modAL` library specifically (which adds a
dependency for what is, underneath, a few lines of numpy over
`predict_proba` outputs already available from this project's models).
"""

from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class UncertaintySample:
    index: int
    uncertainty_score: float
    predicted_class: int
    predicted_probability: float


def uncertainty_sampling(model: Any, X_unlabeled: np.ndarray, n_samples: int = 10) -> List[UncertaintySample]:
    """
    Least-confidence uncertainty sampling: ranks unlabeled samples by how
    close the model's top predicted probability is to a coin flip (for
    binary classification, closest to 0.5 = most uncertain), returning the
    `n_samples` most valuable candidates for manual labeling.
    """
    probabilities = model.predict_proba(X_unlabeled)
    top_probability = np.max(probabilities, axis=1)
    predicted_class = np.argmax(probabilities, axis=1)

    uncertainty = 1.0 - top_probability  # higher = more uncertain
    ranked_indices = np.argsort(uncertainty)[::-1][:n_samples]

    return [
        UncertaintySample(
            index=int(idx),
            uncertainty_score=round(float(uncertainty[idx]), 6),
            predicted_class=int(predicted_class[idx]),
            predicted_probability=round(float(top_probability[idx]), 6),
        )
        for idx in ranked_indices
    ]


def query_by_committee(models: List[Any], X_unlabeled: np.ndarray, n_samples: int = 10) -> List[UncertaintySample]:
    """
    Ranks unlabeled samples by disagreement across an ensemble of already-
    trained models (e.g. the base learners inside ml/ensemble.py's stacking
    classifier) - vote entropy across the committee's predictions, rather
    than a single model's own confidence.
    """
    all_predictions = np.array([model.predict(X_unlabeled) for model in models])  # (n_models, n_samples)
    n_models = len(models)

    disagreement_scores = []
    for col in range(all_predictions.shape[1]):
        votes = all_predictions[:, col]
        _, counts = np.unique(votes, return_counts=True)
        vote_fractions = counts / n_models
        entropy = -np.sum(vote_fractions * np.log2(vote_fractions + 1e-12))
        disagreement_scores.append(entropy)

    disagreement_scores = np.array(disagreement_scores)
    ranked_indices = np.argsort(disagreement_scores)[::-1][:n_samples]

    majority_predictions = np.array(
        [np.bincount(all_predictions[:, i].astype(int)).argmax() for i in range(all_predictions.shape[1])]
    )

    return [
        UncertaintySample(
            index=int(idx),
            uncertainty_score=round(float(disagreement_scores[idx]), 6),
            predicted_class=int(majority_predictions[idx]),
            predicted_probability=float(np.mean(all_predictions[:, idx] == majority_predictions[idx])),
        )
        for idx in ranked_indices
    ]


def self_training_pseudo_labels(
    model: Any,
    X_unlabeled: np.ndarray,
    confidence_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pseudo-labels only the unlabeled samples the model is highly confident
    about (>= confidence_threshold), for a retraining round that folds them
    into the labeled set. Returns (pseudo_labeled_X, pseudo_labels,
    original_indices) so the caller can trace which unlabeled samples were
    used and re-verify them later if needed - pseudo-labels should always be
    treated as lower-trust than human-confirmed labels.
    """
    probabilities = model.predict_proba(X_unlabeled)
    top_probability = np.max(probabilities, axis=1)
    predicted_class = np.argmax(probabilities, axis=1)

    confident_mask = top_probability >= confidence_threshold
    confident_indices = np.where(confident_mask)[0]

    return X_unlabeled[confident_mask], predicted_class[confident_mask], confident_indices
