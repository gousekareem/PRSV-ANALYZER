from __future__ import annotations

"""
Trust scores (v3.1): flags predictions that are technically high-confidence
(per the model's own softmax/predict_proba output) but sit in a sparse,
poorly-represented region of the training feature space - a case where the
model's stated confidence shouldn't be fully trusted, since it's
extrapolating rather than interpolating.

Implementation: for a given prediction, compares the distance to the
nearest same-predicted-class training point against the distance to the
nearest different-class training point (the actual "trust score" formulation
from Jiang et al., 2018, "To Trust Or Not To Trust A Classifier") - a ratio
> 1 means the point is closer to a different class than to its own
predicted class, a red flag regardless of what predict_proba says.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class TrustScoreResult:
    trust_score: float
    distance_to_predicted_class: float
    distance_to_nearest_other_class: float
    is_low_trust: bool


class TrustScorer:
    def __init__(self, X_train_scaled: np.ndarray, y_train: np.ndarray, k: int = 5) -> None:
        self.classes = np.unique(y_train)
        self.neighbor_indices_by_class = {}
        for cls in self.classes:
            class_points = X_train_scaled[y_train == cls]
            n_neighbors = min(k, len(class_points))
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(class_points)
            self.neighbor_indices_by_class[cls] = nn

    def score(self, x_scaled: np.ndarray, predicted_class: int, low_trust_threshold: float = 1.2) -> TrustScoreResult:
        x_scaled = x_scaled.reshape(1, -1)

        own_class_nn = self.neighbor_indices_by_class[predicted_class]
        own_distances, _ = own_class_nn.kneighbors(x_scaled)
        distance_to_predicted = float(np.mean(own_distances))

        other_class_distances = []
        for cls, nn in self.neighbor_indices_by_class.items():
            if cls == predicted_class:
                continue
            distances, _ = nn.kneighbors(x_scaled)
            other_class_distances.append(float(np.mean(distances)))

        distance_to_other = min(other_class_distances) if other_class_distances else float("inf")

        trust_score = distance_to_other / (distance_to_predicted + 1e-8)

        return TrustScoreResult(
            trust_score=round(trust_score, 6),
            distance_to_predicted_class=round(distance_to_predicted, 6),
            distance_to_nearest_other_class=round(distance_to_other, 6),
            is_low_trust=trust_score < low_trust_threshold,
        )
