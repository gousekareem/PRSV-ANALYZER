from __future__ import annotations

"""
Feature quality analysis (v3.1): checks whether the 12 engineered features
(ml/feature_schema.py) are genuinely independent/informative, or partially
redundant - runnable directly against the existing feature CSV
(data/training_features.csv), no new data needed.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


@dataclass
class FeatureQualityReport:
    pca_explained_variance_ratio: List[float]
    pca_n_components_for_95pct: int
    mutual_information: Dict[str, float]
    rfecv_selected_features: List[str]
    rfecv_ranking: Dict[str, int]
    lda_class_separation_score: float


def run_pca_analysis(X: np.ndarray) -> tuple[List[float], int]:
    pca = PCA()
    pca.fit(X)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_for_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    return [round(float(v), 6) for v in explained], n_for_95


def run_mutual_information(X: np.ndarray, y: np.ndarray, feature_names: List[str], random_state: int = 42) -> Dict[str, float]:
    scores = mutual_info_classif(X, y, random_state=random_state)
    return {name: round(float(score), 6) for name, score in zip(feature_names, scores)}


def run_rfecv(X: np.ndarray, y: np.ndarray, feature_names: List[str], random_state: int = 42) -> tuple[List[str], Dict[str, int]]:
    min_class_count = int(np.min(np.bincount(y)))
    cv_folds = max(2, min(5, min_class_count))

    estimator = SVC(kernel="linear", class_weight="balanced", random_state=random_state)
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring="f1_macro",
        n_jobs=-1,
    )
    selector.fit(X, y)

    selected = [name for name, keep in zip(feature_names, selector.support_) if keep]
    ranking = {name: int(rank) for name, rank in zip(feature_names, selector.ranking_)}
    return selected, ranking


def run_lda_separation(X: np.ndarray, y: np.ndarray) -> float:
    """
    Fits LDA and reports its own training accuracy as a rough proxy for how
    linearly separable the classes are in this 12-dimensional feature
    space - a quick sanity check, not a generalization estimate.
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    return round(float(lda.score(X, y)), 6)


def analyze_feature_quality(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> FeatureQualityReport:
    explained_variance, n_components_95 = run_pca_analysis(X)
    mutual_information = run_mutual_information(X, y, feature_names)
    rfecv_selected, rfecv_ranking = run_rfecv(X, y, feature_names)
    lda_score = run_lda_separation(X, y)

    return FeatureQualityReport(
        pca_explained_variance_ratio=explained_variance,
        pca_n_components_for_95pct=n_components_95,
        mutual_information=mutual_information,
        rfecv_selected_features=rfecv_selected,
        rfecv_ranking=rfecv_ranking,
        lda_class_separation_score=lda_score,
    )
