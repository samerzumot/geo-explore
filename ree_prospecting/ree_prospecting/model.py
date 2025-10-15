"""Model training and prediction for REE prospectivity MVP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: RandomForestClassifier
    feature_names: List[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    report: str
    conf_matrix: np.ndarray
    importance: np.ndarray
    proba_std: np.ndarray


def make_labels_from_points(df_features: pd.DataFrame, positive_mask: np.ndarray, negative_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.full(df_features.shape[0], -1, dtype=int)
    y[positive_mask] = 1
    y[negative_mask] = 0
    labeled_idx = (y >= 0)
    return y[labeled_idx], labeled_idx


def train_rf(df_features: pd.DataFrame, feature_names: List[str], labeled_mask: np.ndarray, y: np.ndarray, random_state: int = 42,
             n_estimators: int = 300, max_depth: int | None = None, class_weight: str | dict | None = 'balanced_subsample', use_smote: bool = True) -> TrainResult:
    X = df_features.loc[labeled_mask, feature_names].to_numpy()
    # Handle NaNs
    nan_mask = np.any(~np.isfinite(X), axis=1)
    keep = ~nan_mask
    X = X[keep]
    y = y[keep]

    if use_smote:
        try:
            smote = SMOTE(random_state=random_state)
            X_res, y_res = smote.fit_resample(X, y)
            X, y = X_res, y_res
        except Exception as e:
            logger.warning("SMOTE failed (%s); proceeding without.", e)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X, y)

    y_proba = rf.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    importance = rf.feature_importances_

    # Uncertainty estimate: std of per-tree probabilities
    tree_probas = np.stack([t.predict_proba(X)[:, 1] for t in rf.estimators_], axis=1)
    proba_std = np.std(tree_probas, axis=1)

    return TrainResult(
        model=rf,
        feature_names=feature_names,
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        report=report,
        conf_matrix=cm,
        importance=importance,
        proba_std=proba_std,
    )


def predict_full_grid(model: RandomForestClassifier, df_features: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X_full = df_features[feature_names].to_numpy()
    # Handle NaNs
    nan_mask = np.any(~np.isfinite(X_full), axis=1)
    X_full = np.where(np.isfinite(X_full), X_full, np.nanmedian(X_full, axis=0))
    proba = model.predict_proba(X_full)[:, 1]
    # Uncertainty proxy: std across trees on full grid
    tree_probas = np.stack([t.predict_proba(X_full)[:, 1] for t in model.estimators_], axis=1)
    std = np.std(tree_probas, axis=1)
    return proba, std
