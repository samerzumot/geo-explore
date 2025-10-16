from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: RandomForestClassifier
    feature_names: List[str]
    report: str
    conf_matrix: np.ndarray
    importance: np.ndarray


def train_rf(
    df_features: pd.DataFrame,
    feature_names: List[str],
    labeled_mask: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
    class_weight: str | dict | None = 'balanced_subsample',
    use_smote: bool = True,
) -> TrainResult:
    X = df_features.loc[labeled_mask, feature_names].to_numpy()
    nan_mask = np.any(~np.isfinite(X), axis=1)
    keep = ~nan_mask
    X = X[keep]
    y = y[keep]

    if use_smote:
        try:
            smote = SMOTE(random_state=random_state)
            X, y = smote.fit_resample(X, y)
        except Exception as e:
            logger.warning("SMOTE failed (%s); proceeding without.", e)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        oob_score=False,
    )
    rf.fit(X, y)

    report = classification_report(y, rf.predict(X))
    cm = confusion_matrix(y, rf.predict(X))
    importance = rf.feature_importances_

    return TrainResult(
        model=rf,
        feature_names=feature_names,
        report=report,
        conf_matrix=cm,
        importance=importance,
    )


def predict_full_grid(model: RandomForestClassifier, df_features: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    X_full = df_features[feature_names].to_numpy()
    X_full = np.where(np.isfinite(X_full), X_full, np.nanmedian(X_full, axis=0))
    return model.predict_proba(X_full)[:, 1]
