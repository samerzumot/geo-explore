from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class CVResult:
    reports: List[str]
    confusion_matrices: List[np.ndarray]


def assign_spatial_blocks(rows: np.ndarray, cols: np.ndarray, k: int) -> np.ndarray:
    # Simple grid-based blocking along rows/cols
    # Create k blocks by quantiles of row and col indices
    row_bins = np.linspace(rows.min(), rows.max() + 1, k + 1, dtype=int)
    col_bins = np.linspace(cols.min(), cols.max() + 1, k + 1, dtype=int)
    row_ids = np.digitize(rows, row_bins) - 1
    col_ids = np.digitize(cols, col_bins) - 1
    return row_ids * k + col_ids  # k*k blocks


def spatial_block_cv(
    df: pd.DataFrame,
    feature_names: List[str],
    labeled_mask: np.ndarray,
    y: np.ndarray,
    k_blocks: int = 5,
    random_state: int = 42,
    rf_params: dict | None = None,
) -> CVResult:
    rf_params = rf_params or {}

    rows = df.loc[labeled_mask, 'row'].to_numpy()
    cols = df.loc[labeled_mask, 'col'].to_numpy()
    blocks = assign_spatial_blocks(rows, cols, k_blocks)

    X = df.loc[labeled_mask, feature_names].to_numpy()
    y_lab = y.copy()

    # Create folds by block id; ensure reproducibility
    unique_blocks = np.unique(blocks)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_blocks)
    folds = np.array_split(unique_blocks, k_blocks)

    reports: List[str] = []
    cms: List[np.ndarray] = []

    for fold_blocks in folds:
        test_mask = np.isin(blocks, fold_blocks)
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        X_tr, y_tr = X[train_mask], y_lab[train_mask]
        X_te, y_te = X[test_mask], y_lab[test_mask]
        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1, **rf_params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        reports.append(
            classification_report(y_te, y_pred)
        )
        cms.append(confusion_matrix(y_te, y_pred))

    return CVResult(reports=reports, confusion_matrices=cms)
