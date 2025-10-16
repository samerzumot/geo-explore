"""Model training, validation, and prediction for the MVP.

Implements a RandomForest baseline with spatial cross-validation and outputs
probability and uncertainty maps.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

from .config import Config, DEFAULT_CONFIG
from .preprocessing import GridDefinition

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def assign_spatial_blocks(
    grid: GridDefinition, points: gpd.GeoDataFrame, blocks_xy: Tuple[int, int]
) -> np.ndarray:
    """Assign a spatial block id (0..N-1) to each point based on grid indices."""
    nx, ny = blocks_xy
    # Use original grid row/col if present; else approximate from geometry
    if {"row", "col"}.issubset(points.columns):
        cols = points["col"].to_numpy()
        rows = points["row"].to_numpy()
    else:
        # Fallback: compute col/row from transform (approx)
        inv = ~grid.transform
        cols = []
        rows = []
        for geom in points.to_crs(grid.crs).geometry:
            c, r = inv * (geom.x, geom.y)
            cols.append(int(np.clip(np.floor(c), 0, grid.width - 1)))
            rows.append(int(np.clip(np.floor(r), 0, grid.height - 1)))
        cols = np.array(cols)
        rows = np.array(rows)

    bx = np.floor(cols * nx / grid.width).astype(int)
    by = np.floor(rows * ny / grid.height).astype(int)
    block_id = by * nx + bx
    return block_id


def train_evaluate_rf(
    features_gdf: gpd.GeoDataFrame,
    label_col: str,
    grid: GridDefinition,
    config: Config = DEFAULT_CONFIG,
) -> Tuple[RandomForestClassifier, SimpleImputer, Dict[str, float], Dict[str, float], np.ndarray, np.ndarray]:
    """Train RandomForest with spatial CV and return fitted model and metrics.

    Returns (model, metrics, feature_importances) where metrics include
    precision, recall, f1, auc, and hit_rate_top10pct.
    """
    feature_cols = [c for c in features_gdf.columns if c not in {label_col, "geometry"}]
    X = features_gdf[feature_cols].to_numpy(dtype=np.float32)
    y = features_gdf[label_col].to_numpy(dtype=np.int32)

    groups = assign_spatial_blocks(grid, features_gdf, config.spatial_blocks)

    gkf = GroupKFold(n_splits=min(5, np.unique(groups).size))

    y_true_all: List[int] = []
    y_prob_all: List[float] = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        # Impute on train only, apply to test
        imputer_cv = SimpleImputer(strategy="median")
        X_train = imputer_cv.fit_transform(X[train_idx])
        X_test = imputer_cv.transform(X[test_idx])

        model = RandomForestClassifier(**config.rf_params)
        model.fit(X_train, y[train_idx])
        prob = model.predict_proba(X_test)[:, 1]
        y_true_all.extend(y[test_idx].tolist())
        y_prob_all.extend(prob.tolist())
        logger.info("Fold %d: AUC=%.3f", fold + 1, roc_auc_score(y[test_idx], prob))

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    y_pred = (y_prob >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_prob)

    # Hit-rate in top 10% scores
    threshold = np.quantile(y_prob, 0.9)
    top_mask = y_prob >= threshold
    hit_rate_top10 = (y_true[top_mask] == 1).mean() if top_mask.any() else float("nan")

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "hit_rate_top10pct": float(hit_rate_top10),
    }

    # Train final model on all data with imputer
    final_imputer = SimpleImputer(strategy="median")
    X_all = final_imputer.fit_transform(X)
    final_model = RandomForestClassifier(**config.rf_params)
    final_model.fit(X_all, y)

    # Feature importances
    importances = {feature_cols[i]: float(v) for i, v in enumerate(final_model.feature_importances_)}

    return final_model, final_imputer, metrics, importances, y_true, y_prob


def predict_full_grid(
    model: RandomForestClassifier,
    imputer: SimpleImputer,
    grid: GridDefinition,
    feature_arrays: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probability and uncertainty for each cell in the grid."""
    feature_cols = list(feature_arrays.keys())
    H, W = grid.height, grid.width
    X = np.column_stack([feature_arrays[c].ravel().astype("float32") for c in feature_cols])
    X_imp = imputer.transform(X)
    proba = model.predict_proba(X_imp)[:, 1]

    # Uncertainty estimate from tree ensemble spread
    est_probs = np.stack([est.predict_proba(X_imp)[:, 1] for est in model.estimators_], axis=0)
    std = est_probs.std(axis=0)

    return proba.reshape(H, W), std.reshape(H, W)


def save_metrics_and_importance(
    metrics: Dict[str, float],
    importances: Dict[str, float],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "feature_importance.json").write_text(json.dumps(importances, indent=2))

    # Bar chart for importances
    names = list(importances.keys())
    vals = [importances[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, vals)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
