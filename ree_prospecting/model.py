from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:  # pragma: no cover - optional
    SMOTE = None  # type: ignore

from .config import Config

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: RandomForestClassifier
    feature_names: list[str]
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    roc_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]
    feature_importances: pd.DataFrame
    permutation_importances: Optional[pd.DataFrame]


def _assign_spatial_blocks(gdf: gpd.GeoDataFrame, block_size_m: int) -> np.ndarray:
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    bx = np.floor(xs / block_size_m).astype(int)
    by = np.floor(ys / block_size_m).astype(int)
    return (bx * 1_000_000 + by)  # unique id per block


def label_training_samples(feature_gdf: gpd.GeoDataFrame, mrds: gpd.GeoDataFrame, config: Config) -> gpd.GeoDataFrame:
    """Label positives within buffer around MRDS points; sample negatives elsewhere."""
    if mrds.empty:
        LOGGER.warning("MRDS is empty; cannot create positive labels. Returning empty training set.")
        feature_gdf["label"] = pd.Series(dtype=int)
        return feature_gdf.iloc[0:0]

    buffer = mrds.buffer(config.model.positive_buffer_m)
    union = buffer.unary_union
    in_pos = feature_gdf.geometry.within(union)

    pos = feature_gdf[in_pos].copy()
    pos["label"] = 1

    neg = feature_gdf[~in_pos].copy()
    n_neg = int(len(pos) * config.model.random_negative_multiplier)
    if n_neg > len(neg):
        n_neg = len(neg)
    neg = neg.sample(n=n_neg, random_state=config.model.random_seed)
    neg["label"] = 0

    train = pd.concat([pos, neg], axis=0)
    return train


def train_random_forest_spatial_cv(train_gdf: gpd.GeoDataFrame, config: Config) -> TrainResult:
    features = [c for c in train_gdf.columns if c not in ("label", "geometry")]
    X = train_gdf[features].values
    y = train_gdf["label"].values.astype(int)

    # Group by spatial block to avoid leakage
    groups = _assign_spatial_blocks(train_gdf, config.model.spatial_block_size_m)
    gkf = GroupKFold(n_splits=min(config.model.cv_folds, len(np.unique(groups))))

    # Optionally handle imbalance with SMOTE on each fold
    use_smote = (SMOTE is not None) and (not config.model.class_weight_balanced)

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_proba_all: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_smote:
            sm = SMOTE(random_state=config.model.random_seed)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        clf = RandomForestClassifier(
            n_estimators=config.model.rf_n_estimators,
            max_depth=config.model.rf_max_depth,
            min_samples_leaf=config.model.rf_min_samples_leaf,
            class_weight=("balanced" if config.model.class_weight_balanced else None),
            random_state=config.model.random_seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_proba_all.extend(proba.tolist())

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="binary", zero_division=0)
    auc = roc_auc_score(y_true_all, y_proba_all) if len(set(y_true_all)) > 1 else float("nan")
    cm = confusion_matrix(y_true_all, y_pred_all)
    fpr, tpr, thr = roc_curve(y_true_all, y_proba_all) if len(set(y_true_all)) > 1 else (np.array([]), np.array([]), np.array([]))

    # Train final model on full dataset
    clf_final = RandomForestClassifier(
        n_estimators=config.model.rf_n_estimators,
        max_depth=config.model.rf_max_depth,
        min_samples_leaf=config.model.rf_min_samples_leaf,
        class_weight=("balanced" if config.model.class_weight_balanced else None),
        random_state=config.model.random_seed,
        n_jobs=-1,
    )
    clf_final.fit(X, y)

    # Feature importances and permutation
    importances = pd.DataFrame({
        "feature": features,
        "importance": clf_final.feature_importances_,
    }).sort_values("importance", ascending=False)

    perm_df = None
    try:
        perm = permutation_importance(clf_final, X, y, n_repeats=5, random_state=config.model.random_seed, n_jobs=-1)
        perm_df = pd.DataFrame({
            "feature": features,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
    except Exception as exc:
        LOGGER.warning("Permutation importance failed: %s", exc)

    metrics = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "roc_auc": float(auc)}

    return TrainResult(
        model=clf_final,
        feature_names=features,
        metrics=metrics,
        confusion_matrix=cm,
        roc_curve=(fpr, tpr, thr),
        feature_importances=importances,
        permutation_importances=perm_df,
    )


def predict_raster_probabilities(model: RandomForestClassifier, features: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray]:
    """Predict probability for each pixel and compute uncertainty as stddev across trees."""
    feat_names = list(features.data_vars)
    X = xr.concat([features[v] for v in feat_names], dim="variable").transpose("y", "x", "variable").values
    X = X.reshape(-1, len(feat_names))

    # Drop rows with NaNs and keep index map
    valid_mask = np.all(np.isfinite(X), axis=1)
    X_valid = X[valid_mask]

    proba = np.full(X.shape[0], np.nan, dtype=np.float32)
    std = np.full(X.shape[0], np.nan, dtype=np.float32)

    if X_valid.size > 0:
        proba_valid = model.predict_proba(X_valid)[:, 1]
        # Tree-wise std
        all_tree_proba = np.vstack([estimator.predict_proba(X_valid)[:, 1] for estimator in model.estimators_])
        std_valid = all_tree_proba.std(axis=0)

        proba[valid_mask] = proba_valid
        std[valid_mask] = std_valid

    proba_da = xr.DataArray(
        proba.reshape((features.sizes["y"], features.sizes["x"])),
        coords={"y": features.y, "x": features.x},
        name="prospectivity",
    )
    proba_da.rio.write_crs(features.rio.crs, inplace=True)

    std_da = xr.DataArray(
        std.reshape((features.sizes["y"], features.sizes["x"])),
        coords={"y": features.y, "x": features.x},
        name="uncertainty_std",
    )
    std_da.rio.write_crs(features.rio.crs, inplace=True)
    return proba_da, std_da
