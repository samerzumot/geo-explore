"""Orchestration script for end-to-end MVP run.

Steps:
1) Load data (DEM, Landsat, MRDS)
2) Build common grid and compute features
3) Create training labels from MRDS buffers and random background
4) Train RandomForest with spatial CV and evaluate
5) Predict full-grid probabilities and uncertainty
6) Save outputs (GeoTIFF, hotspots, folium map, charts)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

import numpy as np
import geopandas as gpd

from .config import DEFAULT_CONFIG, Config
from .data_loader import (
    RasterLayer,
    apply_landsat_scaling,
    download_dem_srtm,
    load_landsat_stack,
    load_mrds_points,
    open_raster_clipped,
)
from .preprocessing import (
    GridDefinition,
    build_feature_gdf,
    compute_dem_features,
    compute_landsat_features,
    make_common_grid,
    resample_to_grid,
)
from .model import predict_full_grid, save_metrics_and_importance, train_evaluate_rf
from .visualization import (
    build_folium_map,
    extract_hotspot_polygons,
    plot_confusion_and_roc,
    write_geotiff,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_training_labels(
    grid: GridDefinition,
    features_gdf: gpd.GeoDataFrame,
    mrds_points: gpd.GeoDataFrame,
    buffer_m: float,
    negative_multiplier: float,
) -> gpd.GeoDataFrame:
    """Assign binary labels using buffers around MRDS points and random negatives."""
    if mrds_points.empty:
        raise RuntimeError("No MRDS points found; cannot create training labels")

    pos_union = mrds_points.to_crs(grid.crs).buffer(buffer_m).unary_union
    in_pos = features_gdf.to_crs(grid.crs).geometry.within(pos_union)

    pos_samples = features_gdf[in_pos].copy()
    neg_candidates = features_gdf[~in_pos].copy()

    n_pos = len(pos_samples)
    n_neg = int(np.ceil(negative_multiplier * n_pos))
    if len(neg_candidates) < n_neg:
        n_neg = len(neg_candidates)
    neg_samples = neg_candidates.sample(n=n_neg, random_state=42)

    pos_samples["label"] = 1
    neg_samples["label"] = 0

    labeled = (
        gpd.GeoDataFrame(pd.concat([pos_samples, neg_samples], ignore_index=True), crs=features_gdf.crs)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    return labeled


def run_pipeline(config: Config = DEFAULT_CONFIG) -> None:
    # 1) Data loading
    logger.info("Loading MRDS points...")
    mrds = load_mrds_points(config)

    logger.info("Downloading DEM...")
    dem_path = download_dem_srtm(config)
    dem_raw = open_raster_clipped(dem_path, config.bbox_wgs84)

    logger.info("Loading Landsat bands...")
    landsat_bands = apply_landsat_scaling(load_landsat_stack(config))

    # 2) Grid + resampling
    grid = make_common_grid(config)

    dem = resample_to_grid(dem_raw, grid, resampling=rasterio.enums.Resampling.bilinear)
    b4 = resample_to_grid(landsat_bands["B4"], grid)
    b5 = resample_to_grid(landsat_bands["B5"], grid)
    b6 = resample_to_grid(landsat_bands["B6"], grid)
    b7 = resample_to_grid(landsat_bands["B7"], grid)

    # 3) Feature engineering
    dem_feats = compute_dem_features(dem, cellsize=config.resolution_m)
    sat_feats = compute_landsat_features({"B4": b4, "B5": b5, "B6": b6, "B7": b7}, config.ndvi_mask_threshold)

    # Assemble features for modeling (exclude NDVI to avoid masking as target leakage)
    feature_arrays = {
        **dem_feats,
        **{k: v for k, v in sat_feats.items() if k != "ndvi"},
    }

    features_gdf = build_feature_gdf(grid, feature_arrays)

    # 4) Labels from MRDS buffers + negatives
    labeled = create_training_labels(
        grid,
        features_gdf,
        mrds_points=mrds,
        buffer_m=config.positive_buffer_m,
        negative_multiplier=config.negative_multiplier,
    )

    # 5) Model training + evaluation
    model, imputer, metrics, importances, y_true, y_prob = train_evaluate_rf(labeled, "label", grid, config)
    logger.info("Metrics: %s", metrics)

    # 6) Predict full grid
    proba_grid, std_grid = predict_full_grid(model, imputer, grid, feature_arrays)

    # 7) Outputs
    write_geotiff(config.output_dir / "prospectivity_prob.tif", proba_grid, grid)
    write_geotiff(config.output_dir / "prospectivity_std.tif", std_grid, grid)

    hotspots = extract_hotspot_polygons(proba_grid, grid, quantile=0.9, top_k=10)
    # Save hotspots
    from .visualization import save_hotspots_geojson
    save_hotspots_geojson(config.output_dir / "hotspots.geojson", hotspots)
    # Metrics plots
    plot_confusion_and_roc(y_true, y_prob, config.output_dir)
    save_metrics_and_importance(metrics, importances, config.output_dir)

    # Simple validation overlay
    build_folium_map(proba_grid, hotspots, mrds, grid, config.output_dir / "prospectivity_map.html")


if __name__ == "__main__":
    run_pipeline()
