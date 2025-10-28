from __future__ import annotations

"""Orchestration for REE prospectivity MVP (Mountain Pass region).

This pipeline:
- Loads config
- Downloads MRDS and SRTM DEM
- Computes features (terrain, proxies; Landsat optional)
- Builds labels from known deposits (buffer around MRDS points as positives; random negatives)
- Trains RandomForest baseline
- Predicts full-grid probabilities and uncertainty
- Writes GeoTIFF, hotspots GeoJSON, plots, and Folium HTML map
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio

from .config import CFG
from . import data_loader as dl
from . import preprocessing as prep
from . import model as mdl
from . import visualization as viz


def main() -> None:
    dl.ensure_logging()
    logger = logging.getLogger("ree.main")
    cfg = CFG
    p = cfg.paths

    # Ensure directories
    dl.ensure_dirs(p.data_dir, p.cache_dir, p.outputs_dir, p.figures_dir)

    # Define region bbox
    bbox_gdf = dl.get_bbox_polygon(cfg.region.min_lon, cfg.region.min_lat, cfg.region.max_lon, cfg.region.max_lat)

    # Download MRDS and filter
    mrds_csv = dl.download_mrds(p.cache_dir).path
    mrds = dl.load_mrds_points(mrds_csv, bbox_gdf)
    logger.info("MRDS points in bbox: %d", len(mrds))

    # Download DEM
    dem_res = dl.download_srtm_dem(bbox_gdf, p.cache_dir).path
    dem_grid = prep.read_raster(dem_res)

    # Reproject DEM to UTM with target resolution
    utm_crs = f"EPSG:{cfg.region.utm_epsg}"
    dem_grid_utm = prep.reproject_match(dem_grid, utm_crs, dst_res=cfg.model.grid_resolution_m)

    # Reproject MRDS to UTM
    mrds_utm = dl.reproject_vector(mrds, cfg.region.utm_epsg)

    # Build features (Landsat optional in MVP)
    df_features, grid_template, feature_names = prep.build_feature_stack(dem_grid_utm)

    # Create labels: positives around MRDS buffers, negatives sampled randomly
    h, w = grid_template.data.shape
    rr, cc = np.indices((h, w))
    # Rasterize positive buffer (e.g., 300 m)
    mrds_buffer = mrds_utm.copy()
    mrds_buffer["geometry"] = mrds_buffer.geometry.buffer(300)
    shapes = ((geom, 1) for geom in mrds_buffer.geometry)
    pos_mask = rasterio.features.rasterize(shapes=shapes, out_shape=(h, w), transform=grid_template.transform, fill=0).astype(bool)

    # Negative mask: random sample outside a larger exclusion (e.g., 1km)
    mrds_excl = mrds_utm.copy()
    mrds_excl["geometry"] = mrds_excl.geometry.buffer(1000)
    excl_shapes = ((g, 1) for g in mrds_excl.geometry)
    excl_mask = rasterio.features.rasterize(shapes=excl_shapes, out_shape=(h, w), transform=grid_template.transform, fill=0).astype(bool)
    rng = np.random.RandomState(cfg.model.random_state)
    neg_candidates = ~excl_mask
    neg_indices = np.where(neg_candidates.reshape(-1))[0]
    # Sample similar count to positives
    num_pos = int(pos_mask.sum())
    num_neg = min(num_pos, len(neg_indices))
    sampled_neg = np.zeros(h * w, dtype=bool)
    if num_neg > 0:
        sampled_idx = rng.choice(neg_indices, size=num_neg, replace=False)
        sampled_neg[sampled_idx] = True
    neg_mask = sampled_neg.reshape(h, w)

    y_labels, labeled_mask = mdl.make_labels_from_points(df_features, pos_mask.reshape(-1), neg_mask.reshape(-1))

    # Train RF
    result = mdl.train_rf(
        df_features, feature_names, labeled_mask, y_labels,
        random_state=cfg.model.random_state,
        n_estimators=cfg.model.rf_n_estimators,
        max_depth=cfg.model.rf_max_depth,
        class_weight=cfg.model.rf_class_weight,
        use_smote=cfg.model.smote_enabled,
    )

    # Predict full grid
    proba, std = mdl.predict_full_grid(result.model, df_features, feature_names)
    proba_grid = proba.reshape(h, w)

    # Outputs
    prob_tif = p.outputs_dir / "prospectivity_prob.tif"
    viz.array_to_geotiff(prob_tif, proba_grid, grid_template.transform, grid_template.crs)

    hotspots_gdf = viz.top_hotspots_to_geojson(proba_grid, grid_template.transform, grid_template.crs, top_k=10)
    hotspots_path = p.outputs_dir / "hotspots.geojson"
    hotspots_gdf.to_file(hotspots_path, driver='GeoJSON')

    # Feature importance
    viz.feature_importance_plot(p.figures_dir / "feature_importance.png", result.feature_names, result.importance)

    # Classification report
    viz.classification_report_to_text(p.outputs_dir / "model_report.txt", result.report, result.conf_matrix)

    # Folium map
    map_html = p.outputs_dir / "interactive_map.html"
    viz.folium_map(map_html, proba_grid, grid_template.transform, grid_template.crs, hotspots_gdf, mrds_utm)

    logger.info("Pipeline complete. Outputs written to %s", p.outputs_dir)


if __name__ == "__main__":
    main()
