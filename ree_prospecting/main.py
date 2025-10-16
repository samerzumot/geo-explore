from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

from .config import CFG
from . import data_loader as dl
from . import preprocessing as prep
from . import model as mdl
from . import spatial_cv as scv
from . import visualization as viz


def main() -> None:
    dl.ensure_logging()
    logger = logging.getLogger("ree.main")
    cfg = CFG
    p = cfg.paths

    dl.ensure_dirs(p.data_dir, p.cache_dir, p.outputs_dir, p.figures_dir)

    bbox_gdf = dl.get_bbox_polygon(cfg.region.min_lon, cfg.region.min_lat, cfg.region.max_lon, cfg.region.max_lat)

    mrds_csv = dl.download_mrds(p.cache_dir).path
    mrds = dl.load_mrds_points(mrds_csv, bbox_gdf)
    logger.info("MRDS points in bbox: %d", len(mrds))

    dem_res = dl.download_srtm_dem(bbox_gdf, p.cache_dir).path
    dem_grid = prep.read_raster(dem_res)

    utm_crs = f"EPSG:{cfg.region.utm_epsg}"
    dem_grid_utm = prep.reproject_match(dem_grid, utm_crs, dst_res=cfg.model.grid_resolution_m)

    mrds_utm = dl.reproject_vector(mrds, cfg.region.utm_epsg)

    df_features, grid_template, feature_names = prep.build_feature_stack(dem_grid_utm, mrds_utm)

    h, w = grid_template.data.shape
    rr, cc = np.indices((h, w))

    # Positive mask: 300 m buffer around MRDS points
    mrds_buffer = mrds_utm.copy()
    mrds_buffer["geometry"] = mrds_buffer.geometry.buffer(300)
    pos_mask = rasterio.features.rasterize(
        shapes=((g, 1) for g in mrds_buffer.geometry),
        out_shape=(h, w),
        transform=grid_template.transform,
        fill=0,
    ).astype(bool)

    # Negative candidates: outside 1 km buffer
    mrds_excl = mrds_utm.copy()
    mrds_excl["geometry"] = mrds_excl.geometry.buffer(1000)
    excl_mask = rasterio.features.rasterize(
        shapes=((g, 1) for g in mrds_excl.geometry),
        out_shape=(h, w),
        transform=grid_template.transform,
        fill=0,
    ).astype(bool)

    rng = np.random.RandomState(cfg.model.random_state)
    neg_candidates = ~excl_mask
    neg_indices = np.where(neg_candidates.reshape(-1))[0]
    num_pos = int(pos_mask.sum())
    num_neg = min(num_pos, len(neg_indices))
    sampled_neg = np.zeros(h * w, dtype=bool)
    if num_neg > 0:
        sampled_idx = rng.choice(neg_indices, size=num_neg, replace=False)
        sampled_neg[sampled_idx] = True
    neg_mask = sampled_neg.reshape(h, w)

    # Labels
    y = np.full(df_features.shape[0], -1, dtype=int)
    y[pos_mask.reshape(-1)] = 1
    y[neg_mask.reshape(-1)] = 0
    labeled_mask = (y >= 0)
    y_labeled = y[labeled_mask]

    # Spatial block CV
    cv = scv.spatial_block_cv(
        df=df_features,
        feature_names=feature_names,
        labeled_mask=labeled_mask,
        y=y_labeled,
        k_blocks=cfg.model.spatial_block_k,
        random_state=cfg.model.random_state,
        rf_params={
            'n_estimators': cfg.model.rf_n_estimators,
            'max_depth': cfg.model.rf_max_depth,
            'class_weight': cfg.model.rf_class_weight,
        }
    )

    # Train final model on all labeled
    result = mdl.train_rf(
        df_features, feature_names, labeled_mask, y_labeled,
        random_state=cfg.model.random_state,
        n_estimators=cfg.model.rf_n_estimators,
        max_depth=cfg.model.rf_max_depth,
        class_weight=cfg.model.rf_class_weight,
        use_smote=cfg.model.smote_enabled,
    )

    # Predict full grid
    proba = mdl.predict_full_grid(result.model, df_features, feature_names)
    proba_grid = proba.reshape(h, w)

    # Outputs
    prob_tif = p.outputs_dir / "prospectivity_prob.tif"
    viz.array_to_geotiff(prob_tif, proba_grid, grid_template.transform, grid_template.crs)

    hotspots_gdf = viz.top_hotspots_to_geojson(proba_grid, grid_template.transform, grid_template.crs, top_k=10)
    hotspots_path = p.outputs_dir / "hotspots.geojson"
    hotspots_gdf.to_file(hotspots_path, driver='GeoJSON')

    viz.feature_importance_plot(p.figures_dir / "feature_importance.png", result.feature_names, result.importance)

    # Write CV reports and training report
    viz.write_cv_reports(p.outputs_dir, cv.reports)
    with open(p.outputs_dir / "model_report.txt", 'w') as f:
        f.write(result.report)
        f.write("\nConfusion matrix:\n")
        f.write(str(result.conf_matrix))

    # Folium map
    map_html = p.outputs_dir / "interactive_map.html"
    viz.folium_map(map_html, hotspots_gdf, mrds_utm)

    logger.info("Pipeline complete. Outputs written to %s", p.outputs_dir)


if __name__ == "__main__":
    main()
