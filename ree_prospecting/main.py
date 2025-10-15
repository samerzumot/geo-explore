from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from .config import get_default_config
from .data_loader import load_landsat_composite, load_dem, load_mrds
from .preprocessing import compute_landsat_indices, compute_dem_features, assemble_feature_stack, add_distance_to_points, features_to_dataframe
from .model import label_training_samples, train_random_forest_spatial_cv, predict_raster_probabilities
from .visualization import save_geotiff, extract_hotspots, save_hotspots_geojson, make_folium_map, plot_feature_importance, plot_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("ree_prospecting")


def run_pipeline() -> None:
    cfg = get_default_config(Path(__file__).resolve().parent)

    LOGGER.info("Loading Landsat composite...")
    l8 = load_landsat_composite(cfg)

    LOGGER.info("Computing spectral indices...")
    ls_idx = compute_landsat_indices(l8)

    LOGGER.info("Loading DEM and computing terrain features...")
    dem = load_dem(cfg)
    dem_feats = compute_dem_features(dem, cfg)

    LOGGER.info("Merging features and adding distance to MRDS...")
    features = assemble_feature_stack(ls_idx, dem_feats, cfg)
    mrds = load_mrds(cfg)
    features = add_distance_to_points(features, mrds, cfg)

    LOGGER.info("Preparing training data...")
    feat_gdf = features_to_dataframe(features)
    train_gdf = label_training_samples(feat_gdf, mrds, cfg)

    if train_gdf.empty:
        LOGGER.error("Training set is empty. Check MRDS availability and region bounds.")
        return

    LOGGER.info("Training RandomForest with spatial CV...")
    result = train_random_forest_spatial_cv(train_gdf, cfg)

    LOGGER.info("Model metrics: %s", result.metrics)

    LOGGER.info("Predicting prospectivity across full raster...")
    proba_da, std_da = predict_raster_probabilities(result.model, features)

    out_dir = cfg.paths.outputs_dir
    save_geotiff(proba_da, out_dir / "prospectivity.tif")
    save_geotiff(std_da, out_dir / "uncertainty_std.tif")

    LOGGER.info("Extracting hotspot polygons and saving GeoJSON...")
    hotspots = extract_hotspots(proba_da, top_percent=10.0)
    save_hotspots_geojson(hotspots, out_dir / "hotspots_top10.geojson", top_n=10)

    LOGGER.info("Creating interactive map...")
    make_folium_map(cfg, hotspots.head(10), mrds, out_dir / "map.html")

    LOGGER.info("Saving feature importance and metrics plots...")
    plot_feature_importance(result.feature_importances, out_dir / "feature_importance.png")
    plot_metrics(result.confusion_matrix, result.roc_curve, out_dir / "model")

    # Save metrics report
    with open((out_dir / "metrics.json").as_posix(), "w") as f:
        import json
        json.dump(result.metrics, f, indent=2)

    LOGGER.info("Pipeline complete. Outputs written to %s", out_dir)


if __name__ == "__main__":
    run_pipeline()
