from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Optional

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from rasterio import features as rio_features
from shapely.geometry import shape

from .config import Config

LOGGER = logging.getLogger(__name__)


def save_geotiff(da: xr.DataArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(path.as_posix(), compress="deflate")


def extract_hotspots(proba: xr.DataArray, top_percent: float = 10.0, min_area_pixels: int = 20) -> gpd.GeoDataFrame:
    """Extract top percentile hotspots as polygons ranked by mean probability."""
    threshold = np.nanpercentile(proba.values[np.isfinite(proba.values)], 100 - top_percent)
    mask = (proba.values >= threshold) & np.isfinite(proba.values)

    shapes = rio_features.shapes(mask.astype("uint8"), mask=None, transform=proba.rio.transform())
    polys = []
    for geom, val in shapes:
        if val != 1:
            continue
        geom_shape = shape(geom)
        area_pixels = int(geom_shape.area)  # in pixel units; approximate
        if area_pixels < min_area_pixels:
            continue
        polys.append(geom_shape)

    if len(polys) == 0:
        return gpd.GeoDataFrame(columns=["geometry", "score"], geometry="geometry", crs=proba.rio.crs)

    gdf = gpd.GeoDataFrame(geometry=polys, crs=proba.rio.crs)
    # Compute mean probability within polygon by sampling raster at polygon mask
    scores = []
    for geom in gdf.geometry:
        mask_geom = rio_features.geometry_mask([geom.__geo_interface__], invert=True, transform=proba.rio.transform(), out_shape=proba.shape)
        vals = proba.values[mask_geom]
        if vals.size == 0:
            scores.append(float("nan"))
        else:
            scores.append(float(np.nanmean(vals)))
    gdf["score"] = scores
    gdf = gdf.sort_values("score", ascending=False).reset_index(drop=True)
    return gdf


def save_hotspots_geojson(hotspots: gpd.GeoDataFrame, path: Path, top_n: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hotspots.head(top_n).to_file(path, driver="GeoJSON")


def make_folium_map(config: Config, hotspots: gpd.GeoDataFrame, mrds: gpd.GeoDataFrame, output_html: Path) -> None:
    """Create an interactive folium map with hotspots and known deposits."""
    # Center map at region center (WGS84)
    minx, miny, maxx, maxy = config.region.bbox_wgs84
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    # Hotspots layer
    if not hotspots.empty:
        hotspots_wgs84 = hotspots.to_crs(4326)
        folium.GeoJson(
            hotspots_wgs84.__geo_interface__,
            name="Predicted hotspots",
            style_function=lambda f: {
                "fillColor": "#e41a1c",
                "color": "#e41a1c",
                "weight": 1,
                "fillOpacity": min(0.9, max(0.3, f["properties"].get("score", 0.5))),
            },
            tooltip=folium.features.GeoJsonTooltip(fields=["score"], aliases=["Hotspot score"]),
        ).add_to(m)

    # MRDS points layer
    if not mrds.empty:
        mrds_wgs84 = mrds.to_crs(4326)
        for _, row in mrds_wgs84.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color="#377eb8",
                fill=True,
                fill_color="#377eb8",
                fill_opacity=0.8,
                popup=row.get("site_name", "MRDS point"),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_html.as_posix())


def plot_feature_importance(importances_df, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    df = importances_df.head(20)
    plt.barh(df["feature"], df["importance"], color="#4daf4a")
    plt.gca().invert_yaxis()
    plt.xlabel("Feature importance")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.as_posix(), dpi=150)
    plt.close()


def plot_metrics(cm: np.ndarray, roc_curve: Tuple[np.ndarray, np.ndarray, np.ndarray], path_prefix: Path) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig((path_prefix.parent / f"{path_prefix.name}_confusion_matrix.png").as_posix(), dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, thr = roc_curve
    if fpr.size > 0:
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig((path_prefix.parent / f"{path_prefix.name}_roc.png").as_posix(), dpi=150)
        plt.close()
