"""Visualization and output helpers for MVP.

- Write GeoTIFF probability/uncertainty rasters
- Extract hotspot polygons and save GeoJSON
- Build an interactive Folium map with overlays
- Plot confusion matrix and ROC curve
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.features import shapes, rasterize
from shapely.geometry import shape
import folium

from .config import Config, DEFAULT_CONFIG
from .preprocessing import GridDefinition

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def write_geotiff(
    path: Path,
    array: np.ndarray,
    grid: GridDefinition,
    nodata: float = np.nan,
) -> None:
    """Write a single-band GeoTIFF aligned to the common grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)
    logger.info("Saved GeoTIFF: %s", path)


def extract_hotspot_polygons(
    proba: np.ndarray,
    grid: GridDefinition,
    quantile: float = 0.9,
    top_k: int = 10,
) -> gpd.GeoDataFrame:
    """Extract top hotspot polygons by taking cells above a probability quantile."""
    thresh = float(np.nanquantile(proba, quantile))
    mask = np.where(np.isnan(proba), 0, (proba >= thresh).astype(np.uint8))

    geom_iter = shapes(mask, mask=mask.astype(bool), transform=grid.transform)
    polys = []
    scores = []
    for geom, val in geom_iter:
        if val != 1:
            continue
        poly = shape(geom)
        polys.append(poly)
        # Rasterize this polygon to get per-polygon mean probability
        poly_mask = rasterize(
            [(poly, 1)],
            out_shape=proba.shape,
            transform=grid.transform,
            fill=0,
            dtype=np.uint8,
        )
        scores.append(float(np.nanmean(proba[poly_mask == 1])))

    gdf = gpd.GeoDataFrame({"score": scores}, geometry=polys, crs=grid.crs)

    # Rank and take top_k
    gdf = gdf.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return gdf


def save_hotspots_geojson(path: Path, hotspots: gpd.GeoDataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hotspots.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    logger.info("Saved hotspots GeoJSON: %s", path)


def _make_probability_png_overlay(
    proba: np.ndarray,
    grid: GridDefinition,
    alpha: float = 0.7,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Create an RGBA image (uint8) and bounds for Folium overlay."""
    # Simple colormap (blue->yellow->red)
    cmap = plt.get_cmap("turbo")
    img = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
    valid = ~np.isnan(proba)
    cmapped = cmap(np.clip(proba, 0, 1))
    rgba = (cmapped * 255).astype(np.uint8)
    img[valid] = rgba[valid]
    img[~valid, 3] = 0
    img[:, :, 3] = (img[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)

    # Compute bounds in WGS84 for overlay
    import geopandas as gpd
    from shapely.geometry import box

    # Build polygon of grid extent
    minx = grid.transform.c
    maxy = grid.transform.f
    maxx = minx + grid.transform.a * grid.width
    miny = maxy + grid.transform.e * grid.height

    extent_poly = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=grid.crs).to_crs("EPSG:4326").geometry.iloc[0].bounds
    # folium expects [[south, west], [north, east]]
    bounds = [[extent_poly[1], extent_poly[0]], [extent_poly[3], extent_poly[2]]]
    return img, bounds


def build_folium_map(
    proba: np.ndarray,
    hotspots: gpd.GeoDataFrame,
    known_points: gpd.GeoDataFrame,
    grid: GridDefinition,
    out_html: Path,
) -> None:
    """Create an interactive folium map with hotspots and known deposits."""
    # Center map on region centroid
    center = hotspots.to_crs("EPSG:4326").unary_union.centroid if not hotspots.empty else known_points.to_crs("EPSG:4326").unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="CartoDB positron")

    # Probability overlay
    img, bounds = _make_probability_png_overlay(proba, grid)
    import PIL.Image
    overlay = PIL.Image.fromarray(img, mode="RGBA")
    folium.raster_layers.ImageOverlay(image=overlay, bounds=bounds, opacity=0.7, name="Prospectivity").add_to(m)

    # Hotspots layer
    if not hotspots.empty:
        hs = hotspots.to_crs("EPSG:4326")
        folium.GeoJson(
            hs.__geo_interface__,
            name="Hotspots (top)",
            style_function=lambda f: {
                "color": "#d73027",
                "fillColor": "#fc8d59",
                "fillOpacity": 0.3,
                "weight": 2,
            },
            tooltip=folium.GeoJsonTooltip(fields=["score"], aliases=["Score"], localize=True),
        ).add_to(m)

    # Known deposits/validation markers
    if not known_points.empty:
        kp = known_points.to_crs("EPSG:4326")
        for _, row in kp.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color="#2b8cbe",
                fill=True,
                fill_opacity=0.9,
                popup=row.get("name", "MRDS point"),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    logger.info("Saved interactive map: %s", out_html)


def plot_confusion_and_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    y_pred = (y_prob >= 0.5).astype(int)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax[0])
    ax[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax[1])
    ax[1].set_title("ROC Curve")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_roc.png", dpi=150)
    plt.close(fig)
