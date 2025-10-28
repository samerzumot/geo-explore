"""Visualization and outputs for REE prospectivity MVP."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import logging
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from shapely.geometry import box, shape, mapping
import folium
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def array_to_geotiff(out_path: Path, data: np.ndarray, transform: Affine, crs: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'height': height,
        'width': width,
        'transform': transform,
        'crs': crs,
        'compress': 'lzw'
    }
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data.astype('float32'), 1)


def top_hotspots_to_geojson(proba: np.ndarray, transform: Affine, crs: str, top_k: int = 10) -> gpd.GeoDataFrame:
    # Get top K pixels; in MVP, convert each pixel to a polygon footprint
    flat_idx = np.argsort(proba.ravel())[::-1][:top_k]
    rows, cols = np.unravel_index(flat_idx, proba.shape)
    polygons = []
    scores = []
    for r, c in zip(rows, cols):
        x = transform.c + c * transform.a
        y = transform.f + r * transform.e
        # Pixel footprint
        poly = box(x, y + transform.e, x + transform.a, y)
        polygons.append(poly)
        scores.append(float(proba[r, c]))
    gdf = gpd.GeoDataFrame({'score': scores}, geometry=polygons, crs=crs)
    return gdf


def feature_importance_plot(out_path: Path, feature_names: List[str], importance: np.ndarray) -> None:
    idx = np.argsort(importance)
    names = [feature_names[i] for i in idx]
    vals = importance[idx]
    plt.figure(figsize=(6, 4))
    plt.barh(names, vals)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def classification_report_to_text(out_path: Path, report: str, conf_matrix: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(conf_matrix))


def folium_map(out_html: Path, proba: np.ndarray, transform: Affine, crs: str, hotspots: gpd.GeoDataFrame, known_points: gpd.GeoDataFrame | None = None) -> None:
    # Build map centered on bbox
    # Convert a few hotspot polygons to lat/lon for display
    hs_ll = hotspots.to_crs(epsg=4326)
    if known_points is not None and not known_points.empty:
        known_ll = known_points.to_crs(epsg=4326)
        center = [known_ll.geometry.y.mean(), known_ll.geometry.x.mean()]
    else:
        bounds = hs_ll.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=10, tiles='CartoDB positron')

    # Add hotspots
    for _, row in hs_ll.iterrows():
        geojson = mapping(row.geometry)
        folium.GeoJson(geojson, name=f"Hotspot score={row['score']:.2f}", style_function=lambda x: {'color': 'red', 'fillColor': 'red', 'weight': 1, 'fillOpacity': 0.4}).add_to(m)

    # Add known points
    if known_points is not None and not known_points.empty:
        for _, row in known_ll.iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=4, color='blue', fill=True, popup=row.get('site_name','known')).add_to(m)

    folium.LayerControl().add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
