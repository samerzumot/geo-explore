from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from shapely.geometry import box
import geopandas as gpd
import folium
import matplotlib.pyplot as plt


def array_to_geotiff(out_path: Path, data: np.ndarray, transform: Affine, crs: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = data.shape
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'height': h, 'width': w, 'transform': transform, 'crs': crs, 'compress': 'lzw'
    }
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data.astype('float32'), 1)


def top_hotspots_to_geojson(proba: np.ndarray, transform: Affine, crs: str, top_k: int = 10) -> gpd.GeoDataFrame:
    idx = np.argsort(proba.ravel())[::-1][:top_k]
    rows, cols = np.unravel_index(idx, proba.shape)
    polys = []
    scores = []
    for r, c in zip(rows, cols):
        x = transform.c + c * transform.a
        y = transform.f + r * transform.e
        # Pixel footprint polygon in the raster CRS
        poly = box(x, y + transform.e, x + transform.a, y)
        polys.append(poly)
        scores.append(float(proba[r, c]))
    return gpd.GeoDataFrame({'score': scores}, geometry=polys, crs=crs)


def feature_importance_plot(out_path: Path, feature_names: List[str], importance: np.ndarray) -> None:
    order = np.argsort(importance)
    names = [feature_names[i] for i in order]
    vals = importance[order]
    plt.figure(figsize=(6, 4))
    plt.barh(names, vals)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_cv_reports(out_dir: Path, reports: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rep in enumerate(reports, 1):
        with open(out_dir / f"cv_report_fold_{i}.txt", 'w') as f:
            f.write(rep)


def folium_map(out_html: Path, hotspots: gpd.GeoDataFrame, known_points: gpd.GeoDataFrame | None = None) -> None:
    hs_ll = hotspots.to_crs(epsg=4326)
    if known_points is not None and not known_points.empty:
        known_ll = known_points.to_crs(epsg=4326)
        center = [known_ll.geometry.y.mean(), known_ll.geometry.x.mean()]
    else:
        bounds = hs_ll.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=10, tiles='CartoDB positron')

    for _, row in hs_ll.iterrows():
        gj = row.geometry.__geo_interface__
        folium.GeoJson(gj, name=f"Hotspot score={row['score']:.2f}",
                       style_function=lambda x: {'color': 'red', 'fillColor': 'red', 'weight': 1, 'fillOpacity': 0.4}).add_to(m)

    if known_points is not None and not known_points.empty:
        for _, row in known_ll.iterrows():
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=4, color='blue', fill=True,
                                popup=str(row.get('site_name', 'known'))).add_to(m)

    folium.LayerControl().add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
