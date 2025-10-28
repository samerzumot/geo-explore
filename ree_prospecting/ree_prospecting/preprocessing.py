"""Feature engineering for REE prospectivity MVP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from rasterio import features
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import from_origin
from scipy.ndimage import sobel

logger = logging.getLogger(__name__)


@dataclass
class RasterGrid:
    data: np.ndarray
    transform: rasterio.transform.Affine
    crs: str


def read_raster(path: Path) -> RasterGrid:
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs.to_string()
    return RasterGrid(data=data, transform=transform, crs=crs)


def reproject_match(src_grid: RasterGrid, dst_crs: str, dst_res: float) -> RasterGrid:
    """Reproject raster to destination CRS and resolution using nearest/bilinear as needed."""
    src_height, src_width = src_grid.data.shape
    src_transform = src_grid.transform
    src_crs = src_grid.crs
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *rasterio.transform.array_bounds(src_height, src_width, src_transform), resolution=dst_res
    )
    dst_data = np.empty((dst_height, dst_width), dtype=src_grid.data.dtype)
    reproject(
        source=src_grid.data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return RasterGrid(data=dst_data, transform=dst_transform, crs=dst_crs)


def compute_terrain_attributes(dem: RasterGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Compute slope (deg) and aspect (deg) from DEM using simple gradient approximations."""
    # Pixel size from transform
    px = dem.transform.a
    py = -dem.transform.e
    z = dem.data.astype(float)
    dzdx = sobel(z, axis=1) / (8 * px)
    dzdy = sobel(z, axis=0) / (8 * py)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(dzdy, -dzdx)) + 360) % 360
    return slope, aspect


def simple_edge_density(dem: RasterGrid, sigma: int = 1) -> np.ndarray:
    """Proxy for lineament density: gradient magnitude as edge strength, smoothed."""
    z = dem.data.astype(float)
    gx = sobel(z, axis=1)
    gy = sobel(z, axis=0)
    mag = np.hypot(gx, gy)
    # Normalize 0-1
    mag = (mag - np.nanmin(mag)) / (np.nanmax(mag) - np.nanmin(mag) + 1e-9)
    return mag


def rasterize_points_distance(points: gpd.GeoDataFrame, template: RasterGrid) -> np.ndarray:
    """Distance to nearest point in meters across template grid (UTM CRS)."""
    # Build a mask of ones at point pixels
    height, width = template.data.shape
    transform = template.transform
    affine = transform
    # Create a binary raster with points
    shapes = ((geom, 1) for geom in points.geometry)
    mask = features.rasterize(shapes=shapes, out_shape=(height, width), transform=affine, fill=0, dtype=np.uint8)
    # Compute distance transform in pixel units via scipy (avoid heavy dependencies)
    from scipy.ndimage import distance_transform_edt
    dist_px = distance_transform_edt(1 - mask)  # distance to nearest 1
    # Convert to meters using average pixel size
    px_size = (abs(affine.a) + abs(affine.e)) / 2.0
    dist_m = dist_px * px_size
    return dist_m


def idw_interpolation(points: gpd.GeoDataFrame, value_col: str, template: RasterGrid, power: float = 2.0, radius: float = 1000.0) -> np.ndarray:
    """Simple IDW over a raster grid. This is a naive implementation for MVP and may be slow for big grids."""
    height, width = template.data.shape
    xs = np.arange(width)
    ys = np.arange(height)
    xv, yv = np.meshgrid(xs, ys)
    affine = template.transform
    # Convert pixel centers to coordinates
    x_coords = affine.c + xv * affine.a + yv * affine.b
    y_coords = affine.f + xv * affine.d + yv * affine.e

    # Points in same CRS as template
    pts = np.array([(geom.x, geom.y) for geom in points.geometry])
    vals = points[value_col].to_numpy()

    out = np.full((height, width), np.nan, dtype=float)
    # Vectorized distance computation can be memory-heavy; do in tiles
    tile = 256
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1 = min(height, y0 + tile)
            x1 = min(width, x0 + tile)
            xt = x_coords[y0:y1, x0:x1][..., None]
            yt = y_coords[y0:y1, x0:x1][..., None]
            dx = xt - pts[:, 0]
            dy = yt - pts[:, 1]
            dist2 = dx * dx + dy * dy
            # Apply radius cutoff
            if radius > 0:
                mask = dist2 <= radius * radius
            else:
                mask = np.ones_like(dist2, dtype=bool)
            w = np.where(mask, 1.0 / (dist2 ** (power / 2.0) + 1e-12), 0.0)
            wsum = np.sum(w, axis=2)
            vsum = np.sum(w * vals, axis=2)
            out[y0:y1, x0:x1] = np.where(wsum > 0, vsum / wsum, np.nan)
    return out


def build_feature_stack(
    dem_grid: RasterGrid,
    landsat_b6: RasterGrid | None = None,
    landsat_b5: RasterGrid | None = None,
    landsat_b7: RasterGrid | None = None,
    ree_points_utm: gpd.GeoDataFrame | None = None,
) -> Tuple[pd.DataFrame, RasterGrid, List[str]]:
    """Compute features and return tabular DataFrame aligned to raster grid plus feature names."""
    # Terrain features
    slope, aspect = compute_terrain_attributes(dem_grid)
    edge = simple_edge_density(dem_grid)

    features_list = [
        (slope, 'slope_deg'),
        (aspect, 'aspect_deg'),
        (edge, 'edge_density'),
    ]

    # Landsat-derived
    if landsat_b6 is not None and landsat_b7 is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            iron_ratio = landsat_b6.data.astype(float) / (landsat_b7.data.astype(float) + 1e-6)
        features_list.append((iron_ratio, 'iron_ratio_b6_b7'))
    if landsat_b6 is not None and landsat_b5 is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            clay_ratio = landsat_b6.data.astype(float) / (landsat_b5.data.astype(float) + 1e-6)
        features_list.append((clay_ratio, 'clay_ratio_b6_b5'))
    if landsat_b5 is not None and landsat_b4 := landsat_b5:  # placeholder for NDVI if B4 present
        # MVP: if we only have B5, skip NDVI; real impl requires B4 (red) and B5 (NIR)
        pass

    # Geochem-derived proxies
    if ree_points_utm is not None and not ree_points_utm.empty:
        dist = rasterize_points_distance(ree_points_utm, dem_grid)
        features_list.append((dist, 'dist_to_ree_m'))
        # If points contain a 'value' column, add IDW; else skip in MVP
        if 'value' in ree_points_utm.columns:
            idw = idw_interpolation(ree_points_utm, 'value', dem_grid)
            features_list.append((idw, 'ree_idw'))

    # Stack to DataFrame
    h, w = dem_grid.data.shape
    rows = []
    names = []
    for arr, name in features_list:
        rows.append(arr.reshape(-1))
        names.append(name)
    X = np.vstack(rows).T
    df = pd.DataFrame(X, columns=names)
    # Add row/col indices to map back later
    rr, cc = np.indices((h, w))
    df['row'] = rr.reshape(-1)
    df['col'] = cc.reshape(-1)
    return df, dem_grid, names
