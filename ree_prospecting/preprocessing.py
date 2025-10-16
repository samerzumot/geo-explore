from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import sobel, distance_transform_edt


@dataclass
class RasterGrid:
    data: np.ndarray
    transform: rasterio.transform.Affine
    crs: str


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs.to_string()
    return RasterGrid(data=data, transform=transform, crs=crs)


def reproject_match(src_grid: RasterGrid, dst_crs: str, dst_res: float) -> RasterGrid:
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
    px = dem.transform.a
    py = -dem.transform.e
    z = dem.data.astype(float)
    dzdx = sobel(z, axis=1) / (8 * px)
    dzdy = sobel(z, axis=0) / (8 * py)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(dzdy, -dzdx)) + 360) % 360
    return slope, aspect


def simple_edge_density(dem: RasterGrid) -> np.ndarray:
    z = dem.data.astype(float)
    gx = sobel(z, axis=1)
    gy = sobel(z, axis=0)
    mag = np.hypot(gx, gy)
    mag = (mag - np.nanmin(mag)) / (np.nanmax(mag) - np.nanmin(mag) + 1e-9)
    return mag


def rasterize_points_distance(points: gpd.GeoDataFrame, template: RasterGrid) -> np.ndarray:
    from rasterio import features
    height, width = template.data.shape
    affine = template.transform
    shapes = ((geom, 1) for geom in points.geometry)
    mask = features.rasterize(shapes=shapes, out_shape=(height, width), transform=affine, fill=0, dtype=np.uint8)
    dist_px = distance_transform_edt(1 - mask)
    px_size = (abs(affine.a) + abs(affine.e)) / 2.0
    return dist_px * px_size


def build_feature_stack(
    dem_grid: RasterGrid,
    ree_points_utm: gpd.GeoDataFrame | None = None,
) -> Tuple[pd.DataFrame, RasterGrid, List[str]]:
    slope, aspect = compute_terrain_attributes(dem_grid)
    edge = simple_edge_density(dem_grid)

    features = [
        (slope, 'slope_deg'),
        (aspect, 'aspect_deg'),
        (edge, 'edge_density'),
    ]

    if ree_points_utm is not None and not ree_points_utm.empty:
        dist = rasterize_points_distance(ree_points_utm, dem_grid)
        features.append((dist, 'dist_to_ree_m'))

    h, w = dem_grid.data.shape
    rows = []
    names = []
    for arr, name in features:
        rows.append(arr.reshape(-1))
        names.append(name)
    X = np.vstack(rows).T
    df = pd.DataFrame(X, columns=names)
    rr, cc = np.indices((h, w))
    df['row'] = rr.reshape(-1)
    df['col'] = cc.reshape(-1)
    return df, dem_grid, names
