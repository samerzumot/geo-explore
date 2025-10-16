"""Feature engineering and grid construction for the MVP.

This module builds a common grid, resamples inputs, computes features, and packs
results into a GeoDataFrame with spatial indexing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import Point
from skimage.filters import sobel
from scipy.ndimage import uniform_filter


def _safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Compute numer/denom with safe division and NaN where invalid."""
    out = numer.astype("float32") / np.clip(denom.astype("float32"), 1e-6, None)
    out[~np.isfinite(out)] = np.nan
    return out

from .config import Config, DEFAULT_CONFIG
from .data_loader import RasterLayer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class GridDefinition:
    """Defines the common raster grid used throughout the pipeline."""

    width: int
    height: int
    transform: Affine
    crs: CRS


def make_common_grid(
    config: Config = DEFAULT_CONFIG,
) -> GridDefinition:
    """Create a projected grid covering the bbox at specified resolution."""
    minx, miny, maxx, maxy = config.bbox_wgs84
    target_crs = CRS.from_epsg(config.target_crs_epsg)

    # Compute destination transform and shape based on resolution
    # Convert bbox polygon into target CRS to get size in meters
    from shapely.geometry import box
    import geopandas as gpd

    bbox = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoSeries([bbox], crs="EPSG:4326").to_crs(target_crs)
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy) in target CRS

    width = int(np.ceil((bounds[2] - bounds[0]) / config.resolution_m))
    height = int(np.ceil((bounds[3] - bounds[1]) / config.resolution_m))

    transform = Affine(
        config.resolution_m,
        0,
        bounds[0],
        0,
        -config.resolution_m,
        bounds[3],
    )

    logger.info("Common grid: %dx%d at %.1fm (EPSG:%d)", width, height, config.resolution_m, target_crs.to_epsg())
    return GridDefinition(width=width, height=height, transform=transform, crs=target_crs)


def resample_to_grid(
    layer: RasterLayer,
    grid: GridDefinition,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject/resample a raster layer to the common grid (single-band)."""
    if layer.array.ndim == 3:
        raise ValueError("resample_to_grid expects a single-band layer")

    dst = np.full((grid.height, grid.width), np.nan, dtype="float32")
    reproject(
        source=layer.array,
        destination=dst,
        src_transform=layer.transform,
        src_crs=layer.crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=resampling,
        src_nodata=layer.nodata,
        dst_nodata=np.nan,
    )
    return dst


def compute_dem_features(dem: np.ndarray, cellsize: float) -> Dict[str, np.ndarray]:
    """Compute slope, aspect, and lineament density proxies from DEM.

    - Slope via gradient magnitude
    - Aspect via arctangent of gradients
    - Lineament density via Sobel edges + local mean filter
    """
    gy, gx = np.gradient(dem.astype("float32"), cellsize, edge_order=1)
    slope = np.sqrt(gx * gx + gy * gy)
    aspect = np.rad2deg(np.arctan2(-gx, gy)) % 360.0

    # Edge detection as a simple proxy for lineaments
    edges = sobel(dem.astype("float32"))
    # Local density using a 15x15 window (adjustable)
    density = uniform_filter(edges, size=15)

    return {
        "dem_slope": slope,
        "dem_aspect": aspect,
        "lineament_density": density,
    }


def compute_landsat_features(
    bands: Dict[str, np.ndarray],
    ndvi_mask_threshold: float,
) -> Dict[str, np.ndarray]:
    """Compute iron/clay alteration indices and NDVI.

    Expected bands keys: B4 (red), B5 (nir), B6 (swir1), B7 (swir2)
    """
    b4 = bands.get("B4")
    b5 = bands.get("B5")
    b6 = bands.get("B6")
    b7 = bands.get("B7")

    if any(arr is None for arr in (b4, b5, b6, b7)):
        raise ValueError("Missing required Landsat bands for feature computation")

    # NDVI
    ndvi = _safe_ratio(b5 - b4, b5 + b4)

    # Iron oxide ratio: SWIR1 / SWIR2
    iron_oxide_ratio = _safe_ratio(b6, b7)

    # Clay alteration ratio: SWIR1 / NIR
    clay_ratio = _safe_ratio(b6, b5)

    # Apply vegetation mask (set features to NaN where NDVI > threshold)
    veg_mask = ndvi > ndvi_mask_threshold
    iron_oxide_ratio = np.where(veg_mask, np.nan, iron_oxide_ratio)
    clay_ratio = np.where(veg_mask, np.nan, clay_ratio)

    return {
        "ndvi": ndvi,
        "iron_oxide_ratio": iron_oxide_ratio,
        "clay_ratio": clay_ratio,
    }


def grid_to_points(grid: GridDefinition) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of cell centers in WGS84, with grid indices."""
    xs = np.arange(grid.width)
    ys = np.arange(grid.height)
    xx, yy = np.meshgrid(xs, ys)
    # Convert to world coordinates
    xw = grid.transform.c + (xx + 0.5) * grid.transform.a
    yw = grid.transform.f + (yy + 0.5) * grid.transform.e

    # Flatten
    xw = xw.ravel()
    yw = yw.ravel()

    # Build GeoDataFrame in grid CRS then reproject to WGS84
    gdf = gpd.GeoDataFrame(
        {"col": xx.ravel(), "row": yy.ravel()},
        geometry=gpd.points_from_xy(xw, yw),
        crs=grid.crs,
    ).to_crs("EPSG:4326")
    return gdf


def build_feature_gdf(
    grid: GridDefinition,
    feature_arrays: Dict[str, np.ndarray],
) -> gpd.GeoDataFrame:
    """Pack feature rasters into a point GeoDataFrame indexed by grid cells."""
    points = grid_to_points(grid)
    for name, arr in feature_arrays.items():
        if arr.shape != (grid.height, grid.width):
            raise ValueError(f"Feature '{name}' shape {arr.shape} != grid {(grid.height, grid.width)}")
        points[name] = arr.ravel()

    # Drop rows where all features are NaN
    feature_cols = list(feature_arrays.keys())
    points = points.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    return points


def compute_distance_to_points(
    grid_points: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    target_crs: CRS,
    column_name: str,
) -> np.ndarray:
    """Compute Euclidean distance (meters) from each grid point to nearest target.

    Uses projected CRS for accurate distance. Returns array aligned with grid_points index.
    """
    if targets.empty:
        return np.full(len(grid_points), np.nan, dtype="float32")

    pts_proj = grid_points.to_crs(target_crs)
    tgt_proj = targets.to_crs(target_crs)

    # Use spatial index for efficiency
    sindex = tgt_proj.sindex
    out = np.empty(len(pts_proj), dtype="float32")
    for i, geom in enumerate(pts_proj.geometry):
        # Query k nearest candidates (k=5) for robustness
        cand_idx = list(sindex.nearest(geom.bounds, num_results=5))
        dmin = min(geom.distance(tgt_proj.geometry.iloc[j]) for j in cand_idx)
        out[i] = dmin
    return out


def idw_interpolation(
    grid_points: gpd.GeoDataFrame,
    sample_points: gpd.GeoDataFrame,
    value_column: str,
    target_crs: CRS,
    power: float = 2.0,
) -> np.ndarray:
    """Inverse distance weighting from point samples onto grid points.

    If no samples are present or values are missing, returns NaNs.
    """
    if sample_points.empty or value_column not in sample_points.columns:
        return np.full(len(grid_points), np.nan, dtype="float32")

    gp = grid_points.to_crs(target_crs)
    sp = sample_points.dropna(subset=[value_column]).to_crs(target_crs)
    if sp.empty:
        return np.full(len(grid_points), np.nan, dtype="float32")

    s_coords = np.column_stack((sp.geometry.x.values, sp.geometry.y.values))
    s_vals = sp[value_column].values.astype("float32")

    # Brute-force IDW for MVP: use nearest 8 samples per grid point
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(8, len(sp)), algorithm="kd_tree").fit(s_coords)

    out_vals = np.empty(len(gp), dtype="float32")
    for i, geom in enumerate(gp.geometry):
        dists, idxs = nn.kneighbors([[geom.x, geom.y]], return_distance=True)
        dists = dists[0]
        idxs = idxs[0]
        # Avoid zero distance by adding small epsilon
        weights = 1.0 / np.power(dists + 1e-6, power)
        out_vals[i] = np.sum(weights * s_vals[idxs]) / np.sum(weights)

    return out_vals
