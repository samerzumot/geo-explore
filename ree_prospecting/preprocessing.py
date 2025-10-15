from __future__ import annotations

import logging
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from scipy.ndimage import uniform_filter
from skimage.feature import canny

from .config import Config

LOGGER = logging.getLogger(__name__)


def compute_landsat_indices(ds: xr.Dataset) -> xr.Dataset:
    """Compute spectral indices: NDVI, iron oxide ratio, clay alteration ratio.

    - Iron oxide ratio: swir1 / swir2
    - Clay alteration ratio: swir1 / nir
    - NDVI: (nir - red) / (nir + red)
    """
    red = ds["red"]
    nir = ds["nir"]
    swir1 = ds["swir1"]
    swir2 = ds["swir2"]

    np.seterr(divide="ignore", invalid="ignore")

    ndvi = (nir - red) / (nir + red)
    iron_oxide_ratio = swir1 / swir2
    clay_ratio = swir1 / nir

    feat = xr.Dataset({
        "ndvi": ndvi.clip(min=-1.0, max=1.0),
        "iron_oxide_ratio": iron_oxide_ratio,
        "clay_ratio": clay_ratio,
    })
    feat.rio.write_crs(ds.rio.crs, inplace=True)
    return feat


def _gradient_slope_aspect(dem: xr.DataArray, pixel_size_m: float) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute slope (degrees) and aspect (radians) from DEM using Horn's method approximation."""
    # Use numpy gradient on DEM values
    dz_dy, dz_dx = np.gradient(dem.values, pixel_size_m, pixel_size_m)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(dz_dy, -dz_dx)
    aspect = np.mod(aspect, 2 * np.pi)

    slope = xr.DataArray(np.degrees(slope_rad), coords=dem.coords, dims=dem.dims, name="slope")
    aspect = xr.DataArray(aspect, coords=dem.coords, dims=dem.dims, name="aspect")
    slope.rio.write_crs(dem.rio.crs, inplace=True)
    aspect.rio.write_crs(dem.rio.crs, inplace=True)
    return slope, aspect


def compute_dem_features(dem: xr.DataArray, config: Config) -> xr.Dataset:
    """Compute slope, aspect (sin/cos), and simple lineament density from DEM."""
    res = config.model.resolution_m
    slope, aspect = _gradient_slope_aspect(dem, res)

    # Edge detection on slope as proxy for lineaments
    slope_norm = (slope - float(slope.min())) / (float((slope.max() - slope.min()) + 1e-6))
    edges = canny(slope_norm.astype("float32"), sigma=1.5)

    radius_pix = max(1, int(config.model.lineament_density_radius_m / res))
    window = radius_pix * 2 + 1
    # Density via mean in window
    density = uniform_filter(edges.astype("float32"), size=window, mode="nearest")

    dem_feats = xr.Dataset({
        "slope": slope,
        "aspect_sin": np.sin(aspect),
        "aspect_cos": np.cos(aspect),
        "lineament_density": xr.DataArray(density, coords=dem.coords, dims=dem.dims),
    })
    dem_feats.rio.write_crs(dem.rio.crs, inplace=True)
    return dem_feats


def assemble_feature_stack(ls_indices: xr.Dataset, dem_features: xr.Dataset, config: Config) -> xr.Dataset:
    """Merge all features into one aligned xarray.Dataset, apply NDVI mask if configured."""
    # Align datasets
    common = xr.align(ls_indices, dem_features, join="inner")
    ls_indices_aligned, dem_features_aligned = common

    features = xr.merge([ls_indices_aligned, dem_features_aligned])
    if config.model.ndvi_mask_threshold is not None:
        mask = features["ndvi"] > config.model.ndvi_mask_threshold
        for var in features.data_vars:
            if var != "ndvi":
                features[var] = features[var].where(~mask)
    return features


def add_distance_to_points(features: xr.Dataset, points: gpd.GeoDataFrame, config: Config) -> xr.Dataset:
    """Compute distance (km) from each grid cell center to nearest point (e.g., MRDS)."""
    if points.empty:
        features["dist_to_mrds_km"] = xr.full_like(features["ndvi"], fill_value=np.nan)
        return features

    # Build KDTree of point coordinates in UTM meters
    from scipy.spatial import cKDTree

    points_xy = np.vstack([points.geometry.x.values, points.geometry.y.values]).T
    tree = cKDTree(points_xy)

    # Prepare grid of centers
    yy, xx = np.meshgrid(features.y.values, features.x.values, indexing="xy")
    grid_xy = np.vstack([xx.ravel(), yy.ravel()]).T

    dist, _ = tree.query(grid_xy, k=1, workers=-1)
    dist = dist.reshape((features.sizes["x"], features.sizes["y"]))
    dist_km = dist / 1000.0
    features["dist_to_mrds_km"] = xr.DataArray(dist_km, coords={"x": features.x, "y": features.y})
    return features


def features_to_dataframe(features: xr.Dataset, sample_fraction: float = 1.0) -> gpd.GeoDataFrame:
    """Flatten features to a GeoDataFrame of pixel centers; optionally subsample."""
    df = features.to_array().transpose("y", "x", "variable").values.reshape(-1, len(features.data_vars))
    columns = list(features.data_vars)

    # Coordinates
    xs = np.repeat(features.x.values.reshape(1, -1), features.sizes["y"], axis=0).ravel()
    ys = np.repeat(features.y.values.reshape(-1, 1), features.sizes["x"], axis=1).ravel()

    gdf = gpd.GeoDataFrame(
        data=df,
        columns=columns,
        geometry=gpd.points_from_xy(xs, ys),
        crs=features.rio.crs,
    )
    # Drop rows with any NaNs (masked areas)
    gdf = gdf.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if 0 < sample_fraction < 1.0:
        gdf = gdf.sample(frac=sample_fraction, random_state=42)
    return gdf
