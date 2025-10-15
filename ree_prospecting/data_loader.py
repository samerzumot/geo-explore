from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Tuple, Optional, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401 (registers rio accessor)
import xarray as xr
from pyproj import CRS
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from pystac_client import Client
import planetary_computer as pc
import stackstac

from .config import Config


LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _region_bbox(config: Config) -> Tuple[float, float, float, float]:
    return config.region.bbox_wgs84


def _region_geom(config: Config) -> BaseGeometry:
    return box(*_region_bbox(config))


def _utm_crs(config: Config) -> CRS:
    return config.region.utm_crs()


def _scale_landsat_sr(sr: xr.DataArray) -> xr.DataArray:
    """Scale Landsat Collection 2 Level-2 surface reflectance DN to reflectance.

    Formula: reflectance = DN * 2.75e-05 - 0.2 (USGS)
    """
    return (sr.astype("float32") * 2.75e-05) - 0.2


def load_landsat_composite(config: Config) -> xr.Dataset:
    """Load a median Landsat 8/9 composite over the region and time window.

    Returns an xarray.Dataset with bands: red, nir, swir1, swir2, ndvi_masked.
    Reprojected to the region UTM CRS at target resolution.
    """
    bbox = _region_bbox(config)
    utm = _utm_crs(config)

    LOGGER.info("Searching Planetary Computer for Landsat items...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=list(config.landsat.collections),
        bbox=bbox,
        datetime=config.landsat.datetime,
        query={"eo:cloud_cover": {"lt": config.landsat.max_cloud_cover}},
    )
    items = list(search.get_items())
    if len(items) == 0:
        raise RuntimeError("No Landsat items found for the specified region and dates")

    LOGGER.info("Found %d Landsat items; stacking bands", len(items))
    signed_items = [pc.sign(item) for item in items]

    assets = list(config.landsat.assets)
    stack = stackstac.stack(
        signed_items,
        assets=assets,
        epsg=utm.to_epsg(),
        resolution=config.model.resolution_m,
        bounds_latlon=bbox,
        dtype="float32",
        chunksize=1024,
    )
    # Rename assets to band names
    band_map = {
        "SR_B4": "red",
        "SR_B5": "nir",
        "SR_B6": "swir1",
        "SR_B7": "swir2",
        "QA_PIXEL": "qa",
    }
    stack = stack.assign_coords(band=[band_map.get(b.item(), str(b.item())) for b in stack.band])

    # Scale SR bands
    for b in ["red", "nir", "swir1", "swir2"]:
        if b in stack.band.values:
            stack.loc[dict(band=b)] = _scale_landsat_sr(stack.loc[dict(band=b)])

    # Cloud mask from QA_PIXEL bits (conservative)
    qa = stack.sel(band="qa").astype("uint32")
    # Bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow
    cloud_mask = (
        ((qa & (1 << 0)) != 0)
        | ((qa & (1 << 1)) != 0)
        | ((qa & (1 << 2)) != 0)
        | ((qa & (1 << 3)) != 0)
        | ((qa & (1 << 4)) != 0)
        | ((qa & (1 << 5)) != 0)
    )

    # Apply cloud mask to SR bands
    bands = stack.sel(band=[b for b in ["red", "nir", "swir1", "swir2"] if b in stack.band.values])
    bands = bands.where(~cloud_mask)

    LOGGER.info("Computing median mosaic across time after cloud masking")
    median = bands.median(dim="time", skipna=True)
    ds = median.to_dataset(dim="band")
    ds.rio.write_crs(utm, inplace=True)

    return ds


def load_dem(config: Config) -> xr.DataArray:
    """Load DEM (Copernicus DEM 30m) mosaicked and reprojected to UTM."""
    bbox = _region_bbox(config)
    utm = _utm_crs(config)

    LOGGER.info("Searching Planetary Computer for DEM tiles...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=[config.dem.collection],
        bbox=bbox,
    )
    items = list(search.get_items())
    if len(items) == 0:
        raise RuntimeError("No DEM items found for the specified region")

    signed_items = [pc.sign(item) for item in items]
    stack = stackstac.stack(
        signed_items,
        assets=[config.dem.asset],
        epsg=utm.to_epsg(),
        resolution=config.model.resolution_m,
        bounds_latlon=bbox,
        dtype="float32",
        chunksize=1024,
    )
    # Remove asset dim and time via median mosaic
    dem = stack.median(dim="time", skipna=True).squeeze(drop=True)
    dem = dem.rio.write_crs(utm)
    dem.name = "elevation"
    return dem


def load_mrds(config: Config) -> gpd.GeoDataFrame:
    """Download and cache USGS MRDS, filter to region and REE-related commodities."""
    cache_dir = config.paths.cache_dir
    _ensure_dir(cache_dir)
    csv_path = cache_dir / config.mrds.cache_filename

    if not csv_path.exists():
        LOGGER.info("Downloading MRDS from %s", config.mrds.url)
        import requests

        resp = requests.get(config.mrds.url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Find CSV inside archive
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, low_memory=False)
        # Cache CSV
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path, low_memory=False)

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    if not {"longitude", "latitude"}.issubset(df.columns):
        # Alternative column names in some exports
        lon_col = "longdec" if "longdec" in df.columns else "longitude"
        lat_col = "latdec" if "latdec" in df.columns else "latitude"
    else:
        lon_col, lat_col = "longitude", "latitude"

    # Filter rows with valid coordinates
    df = df[np.isfinite(df[lon_col]) & np.isfinite(df[lat_col])]

    # Filter to bbox
    minx, miny, maxx, maxy = _region_bbox(config)
    df = df[(df[lon_col] >= minx) & (df[lon_col] <= maxx) & (df[lat_col] >= miny) & (df[lat_col] <= maxy)]

    if df.empty:
        LOGGER.warning("MRDS has no entries within region bbox. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy([], []), crs="EPSG:4326").to_crs(config.region.utm_crs())

    # Filter to REE commodities
    comm_cols = [c for c in df.columns if "commod" in c]
    pattern = "|".join(config.mrds.commodity_keywords)
    mask = pd.Series(False, index=df.index)
    for cc in comm_cols:
        mask = mask | df[cc].astype(str).str.lower().str.contains(pattern, na=False)
    df = df[mask]

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    gdf = gdf.to_crs(config.region.utm_crs())
    return gdf
