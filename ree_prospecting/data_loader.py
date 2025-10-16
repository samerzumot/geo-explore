"""Data loaders for the REE prospectivity MVP.

This module focuses on lightweight, public data sources with minimal friction:
- Landsat 8/9 C2 L2 via Microsoft Planetary Computer STAC (no key required)
- USGS MRDS (Mineral Resources Data System) CSV
- SRTM DEM via the `elevation` package (SRTM 1 arc-second)

Each function caches downloads in `config.cache_dir` and returns data in
convenient forms for preprocessing. All spatial operations assume input bbox in
WGS84 (EPSG:4326) and return rasters with their native georeferencing; alignment
is handled later during preprocessing.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from shapely.geometry import Point, box
import geopandas as gpd

from .config import Config, DEFAULT_CONFIG

try:
    # STAC + Planetary Computer for Landsat
    import planetary_computer as pc
    from pystac_client import Client
except Exception:  # pragma: no cover - optional dependency during scaffold
    pc = None
    Client = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RasterLayer:
    """Simple container for a raster and its georeferencing."""

    array: np.ndarray  # shape: (height, width) or (bands, height, width)
    transform: rasterio.Affine
    crs: CRS
    nodata: Optional[float]


# -------------------------
# MRDS (USGS) loader
# -------------------------
MRDS_CSV_ZIP_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"


def load_mrds_points(
    config: Config = DEFAULT_CONFIG,
    keywords: Optional[Iterable[str]] = None,
) -> gpd.GeoDataFrame:
    """Download and filter USGS MRDS points within the region bbox.

    Parameters
    - config: configuration with bbox and cache path
    - keywords: commodity keywords to filter (default targets REE commodities)

    Returns
    - GeoDataFrame in EPSG:4326 with columns: ['name','commods','source','geometry']
    """
    cache_path = config.cache_dir / "mrds-csv.zip"

    if not cache_path.exists():
        logger.info("Downloading MRDS CSV archive... %s", MRDS_CSV_ZIP_URL)
        resp = requests.get(MRDS_CSV_ZIP_URL, timeout=120)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        logger.info("Saved: %s (%.1f MB)", cache_path, cache_path.stat().st_size / 1e6)
    else:
        logger.info("Using cached MRDS at %s", cache_path)

    # Read the CSV within the ZIP without extracting to disk
    logger.info("Reading MRDS CSV from zip...")
    import zipfile

    with zipfile.ZipFile(cache_path) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, dtype=str, low_memory=False)

    # Normalize columns
    for col in ("latitude", "longitude"):
        if col not in df.columns:
            raise RuntimeError(f"MRDS CSV missing required column '{col}'")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Merge commodities columns into one for filtering
    commodity_cols = [c for c in df.columns if c.lower().startswith("commod")]
    df["commods"] = df[commodity_cols].fillna("").agg(";".join, axis=1).str.upper()

    # Bounding box filter first (fast)
    minx, miny, maxx, maxy = config.bbox_wgs84
    bbox_poly = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame(
        df[["site_name", "commods", "source", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf = gdf[gdf.within(bbox_poly)].copy()

    # REE filtering
    if keywords is None:
        keywords = (
            "REE",
            "RARE EARTH",
            "LANTHANIDE",
            "CERIUM",
            "LANTHANUM",
            "NEODYMIUM",
            "YTTRIUM",
            "SCANDIUM",
        )
    pattern = re.compile("|".join(map(re.escape, keywords)))
    gdf = gdf[gdf["commods"].str.contains(pattern, na=False)].copy()

    gdf.rename(columns={"site_name": "name"}, inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    logger.info("MRDS points in bbox (REE-filtered): %d", len(gdf))
    return gdf[["name", "commods", "source", "geometry"]]


# -------------------------
# DEM loader (SRTM via elevation)
# -------------------------

def download_dem_srtm(
    config: Config = DEFAULT_CONFIG,
    product: str = "SRTM1",
) -> Path:
    """Download/clip SRTM DEM for bbox to GeoTIFF.

    Tries `elevation` package first. If unavailable or GDAL is missing,
    falls back to OpenTopography Global DEM API (SRTMGL1) which may require
    an API key via environment var `OPENTOPO_API_KEY`.
    """
    minx, miny, maxx, maxy = config.bbox_wgs84
    out_path = config.data_dir / f"dem_srtm_{config.region_name.replace(' ', '_')}.tif"
    if out_path.exists():
        logger.info("Using cached DEM at %s", out_path)
        return out_path

    # Attempt elevation first
    try:
        import elevation  # type: ignore

        logger.info("Downloading and clipping SRTM DEM via 'elevation'...")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        elevation.clip(
            bounds=(minx, miny, maxx, maxy),
            output=str(out_path),
            product=product,
        )
        logger.info("DEM saved: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        return out_path
    except Exception as exc:
        logger.warning("elevation-based DEM download failed (%s); trying OpenTopography API...", exc)

    # Fallback: OpenTopography API (SRTMGL1)
    import os
    api_key = os.environ.get("OPENTOPO_API_KEY", "")
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=SRTMGL1&south={miny}&north={maxy}&west={minx}&east={maxx}"
        "&outputFormat=GTiff"
        + (f"&API_Key={api_key}" if api_key else "")
    )
    logger.info("Requesting DEM from OpenTopography...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    logger.info("DEM saved: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def open_raster_clipped(path: Path, bbox_wgs84: Tuple[float, float, float, float]) -> RasterLayer:
    """Open a raster and clip it to provided WGS84 bbox, returning array + georef.

    For DEM rasters, returns a single band (H, W). For multi-band rasters,
    returns (bands, H, W).
    """
    with rasterio.open(path) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError("Raster has no CRS")
        # Create a mask geometry from bbox in source CRS
        bbox = box(*bbox_wgs84)
        geom = gpd.GeoSeries([bbox], crs="EPSG:4326").to_crs(src_crs)
        shapes = [mapping(geom.geometry.iloc[0])]
        out_image, out_transform = rio_mask(src, shapes, crop=True)
        out_nodata = src.nodata
        if src.count == 1:
            out_image = out_image[0]
        return RasterLayer(out_image, out_transform, src_crs, out_nodata)


# -------------------------
# Landsat 8/9 via Planetary Computer STAC
# -------------------------

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _select_best_landsat_item(items, prefer_sensors: Tuple[str, ...]) -> Optional[object]:
    """Pick the lowest-cloud item preferring specified sensors (e.g., LC09 then LC08)."""
    if not items:
        return None
    # Group by preferred sensors
    ranked = []
    for sensor in prefer_sensors:
        for it in items:
            if it.id.startswith(sensor):
                ranked.append(it)
    # Fallback: any
    if not ranked:
        ranked = list(items)
    # Sort by cloud cover
    ranked.sort(key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)))
    return ranked[0]


def load_landsat_stack(
    config: Config = DEFAULT_CONFIG,
) -> Dict[str, RasterLayer]:
    """Query STAC and load Landsat bands for the bbox.

    Returns a dict of band name -> RasterLayer for required bands (B4,B5,B6,B7)
    and the QA_PIXEL band when available.
    """
    if Client is None or pc is None:
        raise RuntimeError(
            "pystac-client and planetary-computer packages are required for Landsat."
        )

    minx, miny, maxx, maxy = config.bbox_wgs84
    bbox = [minx, miny, maxx, maxy]

    logger.info("Searching Landsat STAC on Planetary Computer for low-cloud scenes...")
    client = Client.open(PC_STAC)
    search = client.search(
        collections=[config.landsat_collection],
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": config.landsat_max_cloud}},
        limit=100,
    )
    items = list(search.get_items())
    if not items:
        raise RuntimeError("No Landsat items found in bbox and cloud constraints")

    item = _select_best_landsat_item(items, config.landsat_prefer_sensors)
    if item is None:
        raise RuntimeError("No suitable Landsat item found")

    item = pc.sign(item)
    logger.info(
        "Selected Landsat item: %s (clouds=%.1f%%)",
        item.id,
        float(item.properties.get("eo:cloud_cover", 100.0)),
    )

    # Required bands for features (L8/9 naming)
    assets = {
        "B4": "SR_B4",  # Red
        "B5": "SR_B5",  # NIR
        "B6": "SR_B6",  # SWIR1
        "B7": "SR_B7",  # SWIR2
        "QA_PIXEL": "QA_PIXEL",
    }

    bands: Dict[str, RasterLayer] = {}
    for key, asset_name in assets.items():
        if asset_name not in item.assets:
            logger.warning("Asset %s not present on item %s", asset_name, item.id)
            continue
        href = item.assets[asset_name].href
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
            with rasterio.open(href) as src:
                src_crs = src.crs
                if src_crs is None:
                    raise ValueError("Landsat asset has no CRS")
                bbox_poly = box(minx, miny, maxx, maxy)
                geom = gpd.GeoSeries([bbox_poly], crs="EPSG:4326").to_crs(src_crs)
                shapes = [mapping(geom.geometry.iloc[0])]
                out, transform = rio_mask(src, shapes, crop=True)
                nodata = src.nodata
                if src.count == 1:
                    out = out[0]
        bands[key] = RasterLayer(out, transform, src_crs, nodata)
        logger.info("Loaded Landsat %s (%s) with shape %s", key, asset_name, np.shape(out))

    return bands


def apply_landsat_scaling(bands: Dict[str, RasterLayer]) -> Dict[str, RasterLayer]:
    """Apply reflectance scaling to Landsat SR bands (C2 L2):
    reflectance = DN * 2.75e-05 - 0.2
    """
    scaled: Dict[str, RasterLayer] = {}
    for key, layer in bands.items():
        if key in {"B4", "B5", "B6", "B7"}:
            arr = layer.array.astype("float32") * 2.75e-05 - 0.2
            scaled[key] = RasterLayer(arr, layer.transform, layer.crs, layer.nodata)
        else:
            scaled[key] = layer
    return scaled
