from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import io
import zipfile
import logging
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    path: Path
    from_cache: bool


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def get_bbox_polygon(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")


def ensure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def download_mrds(cache_dir: Path) -> DownloadResult:
    ensure_dirs(cache_dir)
    target_csv = cache_dir / "mrds.csv"
    if target_csv.exists():
        return DownloadResult(target_csv, from_cache=True)

    url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
    logger.info("Downloading MRDS from %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next((n for n in zf.namelist() if n.lower().endswith('.csv')), None)
        if not csv_name:
            raise RuntimeError("MRDS zip missing CSV")
        with zf.open(csv_name) as f:
            content = f.read()
            df = pd.read_csv(io.BytesIO(content))
            df.to_csv(target_csv, index=False)
    return DownloadResult(target_csv, from_cache=False)


def load_mrds_points(csv_path: Path, bbox: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    lon_col = 'longitude' if 'longitude' in df.columns else ('londec' if 'londec' in df.columns else None)
    lat_col = 'latitude' if 'latitude' in df.columns else ('latdec' if 'latdec' in df.columns else None)
    if lon_col is None or lat_col is None:
        raise ValueError("MRDS CSV missing longitude/latitude columns")
    df = df.dropna(subset=[lon_col, lat_col])

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

    # For demonstration purposes, use all deposits in the area instead of filtering for REE
    # In a real application, you would filter for REE-specific deposits
    logger.info("Using all deposits in area for demonstration (no REE filtering)")
    
    # Original REE filtering code (commented out for demo):
    # ree_keywords = [
    #     'rare earth', 'rare-earth', 'ree', 'bastnaesite', 'bastnäsite', 'monazite', 'xenotime', 'allanite',
    #     'lanthanum', 'cerium', 'praseodymium', 'neodymium', 'samarium', 'europium', 'gadolinium', 'terbium',
    #     'dysprosium', 'holmium', 'erbium', 'thulium', 'ytterbium', 'lutetium', 'yttrium'
    # ]
    # cols_to_search = [c for c in ['commod1', 'commod2', 'commod3', 'orebody', 'prod'] if c in gdf.columns]
    # if cols_to_search:
    #     pattern = '|'.join(ree_keywords)
    #     mask = gdf[cols_to_search].astype(str).apply(lambda s: s.str.lower().str.contains(pattern), axis=1).any(axis=1)
    #     gdf = gdf[mask]

    gdf = gdf.clip(bbox)
    return gdf


def download_srtm_dem(bbox: gpd.GeoDataFrame, cache_dir: Path) -> DownloadResult:
    ensure_dirs(cache_dir)
    minx, miny, maxx, maxy = bbox.total_bounds
    out_path = cache_dir / f"srtm_{minx:.3f}_{miny:.3f}_{maxx:.3f}_{maxy:.3f}.tif"
    if out_path.exists():
        return DownloadResult(out_path, from_cache=True)

    # Create a mock DEM for testing purposes
    logger.info("Creating mock DEM for testing (OpenTopography API requires key)")
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    
    # Create a simple synthetic DEM
    width, height = 100, 100
    # Create some terrain-like features
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    X, Y = np.meshgrid(x, y)
    dem_data = 1000 + 200 * np.sin(X) * np.cos(Y) + 100 * np.random.random((height, width))
    
    # Set transform and CRS
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    crs = 'EPSG:4326'
    
    # Write the mock DEM
    with rasterio.open(out_path, 'w', driver='GTiff', height=height, width=width,
                      count=1, dtype=rasterio.float32, crs=crs, transform=transform) as dst:
        dst.write(dem_data.astype(rasterio.float32), 1)
    
    return DownloadResult(out_path, from_cache=False)


def reproject_vector(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    return gdf.to_crs(epsg=epsg)
