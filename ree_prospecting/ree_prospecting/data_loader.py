"""Data ingestion for REE prospectivity MVP.

Sources used in MVP (2-3 easiest):
- Landsat 8/9 (AWS Open Data via requests/rasterio) or Google Earth Engine optional
- SRTM DEM (NASA SRTM 1 Arc-Second via AWS or OpenTopography)
- USGS MRDS (Mineral Resources Data System) for known deposits

Notes:
- To keep MVP self-contained, we avoid heavy dependencies on Earth Engine auth.
- We cache downloads in `data/cache` and re-use them if present.
- CRS handling: Inputs in WGS84; we provide utilities to reproject rasters/vectors to UTM.

Data sources (documented):
- Landsat (Collection 2 Level-2 on AWS): `https://registry.opendata.aws/landsat-8/`
- SRTM 1 Arc-Second tiles: `https://registry.opendata.aws/terrain-tiles/` or `https://opentopography.org`
- USGS MRDS download: `https://mrdata.usgs.gov/mrds/mrds-csv.zip`
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import io
import zipfile
import logging
import requests
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS

# rasterio imported lazily where needed to speed import time

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    path: Path
    from_cache: bool


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def get_bbox_polygon(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")
    return gdf


def download_mrds(cache_dir: Path) -> DownloadResult:
    """Download MRDS CSV zip and extract to cache.

    Returns the path to the extracted CSV.
    """
    ensure_dirs(cache_dir)
    target_csv = cache_dir / "mrds.csv"
    if target_csv.exists():
        return DownloadResult(target_csv, from_cache=True)

    url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
    logger.info("Downloading MRDS from %s", url)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # find main CSV
            csv_name = next((n for n in zf.namelist() if n.lower().endswith('.csv')), None)
            if not csv_name:
                raise RuntimeError("MRDS zip missing CSV")
            with zf.open(csv_name) as f, open(target_csv, 'wb') as out:
                out.write(f.read())
    except Exception as e:
        logger.exception("Failed to download MRDS: %s", e)
        raise

    return DownloadResult(target_csv, from_cache=False)


def load_mrds_points(csv_path: Path, bbox: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load MRDS and filter to bbox. Returns WGS84 GeoDataFrame of point deposits.
    We filter to records with valid coordinates and REE-related commodities/keywords.
    """
    df = gpd.read_file(csv_path)
    # MRDS has 'longitude'/'latitude' fields (sometimes named 'londec'/'latdec'); handle both
    lon_col = 'longitude' if 'longitude' in df.columns else ('londec' if 'londec' in df.columns else None)
    lat_col = 'latitude' if 'latitude' in df.columns else ('latdec' if 'latdec' in df.columns else None)
    if lon_col is None or lat_col is None:
        raise ValueError("MRDS CSV missing longitude/latitude columns")
    df = df.dropna(subset=[lon_col, lat_col])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

    # Filter to REE-related commodities keywords in 'commod1','commod2','commod3','orebody' etc.
    ree_keywords = [
        'rare earth', 'rare-earth', 'REE', 'bastnaesite', 'bastnäsite', 'monazite', 'xenotime', 'allanite',
        'lanthanum', 'cerium', 'praseodymium', 'neodymium', 'samarium', 'europium', 'gadolinium', 'terbium',
        'dysprosium', 'holmium', 'erbium', 'thulium', 'ytterbium', 'lutetium', 'yttrium'
    ]
    cols_to_search = [c for c in ['commod1', 'commod2', 'commod3', 'orebody', 'prod'] if c in gdf.columns]
    if cols_to_search:
        mask = gdf[cols_to_search].astype(str).apply(lambda s: s.str.lower().str.contains('|'.join(ree_keywords)), axis=1).any(axis=1)
        gdf = gdf[mask]

    gdf = gdf.clip(bbox)
    return gdf


def download_srtm_dem(bbox: gpd.GeoDataFrame, cache_dir: Path) -> DownloadResult:
    """Download SRTM DEM for the bbox using elevation API from OpenTopography (simple).
    For MVP, use 1 arc-second (~30m) SRTM if available. Fallback to 3 arc-second (~90m).

    We use OpenTopography GlobalDEM API: https://portal.opentopography.org/apidocs/#/Public/getGlobalDem
    Example: https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=...&north=...&west=...&east=...&outputFormat=GTiff
    """
    ensure_dirs(cache_dir)
    minx, miny, maxx, maxy = bbox.total_bounds
    out_path = cache_dir / f"srtm_{minx:.3f}_{miny:.3f}_{maxx:.3f}_{maxy:.3f}.tif"
    if out_path.exists():
        return DownloadResult(out_path, from_cache=True)

    base = "https://portal.opentopography.org/API/globaldem"
    params = {
        'demtype': 'SRTMGL1',  # 1 arc-second
        'south': f"{miny}",
        'north': f"{maxy}",
        'west': f"{minx}",
        'east': f"{maxx}",
        'outputFormat': 'GTiff',
    }
    try:
        logger.info("Requesting SRTM DEM from OpenTopography")
        r = requests.get(base, params=params, timeout=120)
        r.raise_for_status()
        # Sometimes returns JSON error; check content-type
        if 'tif' not in r.headers.get('Content-Type', '') and not out_path.suffix.lower().endswith('.tif'):
            # Fallback to SRTMGL3
            params['demtype'] = 'SRTMGL3'
            r = requests.get(base, params=params, timeout=120)
            r.raise_for_status()
        with open(out_path, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        logger.exception("Failed to download SRTM DEM: %s", e)
        raise
    return DownloadResult(out_path, from_cache=False)


def search_aws_landsat_scene_ids() -> List[str]:
    """Placeholder: In a full implementation we'd query STAC for Landsat scenes.
    For MVP robustness, we will skip dynamic search and expect preselected scene IDs covering the bbox.
    Mountain Pass area is covered by Landsat Path/Row ~ 040/035 (example).
    """
    # Preselected sample L2 scenes (could be updated):
    return [
        # Users can replace with recent cloud-free IDs
        # Format: LC08_L2SP_<pathrow>_<date>_... or LC09_L2SP_...
    ]


def download_landsat_bands(cache_dir: Path, scene_ids: List[str], bands: List[str]) -> List[Path]:
    """Download selected Landsat bands (Surface Reflectance) from AWS if scene_ids provided.
    If no scene IDs provided, this function returns an empty list; MVP can proceed with DEM+MRDS only.
    """
    ensure_dirs(cache_dir)
    urls: List[Tuple[str, Path]] = []
    base = "https://landsat-pds.s3.amazonaws.com/c1/L8/"
    out_paths: List[Path] = []
    if not scene_ids:
        logger.warning("No Landsat scene IDs provided; skipping satellite-derived features in MVP.")
        return out_paths
    for sid in scene_ids:
        # In practice we'd parse path/row from ID; here we assume precomputed URL structure is known.
        for band in bands:
            # This placeholder won't download unless valid IDs are provided.
            url = f"{base}{sid}/{sid}_{band}.TIF"
            out = cache_dir / f"{sid}_{band}.TIF"
            urls.append((url, out))
    for url, out in urls:
        if out.exists():
            out_paths.append(out)
            continue
        try:
            logger.info("Downloading %s", url)
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 1024:
                with open(out, 'wb') as f:
                    f.write(r.content)
                out_paths.append(out)
            else:
                logger.warning("Failed to fetch %s (status %s)", url, r.status_code)
        except Exception:
            logger.exception("Error downloading %s", url)
    return out_paths


def reproject_vector(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    return gdf.to_crs(epsg=epsg)


def ensure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
