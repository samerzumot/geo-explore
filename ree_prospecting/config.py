from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import logging
from shapely.geometry import box, Polygon
from pyproj import CRS


@dataclass(frozen=True)
class Region:
    """Geospatial region configuration.

    Bounding box is in WGS84 lon/lat: (minx, miny, maxx, maxy).
    """

    name: str
    bbox_wgs84: Tuple[float, float, float, float]

    def polygon_wgs84(self) -> Polygon:
        return box(*self.bbox_wgs84)

    def utm_crs(self) -> CRS:
        """Derive UTM CRS from region centroid."""
        minx, miny, maxx, maxy = self.bbox_wgs84
        lon = (minx + maxx) / 2.0
        lat = (miny + maxy) / 2.0
        zone_number = int((lon + 180) / 6) + 1
        is_northern = lat >= 0
        return CRS.from_dict({"proj": "utm", "zone": zone_number, "south": not is_northern})


@dataclass(frozen=True)
class Paths:
    root_dir: Path
    data_dir: Path
    cache_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class LandsatSettings:
    collections: Tuple[str, ...] = ("landsat-c2-l2",)
    datetime: str = "2019-01-01/2024-12-31"
    max_cloud_cover: int = 20
    # Target assets for indices: red, nir, swir1, swir2, qa
    assets: Tuple[str, ...] = ("SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL")


@dataclass(frozen=True)
class DEMSettings:
    # Planetary Computer Copernicus DEM 30m
    collection: str = "cop-dem-glo-30"
    asset: str = "data"


@dataclass(frozen=True)
class MRDSSettings:
    # USGS MRDS CSV export
    url: str = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
    cache_filename: str = "mrds.csv"
    commodity_keywords: Tuple[str, ...] = (
        "rare earth", "rare-earth", "ree", "lanthanum", "cerium", "neodymium",
        "praseodymium", "yttrium", "dysprosium", "terbium", "europium",
    )


@dataclass(frozen=True)
class ModelingSettings:
    resolution_m: int = 90
    ndvi_mask_threshold: float = 0.2
    positive_buffer_m: int = 1000
    random_negative_multiplier: float = 2.0
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 1
    class_weight_balanced: bool = True
    spatial_block_size_m: int = 5000
    cv_folds: int = 5
    random_seed: int = 42
    lineament_density_radius_m: int = 1000


@dataclass(frozen=True)
class Config:
    region: Region
    paths: Paths
    landsat: LandsatSettings
    dem: DEMSettings
    mrds: MRDSSettings
    model: ModelingSettings


def get_default_config(root: Optional[Path] = None) -> Config:
    """Create default configuration for Mountain Pass, CA (50km x 50km).

    Approximate center: (-115.531, 35.482). Half-side ~25 km.
    """
    if root is None:
        root = Path(__file__).resolve().parent
    # Bounding box in degrees
    bbox = (-115.806, 35.257, -115.256, 35.707)
    region = Region(name="mountain_pass_ca", bbox_wgs84=bbox)

    project_root = root
    paths = Paths(
        root_dir=project_root,
        data_dir=project_root.parent / "data",
        cache_dir=project_root.parent / "cache",
        outputs_dir=project_root.parent / "outputs",
    )

    # Ensure directories exist (best-effort; not fatal)
    for p in [paths.data_dir, paths.cache_dir, paths.outputs_dir]:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logging.warning("Failed to create directory %s: %s", p, exc)

    return Config(
        region=region,
        paths=paths,
        landsat=LandsatSettings(),
        dem=DEMSettings(),
        mrds=MRDSSettings(),
        model=ModelingSettings(),
    )
