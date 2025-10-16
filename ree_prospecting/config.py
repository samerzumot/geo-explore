"""Configuration for REE prospectivity MVP (Mountain Pass, CA).

This module centralizes all parameters (region, paths, resolution, and model
hyperparameters). Change values here to run the pipeline for a different region.

Data sources referenced for the MVP (documented also in README):
- Landsat 8/9 Collection 2 Level-2 via Microsoft Planetary Computer STAC
  API: https://planetarycomputer.microsoft.com/api/stac/v1
- USGS MRDS (Mineral Resources Data System): https://mrdata.usgs.gov/mrds/
  CSV: https://mrdata.usgs.gov/mrds/mrds-csv.zip
- SRTM DEM (1 arc-second) via the `elevation` package (SRTM tiles)
  Package: https://pypi.org/project/elevation/

Notes:
- This is a proof-of-concept for a single pilot region (~50km x 50km)
- Resolution defaults to 90m for speed; switch to 30m if your machine allows it
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class Config:
    """Top-level configuration container for the MVP.

    Adjust `bbox_wgs84` to change the pilot region. Default is the Mountain Pass,
    CA area (approximate bounding box ~50 km x 50 km).
    """

    # Region / CRS
    region_name: str = "Mountain Pass, CA"
    # WGS84 lon/lat bbox: (minx, miny, maxx, maxy)
    # Approx Mountain Pass region (~50 km box around 35.48N, -115.53E)
    bbox_wgs84: Tuple[float, float, float, float] = (
        -115.90, 35.25, -115.15, 35.75
    )
    # Target projected CRS; if None, will be computed as UTM zone from bbox center.
    target_epsg: Optional[int] = None

    # Resolution and grid
    # Use 90.0 by default; can try 30.0 if you have resources/time
    resolution_m: float = 90.0

    # IO paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    # Landsat settings
    landsat_collection: str = "landsat-c2-l2"
    landsat_max_cloud: float = 20.0
    landsat_prefer_sensors: Tuple[str, ...] = ("LC09", "LC08")  # L9 then L8

    # Vegetation masking
    ndvi_mask_threshold: float = 0.2

    # Training data and modeling
    positive_buffer_m: float = 1000.0
    negative_multiplier: float = 2.0  # negatives = multiplier * positives
    random_state: int = 42
    spatial_blocks: Tuple[int, int] = (4, 4)  # block grid for spatial CV

    # Random Forest hyperparameters
    rf_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )

    # Optional geochemistry source (None to skip for MVP)
    geochem_csv_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.data_dir = self.base_dir / "data"
        self.cache_dir = self.base_dir / "_cache"
        self.output_dir = self.base_dir / "outputs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def bbox_polygon_wgs84(self):
        """Shapely polygon of the WGS84 bounding box (lazy import to avoid heavy deps)."""
        from shapely.geometry import box

        minx, miny, maxx, maxy = self.bbox_wgs84
        return box(minx, miny, maxx, maxy)

    @property
    def target_crs_epsg(self) -> int:
        """Return the target EPSG code; derive UTM if not set.

        The region is in southern California, which is UTM zone 11N (EPSG:32611).
        If `target_epsg` is None, we compute from bbox center.
        """
        if self.target_epsg is not None:
            return self.target_epsg

        # Compute UTM from bbox center
        minx, miny, maxx, maxy = self.bbox_wgs84
        center_lon = (minx + maxx) / 2.0
        center_lat = (miny + maxy) / 2.0

        utm_zone = int((center_lon + 180) // 6) + 1
        is_northern = center_lat >= 0
        return 32600 + utm_zone if is_northern else 32700 + utm_zone


# Singleton default config for simple scripts
DEFAULT_CONFIG = Config()
