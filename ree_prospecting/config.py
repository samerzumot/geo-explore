from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Region:
    name: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    utm_epsg: int


@dataclass
class Paths:
    base_dir: Path
    data_dir: Path
    cache_dir: Path
    outputs_dir: Path
    figures_dir: Path


@dataclass
class ModelParams:
    random_state: int
    grid_resolution_m: int
    rf_n_estimators: int
    rf_max_depth: int | None
    rf_class_weight: str | dict[str, float] | None
    smote_enabled: bool
    spatial_block_k: int


@dataclass
class Config:
    region: Region
    paths: Paths
    model: ModelParams

    @staticmethod
    def default(base_dir: str | Path = Path(__file__).resolve().parents[1]) -> "Config":
        base = Path(base_dir)
        region = Region(
            name="Mountain_Pass_CA",
            min_lon=-116.6,
            min_lat=35.3,
            max_lon=-115.9,
            max_lat=35.8,
            utm_epsg=32611,
        )
        paths = Paths(
            base_dir=base,
            data_dir=base / "data",
            cache_dir=base / "data" / "cache",
            outputs_dir=base / "outputs",
            figures_dir=base / "figures",
        )
        model = ModelParams(
            random_state=42,
            grid_resolution_m=30,
            rf_n_estimators=300,
            rf_max_depth=None,
            rf_class_weight="balanced_subsample",
            smote_enabled=True,
            spatial_block_k=5,
        )
        return Config(region=region, paths=paths, model=model)


CFG = Config.default()
