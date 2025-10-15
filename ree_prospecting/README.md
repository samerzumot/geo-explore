# REE Prospectivity Mapping (Prototype)

A minimal end-to-end prototype framework for REE (Rare Earth Elements) prospectivity mapping over a single pilot region (Mountain Pass, CA; ~50 km x 50 km).

This MVP demonstrates data ingestion, feature engineering, a baseline Random Forest model, and visualization of predicted hotspots.

## Scope

- Single pilot region only (Mountain Pass, CA). Modular design to extend to other regions via `config`.
- Data sources used:
  - Landsat 8/9 Collection-2 Level-2 Surface Reflectance (via Microsoft Planetary Computer STAC)
  - Copernicus DEM GLO-30 (DEM via Planetary Computer)
  - USGS MRDS (Mineral Resources Data System) for known deposits
- Optional data (future work): USGS NGDB geochemistry; hyperspectral; aeromagnetics.

## Installation

Recommended: use Conda or mamba.

```bash
conda create -n ree-prospectivity python=3.10 -y
conda activate ree-prospectivity
pip install -r requirements.txt
```

If `rasterio` or `geopandas` wheels are unavailable for your platform, consider using `conda-forge` packages:

```bash
conda install -c conda-forge geopandas rasterio rioxarray scikit-image pyproj shapely fiona gdal -y
pip install -r requirements.txt --no-deps
```

## Data Access

- Microsoft Planetary Computer (MPC) public STAC API is used; no API key required for small-scale access.
  - Catalog: `https://planetarycomputer.microsoft.com/api/stac/v1`
  - Collections: `landsat-c2-l2`, `cop-dem-glo-30`
- USGS MRDS CSV is downloaded and cached locally.

## Running the Pipeline

```bash
python -m ree_prospecting.main
```

Outputs are written to `outputs/`:
- `prospectivity.tif`: Probability (0-1) raster
- `uncertainty_std.tif`: Random Forest prediction stddev raster
- `hotspots_top10.geojson`: Top 10 polygons by mean probability
- `map.html`: Interactive map (hotspots + MRDS points)
- `feature_importance.png`: Feature importance
- `model_confusion_matrix.png`, `model_roc.png`: Metrics plots
- `metrics.json`: Precision, recall, F1, ROC AUC

## Configuration

Configuration is defined in `config.py`. Defaults target Mountain Pass, CA. Key parameters:
- Spatial resolution (m)
- Landsat date range and cloud cover threshold
- MRDS commodity filters and training buffer radius
- Spatial cross-validation and block size

To adapt to a new region, edit `Region` bbox in `get_default_config()` or provide a custom `Config`.

## Feature Engineering

- Landsat indices:
  - Iron oxide ratio = SWIR1 / SWIR2
  - Clay alteration ratio = SWIR1 / NIR
  - NDVI for vegetation masking
- DEM derivatives:
  - Slope, aspect (sin/cos)
  - Simple lineament density (Canny edges on slope + local mean)
- Proximity:
  - Distance to nearest MRDS point (km)

Note: Geoscience assumptions are simplified for MVP; indices are proxies for alteration and may need refinement.

## Modeling

- Random Forest classifier (tabular features)
- Labels: positives = pixels within buffer around MRDS deposits; negatives = random background pixels
- Spatial cross-validation via block groups to reduce spatial leakage
- Class imbalance addressed with class weights or SMOTE (optional)
- Uncertainty = stddev across trees

## Validation

- Metrics: precision, recall, F1, ROC AUC
- Hit-rate analysis recommendation: inspect MRDS points vs high-scoring hotspots on `map.html`

## Limitations

- Not production-ready; designed for a single region demo
- Spectral indices and lineament proxy are simplistic
- MRDS may be incomplete; deposit locations approximate
- Geochemistry (NGDB) not integrated in MVP due to access/size; can be added
- CNN image model phase omitted for brevity; ensemble left as future work

## Future Improvements

- Add USGS NGDB geochemistry (distance to REE samples; IDW interpolation)
- Include additional remote-sensing layers (ASTER indices, hyperspectral)
- Incorporate geologic map features (lithology, structure)
- More robust cloud/shadow masking and temporal compositing
- Proper spatial blocking (e.g., k-fold with hexagonal bins)
- Probabilistic models and physics-informed priors
