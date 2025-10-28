# REE Prospectivity Mapping Prototype (MVP)

Prototype framework for rare earth element (REE) prospectivity mapping over a single pilot region (Mountain Pass, CA ~50x50 km). Demonstrates end-to-end data ingestion, feature engineering, baseline modeling, and interactive outputs.

## Key Choices and Scope
- Region: Mountain Pass, CA (WGS84 bbox in `config.py`).
- Data sources (MVP):
  - SRTM DEM via OpenTopography GlobalDEM API (1 arc-sec, fallback to 3 arc-sec).
  - USGS MRDS (Mineral Resources Data System) for known deposits.
  - Landsat bands optional (stubbed for AWS download; disabled by default for robustness). Future: enable STAC/EE.
- Resolution: 30 m target grid in local UTM (EPSG:32611).
- Model: Random Forest classifier on tabular features.
- Validation: simple hold-in evaluation on resampled labeled pixels; spatial CV is future work for MVP.

## Install

Using conda (recommended):

```bash
conda create -n ree python=3.10 -y
conda activate ree
pip install -r requirements.txt
```

If `rtree` install fails, install system deps or skip spatial index acceleration.

## Data Sources
- SRTM Global DEM: `https://portal.opentopography.org/apidocs/#/Public/getGlobalDem`
- USGS MRDS CSV: `https://mrdata.usgs.gov/mrds/mrds-csv.zip`
- Landsat 8/9 AWS Open Data: `https://registry.opendata.aws/landsat-8/`

## How to Run

```bash
python -m ree_prospecting.main
```

Outputs will be saved under `outputs/` and figures under `figures/`.

Artifacts:
- `outputs/prospectivity_prob.tif`: GeoTIFF of probability (0-1).
- `outputs/hotspots.geojson`: top 10 pixel polygons ranked by score.
- `outputs/interactive_map.html`: Folium map with hotspots and known deposits.
- `outputs/model_report.txt`: precision/recall/F1 and confusion matrix.
- `figures/feature_importance.png`: model feature importance.

## Configuration

Edit `ree_prospecting/config.py` to change:
- Region bounds and UTM EPSG
- Grid resolution
- Model hyperparameters (RF estimators, class weighting, SMOTE use)

## Limitations and Assumptions
- MRDS used as proxy for positive examples; may include non-REE sites even after keyword filtering.
- No rigorous spatial cross-validation in MVP; add spatial blocking for robust metrics.
- Landsat-derived features disabled by default; DEM proxies used (slope, aspect, edge density) and proximity to MRDS.
- IDW geochemistry interpolation included only if points contain a `value` column (not present in MRDS).
- DEM edge density is a crude lineament proxy (edge magnitude), not a true structural interpretation.

## Future Improvements
- Integrate Landsat via STAC or GEE with cloud masking and NDVI.
- Add aeromagnetic and hyperspectral datasets when available.
- Spatial cross-validation with block bootstrap or k-fold.
- Ensemble with simple CNN on image patches around deposits.
- Explainability: SHAP on RF to complement permutation importance.

## Runtime
Target < 2 hours on a laptop for this single region. The current MVP typically runs in tens of minutes depending on network and MRDS density.
