# REE Prospectivity MVP (Mountain Pass, CA)

A minimal, modular prototype to demonstrate feasibility of REE (Rare Earth Element) prospectivity mapping using publicly available geospatial data over a single pilot region (~50x50 km around Mountain Pass, California).

This MVP prioritizes working code and clarity over completeness. It is built to be extended to new regions by editing configuration only.

## Key Features
- Ingestion of Landsat 8/9 (via Microsoft Planetary Computer STAC), USGS MRDS, and SRTM DEM (via `elevation`)
- Feature engineering: iron/clay spectral ratios, NDVI mask, slope, aspect, lineament density
- Baseline model: Random Forest with spatial cross-validation and uncertainty
- Outputs: GeoTIFF prospectivity map, hotspot polygons (GeoJSON), interactive Folium map, feature importance chart, metrics report

## Project Structure
```
ree_prospecting/
├── config.py          # Region bounds, data paths, model hyperparameters
├── data_loader.py     # Download and cache data sources
├── preprocessing.py   # Feature engineering functions
├── model.py           # ML model training and prediction
├── visualization.py   # Map generation and plotting
├── main.py            # Orchestration script
└── README.md          # This file
```

## Installation
Use a fresh Python 3.10+ environment.

Option A: Conda
```bash
conda create -n ree-mvp python=3.10 -y
conda activate ree-mvp
pip install -r requirements.txt
```

Option B: venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `requirements.txt` at repo root (provided) and install. The MVP uses:
- geopandas, rasterio, shapely, numpy, pandas, scipy, scikit-image
- scikit-learn, matplotlib, folium
- pystac-client, planetary-computer (for Landsat access)
- elevation (for SRTM DEM)

## Data Sources
- Landsat 8/9 Collection 2 Level-2 via Microsoft Planetary Computer STAC: `https://planetarycomputer.microsoft.com/api/stac/v1`
- USGS MRDS (Mineral Resources Data System) CSV: `https://mrdata.usgs.gov/mrds/mrds-csv.zip`
- SRTM DEM via `elevation` (SRTM 1 arc-second): `https://pypi.org/project/elevation/`

Note: For MVP we skip hyperspectral and aeromagnetics but the code is modular for future integration.

## How to Run
1. Ensure dependencies are installed and internet access is available.
2. Edit `ree_prospecting/config.py` if you want to change the region or resolution.
3. Run the full pipeline:
```bash
python -m ree_prospecting.main
```

Outputs will be written under `ree_prospecting/outputs/`:
- `prospectivity_prob.tif` — probability (0-1)
- `prospectivity_std.tif` — model uncertainty (std across trees)
- `hotspots.geojson` — top hotspots polygons
- `prospectivity_map.html` — interactive Folium map
- `feature_importance.png`, `metrics.json`, `feature_importance.json`, `confusion_roc.png`

## Validation & Interpretability
- Positives: buffers around MRDS REE points; negatives: random background
- Spatial cross-validation using block-based folds
- Metrics: precision, recall, F1, ROC AUC, hit rate in top 10% scores
- Explainability: permutation importance + RF feature importances

## Limitations & Assumptions
- Single pilot region only; default 90 m resolution to keep runtime reasonable (< 2 hours on a laptop)
- Landsat cloud masking is minimal; low-cloud scenes are selected but residual clouds may remain
- Geochemistry features are optional and disabled by default (no reliable, small-area API); future work could integrate NGDB subsets
- DEM lineament density uses simple edge detection; more advanced structural geology methods are left for future work

## Future Improvements
- Add hyperspectral indices and aeromagnetic textures
- Use physics-based spectral unmixing and mineral mapping
- Incorporate more robust cloud/shadow masking (e.g., QA bits)
- Implement spatial ensembling and uncertainty quantification beyond RF
- Add CNN for 64x64 patches and ensemble with tabular model
