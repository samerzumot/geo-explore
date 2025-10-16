# geo-explore

Welcome to the geo-explore repository! This project is aimed at exploring geographical data and tools.

## REE Prospectivity Mapping MVP

This repo includes a prototype framework for rare earth element (REE) prospectivity mapping over a pilot region (Mountain Pass, CA). It demonstrates end-to-end data ingestion (MRDS, SRTM DEM), feature engineering (terrain attributes, distance to deposits), a baseline Random Forest model with simple spatial block cross-validation, and outputs including a prospectivity GeoTIFF, hotspots GeoJSON, and an interactive Folium map.

### How to run

```bash
# create env (example)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run pipeline
python -m ree_prospecting.main
```

Outputs go to `outputs/` and `figures/`.

### Notes
- Landsat-derived features are optional and not enabled by default in this MVP.
- Spatial CV here uses simple grid-based blocks for a quick sanity check, not a rigorous geostatistical evaluation.
