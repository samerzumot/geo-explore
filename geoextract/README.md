# GeoExtract

Open-source OCR + LLM system to extract structured geological data from legacy reports (1950s–2000s).

## Quick start

```bash
# With Poetry
poetry install -E ocr -E llm -E geospatial
poetry run geoextract --help

# Process a single file
poetry run geoextract process examples/sample_reports/report.pdf --output results/
```

## Features (MVP)
- PDF → images → preprocessing (deskew/denoise) → OCR (PaddleOCR primary, Tesseract fallback)
- LLM-assisted extraction (Ollama Llama 3.1 8B or OpenAI) for coordinates and assay tables
- CSV export, with GeoJSON stub
- Typer CLI and Streamlit UI stubs

See `docs/` for architecture, schema, and roadmap.
