# GeoExtract: Open-Source Geological Report Data Extraction System

GeoExtract is a Python-based CLI tool and web interface that uses OCR + LLMs to automatically extract structured geological data from legacy PDF reports (1950s-2000s). The tool handles scanned documents, typed reports, and mixed formats, outputting standardized geospatial data formats.

## Features

- **Multi-format Input**: PDFs, scanned images (TIF, JPG, PNG)
- **Advanced OCR**: PaddleOCR + Tesseract with layout preservation
- **LLM Extraction**: Local Ollama or OpenAI API integration
- **Geological Focus**: Specialized for mining/exploration reports
- **Multiple Outputs**: GeoJSON, CSV, GeoPackage, JSON-LD
- **Web Interface**: Streamlit-based UI for easy document processing
- **API Server**: FastAPI with async job processing
- **Batch Processing**: Handle directories of documents

## Quick Start

### Installation

```bash
# Install with Poetry
git clone https://github.com/your-org/geoextract.git
cd geoextract
poetry install

# Or install with pip
pip install geoextract
```

### Basic Usage

```bash
# Process a single PDF
geoextract process report.pdf --output results/

# Batch process directory
geoextract batch input_dir/ --output results/ --format geojson,csv

# Start web interface
geoextract ui

# Start API server
geoextract serve --port 8000
```

## Supported Data Types

- **Location Data**: Coordinates (decimal degrees, DMS, UTM, township-range-section)
- **Geological Data**: Rock types, mineral occurrences, assay results
- **Drilling Data**: Hole IDs, depth intervals, sample data
- **Temporal Data**: Report dates, survey dates
- **Metadata**: Report titles, authors, references

## Output Formats

- **GeoJSON**: For immediate mapping with proper CRS metadata
- **CSV**: Tabular data for analysis
- **GeoPackage**: For GIS integration
- **JSON-LD**: With schema.org geological vocabulary

## Configuration

```bash
# Set LLM provider
geoextract config --set llm.provider=ollama
geoextract config --set llm.model=llama3.1:8b

# Configure OCR engine
geoextract config --set ocr.engine=paddle
geoextract config --set ocr.confidence_threshold=0.8
```

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run tests
pytest

# Format code
black geoextract/
isort geoextract/

# Type checking
mypy geoextract/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- GitHub Issues: [Report bugs and request features](https://github.com/your-org/geoextract/issues)
- Documentation: [Full documentation](https://geoextract.readthedocs.io)
- Community: [Join our discussions](https://github.com/your-org/geoextract/discussions)