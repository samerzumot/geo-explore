from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.progress import Progress

from .config import settings

app = typer.Typer(add_completion=False, help="GeoExtract CLI")


@app.command()
def process(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="PDF or image file"),
    output: Path = typer.Option(Path("results"), file_okay=False, help="Output directory"),
    llm_provider: Optional[str] = typer.Option(None, help="ollama|openai"),
    llm_model: Optional[str] = typer.Option(None, help="e.g., llama3.1:8b"),
    ocr_engine: Optional[str] = typer.Option(None, help="paddle|tesseract|both"),
    dpi: Optional[int] = typer.Option(None, help="PDF to image DPI"),
    confidence_threshold: Optional[float] = typer.Option(None, help="0-1 min confidence"),
    debug: bool = typer.Option(False, help="Save intermediates"),
):
    """Process a single file through the MVP pipeline and export CSV."""
    # Apply CLI overrides to settings
    if llm_provider:
        settings.llm_provider = llm_provider
    if llm_model:
        settings.llm_model = llm_model
    if ocr_engine:
        settings.ocr_engine = ocr_engine
    if dpi is not None:
        settings.pdf_dpi = dpi
    if confidence_threshold is not None:
        settings.confidence_threshold = confidence_threshold
    if debug:
        settings.debug = True

    output.mkdir(parents=True, exist_ok=True)

    from .preprocessing.pdf_handler import convert_to_images
    from .preprocessing.image_clean import preprocess_image
    from .ocr.ocr_manager import OCRManager
    from .extraction.entity_extractor import extract_entities_mvp
    from .export.csv_writer import write_csv

    with Progress() as progress:
        task = progress.add_task("Processing", total=5)

        # 1) Convert to images
        images = convert_to_images(input_path, dpi=settings.pdf_dpi)
        progress.advance(task)

        # 2) Preprocess images
        preprocessed = [preprocess_image(img) for img in images]
        progress.advance(task)

        # 3) OCR
        ocr = OCRManager(engine_preference=settings.ocr_engine, language=settings.ocr_language)
        ocr_blocks = ocr.run(preprocessed)
        progress.advance(task)

        # 4) LLM extraction (coords + assays)
        result = extract_entities_mvp(ocr_blocks, confidence_threshold=settings.confidence_threshold)
        progress.advance(task)

        # 5) Export CSV
        csv_path = output / f"{input_path.stem}_extractions.csv"
        write_csv(result, csv_path)
        progress.advance(task)

    print(f"[green]Done. CSV written to[/green] {csv_path}")


@app.command()
def ui():
    """Launch Streamlit UI."""
    import subprocess

    subprocess.run(["streamlit", "run", "geoextract/ui/streamlit_app.py"])  # nosec


@app.command()
def serve(port: int = 8000):
    """Start FastAPI server."""
    import subprocess

    subprocess.run(["uvicorn", "geoextract.api.main:app", "--host", "0.0.0.0", "--port", str(port), "--reload"])  # nosec


if __name__ == "__main__":
    app()
