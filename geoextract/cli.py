"""Command-line interface for GeoExtract."""

import logging
import sys
from pathlib import Path
from typing import Optional, List
import asyncio

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from geoextract.config import settings
from geoextract.preprocessing.pdf_handler import PDFHandler
from geoextract.preprocessing.image_clean import ImageCleaner
from geoextract.ocr.ocr_manager import OCRManager
from geoextract.extraction.entity_extractor import EntityExtractor
from geoextract.export.geojson_writer import GeoJSONWriter
from geoextract.export.csv_writer import CSVWriter
from geoextract.schemas.document import GeologicalDocument, DocumentMetadata, ProcessingStats

# Initialize Typer app
app = typer.Typer(
    name="geoextract",
    help="Open-Source Geological Report Data Extraction System",
    add_completion=False
)

# Initialize console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Path to PDF file to process"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
    format: str = typer.Option("geojson", "--format", "-f", help="Output format (geojson, csv, both)"),
    llm_provider: str = typer.Option(None, "--llm-provider", help="LLM provider (ollama, openai, anthropic)"),
    llm_model: str = typer.Option(None, "--llm-model", help="LLM model name"),
    ocr_engine: str = typer.Option(None, "--ocr-engine", help="OCR engine (paddle, tesseract, both)"),
    confidence_threshold: float = typer.Option(None, "--confidence-threshold", help="Minimum confidence threshold"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    save_intermediate: bool = typer.Option(False, "--save-intermediate", help="Save intermediate processing files")
):
    """Process a single PDF file and extract geological data."""
    
    # Update settings if provided
    if llm_provider:
        settings.llm_provider = llm_provider
    if llm_model:
        settings.llm_model = llm_model
    if ocr_engine:
        settings.ocr_engine = ocr_engine
    if confidence_threshold:
        settings.ocr_confidence_threshold = confidence_threshold
    if debug:
        settings.debug = debug
        settings.save_intermediate_outputs = save_intermediate
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if not input_file.suffix.lower() == '.pdf':
        console.print(f"[red]Error: Input file must be a PDF[/red]")
        raise typer.Exit(1)
    
    # Process the file
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: PDF to images
            task1 = progress.add_task("Converting PDF to images...", total=None)
            pdf_handler = PDFHandler()
            images = pdf_handler.extract_images_from_pdf(input_file)
            metadata = pdf_handler.get_pdf_metadata(input_file)
            progress.update(task1, description=f"Converted {len(images)} pages")
            
            # Step 2: Image preprocessing
            task2 = progress.add_task("Preprocessing images...", total=None)
            image_cleaner = ImageCleaner(save_intermediate=settings.save_intermediate_outputs)
            processed_images = image_cleaner.batch_preprocess(images)
            progress.update(task2, description=f"Preprocessed {len(processed_images)} images")
            
            # Step 3: OCR
            task3 = progress.add_task("Running OCR...", total=None)
            ocr_manager = OCRManager()
            ocr_results = ocr_manager.batch_extract(processed_images)
            progress.update(task3, description=f"OCR completed for {len(ocr_results)} pages")
            
            # Step 4: Entity extraction
            task4 = progress.add_task("Extracting geological entities...", total=None)
            entity_extractor = EntityExtractor()
            
            all_entities = {
                "locations": [],
                "samples": [],
                "observations": [],
                "metadata": {}
            }
            
            for i, ocr_result in enumerate(ocr_results):
                if ocr_result.get("blocks"):
                    page_entities = entity_extractor.extract_from_ocr_blocks(ocr_result["blocks"])
                    
                    # Merge entities
                    all_entities["locations"].extend(page_entities["locations"])
                    all_entities["samples"].extend(page_entities["samples"])
                    all_entities["observations"].extend(page_entities["observations"])
                    
                    # Use first non-empty metadata
                    if not all_entities["metadata"] and page_entities["metadata"]:
                        all_entities["metadata"] = page_entities["metadata"]
            
            # Link samples to locations
            all_entities["samples"] = entity_extractor.link_samples_to_locations(
                all_entities["locations"], all_entities["samples"]
            )
            
            progress.update(task4, description=f"Extracted {len(all_entities['locations'])} locations, {len(all_entities['samples'])} samples")
            
            # Step 5: Create document
            task5 = progress.add_task("Creating document...", total=None)
            
            # Create processing stats
            processing_stats = ProcessingStats(
                pages_processed=len(images),
                ocr_confidence_avg=sum(r.get("confidence", 0) for r in ocr_results) / len(ocr_results) if ocr_results else 0,
                extraction_confidence_avg=0.8,  # Placeholder
                processing_time_seconds=0,  # Placeholder
                errors=[],
                warnings=[]
            )
            
            # Create document metadata
            doc_metadata = DocumentMetadata(
                source_file=input_file,
                file_size_bytes=input_file.stat().st_size,
                ocr_engine=settings.ocr_engine,
                llm_model=settings.llm_model,
                language=settings.ocr_language,
                page_count=len(images),
                processing_stats=processing_stats,
                **all_entities["metadata"]
            )
            
            # Create geological document
            document = GeologicalDocument(
                metadata=doc_metadata,
                locations=all_entities["locations"],
                samples=all_entities["samples"],
                observations=all_entities["observations"]
            )
            
            progress.update(task5, description="Document created")
            
            # Step 6: Export
            task6 = progress.add_task("Exporting results...", total=None)
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if format in ["geojson", "both"]:
                geojson_writer = GeoJSONWriter()
                geojson_path = output_dir / f"{input_file.stem}.geojson"
                geojson_writer.write_document(document, geojson_path)
                console.print(f"[green]GeoJSON exported to {geojson_path}[/green]")
            
            if format in ["csv", "both"]:
                csv_writer = CSVWriter()
                csv_dir = output_dir / f"{input_file.stem}_csv"
                csv_writer.write_document(document, csv_dir)
                console.print(f"[green]CSV files exported to {csv_dir}[/green]")
            
            progress.update(task6, description="Export completed")
        
        # Display summary
        console.print("\n[bold green]Processing completed successfully![/bold green]")
        
        # Create summary table
        table = Table(title="Extraction Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Pages processed", str(len(images)))
        table.add_row("Locations found", str(len(all_entities["locations"])))
        table.add_row("Samples found", str(len(all_entities["samples"])))
        table.add_row("Observations found", str(len(all_entities["observations"])))
        table.add_row("OCR Engine", settings.ocr_engine)
        table.add_row("LLM Model", settings.llm_model)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error processing file: {e}[/red]")
        if settings.debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing PDF files"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
    format: str = typer.Option("geojson", "--format", "-f", help="Output format (geojson, csv, both)"),
    llm_provider: str = typer.Option(None, "--llm-provider", help="LLM provider (ollama, openai, anthropic)"),
    llm_model: str = typer.Option(None, "--llm-model", help="LLM model name"),
    ocr_engine: str = typer.Option(None, "--ocr-engine", help="OCR engine (paddle, tesseract, both)"),
    confidence_threshold: float = typer.Option(None, "--confidence-threshold", help="Minimum confidence threshold"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Process multiple PDF files in a directory."""
    
    # Update settings if provided
    if llm_provider:
        settings.llm_provider = llm_provider
    if llm_model:
        settings.llm_model = llm_model
    if ocr_engine:
        settings.ocr_engine = ocr_engine
    if confidence_threshold:
        settings.ocr_confidence_threshold = confidence_threshold
    if debug:
        settings.debug = debug
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory {input_dir} does not exist[/red]")
        raise typer.Exit(1)
    
    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]Error: No PDF files found in {input_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Found {len(pdf_files)} PDF files to process[/blue]")
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        console.print(f"\n[blue]Processing {i}/{len(pdf_files)}: {pdf_file.name}[/blue]")
        
        try:
            # Create output subdirectory for this file
            file_output_dir = output_dir / pdf_file.stem
            
            # Process the file
            process(
                input_file=pdf_file,
                output_dir=file_output_dir,
                format=format,
                llm_provider=settings.llm_provider,
                llm_model=settings.llm_model,
                ocr_engine=settings.ocr_engine,
                confidence_threshold=settings.ocr_confidence_threshold,
                debug=settings.debug,
                save_intermediate=False
            )
            
            successful += 1
            console.print(f"[green]✓ Successfully processed {pdf_file.name}[/green]")
            
        except Exception as e:
            failed += 1
            console.print(f"[red]✗ Failed to process {pdf_file.name}: {e}[/red]")
            if settings.debug:
                console.print_exception()
    
    # Display summary
    console.print(f"\n[bold]Batch processing completed![/bold]")
    console.print(f"[green]Successful: {successful}[/green]")
    console.print(f"[red]Failed: {failed}[/red]")


@app.command()
def config(
    set_param: Optional[str] = typer.Option(None, "--set", help="Set configuration parameter (key=value)"),
    show: bool = typer.Option(False, "--show", help="Show current configuration")
):
    """Manage configuration settings."""
    
    if show:
        # Display current configuration
        table = Table(title="Current Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("LLM Provider", settings.llm_provider)
        table.add_row("LLM Model", settings.llm_model)
        table.add_row("OCR Engine", settings.ocr_engine)
        table.add_row("OCR Language", settings.ocr_language)
        table.add_row("OCR Confidence Threshold", str(settings.ocr_confidence_threshold))
        table.add_row("PDF DPI", str(settings.pdf_dpi))
        table.add_row("Max File Size (MB)", str(settings.max_file_size_mb))
        table.add_row("Output Directory", str(settings.output_dir))
        table.add_row("Debug Mode", str(settings.debug))
        
        console.print(table)
    
    elif set_param:
        # Set configuration parameter
        if "=" not in set_param:
            console.print("[red]Error: Parameter must be in format key=value[/red]")
            raise typer.Exit(1)
        
        key, value = set_param.split("=", 1)
        
        # Map parameter names to settings attributes
        param_map = {
            "llm.provider": "llm_provider",
            "llm.model": "llm_model",
            "ocr.engine": "ocr_engine",
            "ocr.language": "ocr_language",
            "ocr.confidence_threshold": "ocr_confidence_threshold",
            "pdf.dpi": "pdf_dpi",
            "max_file_size_mb": "max_file_size_mb",
            "output_dir": "output_dir",
            "debug": "debug"
        }
        
        if key not in param_map:
            console.print(f"[red]Error: Unknown parameter '{key}'[/red]")
            console.print(f"Available parameters: {', '.join(param_map.keys())}")
            raise typer.Exit(1)
        
        # Set the parameter
        attr_name = param_map[key]
        setattr(settings, attr_name, value)
        
        console.print(f"[green]Set {key} = {value}[/green]")
    
    else:
        console.print("[red]Error: Must specify --set or --show[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Path to extracted data file to validate")
):
    """Validate extracted geological data."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    try:
        # Load and validate the data
        if input_file.suffix.lower() == '.geojson':
            import json
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Basic validation
            if "features" not in data:
                console.print("[red]Error: Invalid GeoJSON format[/red]")
                raise typer.Exit(1)
            
            console.print(f"[green]✓ Valid GeoJSON format[/green]")
            console.print(f"Features: {len(data['features'])}")
            
        else:
            console.print(f"[red]Error: Unsupported file format: {input_file.suffix}[/red]")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]Error validating file: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ui():
    """Launch the web interface."""
    console.print("[blue]Launching GeoExtract web interface...[/blue]")
    
    try:
        import subprocess
        import sys
        
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "geoextract/ui/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    
    except Exception as e:
        console.print(f"[red]Error launching web interface: {e}[/red]")
        console.print("[yellow]Make sure Streamlit is installed: pip install streamlit[/yellow]")
        raise typer.Exit(1)


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the API server on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind the server to")
):
    """Start the API server."""
    console.print(f"[blue]Starting API server on {host}:{port}...[/blue]")
    
    try:
        import uvicorn
        from geoextract.api.main import app as api_app
        
        uvicorn.run(api_app, host=host, port=port)
    
    except Exception as e:
        console.print(f"[red]Error starting API server: {e}[/red]")
        console.print("[yellow]Make sure FastAPI and uvicorn are installed[/yellow]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()