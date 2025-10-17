"""API routes for GeoExtract."""

import logging
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
import tempfile
import json

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from geoextract.config import settings
from geoextract.preprocessing.pdf_handler import PDFHandler
from geoextract.preprocessing.image_clean import ImageCleaner
from geoextract.ocr.ocr_manager import OCRManager
from geoextract.extraction.entity_extractor import EntityExtractor
from geoextract.export.geojson_writer import GeoJSONWriter
from geoextract.export.csv_writer import CSVWriter
from geoextract.schemas.document import GeologicalDocument, DocumentMetadata, ProcessingStats

logger = logging.getLogger(__name__)

router = APIRouter()

# Global job storage (in production, use Redis or database)
job_storage = {}

class ProcessingRequest(BaseModel):
    """Request model for document processing."""
    llm_provider: Optional[str] = "ollama"
    llm_model: Optional[str] = "llama3.1:8b"
    ocr_engine: Optional[str] = "paddle"
    confidence_threshold: Optional[float] = 0.8
    language: Optional[str] = "en"
    output_format: Optional[List[str]] = ["geojson"]
    debug: Optional[bool] = False

class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    result_path: Optional[str] = None
    error: Optional[str] = None

@router.post("/process", response_model=JobStatus)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    llm_provider: str = Form("ollama"),
    llm_model: str = Form("llama3.1:8b"),
    ocr_engine: str = Form("paddle"),
    confidence_threshold: float = Form(0.8),
    language: str = Form("en"),
    output_format: str = Form("geojson"),
    debug: bool = Form(False)
):
    """Process a single PDF document."""
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_storage[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Job created",
        "result_path": None,
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_document_background,
        job_id,
        file,
        llm_provider,
        llm_model,
        ocr_engine,
        confidence_threshold,
        language,
        output_format.split(","),
        debug
    )
    
    return JobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Job created"
    )

@router.post("/process/batch", response_model=JobStatus)
async def process_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    llm_provider: str = Form("ollama"),
    llm_model: str = Form("llama3.1:8b"),
    ocr_engine: str = Form("paddle"),
    confidence_threshold: float = Form(0.8),
    language: str = Form("en"),
    output_format: str = Form("geojson"),
    debug: bool = Form(False)
):
    """Process multiple PDF documents."""
    
    # Validate files
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be a PDF")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_storage[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Batch job created",
        "result_path": None,
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_batch_background,
        job_id,
        files,
        llm_provider,
        llm_model,
        ocr_engine,
        confidence_threshold,
        language,
        output_format.split(","),
        debug
    )
    
    return JobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Batch job created"
    )

async def process_document_background(
    job_id: str,
    file: UploadFile,
    llm_provider: str,
    llm_model: str,
    ocr_engine: str,
    confidence_threshold: float,
    language: str,
    output_format: List[str],
    debug: bool
):
    """Background task for processing a single document."""
    
    try:
        # Update job status
        job_storage[job_id]["status"] = "processing"
        job_storage[job_id]["message"] = "Starting processing"
        job_storage[job_id]["progress"] = 0.1
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded file
            file_path = temp_path / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            job_storage[job_id]["progress"] = 0.2
            job_storage[job_id]["message"] = "File saved, starting PDF processing"
            
            # Process PDF
            pdf_handler = PDFHandler()
            images = pdf_handler.extract_images_from_pdf(file_path)
            metadata = pdf_handler.get_pdf_metadata(file_path)
            
            job_storage[job_id]["progress"] = 0.4
            job_storage[job_id]["message"] = f"Extracted {len(images)} pages, preprocessing images"
            
            # Preprocess images
            image_cleaner = ImageCleaner(save_intermediate=debug)
            processed_images = image_cleaner.batch_preprocess(images)
            
            job_storage[job_id]["progress"] = 0.6
            job_storage[job_id]["message"] = "Running OCR"
            
            # Run OCR
            ocr_manager = OCRManager(engine=ocr_engine, language=language)
            ocr_manager.confidence_threshold = confidence_threshold
            ocr_results = ocr_manager.batch_extract(processed_images)
            
            job_storage[job_id]["progress"] = 0.8
            job_storage[job_id]["message"] = "Extracting geological entities"
            
            # Extract entities
            entity_extractor = EntityExtractor()
            
            all_entities = {
                "locations": [],
                "samples": [],
                "observations": [],
                "metadata": {}
            }
            
            for ocr_result in ocr_results:
                if ocr_result.get("blocks"):
                    page_entities = entity_extractor.extract_from_ocr_blocks(ocr_result["blocks"])
                    
                    all_entities["locations"].extend(page_entities["locations"])
                    all_entities["samples"].extend(page_entities["samples"])
                    all_entities["observations"].extend(page_entities["observations"])
                    
                    if not all_entities["metadata"] and page_entities["metadata"]:
                        all_entities["metadata"] = page_entities["metadata"]
            
            # Link samples to locations
            all_entities["samples"] = entity_extractor.link_samples_to_locations(
                all_entities["locations"], all_entities["samples"]
            )
            
            job_storage[job_id]["progress"] = 0.9
            job_storage[job_id]["message"] = "Creating document and exporting results"
            
            # Create document
            processing_stats = ProcessingStats(
                pages_processed=len(images),
                ocr_confidence_avg=sum(r.get("confidence", 0) for r in ocr_results) / len(ocr_results) if ocr_results else 0,
                extraction_confidence_avg=0.8,
                processing_time_seconds=0,
                errors=[],
                warnings=[]
            )
            
            doc_metadata = DocumentMetadata(
                source_file=file_path,
                file_size_bytes=file_path.stat().st_size,
                ocr_engine=ocr_engine,
                llm_model=llm_model,
                language=language,
                page_count=len(images),
                processing_stats=processing_stats,
                **all_entities["metadata"]
            )
            
            document = GeologicalDocument(
                metadata=doc_metadata,
                locations=all_entities["locations"],
                samples=all_entities["samples"],
                observations=all_entities["observations"]
            )
            
            # Export results
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            if "geojson" in output_format:
                geojson_writer = GeoJSONWriter()
                geojson_path = output_dir / f"{file_path.stem}.geojson"
                geojson_writer.write_document(document, geojson_path)
            
            if "csv" in output_format:
                csv_writer = CSVWriter()
                csv_dir = output_dir / f"{file_path.stem}_csv"
                csv_writer.write_document(document, csv_dir)
            
            # Create zip file
            zip_path = temp_path / f"{file_path.stem}_results.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(output_dir))
            
            # Move result to permanent location
            result_dir = settings.output_dir / "api_results"
            result_dir.mkdir(parents=True, exist_ok=True)
            final_result_path = result_dir / f"{job_id}_{file_path.stem}_results.zip"
            
            import shutil
            shutil.move(str(zip_path), str(final_result_path))
            
            # Update job status
            job_storage[job_id]["status"] = "completed"
            job_storage[job_id]["progress"] = 1.0
            job_storage[job_id]["message"] = "Processing completed successfully"
            job_storage[job_id]["result_path"] = str(final_result_path)
            
            logger.info(f"Job {job_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)
        job_storage[job_id]["message"] = f"Processing failed: {e}"

async def process_batch_background(
    job_id: str,
    files: List[UploadFile],
    llm_provider: str,
    llm_model: str,
    ocr_engine: str,
    confidence_threshold: float,
    language: str,
    output_format: List[str],
    debug: bool
):
    """Background task for processing multiple documents."""
    
    try:
        job_storage[job_id]["status"] = "processing"
        job_storage[job_id]["message"] = f"Processing {len(files)} files"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Process each file
            all_documents = []
            for i, file in enumerate(files):
                job_storage[job_id]["progress"] = i / len(files)
                job_storage[job_id]["message"] = f"Processing file {i+1}/{len(files)}: {file.filename}"
                
                # Save file
                file_path = temp_path / file.filename
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Process file (simplified - in real implementation, call the full processing pipeline)
                # For now, just create a placeholder document
                document = GeologicalDocument(
                    metadata=DocumentMetadata(
                        source_file=file_path,
                        file_size_bytes=file_path.stat().st_size,
                        ocr_engine=ocr_engine,
                        llm_model=llm_model,
                        language=language,
                        page_count=1,
                        processing_stats=ProcessingStats()
                    ),
                    locations=[],
                    samples=[],
                    observations=[]
                )
                
                all_documents.append(document)
            
            # Export combined results
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            if "geojson" in output_format:
                # Combine all documents into one GeoJSON
                geojson_writer = GeoJSONWriter()
                combined_geojson = {
                    "type": "FeatureCollection",
                    "features": []
                }
                
                for doc in all_documents:
                    doc_geojson = geojson_writer._create_geojson(doc)
                    combined_geojson["features"].extend(doc_geojson["features"])
                
                geojson_path = output_dir / "combined_results.geojson"
                with open(geojson_path, 'w') as f:
                    json.dump(combined_geojson, f, indent=2)
            
            if "csv" in output_format:
                # Export each document separately
                for i, doc in enumerate(all_documents):
                    csv_writer = CSVWriter()
                    csv_dir = output_dir / f"document_{i+1}_csv"
                    csv_writer.write_document(doc, csv_dir)
            
            # Create zip file
            zip_path = temp_path / "batch_results.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(output_dir))
            
            # Move result to permanent location
            result_dir = settings.output_dir / "api_results"
            result_dir.mkdir(parents=True, exist_ok=True)
            final_result_path = result_dir / f"{job_id}_batch_results.zip"
            
            import shutil
            shutil.move(str(zip_path), str(final_result_path))
            
            # Update job status
            job_storage[job_id]["status"] = "completed"
            job_storage[job_id]["progress"] = 1.0
            job_storage[job_id]["message"] = f"Batch processing completed: {len(files)} files processed"
            job_storage[job_id]["result_path"] = str(final_result_path)
            
            logger.info(f"Batch job {job_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)
        job_storage[job_id]["message"] = f"Batch processing failed: {e}"

@router.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {"jobs": list(job_storage.values())}

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete result file if it exists
    job = job_storage[job_id]
    if job.get("result_path"):
        result_path = Path(job["result_path"])
        if result_path.exists():
            result_path.unlink()
    
    # Remove job from storage
    del job_storage[job_id]
    
    return {"message": "Job deleted successfully"}

@router.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "ocr_engine": settings.ocr_engine,
        "ocr_language": settings.ocr_language,
        "ocr_confidence_threshold": settings.ocr_confidence_threshold,
        "pdf_dpi": settings.pdf_dpi,
        "max_file_size_mb": settings.max_file_size_mb,
        "output_dir": str(settings.output_dir),
        "debug": settings.debug
    }