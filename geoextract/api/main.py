"""FastAPI main application."""

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import uuid
import json
from typing import List, Optional
import asyncio

from geoextract.config import settings
from geoextract.api.routes import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GeoExtract API",
    description="Open-Source Geological Report Data Extraction System",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")

# Global job storage (in production, use Redis or database)
job_storage = {}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "GeoExtract API",
        "version": "0.1.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0"
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_storage[job_id]

@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get job result."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Return result file
    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        path=str(result_path),
        filename=result_path.name,
        media_type="application/octet-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "geoextract.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )