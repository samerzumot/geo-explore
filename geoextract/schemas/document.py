"""Pydantic models for document metadata and processing results."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator

from geoextract.schemas.geological import Location, Sample, GeologicalObservation


class ProcessingStats(BaseModel):
    """Statistics about document processing."""
    
    pages_processed: int = Field(default=0)
    ocr_confidence_avg: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_confidence_avg: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_seconds: float = Field(default=0.0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Metadata about the source document and processing."""
    
    source_file: Path = Field(..., description="Path to source document")
    file_size_bytes: int = Field(..., description="File size in bytes")
    processing_date: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    ocr_engine: str = Field(..., description="OCR engine used")
    llm_model: str = Field(..., description="LLM model used")
    language: str = Field(default="en", description="Document language")
    page_count: int = Field(default=1, description="Number of pages")
    document_type: Optional[str] = Field(None, description="Type of geological document")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    company: Optional[str] = Field(None, description="Company or organization")
    report_date: Optional[datetime] = Field(None, description="Report date")
    report_type: Optional[str] = Field(None, description="Type of report")
    processing_stats: ProcessingStats = Field(default_factory=ProcessingStats)
    
    @validator('file_size_bytes')
    def validate_file_size(cls, v):
        """Ensure file size is positive."""
        if v <= 0:
            raise ValueError('File size must be positive')
        return v


class GeologicalDocument(BaseModel):
    """Complete geological document with extracted data."""
    
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    locations: List[Location] = Field(default_factory=list)
    samples: List[Sample] = Field(default_factory=list)
    observations: List[GeologicalObservation] = Field(default_factory=list)
    raw_ocr_text: Optional[str] = Field(None, description="Raw OCR text")
    extraction_notes: List[str] = Field(default_factory=list)
    
    def get_locations_by_type(self, location_type: str) -> List[Location]:
        """Get locations filtered by type."""
        return [loc for loc in self.locations if loc.location_type == location_type]
    
    def get_samples_by_location(self, location_id: UUID) -> List[Sample]:
        """Get samples associated with a specific location."""
        return [samp for samp in self.samples if samp.location_id == location_id]
    
    def get_observations_by_type(self, feature_type: str) -> List[GeologicalObservation]:
        """Get observations filtered by feature type."""
        return [obs for obs in self.observations if obs.feature_type == feature_type]
    
    def get_all_coordinates(self) -> List[Location]:
        """Get all locations that have coordinate data."""
        return [loc for loc in self.locations if loc.coordinates]
    
    def get_assay_data(self) -> List[Sample]:
        """Get all samples that have assay data."""
        return [samp for samp in self.samples if samp.assays]
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON format."""
        features = []
        
        for location in self.locations:
            feature = {
                "type": "Feature",
                "geometry": location.geometry.dict(),
                "properties": {
                    "id": str(location.id),
                    "name": location.name,
                    "location_type": location.location_type,
                    "confidence": location.confidence,
                    "county": location.county,
                    "state_province": location.state_province,
                    "country": location.country,
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
            "features": features,
            "metadata": {
                "source_file": str(self.metadata.source_file),
                "processing_date": self.metadata.processing_date.isoformat(),
                "confidence_score": self.metadata.confidence_score,
                "total_locations": len(self.locations),
                "total_samples": len(self.samples),
                "total_observations": len(self.observations),
            }
        }