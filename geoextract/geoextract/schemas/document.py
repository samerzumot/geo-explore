from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source_file: str
    processing_date: str
    confidence_score: float
    ocr_engine: str
    llm_model: str


class Location(BaseModel):
    id: str
    geometry: dict
    crs: str
    location_type: str
    name: Optional[str] = None
    confidence: float = 0.0


class Assay(BaseModel):
    element: str
    value: float
    unit: str
    detection_limit: Optional[float] = None


class Sample(BaseModel):
    id: str
    location_id: Optional[str] = None
    depth_from: Optional[float] = None
    depth_to: Optional[float] = None
    depth_unit: Optional[str] = None
    assays: List[Assay] = Field(default_factory=list)
    lithology: Optional[str] = None
    confidence: float = 0.0


class GeologicalMeasurements(BaseModel):
    strike: Optional[float] = None
    dip: Optional[float] = None
    trend: Optional[float] = None
    plunge: Optional[float] = None


class GeologicalObservation(BaseModel):
    feature_type: str
    description: Optional[str] = None
    location_id: Optional[str] = None
    measurements: Optional[GeologicalMeasurements] = None


class ExtractionDocument(BaseModel):
    document_metadata: DocumentMetadata
    locations: List[Location] = Field(default_factory=list)
    samples: List[Sample] = Field(default_factory=list)
    geological_observations: List[GeologicalObservation] = Field(default_factory=list)
