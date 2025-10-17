"""Data schemas for geological document processing."""

from geoextract.schemas.document import DocumentMetadata, GeologicalDocument
from geoextract.schemas.geological import (
    Location,
    Sample,
    GeologicalObservation,
    AssayResult,
    Coordinate,
    StructuralMeasurement,
)

__all__ = [
    "DocumentMetadata",
    "GeologicalDocument",
    "Location",
    "Sample", 
    "GeologicalObservation",
    "AssayResult",
    "Coordinate",
    "StructuralMeasurement",
]