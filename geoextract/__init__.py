"""
GeoExtract: Open-Source Geological Report Data Extraction System

A Python-based CLI tool and web interface that uses OCR + LLMs to automatically
extract structured geological data from legacy PDF reports.
"""

__version__ = "0.1.0"
__author__ = "GeoExtract Team"
__email__ = "team@geoextract.org"

from geoextract.config import settings
from geoextract.schemas.document import DocumentMetadata, GeologicalDocument
from geoextract.schemas.geological import Location, Sample, GeologicalObservation

__all__ = [
    "settings",
    "DocumentMetadata",
    "GeologicalDocument", 
    "Location",
    "Sample",
    "GeologicalObservation",
]