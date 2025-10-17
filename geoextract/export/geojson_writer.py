"""GeoJSON export functionality."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from geoextract.schemas.document import GeologicalDocument
from geoextract.schemas.geological import Location, Sample, GeologicalObservation

logger = logging.getLogger(__name__)


class GeoJSONWriter:
    """Writes geological data to GeoJSON format."""
    
    def __init__(self, include_metadata: bool = True):
        """Initialize GeoJSON writer.
        
        Args:
            include_metadata: Whether to include metadata in output
        """
        self.include_metadata = include_metadata
    
    def write_document(self, document: GeologicalDocument, output_path: Path) -> None:
        """Write geological document to GeoJSON file.
        
        Args:
            document: Geological document to export
            output_path: Path to output file
        """
        try:
            geojson_data = self._create_geojson(document)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"GeoJSON exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write GeoJSON: {e}")
            raise
    
    def _create_geojson(self, document: GeologicalDocument) -> Dict[str, Any]:
        """Create GeoJSON structure from document.
        
        Args:
            document: Geological document
            
        Returns:
            GeoJSON dictionary
        """
        features = []
        
        # Add location features
        for location in document.locations:
            feature = self._create_location_feature(location, document)
            if feature:
                features.append(feature)
        
        # Add sample features (as points at locations)
        for sample in document.samples:
            feature = self._create_sample_feature(sample, document)
            if feature:
                features.append(feature)
        
        # Create FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::4326"
                }
            },
            "features": features
        }
        
        # Add metadata if requested
        if self.include_metadata:
            geojson["metadata"] = self._create_metadata(document)
        
        return geojson
    
    def _create_location_feature(self, location: Location, document: GeologicalDocument) -> Optional[Dict[str, Any]]:
        """Create GeoJSON feature for location.
        
        Args:
            location: Location object
            document: Parent document
            
        Returns:
            GeoJSON feature or None
        """
        try:
            # Get geometry
            geometry = location.geometry
            
            # Create properties
            properties = {
                "id": str(location.id),
                "name": location.name,
                "location_type": location.location_type,
                "confidence": location.confidence,
                "county": location.county,
                "state_province": location.state_province,
                "country": location.country,
                "elevation": location.elevation,
                "elevation_unit": location.elevation_unit
            }
            
            # Add coordinate information
            if location.coordinates:
                properties["coordinates"] = []
                for coord in location.coordinates:
                    coord_info = {
                        "latitude": coord.latitude,
                        "longitude": coord.longitude,
                        "easting": coord.easting,
                        "northing": coord.northing,
                        "utm_zone": coord.utm_zone,
                        "township": coord.township,
                        "range": coord.range,
                        "section": coord.section,
                        "coordinate_system": coord.coordinate_system,
                        "confidence": coord.confidence
                    }
                    # Remove None values
                    coord_info = {k: v for k, v in coord_info.items() if v is not None}
                    properties["coordinates"].append(coord_info)
            
            # Add samples associated with this location
            samples = [s for s in document.samples if s.location_id == location.id]
            if samples:
                properties["samples"] = []
                for sample in samples:
                    sample_info = {
                        "id": sample.id,
                        "sample_type": sample.sample_type,
                        "depth_from": sample.depth_from,
                        "depth_to": sample.depth_to,
                        "depth_unit": sample.depth_unit,
                        "lithology": sample.lithology,
                        "alteration": sample.alteration,
                        "mineralization": sample.mineralization,
                        "collection_date": sample.collection_date.isoformat() if sample.collection_date else None,
                        "confidence": sample.confidence
                    }
                    # Remove None values
                    sample_info = {k: v for k, v in sample_info.items() if v is not None}
                    properties["samples"].append(sample_info)
            
            return {
                "type": "Feature",
                "geometry": geometry.dict() if hasattr(geometry, 'dict') else geometry,
                "properties": properties
            }
            
        except Exception as e:
            logger.warning(f"Failed to create location feature: {e}")
            return None
    
    def _create_sample_feature(self, sample: Sample, document: GeologicalDocument) -> Optional[Dict[str, Any]]:
        """Create GeoJSON feature for sample.
        
        Args:
            sample: Sample object
            document: Parent document
            
        Returns:
            GeoJSON feature or None
        """
        try:
            # Find associated location
            location = None
            if sample.location_id:
                location = next((l for l in document.locations if l.id == sample.location_id), None)
            
            if not location:
                return None
            
            # Use location geometry
            geometry = location.geometry
            
            # Create properties
            properties = {
                "id": sample.id,
                "sample_type": sample.sample_type,
                "location_id": str(sample.location_id),
                "location_name": location.name,
                "depth_from": sample.depth_from,
                "depth_to": sample.depth_to,
                "depth_unit": sample.depth_unit,
                "lithology": sample.lithology,
                "alteration": sample.alteration,
                "mineralization": sample.mineralization,
                "collection_date": sample.collection_date.isoformat() if sample.collection_date else None,
                "confidence": sample.confidence
            }
            
            # Add assay data
            if sample.assays:
                properties["assays"] = []
                for assay in sample.assays:
                    assay_info = {
                        "element": assay.element,
                        "value": assay.value,
                        "unit": assay.unit,
                        "detection_limit": assay.detection_limit,
                        "method": assay.method,
                        "confidence": assay.confidence
                    }
                    # Remove None values
                    assay_info = {k: v for k, v in assay_info.items() if v is not None}
                    properties["assays"].append(assay_info)
            
            return {
                "type": "Feature",
                "geometry": geometry.dict() if hasattr(geometry, 'dict') else geometry,
                "properties": properties
            }
            
        except Exception as e:
            logger.warning(f"Failed to create sample feature: {e}")
            return None
    
    def _create_metadata(self, document: GeologicalDocument) -> Dict[str, Any]:
        """Create metadata for GeoJSON.
        
        Args:
            document: Geological document
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "source_file": str(document.metadata.source_file),
            "processing_date": document.metadata.processing_date.isoformat(),
            "confidence_score": document.metadata.confidence_score,
            "ocr_engine": document.metadata.ocr_engine,
            "llm_model": document.metadata.llm_model,
            "language": document.metadata.language,
            "page_count": document.metadata.page_count,
            "document_type": document.metadata.document_type,
            "title": document.metadata.title,
            "author": document.metadata.author,
            "company": document.metadata.company,
            "report_date": document.metadata.report_date.isoformat() if document.metadata.report_date else None,
            "report_type": document.metadata.report_type,
            "total_locations": len(document.locations),
            "total_samples": len(document.samples),
            "total_observations": len(document.observations),
            "processing_stats": {
                "pages_processed": document.metadata.processing_stats.pages_processed,
                "ocr_confidence_avg": document.metadata.processing_stats.ocr_confidence_avg,
                "extraction_confidence_avg": document.metadata.processing_stats.extraction_confidence_avg,
                "processing_time_seconds": document.metadata.processing_stats.processing_time_seconds,
                "errors": document.metadata.processing_stats.errors,
                "warnings": document.metadata.processing_stats.warnings
            }
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return metadata
    
    def write_locations_only(self, locations: List[Location], output_path: Path) -> None:
        """Write only locations to GeoJSON file.
        
        Args:
            locations: List of Location objects
            output_path: Path to output file
        """
        try:
            features = []
            
            for location in locations:
                feature = self._create_location_feature(location, None)
                if feature:
                    features.append(feature)
            
            geojson = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": "urn:ogc:def:crs:EPSG::4326"
                    }
                },
                "features": features
            }
            
            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Locations GeoJSON exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write locations GeoJSON: {e}")
            raise
    
    def write_samples_only(self, samples: List[Sample], locations: List[Location], output_path: Path) -> None:
        """Write only samples to GeoJSON file.
        
        Args:
            samples: List of Sample objects
            locations: List of Location objects for geometry
            output_path: Path to output file
        """
        try:
            features = []
            
            for sample in samples:
                feature = self._create_sample_feature(sample, None)
                if feature:
                    features.append(feature)
            
            geojson = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": "urn:ogc:def:crs:EPSG::4326"
                    }
                },
                "features": features
            }
            
            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Samples GeoJSON exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write samples GeoJSON: {e}")
            raise