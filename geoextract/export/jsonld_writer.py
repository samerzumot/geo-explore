"""JSON-LD export functionality with geological vocabulary."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from geoextract.schemas.document import GeologicalDocument
from geoextract.schemas.geological import Location, Sample, GeologicalObservation

logger = logging.getLogger(__name__)


class JSONLDWriter:
    """Writes geological data to JSON-LD format with schema.org vocabulary."""
    
    def __init__(self, base_url: str = "https://geoextract.org"):
        """Initialize JSON-LD writer.
        
        Args:
            base_url: Base URL for the application
        """
        self.base_url = base_url
        self.context = self._get_context()
    
    def _get_context(self) -> Dict[str, Any]:
        """Get JSON-LD context with geological vocabulary."""
        return {
            "@context": {
                "@vocab": "https://schema.org/",
                "geo": "https://www.w3.org/2003/01/geo/wgs84_pos#",
                "geological": "https://geoextract.org/vocab#",
                "mining": "https://geoextract.org/mining#",
                "coordinates": {
                    "@id": "geo:coordinates",
                    "@type": "geo:Point"
                },
                "latitude": {
                    "@id": "geo:lat",
                    "@type": "xsd:double"
                },
                "longitude": {
                    "@id": "geo:long",
                    "@type": "xsd:double"
                },
                "elevation": {
                    "@id": "geo:alt",
                    "@type": "xsd:double"
                },
                "location_type": {
                    "@id": "geological:locationType",
                    "@type": "geological:LocationType"
                },
                "sample_type": {
                    "@id": "geological:sampleType",
                    "@type": "geological:SampleType"
                },
                "feature_type": {
                    "@id": "geological:featureType",
                    "@type": "geological:FeatureType"
                },
                "assay_result": {
                    "@id": "geological:assayResult",
                    "@type": "geological:AssayResult"
                },
                "element": {
                    "@id": "geological:element",
                    "@type": "geological:ChemicalElement"
                },
                "confidence": {
                    "@id": "geological:confidence",
                    "@type": "xsd:double"
                }
            }
        }
    
    def write_document(self, document: GeologicalDocument, output_path: Path) -> None:
        """Write geological document to JSON-LD file.
        
        Args:
            document: Geological document to export
            output_path: Path to output file
        """
        try:
            # Create JSON-LD structure
            jsonld_data = self._create_jsonld(document)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(jsonld_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON-LD exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write JSON-LD: {e}")
            raise
    
    def _create_jsonld(self, document: GeologicalDocument) -> Dict[str, Any]:
        """Create JSON-LD structure from document.
        
        Args:
            document: Geological document
            
        Returns:
            JSON-LD dictionary
        """
        # Create main document
        jsonld = {
            **self.context,
            "@id": f"{self.base_url}/document/{document.metadata.source_file.stem}",
            "@type": "geological:GeologicalDocument",
            "name": document.metadata.title or f"Geological Report: {document.metadata.source_file.name}",
            "description": f"Geological data extracted from {document.metadata.source_file.name}",
            "dateCreated": document.metadata.processing_date.isoformat(),
            "author": {
                "@type": "Person",
                "name": document.metadata.author or "Unknown"
            },
            "publisher": {
                "@type": "Organization",
                "name": document.metadata.company or "GeoExtract"
            },
            "about": self._create_locations_jsonld(document.locations),
            "hasPart": self._create_samples_jsonld(document.samples),
            "mentions": self._create_observations_jsonld(document.observations),
            "additionalProperty": self._create_metadata_jsonld(document.metadata)
        }
        
        return jsonld
    
    def _create_locations_jsonld(self, locations: List[Location]) -> List[Dict[str, Any]]:
        """Create JSON-LD for locations.
        
        Args:
            locations: List of Location objects
            
        Returns:
            List of location JSON-LD objects
        """
        location_objects = []
        
        for location in locations:
            # Get primary coordinates
            primary_coord = location.coordinates[0] if location.coordinates else None
            
            location_obj = {
                "@id": f"{self.base_url}/location/{location.id}",
                "@type": "geological:GeologicalLocation",
                "name": location.name or f"Location {location.id}",
                "description": f"Geological {location.location_type}",
                "geo": {
                    "@type": "GeoCoordinates",
                    "latitude": primary_coord.latitude if primary_coord else None,
                    "longitude": primary_coord.longitude if primary_coord else None,
                    "elevation": location.elevation
                },
                "address": {
                    "@type": "PostalAddress",
                    "addressLocality": location.county or "",
                    "addressRegion": location.state_province or "",
                    "addressCountry": location.country or ""
                },
                "additionalProperty": [
                    {
                        "@type": "PropertyValue",
                        "name": "location_type",
                        "value": location.location_type
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "confidence",
                        "value": location.confidence
                    }
                ]
            }
            
            # Add coordinate information
            if primary_coord:
                coord_props = []
                if primary_coord.easting:
                    coord_props.append({
                        "@type": "PropertyValue",
                        "name": "easting",
                        "value": primary_coord.easting
                    })
                if primary_coord.northing:
                    coord_props.append({
                        "@type": "PropertyValue",
                        "name": "northing",
                        "value": primary_coord.northing
                    })
                if primary_coord.utm_zone:
                    coord_props.append({
                        "@type": "PropertyValue",
                        "name": "utm_zone",
                        "value": primary_coord.utm_zone
                    })
                if primary_coord.township:
                    coord_props.append({
                        "@type": "PropertyValue",
                        "name": "township",
                        "value": primary_coord.township
                    })
                if primary_coord.range:
                    coord_props.append({
                        "@type": "PropertyValue",
                        "name": "range",
                        "value": primary_coord.range
                    })
                if primary_coord.section:
                    coord_props.append({
                        "@type": "PropertyValue",
                        "name": "section",
                        "value": primary_coord.section
                    })
                
                location_obj["additionalProperty"].extend(coord_props)
            
            # Remove None values
            location_obj = self._remove_none_values(location_obj)
            location_objects.append(location_obj)
        
        return location_objects
    
    def _create_samples_jsonld(self, samples: List[Sample]) -> List[Dict[str, Any]]:
        """Create JSON-LD for samples.
        
        Args:
            samples: List of Sample objects
            
        Returns:
            List of sample JSON-LD objects
        """
        sample_objects = []
        
        for sample in samples:
            sample_obj = {
                "@id": f"{self.base_url}/sample/{sample.id}",
                "@type": "geological:GeologicalSample",
                "name": f"Sample {sample.id}",
                "description": f"{sample.sample_type} sample",
                "additionalProperty": [
                    {
                        "@type": "PropertyValue",
                        "name": "sample_type",
                        "value": sample.sample_type
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "confidence",
                        "value": sample.confidence
                    }
                ]
            }
            
            # Add depth information
            if sample.depth_from is not None:
                sample_obj["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "depth_from",
                    "value": sample.depth_from
                })
            if sample.depth_to is not None:
                sample_obj["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "depth_to",
                    "value": sample.depth_to
                })
            if sample.depth_unit:
                sample_obj["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "depth_unit",
                    "value": sample.depth_unit
                })
            
            # Add lithology
            if sample.lithology:
                sample_obj["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "lithology",
                    "value": sample.lithology
                })
            
            # Add alteration
            if sample.alteration:
                sample_obj["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "alteration",
                    "value": sample.alteration
                })
            
            # Add mineralization
            if sample.mineralization:
                sample_obj["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "mineralization",
                    "value": sample.mineralization
                })
            
            # Add collection date
            if sample.collection_date:
                sample_obj["dateCreated"] = sample.collection_date.isoformat()
            
            # Add assays
            if sample.assays:
                assay_objects = []
                for assay in sample.assays:
                    assay_obj = {
                        "@type": "geological:AssayResult",
                        "element": {
                            "@type": "geological:ChemicalElement",
                            "name": assay.element
                        },
                        "value": assay.value,
                        "unit": assay.unit,
                        "confidence": assay.confidence
                    }
                    
                    if assay.detection_limit:
                        assay_obj["detectionLimit"] = assay.detection_limit
                    
                    if assay.method:
                        assay_obj["method"] = assay.method
                    
                    assay_objects.append(assay_obj)
                
                sample_obj["hasPart"] = assay_objects
            
            # Link to location
            if sample.location_id:
                sample_obj["isLocatedAt"] = {
                    "@id": f"{self.base_url}/location/{sample.location_id}"
                }
            
            # Remove None values
            sample_obj = self._remove_none_values(sample_obj)
            sample_objects.append(sample_obj)
        
        return sample_objects
    
    def _create_observations_jsonld(self, observations: List[GeologicalObservation]) -> List[Dict[str, Any]]:
        """Create JSON-LD for observations.
        
        Args:
            observations: List of GeologicalObservation objects
            
        Returns:
            List of observation JSON-LD objects
        """
        observation_objects = []
        
        for obs in observations:
            observation_obj = {
                "@id": f"{self.base_url}/observation/{obs.id}",
                "@type": "geological:GeologicalObservation",
                "name": f"{obs.feature_type} observation",
                "description": obs.description,
                "additionalProperty": [
                    {
                        "@type": "PropertyValue",
                        "name": "feature_type",
                        "value": obs.feature_type
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "confidence",
                        "value": obs.confidence
                    }
                ]
            }
            
            # Add rock types
            if obs.rock_types:
                observation_obj["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "rock_types",
                    "value": "; ".join(obs.rock_types)
                })
            
            # Add minerals
            if obs.minerals:
                observation_obj["additionalProperty"].append({
                    "@type": "PropertyValue",
                    "name": "minerals",
                    "value": "; ".join(obs.minerals)
                })
            
            # Add measurements
            if obs.measurements:
                measurements = []
                if obs.measurements.strike is not None:
                    measurements.append({
                        "@type": "PropertyValue",
                        "name": "strike",
                        "value": obs.measurements.strike
                    })
                if obs.measurements.dip is not None:
                    measurements.append({
                        "@type": "PropertyValue",
                        "name": "dip",
                        "value": obs.measurements.dip
                    })
                if obs.measurements.trend is not None:
                    measurements.append({
                        "@type": "PropertyValue",
                        "name": "trend",
                        "value": obs.measurements.trend
                    })
                if obs.measurements.plunge is not None:
                    measurements.append({
                        "@type": "PropertyValue",
                        "name": "plunge",
                        "value": obs.measurements.plunge
                    })
                
                observation_obj["additionalProperty"].extend(measurements)
            
            # Link to location
            if obs.location_id:
                observation_obj["isLocatedAt"] = {
                    "@id": f"{self.base_url}/location/{obs.location_id}"
                }
            
            # Remove None values
            observation_obj = self._remove_none_values(observation_obj)
            observation_objects.append(observation_obj)
        
        return observation_objects
    
    def _create_metadata_jsonld(self, metadata) -> List[Dict[str, Any]]:
        """Create JSON-LD for metadata.
        
        Args:
            metadata: DocumentMetadata object
            
        Returns:
            List of metadata JSON-LD objects
        """
        metadata_objects = [
            {
                "@type": "PropertyValue",
                "name": "source_file",
                "value": str(metadata.source_file)
            },
            {
                "@type": "PropertyValue",
                "name": "file_size_bytes",
                "value": metadata.file_size_bytes
            },
            {
                "@type": "PropertyValue",
                "name": "processing_date",
                "value": metadata.processing_date.isoformat()
            },
            {
                "@type": "PropertyValue",
                "name": "confidence_score",
                "value": metadata.confidence_score
            },
            {
                "@type": "PropertyValue",
                "name": "ocr_engine",
                "value": metadata.ocr_engine
            },
            {
                "@type": "PropertyValue",
                "name": "llm_model",
                "value": metadata.llm_model
            },
            {
                "@type": "PropertyValue",
                "name": "language",
                "value": metadata.language
            },
            {
                "@type": "PropertyValue",
                "name": "page_count",
                "value": metadata.page_count
            }
        ]
        
        # Add optional metadata
        if metadata.document_type:
            metadata_objects.append({
                "@type": "PropertyValue",
                "name": "document_type",
                "value": metadata.document_type
            })
        
        if metadata.report_type:
            metadata_objects.append({
                "@type": "PropertyValue",
                "name": "report_type",
                "value": metadata.report_type
            })
        
        return metadata_objects
    
    def _remove_none_values(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from dictionary recursively.
        
        Args:
            obj: Dictionary to clean
            
        Returns:
            Cleaned dictionary
        """
        if isinstance(obj, dict):
            return {k: self._remove_none_values(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [self._remove_none_values(item) for item in obj if item is not None]
        else:
            return obj