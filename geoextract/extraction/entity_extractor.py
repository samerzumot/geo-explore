"""Entity extraction from geological text using LLM."""

import json
import logging
from typing import Dict, Any, List, Optional
import re
from datetime import datetime

from geoextract.extraction.llm_client import LLMClient
from geoextract.extraction.prompts import PromptManager
from geoextract.extraction.coordinate_parser import CoordinateParser
from geoextract.extraction.validators import DataValidator
from geoextract.schemas.geological import Location, Sample, GeologicalObservation, AssayResult, Coordinate
from geoextract.schemas.document import DocumentMetadata

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts geological entities from text using LLM."""
    
    def __init__(self, llm_client: LLMClient = None):
        """Initialize entity extractor.
        
        Args:
            llm_client: LLM client for extraction
        """
        self.llm_client = llm_client or LLMClient()
        self.prompt_manager = PromptManager()
        self.coordinate_parser = CoordinateParser()
        self.validator = DataValidator()
    
    def extract_from_text(self, text: str, page_number: int = 1) -> Dict[str, Any]:
        """Extract geological entities from text.
        
        Args:
            text: Input text to process
            page_number: Page number for reference
            
        Returns:
            Dictionary with extracted entities
        """
        try:
            # Get system prompt
            system_prompt = self.prompt_manager.get_prompt("system_prompt")
            
            # Extract entities using LLM
            llm_result = self.llm_client.extract_entities_sync(text, system_prompt)
            
            if "error" in llm_result:
                logger.error(f"LLM extraction failed: {llm_result['error']}")
                return self._create_empty_result()
            
            entities = llm_result.get("entities", {})
            
            # Process and validate entities
            processed_entities = self._process_entities(entities, text)
            
            return {
                "locations": processed_entities.get("locations", []),
                "samples": processed_entities.get("samples", []),
                "observations": processed_entities.get("geological_observations", []),
                "metadata": processed_entities.get("document_metadata", {}),
                "confidence": llm_result.get("confidence", 0.0),
                "page_number": page_number,
                "raw_entities": entities
            }
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._create_empty_result()
    
    def _process_entities(self, entities: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        """Process and validate extracted entities.
        
        Args:
            entities: Raw entities from LLM
            source_text: Original text for context
            
        Returns:
            Processed entities
        """
        processed = {
            "locations": [],
            "samples": [],
            "geological_observations": [],
            "document_metadata": {}
        }
        
        # Process locations
        if "locations" in entities:
            for loc_data in entities["locations"]:
                try:
                    location = self._create_location(loc_data, source_text)
                    if location:
                        processed["locations"].append(location)
                except Exception as e:
                    logger.warning(f"Failed to process location: {e}")
        
        # Process samples
        if "samples" in entities:
            for sample_data in entities["samples"]:
                try:
                    sample = self._create_sample(sample_data, source_text)
                    if sample:
                        processed["samples"].append(sample)
                except Exception as e:
                    logger.warning(f"Failed to process sample: {e}")
        
        # Process geological observations
        if "geological_observations" in entities:
            for obs_data in entities["geological_observations"]:
                try:
                    observation = self._create_geological_observation(obs_data, source_text)
                    if observation:
                        processed["geological_observations"].append(observation)
                except Exception as e:
                    logger.warning(f"Failed to process observation: {e}")
        
        # Process metadata
        if "document_metadata" in entities:
            processed["document_metadata"] = self._process_metadata(entities["document_metadata"])
        
        return processed
    
    def _create_location(self, loc_data: Dict[str, Any], source_text: str) -> Optional[Location]:
        """Create Location object from extracted data.
        
        Args:
            loc_data: Location data from LLM
            source_text: Original text for context
            
        Returns:
            Location object or None
        """
        try:
            # Extract coordinates
            coordinates = []
            if "coordinates" in loc_data:
                for coord_data in loc_data["coordinates"]:
                    coord = self.coordinate_parser.parse_coordinate(coord_data)
                    if coord:
                        coordinates.append(coord)
            
            # Create geometry from coordinates
            geometry = self._create_geometry_from_coordinates(coordinates)
            
            # Create location
            location = Location(
                name=loc_data.get("name"),
                location_type=loc_data.get("location_type", "sample_site"),
                geometry=geometry,
                coordinates=coordinates,
                county=loc_data.get("county"),
                state_province=loc_data.get("state_province"),
                country=loc_data.get("country"),
                confidence=loc_data.get("confidence", 0.5),
                source_text=source_text
            )
            
            return location
            
        except Exception as e:
            logger.warning(f"Failed to create location: {e}")
            return None
    
    def _create_sample(self, sample_data: Dict[str, Any], source_text: str) -> Optional[Sample]:
        """Create Sample object from extracted data.
        
        Args:
            sample_data: Sample data from LLM
            source_text: Original text for context
            
        Returns:
            Sample object or None
        """
        try:
            # Process assays
            assays = []
            if "assays" in sample_data:
                for assay_data in sample_data["assays"]:
                    assay = AssayResult(
                        element=assay_data.get("element", ""),
                        value=assay_data.get("value", 0.0),
                        unit=assay_data.get("unit", "ppm"),
                        detection_limit=assay_data.get("detection_limit"),
                        confidence=assay_data.get("confidence", 0.5)
                    )
                    assays.append(assay)
            
            # Parse collection date
            collection_date = None
            if "collection_date" in sample_data:
                try:
                    collection_date = datetime.fromisoformat(sample_data["collection_date"])
                except ValueError:
                    # Try other date formats
                    date_str = sample_data["collection_date"]
                    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                        try:
                            collection_date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
            
            # Create sample
            sample = Sample(
                id=sample_data.get("id", ""),
                location_id=None,  # Will be linked later
                sample_type=sample_data.get("sample_type", "rock"),
                depth_from=sample_data.get("depth_from"),
                depth_to=sample_data.get("depth_to"),
                depth_unit=sample_data.get("depth_unit", "m"),
                lithology=sample_data.get("lithology"),
                alteration=sample_data.get("alteration"),
                mineralization=sample_data.get("mineralization"),
                assays=assays,
                collection_date=collection_date,
                confidence=sample_data.get("confidence", 0.5),
                source_text=source_text
            )
            
            return sample
            
        except Exception as e:
            logger.warning(f"Failed to create sample: {e}")
            return None
    
    def _create_geological_observation(self, obs_data: Dict[str, Any], source_text: str) -> Optional[GeologicalObservation]:
        """Create GeologicalObservation object from extracted data.
        
        Args:
            obs_data: Observation data from LLM
            source_text: Original text for context
            
        Returns:
            GeologicalObservation object or None
        """
        try:
            # Process measurements
            measurements = None
            if "measurements" in obs_data:
                from geoextract.schemas.geological import StructuralMeasurement
                measurements = StructuralMeasurement(
                    strike=obs_data["measurements"].get("strike"),
                    dip=obs_data["measurements"].get("dip"),
                    trend=obs_data["measurements"].get("trend"),
                    plunge=obs_data["measurements"].get("plunge"),
                    confidence=obs_data["measurements"].get("confidence", 0.5)
                )
            
            # Create observation
            observation = GeologicalObservation(
                feature_type=obs_data.get("feature_type", "contact"),
                description=obs_data.get("description", ""),
                measurements=measurements,
                rock_types=obs_data.get("rock_types", []),
                minerals=obs_data.get("minerals", []),
                confidence=obs_data.get("confidence", 0.5),
                source_text=source_text
            )
            
            return observation
            
        except Exception as e:
            logger.warning(f"Failed to create geological observation: {e}")
            return None
    
    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process document metadata.
        
        Args:
            metadata: Raw metadata from LLM
            
        Returns:
            Processed metadata
        """
        processed = {}
        
        # Process date fields
        if "report_date" in metadata:
            try:
                date_str = metadata["report_date"]
                # Try different date formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y"]:
                    try:
                        processed["report_date"] = datetime.strptime(date_str, fmt).isoformat()
                        break
                    except ValueError:
                        continue
            except Exception:
                processed["report_date"] = metadata["report_date"]
        
        # Copy other fields
        for key, value in metadata.items():
            if key != "report_date" and value:
                processed[key] = value
        
        return processed
    
    def _create_geometry_from_coordinates(self, coordinates: List[Coordinate]) -> Dict[str, Any]:
        """Create GeoJSON geometry from coordinates.
        
        Args:
            coordinates: List of Coordinate objects
            
        Returns:
            GeoJSON geometry object
        """
        if not coordinates:
            return {"type": "Point", "coordinates": [0, 0]}
        
        # Use first coordinate for point geometry
        coord = coordinates[0]
        if coord.latitude is not None and coord.longitude is not None:
            return {
                "type": "Point",
                "coordinates": [coord.longitude, coord.latitude]
            }
        elif coord.easting is not None and coord.northing is not None:
            # Convert UTM to lat/lon (simplified)
            # In practice, you'd use pyproj for proper conversion
            return {
                "type": "Point",
                "coordinates": [0, 0]  # Placeholder
            }
        else:
            return {"type": "Point", "coordinates": [0, 0]}
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure.
        
        Returns:
            Empty result dictionary
        """
        return {
            "locations": [],
            "samples": [],
            "observations": [],
            "metadata": {},
            "confidence": 0.0,
            "page_number": 1,
            "raw_entities": {}
        }
    
    def extract_from_ocr_blocks(self, ocr_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract entities from OCR blocks.
        
        Args:
            ocr_blocks: List of OCR text blocks
            
        Returns:
            Dictionary with extracted entities
        """
        all_entities = {
            "locations": [],
            "samples": [],
            "observations": [],
            "metadata": {}
        }
        
        for i, block in enumerate(ocr_blocks):
            text = block.get("text", "")
            if not text.strip():
                continue
            
            # Extract entities from this block
            block_entities = self.extract_from_text(text, i + 1)
            
            # Merge entities
            all_entities["locations"].extend(block_entities["locations"])
            all_entities["samples"].extend(block_entities["samples"])
            all_entities["observations"].extend(block_entities["observations"])
            
            # Merge metadata (take first non-empty)
            if not all_entities["metadata"] and block_entities["metadata"]:
                all_entities["metadata"] = block_entities["metadata"]
        
        return all_entities
    
    def link_samples_to_locations(self, locations: List[Location], samples: List[Sample]) -> List[Sample]:
        """Link samples to their nearest locations.
        
        Args:
            locations: List of Location objects
            samples: List of Sample objects
            
        Returns:
            List of samples with linked location IDs
        """
        linked_samples = []
        
        for sample in samples:
            # Find nearest location based on text similarity or coordinates
            best_location = None
            best_score = 0.0
            
            for location in locations:
                # Simple text similarity (could be improved)
                if sample.source_text and location.source_text:
                    # Check if sample text is near location text
                    if abs(sample.source_text.find(location.name or "") - 
                           location.source_text.find(sample.id or "")) < 100:
                        best_location = location
                        best_score = 0.8
                        break
            
            # Link to best location
            if best_location:
                sample.location_id = best_location.id
            
            linked_samples.append(sample)
        
        return linked_samples