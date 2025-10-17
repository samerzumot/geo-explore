"""Prompt management for geological data extraction."""

import json
from typing import Dict, Any, List
from pathlib import Path

from geoextract.config import settings

class PromptManager:
    """Manages prompts for geological data extraction."""
    
    def __init__(self):
        """Initialize prompt manager."""
        self.prompts = self._load_default_prompts()
    
    def _load_default_prompts(self) -> Dict[str, str]:
        """Load default prompts for geological extraction."""
        return {
            "system_prompt": self._get_system_prompt(),
            "coordinate_extraction": self._get_coordinate_prompt(),
            "assay_extraction": self._get_assay_prompt(),
            "drill_hole_extraction": self._get_drill_hole_prompt(),
            "geological_observation": self._get_geological_observation_prompt(),
            "metadata_extraction": self._get_metadata_prompt(),
            "table_extraction": self._get_table_prompt()
        }
    
    def _get_system_prompt(self) -> str:
        """Get the main system prompt for geological extraction."""
        return """You are a specialized AI assistant for extracting structured geological data from OCR text of legacy mining and exploration reports. Your task is to identify and extract specific geological entities with high precision.

Focus on:
1. Spatial coordinates in any format (decimal degrees, DMS, UTM, township-range-section)
2. Mineral assay data (element, value, unit)
3. Drill hole information (ID, depth intervals, lithology)
4. Geological observations (rock types, structures, mineralization)

Rules:
- Only extract information explicitly stated in the text
- Provide confidence scores (0-1) for each extraction
- Flag ambiguous or potentially OCR-corrupted values
- Preserve original units and notation
- When coordinates appear in multiple formats, extract all versions
- For tables, maintain row-column relationships
- Use standard geological terminology and symbols

Output valid JSON matching the provided schema."""
    
    def _get_coordinate_prompt(self) -> str:
        """Get prompt for coordinate extraction."""
        return """Extract all coordinate information from the following text. Look for:

1. Decimal degrees: 45.1234°N, 123.4567°W
2. Degrees Minutes Seconds: 45°07'24.2"N, 123°27'24.1"W
3. UTM coordinates: 500000E, 5000000N (Zone 10N)
4. Township/Range/Section: T2N R3W Section 15
5. State Plane coordinates

For each coordinate found, provide:
- latitude/longitude in decimal degrees
- original format and text
- coordinate system (WGS84, NAD83, etc.)
- confidence score
- location name if mentioned

Output as JSON array of coordinate objects."""
    
    def _get_assay_prompt(self) -> str:
        """Get prompt for assay data extraction."""
        return """Extract all assay data from the following text. Look for:

1. Element symbols: Au, Ag, Cu, Pb, Zn, etc.
2. Values with units: 2.5 g/t, 150 ppm, 0.5%
3. Detection limits: <0.1 g/t, >1000 ppm
4. Sample IDs and depth intervals
5. Analytical methods if mentioned

For each assay found, provide:
- element name and symbol
- value (numeric)
- unit (g/t, ppm, %, oz/t, etc.)
- detection limit if applicable
- sample ID
- depth interval (from-to)
- confidence score

Output as JSON array of assay objects."""
    
    def _get_drill_hole_prompt(self) -> str:
        """Get prompt for drill hole data extraction."""
        return """Extract drill hole information from the following text. Look for:

1. Hole IDs: DH-01, DDH-123, etc.
2. Depth intervals: 0-10m, 15.5-20.2m
3. Drill types: core, RC, rotary, etc.
4. Sample IDs and descriptions
5. Lithology descriptions
6. Alteration descriptions

For each drill hole found, provide:
- hole ID
- depth from/to
- depth unit (m, ft)
- drill type
- sample IDs
- lithology description
- alteration description
- confidence score

Output as JSON array of drill hole objects."""
    
    def _get_geological_observation_prompt(self) -> str:
        """Get prompt for geological observations extraction."""
        return """Extract geological observations from the following text. Look for:

1. Rock types: granite, basalt, sandstone, etc.
2. Structural features: faults, folds, joints, bedding
3. Mineralization: veins, stockwork, disseminated
4. Alteration: silicification, sericitization, etc.
5. Measurements: strike/dip, trend/plunge

For each observation found, provide:
- feature type (fault, fold, contact, mineralization, etc.)
- description
- rock types involved
- minerals present
- structural measurements (strike, dip, trend, plunge)
- confidence score

Output as JSON array of geological observation objects."""
    
    def _get_metadata_prompt(self) -> str:
        """Get prompt for document metadata extraction."""
        return """Extract document metadata from the following text. Look for:

1. Report title
2. Author/company names
3. Report dates
4. Report type (assessment, exploration, resource estimate)
5. Location references (county, state, country)
6. References to other reports

For each metadata item found, provide:
- field name
- value
- confidence score

Output as JSON object with metadata fields."""
    
    def _get_table_prompt(self) -> str:
        """Get prompt for table data extraction."""
        return """Extract tabular data from the following text. Look for:

1. Assay tables with element columns
2. Drill hole tables with depth and sample data
3. Coordinate tables with location data
4. Summary tables with key statistics

For each table found, provide:
- table title/description
- column headers
- row data as arrays
- data types (numeric, text, date)
- confidence score

Output as JSON array of table objects."""
    
    def get_prompt(self, prompt_type: str) -> str:
        """Get a specific prompt by type.
        
        Args:
            prompt_type: Type of prompt to retrieve
            
        Returns:
            Prompt text
        """
        return self.prompts.get(prompt_type, self.prompts["system_prompt"])
    
    def get_extraction_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for extraction output.
        
        Returns:
            JSON schema for geological data extraction
        """
        return {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "location_type": {"type": "string", "enum": ["drill_hole", "sample_site", "mine", "claim", "prospect", "outcrop"]},
                            "coordinates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "latitude": {"type": "number"},
                                        "longitude": {"type": "number"},
                                        "easting": {"type": "number"},
                                        "northing": {"type": "number"},
                                        "utm_zone": {"type": "string"},
                                        "township": {"type": "string"},
                                        "range": {"type": "string"},
                                        "section": {"type": "string"},
                                        "coordinate_system": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                        "source_text": {"type": "string"}
                                    }
                                }
                            },
                            "county": {"type": "string"},
                            "state_province": {"type": "string"},
                            "country": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["location_type", "coordinates"]
                    }
                },
                "samples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "sample_type": {"type": "string", "enum": ["core", "chip", "grab", "channel", "soil", "rock", "drill_cuttings"]},
                            "depth_from": {"type": "number"},
                            "depth_to": {"type": "number"},
                            "depth_unit": {"type": "string", "enum": ["m", "ft"]},
                            "lithology": {"type": "string"},
                            "alteration": {"type": "string"},
                            "mineralization": {"type": "string"},
                            "assays": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "element": {"type": "string"},
                                        "value": {"type": "number"},
                                        "unit": {"type": "string", "enum": ["ppm", "%", "g/t", "oz/t", "ppb", "wt%"]},
                                        "detection_limit": {"type": "number"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    },
                                    "required": ["element", "value", "unit"]
                                }
                            },
                            "collection_date": {"type": "string", "format": "date-time"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["id", "sample_type"]
                    }
                },
                "geological_observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature_type": {"type": "string", "enum": ["fault", "fold", "contact", "mineralization", "alteration", "vein", "dike", "sill", "unconformity", "bedding"]},
                            "description": {"type": "string"},
                            "rock_types": {"type": "array", "items": {"type": "string"}},
                            "minerals": {"type": "array", "items": {"type": "string"}},
                            "measurements": {
                                "type": "object",
                                "properties": {
                                    "strike": {"type": "number"},
                                    "dip": {"type": "number"},
                                    "trend": {"type": "number"},
                                    "plunge": {"type": "number"}
                                }
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["feature_type", "description"]
                    }
                },
                "document_metadata": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "company": {"type": "string"},
                        "report_date": {"type": "string", "format": "date"},
                        "report_type": {"type": "string"},
                        "county": {"type": "string"},
                        "state_province": {"type": "string"},
                        "country": {"type": "string"}
                    }
                }
            },
            "required": ["locations", "samples", "geological_observations"]
        }
    
    def create_custom_prompt(self, prompt_type: str, prompt_text: str) -> None:
        """Create a custom prompt.
        
        Args:
            prompt_type: Type of prompt
            prompt_text: Custom prompt text
        """
        self.prompts[prompt_type] = prompt_text
    
    def save_prompts(self, file_path: Path) -> None:
        """Save prompts to file.
        
        Args:
            file_path: Path to save prompts
        """
        with open(file_path, 'w') as f:
            json.dump(self.prompts, f, indent=2)
    
    def load_prompts(self, file_path: Path) -> None:
        """Load prompts from file.
        
        Args:
            file_path: Path to load prompts from
        """
        with open(file_path, 'r') as f:
            self.prompts = json.load(f)