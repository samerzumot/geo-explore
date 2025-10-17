"""Data validation utilities for geological data."""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from geoextract.schemas.geological import Coordinate, AssayResult, Sample, Location

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates geological data for consistency and accuracy."""
    
    def __init__(self):
        """Initialize data validator."""
        self.element_symbols = self._load_element_symbols()
        self.unit_conversions = self._load_unit_conversions()
    
    def _load_element_symbols(self) -> Dict[str, str]:
        """Load periodic table element symbols."""
        return {
            'H': 'Hydrogen', 'He': 'Helium', 'Li': 'Lithium', 'Be': 'Beryllium',
            'B': 'Boron', 'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen',
            'F': 'Fluorine', 'Ne': 'Neon', 'Na': 'Sodium', 'Mg': 'Magnesium',
            'Al': 'Aluminum', 'Si': 'Silicon', 'P': 'Phosphorus', 'S': 'Sulfur',
            'Cl': 'Chlorine', 'Ar': 'Argon', 'K': 'Potassium', 'Ca': 'Calcium',
            'Sc': 'Scandium', 'Ti': 'Titanium', 'V': 'Vanadium', 'Cr': 'Chromium',
            'Mn': 'Manganese', 'Fe': 'Iron', 'Co': 'Cobalt', 'Ni': 'Nickel',
            'Cu': 'Copper', 'Zn': 'Zinc', 'Ga': 'Gallium', 'Ge': 'Germanium',
            'As': 'Arsenic', 'Se': 'Selenium', 'Br': 'Bromine', 'Kr': 'Krypton',
            'Rb': 'Rubidium', 'Sr': 'Strontium', 'Y': 'Yttrium', 'Zr': 'Zirconium',
            'Nb': 'Niobium', 'Mo': 'Molybdenum', 'Tc': 'Technetium', 'Ru': 'Ruthenium',
            'Rh': 'Rhodium', 'Pd': 'Palladium', 'Ag': 'Silver', 'Cd': 'Cadmium',
            'In': 'Indium', 'Sn': 'Tin', 'Sb': 'Antimony', 'Te': 'Tellurium',
            'I': 'Iodine', 'Xe': 'Xenon', 'Cs': 'Cesium', 'Ba': 'Barium',
            'La': 'Lanthanum', 'Ce': 'Cerium', 'Pr': 'Praseodymium', 'Nd': 'Neodymium',
            'Pm': 'Promethium', 'Sm': 'Samarium', 'Eu': 'Europium', 'Gd': 'Gadolinium',
            'Tb': 'Terbium', 'Dy': 'Dysprosium', 'Ho': 'Holmium', 'Er': 'Erbium',
            'Tm': 'Thulium', 'Yb': 'Ytterbium', 'Lu': 'Lutetium', 'Hf': 'Hafnium',
            'Ta': 'Tantalum', 'W': 'Tungsten', 'Re': 'Rhenium', 'Os': 'Osmium',
            'Ir': 'Iridium', 'Pt': 'Platinum', 'Au': 'Gold', 'Hg': 'Mercury',
            'Tl': 'Thallium', 'Pb': 'Lead', 'Bi': 'Bismuth', 'Po': 'Polonium',
            'At': 'Astatine', 'Rn': 'Radon', 'Fr': 'Francium', 'Ra': 'Radium',
            'Ac': 'Actinium', 'Th': 'Thorium', 'Pa': 'Protactinium', 'U': 'Uranium',
            'Np': 'Neptunium', 'Pu': 'Plutonium', 'Am': 'Americium', 'Cm': 'Curium',
            'Bk': 'Berkelium', 'Cf': 'Californium', 'Es': 'Einsteinium', 'Fm': 'Fermium',
            'Md': 'Mendelevium', 'No': 'Nobelium', 'Lr': 'Lawrencium', 'Rf': 'Rutherfordium',
            'Db': 'Dubnium', 'Sg': 'Seaborgium', 'Bh': 'Bohrium', 'Hs': 'Hassium',
            'Mt': 'Meitnerium', 'Ds': 'Darmstadtium', 'Rg': 'Roentgenium', 'Cn': 'Copernicium',
            'Nh': 'Nihonium', 'Fl': 'Flerovium', 'Mc': 'Moscovium', 'Lv': 'Livermorium',
            'Ts': 'Tennessine', 'Og': 'Oganesson'
        }
    
    def _load_unit_conversions(self) -> Dict[str, float]:
        """Load unit conversion factors to base units."""
        return {
            # Mass per mass ratios
            'ppm': 1e-6,      # parts per million
            'ppb': 1e-9,      # parts per billion
            'ppt': 1e-12,     # parts per trillion
            '%': 1e-2,        # percent
            'wt%': 1e-2,      # weight percent
            
            # Mass per volume ratios
            'g/t': 1e-6,      # grams per tonne
            'mg/t': 1e-9,     # milligrams per tonne
            'oz/t': 3.11035e-5,  # ounces per tonne
            'lb/t': 1.10231e-6,  # pounds per tonne
            
            # Volume per volume ratios
            'vol%': 1e-2,     # volume percent
        }
    
    def validate_coordinate(self, coord: Coordinate) -> Tuple[bool, List[str]]:
        """Validate coordinate data.
        
        Args:
            coord: Coordinate object to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check latitude range
        if coord.latitude is not None:
            if not (-90 <= coord.latitude <= 90):
                errors.append(f"Invalid latitude: {coord.latitude}")
        
        # Check longitude range
        if coord.longitude is not None:
            if not (-180 <= coord.longitude <= 180):
                errors.append(f"Invalid longitude: {coord.longitude}")
        
        # Check UTM coordinates
        if coord.easting is not None:
            if not (0 <= coord.easting <= 1000000):
                errors.append(f"Invalid UTM easting: {coord.easting}")
        
        if coord.northing is not None:
            if not (0 <= coord.northing <= 10000000):
                errors.append(f"Invalid UTM northing: {coord.northing}")
        
        # Check confidence score
        if not (0 <= coord.confidence <= 1):
            errors.append(f"Invalid confidence score: {coord.confidence}")
        
        return len(errors) == 0, errors
    
    def validate_assay_result(self, assay: AssayResult) -> Tuple[bool, List[str]]:
        """Validate assay result data.
        
        Args:
            assay: AssayResult object to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check element symbol
        if not assay.element:
            errors.append("Element symbol is required")
        elif assay.element not in self.element_symbols:
            # Check if it's a valid element name
            element_name = self.element_symbols.get(assay.element.upper())
            if not element_name:
                errors.append(f"Unknown element: {assay.element}")
        
        # Check value range
        if assay.value < 0:
            errors.append(f"Negative assay value: {assay.value}")
        
        # Check for unrealistic values (could be OCR errors)
        if assay.value > 1000000:  # 1 million ppm
            errors.append(f"Unrealistically high assay value: {assay.value}")
        
        # Check unit validity
        if assay.unit not in self.unit_conversions:
            errors.append(f"Unknown unit: {assay.unit}")
        
        # Check detection limit
        if assay.detection_limit is not None:
            if assay.detection_limit < 0:
                errors.append(f"Negative detection limit: {assay.detection_limit}")
            if assay.detection_limit > assay.value:
                errors.append(f"Detection limit greater than value: {assay.detection_limit} > {assay.value}")
        
        # Check confidence score
        if not (0 <= assay.confidence <= 1):
            errors.append(f"Invalid confidence score: {assay.confidence}")
        
        return len(errors) == 0, errors
    
    def validate_sample(self, sample: Sample) -> Tuple[bool, List[str]]:
        """Validate sample data.
        
        Args:
            sample: Sample object to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if not sample.id:
            errors.append("Sample ID is required")
        
        # Check depth range
        if sample.depth_from is not None and sample.depth_to is not None:
            if sample.depth_from >= sample.depth_to:
                errors.append(f"Invalid depth range: {sample.depth_from} >= {sample.depth_to}")
        
        # Check depth values
        if sample.depth_from is not None and sample.depth_from < 0:
            errors.append(f"Negative depth_from: {sample.depth_from}")
        
        if sample.depth_to is not None and sample.depth_to < 0:
            errors.append(f"Negative depth_to: {sample.depth_to}")
        
        # Validate assays
        for i, assay in enumerate(sample.assays):
            is_valid, assay_errors = self.validate_assay_result(assay)
            if not is_valid:
                errors.extend([f"Assay {i}: {error}" for error in assay_errors])
        
        # Check confidence score
        if not (0 <= sample.confidence <= 1):
            errors.append(f"Invalid confidence score: {sample.confidence}")
        
        return len(errors) == 0, errors
    
    def validate_location(self, location: Location) -> Tuple[bool, List[str]]:
        """Validate location data.
        
        Args:
            location: Location object to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if not location.location_type:
            errors.append("Location type is required")
        
        # Validate coordinates
        for i, coord in enumerate(location.coordinates):
            is_valid, coord_errors = self.validate_coordinate(coord)
            if not is_valid:
                errors.extend([f"Coordinate {i}: {error}" for error in coord_errors])
        
        # Check elevation
        if location.elevation is not None:
            if location.elevation < -1000 or location.elevation > 10000:
                errors.append(f"Unrealistic elevation: {location.elevation}")
        
        # Check confidence score
        if not (0 <= location.confidence <= 1):
            errors.append(f"Invalid confidence score: {location.confidence}")
        
        return len(errors) == 0, errors
    
    def detect_ocr_errors(self, text: str) -> List[str]:
        """Detect potential OCR errors in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of potential OCR errors
        """
        errors = []
        
        # Common OCR character substitutions
        ocr_substitutions = {
            'O': '0',  # Letter O to number 0
            'l': '1',  # Lowercase l to number 1
            'I': '1',  # Uppercase I to number 1
            'S': '5',  # Letter S to number 5
            'B': '8',  # Letter B to number 8
            'G': '6',  # Letter G to number 6
        }
        
        # Check for suspicious patterns
        for char, replacement in ocr_substitutions.items():
            if char in text and replacement in text:
                # Check if they appear in similar contexts
                pattern = f'[{char}{replacement}]'
                if re.search(pattern, text):
                    errors.append(f"Possible OCR error: '{char}' vs '{replacement}'")
        
        # Check for unrealistic numbers
        numbers = re.findall(r'\d+\.?\d*', text)
        for num_str in numbers:
            try:
                num = float(num_str)
                if num > 1000000:  # Very large numbers might be OCR errors
                    errors.append(f"Unrealistically large number: {num}")
            except ValueError:
                continue
        
        return errors
    
    def validate_geological_terms(self, text: str) -> List[str]:
        """Validate geological terminology.
        
        Args:
            text: Text to validate
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Common geological terms
        geological_terms = [
            'granite', 'basalt', 'sandstone', 'limestone', 'shale', 'schist',
            'gneiss', 'marble', 'quartzite', 'fault', 'fold', 'joint',
            'bedding', 'cleavage', 'foliation', 'vein', 'stockwork', 'disseminated',
            'silicification', 'sericitization', 'chloritization', 'propylitization'
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in geological_terms if term in text_lower]
        
        if not found_terms:
            warnings.append("No geological terminology detected")
        
        return warnings
    
    def cross_validate_data(self, locations: List[Location], samples: List[Sample]) -> List[str]:
        """Cross-validate data between locations and samples.
        
        Args:
            locations: List of Location objects
            samples: List of Sample objects
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check for samples without locations
        samples_without_locations = [s for s in samples if s.location_id is None]
        if samples_without_locations:
            warnings.append(f"{len(samples_without_locations)} samples without location references")
        
        # Check for locations without samples
        location_ids = {loc.id for loc in locations}
        samples_with_locations = [s for s in samples if s.location_id in location_ids]
        if len(samples_with_locations) < len(samples):
            warnings.append("Some samples reference non-existent locations")
        
        # Check for coordinate consistency
        for location in locations:
            if len(location.coordinates) > 1:
                # Check if coordinates are consistent
                lats = [c.latitude for c in location.coordinates if c.latitude is not None]
                lons = [c.longitude for c in location.coordinates if c.longitude is not None]
                
                if lats and (max(lats) - min(lats)) > 0.1:  # More than 0.1 degrees difference
                    warnings.append(f"Location {location.name} has inconsistent latitude values")
                
                if lons and (max(lons) - min(lons)) > 0.1:
                    warnings.append(f"Location {location.name} has inconsistent longitude values")
        
        return warnings
    
    def get_validation_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get validation summary for all data.
        
        Args:
            data: Dictionary with locations, samples, and observations
            
        Returns:
            Validation summary
        """
        summary = {
            "total_errors": 0,
            "total_warnings": 0,
            "validation_results": {}
        }
        
        # Validate locations
        locations = data.get("locations", [])
        location_errors = 0
        for location in locations:
            is_valid, errors = self.validate_location(location)
            if not is_valid:
                location_errors += len(errors)
        
        summary["validation_results"]["locations"] = {
            "count": len(locations),
            "errors": location_errors,
            "valid": location_errors == 0
        }
        summary["total_errors"] += location_errors
        
        # Validate samples
        samples = data.get("samples", [])
        sample_errors = 0
        for sample in samples:
            is_valid, errors = self.validate_sample(sample)
            if not is_valid:
                sample_errors += len(errors)
        
        summary["validation_results"]["samples"] = {
            "count": len(samples),
            "errors": sample_errors,
            "valid": sample_errors == 0
        }
        summary["total_errors"] += sample_errors
        
        # Cross-validation
        cross_warnings = self.cross_validate_data(locations, samples)
        summary["total_warnings"] += len(cross_warnings)
        summary["validation_results"]["cross_validation"] = {
            "warnings": cross_warnings
        }
        
        return summary