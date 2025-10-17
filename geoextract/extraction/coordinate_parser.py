"""Coordinate parsing and conversion utilities."""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
import math

from geoextract.schemas.geological import Coordinate

logger = logging.getLogger(__name__)


class CoordinateParser:
    """Parses and converts coordinates from various formats."""
    
    def __init__(self):
        """Initialize coordinate parser."""
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for coordinate detection."""
        return {
            # Decimal degrees with direction
            "decimal_degrees": re.compile(
                r'(\d+\.?\d*)\s*[°\s]*([NSEW])\s*,?\s*(\d+\.?\d*)\s*[°\s]*([NSEW])',
                re.IGNORECASE
            ),
            # Degrees Minutes Seconds
            "dms": re.compile(
                r'(\d+)°\s*(\d+)\'\s*(\d+\.?\d*)"\s*([NSEW])\s*,?\s*(\d+)°\s*(\d+)\'\s*(\d+\.?\d*)"\s*([NSEW])',
                re.IGNORECASE
            ),
            # UTM coordinates
            "utm": re.compile(
                r'(\d{6,7})\s*[Ee]\s*,?\s*(\d{7,8})\s*[Nn]',
                re.IGNORECASE
            ),
            # Township/Range/Section
            "township_range": re.compile(
                r'T(\d+)\s*[Nn]\s*R(\d+)\s*[EW]?\s*,?\s*Section\s*(\d+)',
                re.IGNORECASE
            ),
            # Simple decimal degrees (no direction)
            "decimal_simple": re.compile(
                r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
            ),
            # State Plane coordinates
            "state_plane": re.compile(
                r'(\d+\.?\d*)\s*[Ee]\s*,?\s*(\d+\.?\d*)\s*[Nn]',
                re.IGNORECASE
            )
        }
    
    def parse_coordinate(self, coord_data: Dict[str, Any]) -> Optional[Coordinate]:
        """Parse coordinate from various formats.
        
        Args:
            coord_data: Coordinate data from LLM or text
            
        Returns:
            Coordinate object or None
        """
        try:
            # If already parsed, return as-is
            if isinstance(coord_data, Coordinate):
                return coord_data
            
            # Extract from text if available
            source_text = coord_data.get("source_text", "")
            if source_text:
                parsed = self.parse_from_text(source_text)
                if parsed:
                    return parsed
            
            # Create from provided data
            return Coordinate(
                latitude=coord_data.get("latitude"),
                longitude=coord_data.get("longitude"),
                easting=coord_data.get("easting"),
                northing=coord_data.get("northing"),
                utm_zone=coord_data.get("utm_zone"),
                township=coord_data.get("township"),
                range=coord_data.get("range"),
                section=coord_data.get("section"),
                quarter_section=coord_data.get("quarter_section"),
                coordinate_system=coord_data.get("coordinate_system", "EPSG:4326"),
                confidence=coord_data.get("confidence", 0.5),
                source_text=source_text
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse coordinate: {e}")
            return None
    
    def parse_from_text(self, text: str) -> Optional[Coordinate]:
        """Parse coordinate from text string.
        
        Args:
            text: Text containing coordinate
            
        Returns:
            Coordinate object or None
        """
        text = text.strip()
        
        # Try different patterns
        for pattern_name, pattern in self.patterns.items():
            match = pattern.search(text)
            if match:
                try:
                    if pattern_name == "decimal_degrees":
                        return self._parse_decimal_degrees(match, text)
                    elif pattern_name == "dms":
                        return self._parse_dms(match, text)
                    elif pattern_name == "utm":
                        return self._parse_utm(match, text)
                    elif pattern_name == "township_range":
                        return self._parse_township_range(match, text)
                    elif pattern_name == "decimal_simple":
                        return self._parse_decimal_simple(match, text)
                    elif pattern_name == "state_plane":
                        return self._parse_state_plane(match, text)
                except Exception as e:
                    logger.warning(f"Failed to parse {pattern_name}: {e}")
                    continue
        
        return None
    
    def _parse_decimal_degrees(self, match: re.Match, text: str) -> Coordinate:
        """Parse decimal degrees format.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Coordinate object
        """
        groups = match.groups()
        lat_val = float(groups[0])
        lat_dir = groups[1].upper()
        lon_val = float(groups[2])
        lon_dir = groups[3].upper()
        
        # Convert to decimal degrees
        latitude = lat_val if lat_dir == 'N' else -lat_val
        longitude = lon_val if lon_dir == 'E' else -lon_val
        
        return Coordinate(
            latitude=latitude,
            longitude=longitude,
            coordinate_system="EPSG:4326",
            confidence=0.9,
            source_text=text
        )
    
    def _parse_dms(self, match: re.Match, text: str) -> Coordinate:
        """Parse degrees minutes seconds format.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Coordinate object
        """
        groups = match.groups()
        lat_d = int(groups[0])
        lat_m = int(groups[1])
        lat_s = float(groups[2])
        lat_dir = groups[3].upper()
        
        lon_d = int(groups[4])
        lon_m = int(groups[5])
        lon_s = float(groups[6])
        lon_dir = groups[7].upper()
        
        # Convert to decimal degrees
        latitude = lat_d + lat_m/60 + lat_s/3600
        longitude = lon_d + lon_m/60 + lon_s/3600
        
        if lat_dir == 'S':
            latitude = -latitude
        if lon_dir == 'W':
            longitude = -longitude
        
        return Coordinate(
            latitude=latitude,
            longitude=longitude,
            coordinate_system="EPSG:4326",
            confidence=0.9,
            source_text=text
        )
    
    def _parse_utm(self, match: re.Match, text: str) -> Coordinate:
        """Parse UTM coordinates.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Coordinate object
        """
        groups = match.groups()
        easting = float(groups[0])
        northing = float(groups[1])
        
        # Extract UTM zone if present in text
        utm_zone = self._extract_utm_zone(text)
        
        return Coordinate(
            easting=easting,
            northing=northing,
            utm_zone=utm_zone,
            coordinate_system="EPSG:32600",  # Default UTM
            confidence=0.8,
            source_text=text
        )
    
    def _parse_township_range(self, match: re.Match, text: str) -> Coordinate:
        """Parse Township/Range/Section format.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Coordinate object
        """
        groups = match.groups()
        township = groups[0]
        range_val = groups[1]
        section = groups[2]
        
        return Coordinate(
            township=township,
            range=range_val,
            section=section,
            coordinate_system="US_PLSS",
            confidence=0.8,
            source_text=text
        )
    
    def _parse_decimal_simple(self, match: re.Match, text: str) -> Coordinate:
        """Parse simple decimal degrees format.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Coordinate object
        """
        groups = match.groups()
        lat = float(groups[0])
        lon = float(groups[1])
        
        return Coordinate(
            latitude=lat,
            longitude=lon,
            coordinate_system="EPSG:4326",
            confidence=0.7,
            source_text=text
        )
    
    def _parse_state_plane(self, match: re.Match, text: str) -> Coordinate:
        """Parse State Plane coordinates.
        
        Args:
            match: Regex match object
            text: Original text
            
        Returns:
            Coordinate object
        """
        groups = match.groups()
        easting = float(groups[0])
        northing = float(groups[1])
        
        return Coordinate(
            easting=easting,
            northing=northing,
            coordinate_system="State_Plane",
            confidence=0.7,
            source_text=text
        )
    
    def _extract_utm_zone(self, text: str) -> Optional[str]:
        """Extract UTM zone from text.
        
        Args:
            text: Text to search
            
        Returns:
            UTM zone string or None
        """
        zone_pattern = re.compile(r'Zone\s*(\d+[NS])', re.IGNORECASE)
        match = zone_pattern.search(text)
        if match:
            return match.group(1)
        return None
    
    def convert_utm_to_latlon(self, easting: float, northing: float, utm_zone: str) -> Tuple[float, float]:
        """Convert UTM coordinates to latitude/longitude.
        
        Args:
            easting: UTM easting
            northing: UTM northing
            utm_zone: UTM zone (e.g., '10N')
            
        Returns:
            Tuple of (latitude, longitude)
        """
        try:
            import pyproj
            
            # Parse UTM zone
            zone_num = int(utm_zone[:-1])
            hemisphere = utm_zone[-1].upper()
            
            # Create UTM projection
            utm_proj = pyproj.Proj(proj='utm', zone=zone_num, ellps='WGS84', 
                                 north=(hemisphere == 'N'))
            
            # Convert to lat/lon
            lon, lat = utm_proj(easting, northing, inverse=True)
            
            return lat, lon
            
        except ImportError:
            logger.warning("pyproj not available for UTM conversion")
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"UTM conversion failed: {e}")
            return 0.0, 0.0
    
    def convert_dms_to_decimal(self, degrees: int, minutes: int, seconds: float, direction: str) -> float:
        """Convert degrees minutes seconds to decimal degrees.
        
        Args:
            degrees: Degrees
            minutes: Minutes
            seconds: Seconds
            direction: Direction (N, S, E, W)
            
        Returns:
            Decimal degrees
        """
        decimal = degrees + minutes/60 + seconds/3600
        
        if direction.upper() in ['S', 'W']:
            decimal = -decimal
        
        return decimal
    
    def validate_coordinate(self, coord: Coordinate) -> bool:
        """Validate coordinate values.
        
        Args:
            coord: Coordinate object to validate
            
        Returns:
            True if valid
        """
        # Validate latitude
        if coord.latitude is not None:
            if not (-90 <= coord.latitude <= 90):
                return False
        
        # Validate longitude
        if coord.longitude is not None:
            if not (-180 <= coord.longitude <= 180):
                return False
        
        # Validate UTM coordinates
        if coord.easting is not None:
            if not (0 <= coord.easting <= 1000000):
                return False
        
        if coord.northing is not None:
            if not (0 <= coord.northing <= 10000000):
                return False
        
        return True
    
    def find_coordinates_in_text(self, text: str) -> List[Coordinate]:
        """Find all coordinates in text.
        
        Args:
            text: Text to search
            
        Returns:
            List of Coordinate objects
        """
        coordinates = []
        
        # Split text into sentences for better parsing
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            coord = self.parse_from_text(sentence)
            if coord:
                coordinates.append(coord)
        
        return coordinates