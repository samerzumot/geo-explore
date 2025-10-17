"""Pydantic models for geological data structures."""

from datetime import datetime
from typing import List, Optional, Union, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from geojson_pydantic import Point, Polygon, LineString
from geojson_pydantic.types import BBox


class Coordinate(BaseModel):
    """Represents a coordinate with multiple format support."""
    
    id: UUID = Field(default_factory=uuid4)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    easting: Optional[float] = Field(None, description="UTM Easting")
    northing: Optional[float] = Field(None, description="UTM Northing")
    utm_zone: Optional[str] = Field(None, description="UTM Zone (e.g., '10N')")
    township: Optional[str] = Field(None, description="Township (US PLSS)")
    range: Optional[str] = Field(None, description="Range (US PLSS)")
    section: Optional[str] = Field(None, description="Section (US PLSS)")
    quarter_section: Optional[str] = Field(None, description="Quarter section")
    coordinate_system: str = Field(default="EPSG:4326", description="CRS identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: Optional[str] = Field(None, description="Original text from document")
    
    @validator('latitude', 'longitude')
    def validate_lat_lon(cls, v):
        """Validate latitude/longitude ranges."""
        if v is not None:
            if not (-90 <= v <= 90) if 'latitude' in str(cls) else not (-180 <= v <= 180):
                raise ValueError('Invalid coordinate range')
        return v


class AssayResult(BaseModel):
    """Represents a single assay result."""
    
    element: str = Field(..., description="Element symbol or name")
    value: float = Field(..., description="Assay value")
    unit: Literal["ppm", "%", "g/t", "oz/t", "ppb", "wt%"] = Field(
        default="ppm", description="Unit of measurement"
    )
    detection_limit: Optional[float] = Field(None, description="Detection limit")
    method: Optional[str] = Field(None, description="Analytical method")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    @validator('value')
    def validate_positive_value(cls, v):
        """Ensure assay values are positive."""
        if v < 0:
            raise ValueError('Assay values must be positive')
        return v


class StructuralMeasurement(BaseModel):
    """Represents structural geological measurements."""
    
    strike: Optional[float] = Field(None, ge=0, le=360, description="Strike in degrees")
    dip: Optional[float] = Field(None, ge=0, le=90, description="Dip in degrees")
    trend: Optional[float] = Field(None, ge=0, le=360, description="Trend in degrees")
    plunge: Optional[float] = Field(None, ge=0, le=90, description="Plunge in degrees")
    measurement_type: Optional[str] = Field(None, description="Type of measurement")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Location(BaseModel):
    """Represents a geological location with spatial data."""
    
    id: UUID = Field(default_factory=uuid4)
    name: Optional[str] = Field(None, description="Location name")
    location_type: Literal[
        "drill_hole", "sample_site", "mine", "claim", "prospect", "outcrop"
    ] = Field(..., description="Type of location")
    geometry: Union[Point, Polygon, LineString] = Field(..., description="GeoJSON geometry")
    crs: str = Field(default="EPSG:4326", description="Coordinate reference system")
    coordinates: List[Coordinate] = Field(default_factory=list)
    elevation: Optional[float] = Field(None, description="Elevation in meters")
    elevation_unit: Literal["m", "ft"] = Field(default="m")
    county: Optional[str] = Field(None, description="County name")
    state_province: Optional[str] = Field(None, description="State or province")
    country: Optional[str] = Field(None, description="Country")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: Optional[str] = Field(None, description="Original text from document")


class Sample(BaseModel):
    """Represents a geological sample with assay data."""
    
    id: str = Field(..., description="Sample identifier")
    location_id: UUID = Field(..., description="Associated location ID")
    sample_type: Literal[
        "core", "chip", "grab", "channel", "soil", "rock", "drill_cuttings"
    ] = Field(..., description="Type of sample")
    depth_from: Optional[float] = Field(None, description="Starting depth")
    depth_to: Optional[float] = Field(None, description="Ending depth")
    depth_unit: Literal["m", "ft"] = Field(default="m", description="Depth unit")
    lithology: Optional[str] = Field(None, description="Rock type description")
    alteration: Optional[str] = Field(None, description="Alteration description")
    mineralization: Optional[str] = Field(None, description="Mineralization description")
    assays: List[AssayResult] = Field(default_factory=list)
    collection_date: Optional[datetime] = Field(None, description="Sample collection date")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: Optional[str] = Field(None, description="Original text from document")
    
    @validator('depth_to')
    def validate_depth_range(cls, v, values):
        """Ensure depth_to is greater than depth_from."""
        if v is not None and 'depth_from' in values and values['depth_from'] is not None:
            if v <= values['depth_from']:
                raise ValueError('depth_to must be greater than depth_from')
        return v


class GeologicalObservation(BaseModel):
    """Represents a geological observation or feature."""
    
    id: UUID = Field(default_factory=uuid4)
    feature_type: Literal[
        "fault", "fold", "contact", "mineralization", "alteration", 
        "vein", "dike", "sill", "unconformity", "bedding"
    ] = Field(..., description="Type of geological feature")
    description: str = Field(..., description="Feature description")
    location_id: Optional[UUID] = Field(None, description="Associated location ID")
    measurements: Optional[StructuralMeasurement] = Field(None)
    rock_types: List[str] = Field(default_factory=list, description="Associated rock types")
    minerals: List[str] = Field(default_factory=list, description="Associated minerals")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: Optional[str] = Field(None, description="Original text from document")