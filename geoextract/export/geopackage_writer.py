"""GeoPackage export functionality."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import pandas as pd

from geoextract.schemas.document import GeologicalDocument
from geoextract.schemas.geological import Location, Sample, GeologicalObservation

logger = logging.getLogger(__name__)


class GeoPackageWriter:
    """Writes geological data to GeoPackage format."""
    
    def __init__(self, include_metadata: bool = True):
        """Initialize GeoPackage writer.
        
        Args:
            include_metadata: Whether to include metadata in output
        """
        self.include_metadata = include_metadata
    
    def write_document(self, document: GeologicalDocument, output_path: Path) -> None:
        """Write geological document to GeoPackage file.
        
        Args:
            document: Geological document to export
            output_path: Path to output file
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create GeoDataFrames
            locations_gdf = self._create_locations_gdf(document.locations)
            samples_gdf = self._create_samples_gdf(document.samples, document.locations)
            observations_gdf = self._create_observations_gdf(document.observations, document.locations)
            
            # Write to GeoPackage
            with gpd.io.file.fiona.Env():
                # Write locations
                if not locations_gdf.empty:
                    locations_gdf.to_file(output_path, layer='locations', driver='GPKG')
                
                # Write samples
                if not samples_gdf.empty:
                    samples_gdf.to_file(output_path, layer='samples', driver='GPKG')
                
                # Write observations
                if not observations_gdf.empty:
                    observations_gdf.to_file(output_path, layer='observations', driver='GPKG')
                
                # Write metadata if requested
                if self.include_metadata:
                    metadata_df = self._create_metadata_df(document.metadata)
                    metadata_df.to_file(output_path, layer='metadata', driver='GPKG')
            
            logger.info(f"GeoPackage exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write GeoPackage: {e}")
            raise
    
    def _create_locations_gdf(self, locations: List[Location]) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from locations.
        
        Args:
            locations: List of Location objects
            
        Returns:
            GeoDataFrame with locations
        """
        if not locations:
            return gpd.GeoDataFrame()
        
        data = []
        geometries = []
        
        for location in locations:
            # Create geometry
            if hasattr(location.geometry, 'dict'):
                geom_dict = location.geometry.dict()
            else:
                geom_dict = location.geometry
            
            if geom_dict["type"] == "Point":
                geom = Point(geom_dict["coordinates"])
            elif geom_dict["type"] == "Polygon":
                geom = Polygon(geom_dict["coordinates"][0])
            elif geom_dict["type"] == "LineString":
                geom = LineString(geom_dict["coordinates"])
            else:
                geom = Point(0, 0)  # Default
            
            geometries.append(geom)
            
            # Create data row
            row = {
                'id': str(location.id),
                'name': location.name or '',
                'location_type': location.location_type,
                'county': location.county or '',
                'state_province': location.state_province or '',
                'country': location.country or '',
                'elevation': location.elevation,
                'elevation_unit': location.elevation_unit,
                'confidence': location.confidence,
                'source_text': location.source_text or ''
            }
            
            # Add coordinate information from first coordinate
            if location.coordinates:
                coord = location.coordinates[0]
                row.update({
                    'latitude': coord.latitude,
                    'longitude': coord.longitude,
                    'easting': coord.easting,
                    'northing': coord.northing,
                    'utm_zone': coord.utm_zone or '',
                    'township': coord.township or '',
                    'range': coord.range or '',
                    'section': coord.section or '',
                    'coordinate_system': coord.coordinate_system
                })
            else:
                row.update({
                    'latitude': None,
                    'longitude': None,
                    'easting': None,
                    'northing': None,
                    'utm_zone': '',
                    'township': '',
                    'range': '',
                    'section': '',
                    'coordinate_system': 'EPSG:4326'
                })
            
            data.append(row)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs='EPSG:4326')
        
        return gdf
    
    def _create_samples_gdf(self, samples: List[Sample], locations: List[Location]) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from samples.
        
        Args:
            samples: List of Sample objects
            locations: List of Location objects for geometry
            
        Returns:
            GeoDataFrame with samples
        """
        if not samples:
            return gpd.GeoDataFrame()
        
        data = []
        geometries = []
        
        # Create location lookup
        location_lookup = {loc.id: loc for loc in locations}
        
        for sample in samples:
            # Get geometry from associated location
            if sample.location_id and sample.location_id in location_lookup:
                location = location_lookup[sample.location_id]
                if hasattr(location.geometry, 'dict'):
                    geom_dict = location.geometry.dict()
                else:
                    geom_dict = location.geometry
                
                if geom_dict["type"] == "Point":
                    geom = Point(geom_dict["coordinates"])
                elif geom_dict["type"] == "Polygon":
                    geom = Polygon(geom_dict["coordinates"][0])
                elif geom_dict["type"] == "LineString":
                    geom = LineString(geom_dict["coordinates"])
                else:
                    geom = Point(0, 0)
            else:
                geom = Point(0, 0)  # Default
            
            geometries.append(geom)
            
            # Create data row
            row = {
                'id': sample.id,
                'location_id': str(sample.location_id) if sample.location_id else '',
                'sample_type': sample.sample_type,
                'depth_from': sample.depth_from,
                'depth_to': sample.depth_to,
                'depth_unit': sample.depth_unit,
                'lithology': sample.lithology or '',
                'alteration': sample.alteration or '',
                'mineralization': sample.mineralization or '',
                'collection_date': sample.collection_date.isoformat() if sample.collection_date else '',
                'confidence': sample.confidence,
                'source_text': sample.source_text or '',
                'assay_count': len(sample.assays)
            }
            
            data.append(row)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs='EPSG:4326')
        
        return gdf
    
    def _create_observations_gdf(self, observations: List[GeologicalObservation], locations: List[Location]) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from observations.
        
        Args:
            observations: List of GeologicalObservation objects
            locations: List of Location objects for geometry
            
        Returns:
            GeoDataFrame with observations
        """
        if not observations:
            return gpd.GeoDataFrame()
        
        data = []
        geometries = []
        
        # Create location lookup
        location_lookup = {loc.id: loc for loc in locations}
        
        for obs in observations:
            # Get geometry from associated location
            if obs.location_id and obs.location_id in location_lookup:
                location = location_lookup[obs.location_id]
                if hasattr(location.geometry, 'dict'):
                    geom_dict = location.geometry.dict()
                else:
                    geom_dict = location.geometry
                
                if geom_dict["type"] == "Point":
                    geom = Point(geom_dict["coordinates"])
                elif geom_dict["type"] == "Polygon":
                    geom = Polygon(geom_dict["coordinates"][0])
                elif geom_dict["type"] == "LineString":
                    geom = LineString(geom_dict["coordinates"])
                else:
                    geom = Point(0, 0)
            else:
                geom = Point(0, 0)  # Default
            
            geometries.append(geom)
            
            # Create data row
            row = {
                'id': str(obs.id),
                'feature_type': obs.feature_type,
                'description': obs.description,
                'location_id': str(obs.location_id) if obs.location_id else '',
                'strike': obs.measurements.strike if obs.measurements else None,
                'dip': obs.measurements.dip if obs.measurements else None,
                'trend': obs.measurements.trend if obs.measurements else None,
                'plunge': obs.measurements.plunge if obs.measurements else None,
                'measurement_type': obs.measurements.measurement_type if obs.measurements else '',
                'rock_types': '; '.join(obs.rock_types) if obs.rock_types else '',
                'minerals': '; '.join(obs.minerals) if obs.minerals else '',
                'confidence': obs.confidence,
                'source_text': obs.source_text or ''
            }
            
            data.append(row)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs='EPSG:4326')
        
        return gdf
    
    def _create_metadata_df(self, metadata) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from metadata.
        
        Args:
            metadata: DocumentMetadata object
            
        Returns:
            GeoDataFrame with metadata
        """
        data = {
            'field': [
                'source_file', 'file_size_bytes', 'processing_date', 'confidence_score',
                'ocr_engine', 'llm_model', 'language', 'page_count', 'document_type',
                'title', 'author', 'company', 'report_date', 'report_type',
                'pages_processed', 'ocr_confidence_avg', 'extraction_confidence_avg',
                'processing_time_seconds'
            ],
            'value': [
                str(metadata.source_file),
                metadata.file_size_bytes,
                metadata.processing_date.isoformat(),
                metadata.confidence_score,
                metadata.ocr_engine,
                metadata.llm_model,
                metadata.language,
                metadata.page_count,
                metadata.document_type or '',
                metadata.title or '',
                metadata.author or '',
                metadata.company or '',
                metadata.report_date.isoformat() if metadata.report_date else '',
                metadata.report_type or '',
                metadata.processing_stats.pages_processed,
                metadata.processing_stats.ocr_confidence_avg,
                metadata.processing_stats.extraction_confidence_avg,
                metadata.processing_stats.processing_time_seconds
            ]
        }
        
        # Create dummy geometry for metadata
        geometries = [Point(0, 0)] * len(data['field'])
        
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs='EPSG:4326')
        
        return gdf