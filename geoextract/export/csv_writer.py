"""CSV export functionality for geological data."""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from geoextract.schemas.document import GeologicalDocument
from geoextract.schemas.geological import Location, Sample, GeologicalObservation, AssayResult

logger = logging.getLogger(__name__)


class CSVWriter:
    """Writes geological data to CSV format."""
    
    def __init__(self, include_metadata: bool = True):
        """Initialize CSV writer.
        
        Args:
            include_metadata: Whether to include metadata in output
        """
        self.include_metadata = include_metadata
    
    def write_document(self, document: GeologicalDocument, output_dir: Path) -> None:
        """Write geological document to CSV files.
        
        Args:
            document: Geological document to export
            output_dir: Directory to write CSV files
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write locations
            if document.locations:
                locations_path = output_dir / "locations.csv"
                self.write_locations(document.locations, locations_path)
            
            # Write samples
            if document.samples:
                samples_path = output_dir / "samples.csv"
                self.write_samples(document.samples, samples_path)
            
            # Write assays
            if document.samples:
                assays_path = output_dir / "assays.csv"
                self.write_assays(document.samples, assays_path)
            
            # Write observations
            if document.observations:
                observations_path = output_dir / "observations.csv"
                self.write_observations(document.observations, observations_path)
            
            # Write metadata
            if self.include_metadata:
                metadata_path = output_dir / "metadata.csv"
                self.write_metadata(document.metadata, metadata_path)
            
            logger.info(f"CSV files exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to write CSV files: {e}")
            raise
    
    def write_locations(self, locations: List[Location], output_path: Path) -> None:
        """Write locations to CSV file.
        
        Args:
            locations: List of Location objects
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'id', 'name', 'location_type', 'latitude', 'longitude',
                    'easting', 'northing', 'utm_zone', 'township', 'range',
                    'section', 'quarter_section', 'coordinate_system',
                    'elevation', 'elevation_unit', 'county', 'state_province',
                    'country', 'confidence', 'source_text'
                ]
                writer.writerow(header)
                
                # Write data
                for location in locations:
                    # Get primary coordinates (first coordinate)
                    primary_coord = location.coordinates[0] if location.coordinates else None
                    
                    row = [
                        str(location.id),
                        location.name or '',
                        location.location_type,
                        primary_coord.latitude if primary_coord else None,
                        primary_coord.longitude if primary_coord else None,
                        primary_coord.easting if primary_coord else None,
                        primary_coord.northing if primary_coord else None,
                        primary_coord.utm_zone if primary_coord else None,
                        primary_coord.township if primary_coord else None,
                        primary_coord.range if primary_coord else None,
                        primary_coord.section if primary_coord else None,
                        primary_coord.quarter_section if primary_coord else None,
                        primary_coord.coordinate_system if primary_coord else None,
                        location.elevation,
                        location.elevation_unit,
                        location.county or '',
                        location.state_province or '',
                        location.country or '',
                        location.confidence,
                        location.source_text or ''
                    ]
                    writer.writerow(row)
            
            logger.info(f"Locations CSV exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write locations CSV: {e}")
            raise
    
    def write_samples(self, samples: List[Sample], output_path: Path) -> None:
        """Write samples to CSV file.
        
        Args:
            samples: List of Sample objects
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'id', 'location_id', 'sample_type', 'depth_from', 'depth_to',
                    'depth_unit', 'lithology', 'alteration', 'mineralization',
                    'collection_date', 'confidence', 'source_text'
                ]
                writer.writerow(header)
                
                # Write data
                for sample in samples:
                    row = [
                        sample.id,
                        str(sample.location_id) if sample.location_id else '',
                        sample.sample_type,
                        sample.depth_from,
                        sample.depth_to,
                        sample.depth_unit,
                        sample.lithology or '',
                        sample.alteration or '',
                        sample.mineralization or '',
                        sample.collection_date.isoformat() if sample.collection_date else '',
                        sample.confidence,
                        sample.source_text or ''
                    ]
                    writer.writerow(row)
            
            logger.info(f"Samples CSV exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write samples CSV: {e}")
            raise
    
    def write_assays(self, samples: List[Sample], output_path: Path) -> None:
        """Write assay data to CSV file.
        
        Args:
            samples: List of Sample objects
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'sample_id', 'element', 'value', 'unit', 'detection_limit',
                    'method', 'confidence'
                ]
                writer.writerow(header)
                
                # Write data
                for sample in samples:
                    for assay in sample.assays:
                        row = [
                            sample.id,
                            assay.element,
                            assay.value,
                            assay.unit,
                            assay.detection_limit,
                            assay.method or '',
                            assay.confidence
                        ]
                        writer.writerow(row)
            
            logger.info(f"Assays CSV exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write assays CSV: {e}")
            raise
    
    def write_observations(self, observations: List[GeologicalObservation], output_path: Path) -> None:
        """Write geological observations to CSV file.
        
        Args:
            observations: List of GeologicalObservation objects
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'id', 'feature_type', 'description', 'location_id',
                    'strike', 'dip', 'trend', 'plunge', 'measurement_type',
                    'rock_types', 'minerals', 'confidence', 'source_text'
                ]
                writer.writerow(header)
                
                # Write data
                for obs in observations:
                    # Format rock types and minerals as semicolon-separated
                    rock_types = '; '.join(obs.rock_types) if obs.rock_types else ''
                    minerals = '; '.join(obs.minerals) if obs.minerals else ''
                    
                    # Get measurements
                    strike = obs.measurements.strike if obs.measurements else None
                    dip = obs.measurements.dip if obs.measurements else None
                    trend = obs.measurements.trend if obs.measurements else None
                    plunge = obs.measurements.plunge if obs.measurements else None
                    measurement_type = obs.measurements.measurement_type if obs.measurements else None
                    
                    row = [
                        str(obs.id),
                        obs.feature_type,
                        obs.description,
                        str(obs.location_id) if obs.location_id else '',
                        strike,
                        dip,
                        trend,
                        plunge,
                        measurement_type or '',
                        rock_types,
                        minerals,
                        obs.confidence,
                        obs.source_text or ''
                    ]
                    writer.writerow(row)
            
            logger.info(f"Observations CSV exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write observations CSV: {e}")
            raise
    
    def write_metadata(self, metadata, output_path: Path) -> None:
        """Write document metadata to CSV file.
        
        Args:
            metadata: DocumentMetadata object
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'field', 'value'
                ]
                writer.writerow(header)
                
                # Write data
                data = [
                    ('source_file', str(metadata.source_file)),
                    ('file_size_bytes', metadata.file_size_bytes),
                    ('processing_date', metadata.processing_date.isoformat()),
                    ('confidence_score', metadata.confidence_score),
                    ('ocr_engine', metadata.ocr_engine),
                    ('llm_model', metadata.llm_model),
                    ('language', metadata.language),
                    ('page_count', metadata.page_count),
                    ('document_type', metadata.document_type or ''),
                    ('title', metadata.title or ''),
                    ('author', metadata.author or ''),
                    ('company', metadata.company or ''),
                    ('report_date', metadata.report_date.isoformat() if metadata.report_date else ''),
                    ('report_type', metadata.report_type or ''),
                    ('pages_processed', metadata.processing_stats.pages_processed),
                    ('ocr_confidence_avg', metadata.processing_stats.ocr_confidence_avg),
                    ('extraction_confidence_avg', metadata.processing_stats.extraction_confidence_avg),
                    ('processing_time_seconds', metadata.processing_stats.processing_time_seconds),
                    ('errors', '; '.join(metadata.processing_stats.errors)),
                    ('warnings', '; '.join(metadata.processing_stats.warnings))
                ]
                
                for field, value in data:
                    writer.writerow([field, value])
            
            logger.info(f"Metadata CSV exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write metadata CSV: {e}")
            raise
    
    def write_combined(self, document: GeologicalDocument, output_path: Path) -> None:
        """Write all data to a single combined CSV file.
        
        Args:
            document: Geological document to export
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'record_type', 'id', 'name', 'location_type', 'latitude', 'longitude',
                    'easting', 'northing', 'utm_zone', 'township', 'range', 'section',
                    'elevation', 'county', 'state_province', 'country',
                    'sample_type', 'depth_from', 'depth_to', 'depth_unit',
                    'lithology', 'alteration', 'mineralization',
                    'element', 'value', 'unit', 'detection_limit',
                    'feature_type', 'description', 'strike', 'dip',
                    'rock_types', 'minerals', 'confidence', 'source_text'
                ]
                writer.writerow(header)
                
                # Write locations
                for location in document.locations:
                    primary_coord = location.coordinates[0] if location.coordinates else None
                    row = [
                        'location',
                        str(location.id),
                        location.name or '',
                        location.location_type,
                        primary_coord.latitude if primary_coord else '',
                        primary_coord.longitude if primary_coord else '',
                        primary_coord.easting if primary_coord else '',
                        primary_coord.northing if primary_coord else '',
                        primary_coord.utm_zone if primary_coord else '',
                        primary_coord.township if primary_coord else '',
                        primary_coord.range if primary_coord else '',
                        primary_coord.section if primary_coord else '',
                        location.elevation or '',
                        location.county or '',
                        location.state_province or '',
                        location.country or '',
                        '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                        location.confidence,
                        location.source_text or ''
                    ]
                    writer.writerow(row)
                
                # Write samples
                for sample in document.samples:
                    row = [
                        'sample',
                        sample.id,
                        '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                        sample.sample_type,
                        sample.depth_from or '',
                        sample.depth_to or '',
                        sample.depth_unit,
                        sample.lithology or '',
                        sample.alteration or '',
                        sample.mineralization or '',
                        '', '', '', '', '', '', '', '', '', '',
                        sample.confidence,
                        sample.source_text or ''
                    ]
                    writer.writerow(row)
                    
                    # Write assays for this sample
                    for assay in sample.assays:
                        row = [
                            'assay',
                            sample.id,
                            '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                            '', '', '', '', '', '', '',
                            assay.element,
                            assay.value,
                            assay.unit,
                            assay.detection_limit or '',
                            '', '', '', '', '', '', '', '',
                            assay.confidence,
                            ''
                        ]
                        writer.writerow(row)
                
                # Write observations
                for obs in document.observations:
                    rock_types = '; '.join(obs.rock_types) if obs.rock_types else ''
                    minerals = '; '.join(obs.minerals) if obs.minerals else ''
                    strike = obs.measurements.strike if obs.measurements else ''
                    dip = obs.measurements.dip if obs.measurements else ''
                    
                    row = [
                        'observation',
                        str(obs.id),
                        '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                        '', '', '', '', '', '', '', '', '', '', '', '',
                        obs.feature_type,
                        obs.description,
                        strike,
                        dip,
                        rock_types,
                        minerals,
                        obs.confidence,
                        obs.source_text or ''
                    ]
                    writer.writerow(row)
            
            logger.info(f"Combined CSV exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write combined CSV: {e}")
            raise