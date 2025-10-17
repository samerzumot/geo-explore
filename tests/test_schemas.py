"""Tests for data schemas."""

import pytest
from datetime import datetime
from uuid import uuid4

from geoextract.schemas.geological import (
    Coordinate, AssayResult, Sample, Location, GeologicalObservation,
    StructuralMeasurement
)
from geoextract.schemas.document import DocumentMetadata, ProcessingStats, GeologicalDocument


class TestCoordinate:
    """Test Coordinate schema."""
    
    def test_coordinate_creation(self):
        """Test coordinate creation."""
        coord = Coordinate(
            latitude=37.7749,
            longitude=-122.4194,
            coordinate_system="EPSG:4326",
            confidence=0.9
        )
        
        assert coord.latitude == 37.7749
        assert coord.longitude == -122.4194
        assert coord.coordinate_system == "EPSG:4326"
        assert coord.confidence == 0.9
    
    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Valid coordinate
        coord = Coordinate(latitude=45.0, longitude=120.0)
        assert coord.latitude == 45.0
        assert coord.longitude == 120.0
        
        # Invalid latitude
        with pytest.raises(ValueError):
            Coordinate(latitude=91.0, longitude=120.0)
        
        # Invalid longitude
        with pytest.raises(ValueError):
            Coordinate(latitude=45.0, longitude=181.0)


class TestAssayResult:
    """Test AssayResult schema."""
    
    def test_assay_result_creation(self):
        """Test assay result creation."""
        assay = AssayResult(
            element="Au",
            value=2.5,
            unit="g/t",
            confidence=0.9
        )
        
        assert assay.element == "Au"
        assert assay.value == 2.5
        assert assay.unit == "g/t"
        assert assay.confidence == 0.9
    
    def test_assay_result_validation(self):
        """Test assay result validation."""
        # Valid assay
        assay = AssayResult(element="Au", value=1.0, unit="g/t")
        assert assay.value == 1.0
        
        # Invalid negative value
        with pytest.raises(ValueError):
            AssayResult(element="Au", value=-1.0, unit="g/t")


class TestSample:
    """Test Sample schema."""
    
    def test_sample_creation(self):
        """Test sample creation."""
        sample = Sample(
            id="S001",
            sample_type="core",
            depth_from=0.0,
            depth_to=1.0,
            depth_unit="m",
            confidence=0.9
        )
        
        assert sample.id == "S001"
        assert sample.sample_type == "core"
        assert sample.depth_from == 0.0
        assert sample.depth_to == 1.0
        assert sample.depth_unit == "m"
        assert sample.confidence == 0.9
    
    def test_sample_validation(self):
        """Test sample validation."""
        # Valid sample
        sample = Sample(id="S001", sample_type="core", depth_from=0.0, depth_to=1.0)
        assert sample.depth_from < sample.depth_to
        
        # Invalid depth range
        with pytest.raises(ValueError):
            Sample(id="S001", sample_type="core", depth_from=1.0, depth_to=0.0)


class TestLocation:
    """Test Location schema."""
    
    def test_location_creation(self):
        """Test location creation."""
        from geojson_pydantic import Point
        
        geometry = Point(coordinates=[-122.4194, 37.7749])
        location = Location(
            name="Test Location",
            location_type="sample_site",
            geometry=geometry,
            confidence=0.9
        )
        
        assert location.name == "Test Location"
        assert location.location_type == "sample_site"
        assert location.confidence == 0.9


class TestGeologicalObservation:
    """Test GeologicalObservation schema."""
    
    def test_observation_creation(self):
        """Test observation creation."""
        observation = GeologicalObservation(
            feature_type="fault",
            description="Normal fault with 45° dip",
            confidence=0.8
        )
        
        assert observation.feature_type == "fault"
        assert observation.description == "Normal fault with 45° dip"
        assert observation.confidence == 0.8


class TestDocumentMetadata:
    """Test DocumentMetadata schema."""
    
    def test_metadata_creation(self):
        """Test metadata creation."""
        metadata = DocumentMetadata(
            source_file=Path("test.pdf"),
            file_size_bytes=1024,
            ocr_engine="paddle",
            llm_model="llama3.1:8b"
        )
        
        assert metadata.source_file == Path("test.pdf")
        assert metadata.file_size_bytes == 1024
        assert metadata.ocr_engine == "paddle"
        assert metadata.llm_model == "llama3.1:8b"
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        # Valid metadata
        metadata = DocumentMetadata(
            source_file=Path("test.pdf"),
            file_size_bytes=1024,
            ocr_engine="paddle",
            llm_model="llama3.1:8b"
        )
        assert metadata.file_size_bytes == 1024
        
        # Invalid file size
        with pytest.raises(ValueError):
            DocumentMetadata(
                source_file=Path("test.pdf"),
                file_size_bytes=0,
                ocr_engine="paddle",
                llm_model="llama3.1:8b"
            )


class TestGeologicalDocument:
    """Test GeologicalDocument schema."""
    
    def test_document_creation(self):
        """Test document creation."""
        metadata = DocumentMetadata(
            source_file=Path("test.pdf"),
            file_size_bytes=1024,
            ocr_engine="paddle",
            llm_model="llama3.1:8b"
        )
        
        document = GeologicalDocument(metadata=metadata)
        
        assert document.metadata == metadata
        assert document.locations == []
        assert document.samples == []
        assert document.observations == []
    
    def test_document_methods(self):
        """Test document methods."""
        metadata = DocumentMetadata(
            source_file=Path("test.pdf"),
            file_size_bytes=1024,
            ocr_engine="paddle",
            llm_model="llama3.1:8b"
        )
        
        # Create sample location
        from geojson_pydantic import Point
        geometry = Point(coordinates=[-122.4194, 37.7749])
        location = Location(
            name="Test Location",
            location_type="sample_site",
            geometry=geometry
        )
        
        document = GeologicalDocument(
            metadata=metadata,
            locations=[location]
        )
        
        # Test get_locations_by_type
        sample_sites = document.get_locations_by_type("sample_site")
        assert len(sample_sites) == 1
        assert sample_sites[0] == location
        
        # Test get_all_coordinates
        coords = document.get_all_coordinates()
        assert len(coords) == 1
        assert coords[0] == location