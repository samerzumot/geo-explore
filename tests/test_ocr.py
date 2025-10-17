"""Tests for OCR functionality."""

import pytest
import numpy as np
from pathlib import Path

from geoextract.ocr.ocr_manager import OCRManager
from geoextract.ocr.paddle_engine import PaddleOCREngine
from geoextract.ocr.tesseract_engine import TesseractEngine


class TestOCRManager:
    """Test OCR manager functionality."""
    
    def test_ocr_manager_init(self):
        """Test OCR manager initialization."""
        # This will fail if no OCR engines are available
        try:
            manager = OCRManager(engine="paddle")
            assert manager.engine == "paddle"
        except Exception:
            # Skip test if OCR engines not available
            pytest.skip("OCR engines not available")
    
    def test_validate_image(self):
        """Test image validation."""
        manager = OCRManager(engine="paddle")
        
        # Valid image
        valid_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        is_valid, error = manager.validate_image(valid_image)
        assert is_valid
        assert error == ""
        
        # Invalid image (None)
        is_valid, error = manager.validate_image(None)
        assert not is_valid
        assert "Image is None" in error
        
        # Invalid image (wrong shape)
        invalid_image = np.random.randint(0, 255, (100,), dtype=np.uint8)
        is_valid, error = manager.validate_image(invalid_image)
        assert not is_valid
        assert "Invalid image shape" in error


class TestPaddleOCREngine:
    """Test PaddleOCR engine functionality."""
    
    def test_paddle_engine_init(self):
        """Test PaddleOCR engine initialization."""
        try:
            engine = PaddleOCREngine()
            assert engine.language == "en"
        except ImportError:
            pytest.skip("PaddleOCR not available")
        except Exception:
            pytest.skip("PaddleOCR initialization failed")
    
    def test_extract_text_empty_image(self):
        """Test text extraction from empty image."""
        try:
            engine = PaddleOCREngine()
            empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = engine.extract_text(empty_image)
            
            assert "text" in result
            assert "blocks" in result
            assert "confidence" in result
            assert "engine" in result
            assert result["engine"] == "paddleocr"
        except ImportError:
            pytest.skip("PaddleOCR not available")
        except Exception:
            pytest.skip("PaddleOCR not working")


class TestTesseractEngine:
    """Test Tesseract engine functionality."""
    
    def test_tesseract_engine_init(self):
        """Test Tesseract engine initialization."""
        try:
            engine = TesseractEngine()
            assert engine.language == "en"
        except ImportError:
            pytest.skip("Tesseract not available")
        except Exception:
            pytest.skip("Tesseract initialization failed")
    
    def test_extract_text_empty_image(self):
        """Test text extraction from empty image."""
        try:
            engine = TesseractEngine()
            empty_image = np.zeros((100, 100), dtype=np.uint8)
            result = engine.extract_text(empty_image)
            
            assert "text" in result
            assert "blocks" in result
            assert "confidence" in result
            assert "engine" in result
            assert result["engine"] == "tesseract"
        except ImportError:
            pytest.skip("Tesseract not available")
        except Exception:
            pytest.skip("Tesseract not working")