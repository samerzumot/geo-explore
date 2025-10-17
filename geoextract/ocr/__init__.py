"""OCR engines for text extraction from images."""

from geoextract.ocr.paddle_engine import PaddleOCREngine
from geoextract.ocr.tesseract_engine import TesseractEngine
from geoextract.ocr.ocr_manager import OCRManager

__all__ = ["PaddleOCREngine", "TesseractEngine", "OCRManager"]