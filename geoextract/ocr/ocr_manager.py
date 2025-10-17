"""OCR manager for coordinating multiple OCR engines."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from geoextract.config import settings
from geoextract.ocr.paddle_engine import PaddleOCREngine
from geoextract.ocr.tesseract_engine import TesseractEngine

logger = logging.getLogger(__name__)


class OCRManager:
    """Manages OCR engines and coordinates text extraction."""
    
    def __init__(self, engine: str = None, language: str = None):
        """Initialize OCR manager.
        
        Args:
            engine: OCR engine to use ('paddle', 'tesseract', 'both')
            language: Language code for OCR
        """
        self.engine = engine or settings.ocr_engine
        self.language = language or settings.ocr_language
        self.confidence_threshold = settings.ocr_confidence_threshold
        
        # Initialize engines
        self.paddle_engine = None
        self.tesseract_engine = None
        
        if self.engine in ["paddle", "both"]:
            try:
                self.paddle_engine = PaddleOCREngine(language=self.language)
                logger.info("PaddleOCR engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
        
        if self.engine in ["tesseract", "both"]:
            try:
                self.tesseract_engine = TesseractEngine(language=self.language)
                logger.info("Tesseract engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Tesseract: {e}")
        
        if not self.paddle_engine and not self.tesseract_engine:
            raise RuntimeError("No OCR engines could be initialized")
    
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image using configured engine(s).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if self.engine == "both":
            return self._extract_with_both_engines(image)
        elif self.engine == "paddle" and self.paddle_engine:
            return self.paddle_engine.extract_text_with_layout(image)
        elif self.engine == "tesseract" and self.tesseract_engine:
            return self.tesseract_engine.extract_text_with_layout(image)
        else:
            raise RuntimeError(f"OCR engine '{self.engine}' not available")
    
    def _extract_with_both_engines(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using both engines and combine results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Combined extraction results
        """
        results = {}
        
        # Extract with PaddleOCR
        if self.paddle_engine:
            try:
                paddle_result = self.paddle_engine.extract_text_with_layout(image)
                results["paddle"] = paddle_result
            except Exception as e:
                logger.warning(f"PaddleOCR extraction failed: {e}")
                results["paddle"] = {"error": str(e)}
        
        # Extract with Tesseract
        if self.tesseract_engine:
            try:
                tesseract_result = self.tesseract_engine.extract_text_with_layout(image)
                results["tesseract"] = tesseract_result
            except Exception as e:
                logger.warning(f"Tesseract extraction failed: {e}")
                results["tesseract"] = {"error": str(e)}
        
        # Combine results
        return self._combine_results(results)
    
    def _combine_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple OCR engines.
        
        Args:
            results: Dictionary with results from each engine
            
        Returns:
            Combined results
        """
        # Choose the best result based on confidence
        best_engine = None
        best_confidence = 0.0
        
        for engine, result in results.items():
            if "error" not in result:
                confidence = result.get("confidence", 0.0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_engine = engine
        
        if best_engine:
            combined_result = results[best_engine].copy()
            combined_result["primary_engine"] = best_engine
            combined_result["all_results"] = results
            return combined_result
        else:
            # Fallback to first available result
            for engine, result in results.items():
                if "error" not in result:
                    result["primary_engine"] = engine
                    result["all_results"] = results
                    return result
        
        # If all failed, return error
        return {
            "text": "",
            "blocks": [],
            "confidence": 0.0,
            "engine": "combined",
            "error": "All OCR engines failed",
            "all_results": results
        }
    
    def extract_text_with_confidence_filter(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text and filter by confidence threshold.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Filtered extraction results
        """
        result = self.extract_text(image)
        
        if "blocks" in result:
            # Filter blocks by confidence
            filtered_blocks = [
                block for block in result["blocks"]
                if block.get("confidence", 0.0) >= self.confidence_threshold
            ]
            
            # Update result
            result["blocks"] = filtered_blocks
            result["filtered_blocks"] = len(result["blocks"])
            
            # Recalculate text from filtered blocks
            filtered_text = " ".join([block["text"] for block in filtered_blocks])
            result["text"] = filtered_text
        
        return result
    
    def batch_extract(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Extract text from multiple images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.extract_text_with_confidence_filter(image)
                result["page_number"] = i + 1
                results.append(result)
                logger.debug(f"Processed page {i + 1}")
            except Exception as e:
                logger.error(f"Failed to process page {i + 1}: {e}")
                results.append({
                    "text": "",
                    "blocks": [],
                    "confidence": 0.0,
                    "engine": self.engine,
                    "page_number": i + 1,
                    "error": str(e)
                })
        
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available OCR engines.
        
        Returns:
            Dictionary with engine information
        """
        info = {
            "configured_engine": self.engine,
            "language": self.language,
            "confidence_threshold": self.confidence_threshold,
            "available_engines": {}
        }
        
        if self.paddle_engine:
            info["available_engines"]["paddle"] = {
                "status": "available",
                "supported_languages": self.paddle_engine.get_supported_languages()
            }
        else:
            info["available_engines"]["paddle"] = {"status": "unavailable"}
        
        if self.tesseract_engine:
            info["available_engines"]["tesseract"] = {
                "status": "available",
                "supported_languages": self.tesseract_engine.get_supported_languages()
            }
        else:
            info["available_engines"]["tesseract"] = {"status": "unavailable"}
        
        return info
    
    def validate_image(self, image: np.ndarray) -> tuple[bool, str]:
        """Validate image for OCR processing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, "Image is None"
        
        if len(image.shape) not in [2, 3]:
            return False, f"Invalid image shape: {image.shape}"
        
        if image.size == 0:
            return False, "Image is empty"
        
        # Check image dimensions
        height, width = image.shape[:2]
        if height < 10 or width < 10:
            return False, f"Image too small: {width}x{height}"
        
        if height > 10000 or width > 10000:
            return False, f"Image too large: {width}x{height}"
        
        return True, ""
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        return image