"""PaddleOCR engine for text extraction."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

from geoextract.config import settings

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """PaddleOCR engine for text extraction with layout preservation."""
    
    def __init__(self, language: str = None, use_gpu: bool = True):
        """Initialize PaddleOCR engine.
        
        Args:
            language: Language code (e.g., 'en', 'ch')
            use_gpu: Whether to use GPU acceleration
        """
        if PaddleOCR is None:
            raise ImportError("PaddleOCR is not installed. Install with: pip install paddleocr")
        
        self.language = language or settings.ocr_language
        self.use_gpu = use_gpu
        
        # Initialize PaddleOCR
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info(f"PaddleOCR initialized for language: {self.language}")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image with layout information.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Run OCR
            result = self.ocr.ocr(image, cls=True)
            
            if not result or not result[0]:
                return {
                    "text": "",
                    "blocks": [],
                    "confidence": 0.0,
                    "engine": "paddleocr"
                }
            
            # Process results
            blocks = []
            full_text = []
            confidences = []
            
            for line in result[0]:
                if not line:
                    continue
                
                # Extract bounding box and text
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (text, confidence)
                
                if len(text_info) >= 2:
                    text = text_info[0]
                    confidence = text_info[1]
                else:
                    text = text_info[0] if isinstance(text_info, str) else ""
                    confidence = 0.0
                
                # Convert bbox to (x, y, w, h) format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                block = {
                    "text": text,
                    "bbox": (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                    "confidence": float(confidence),
                    "polygon": bbox
                }
                
                blocks.append(block)
                full_text.append(text)
                confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "text": " ".join(full_text),
                "blocks": blocks,
                "confidence": avg_confidence,
                "engine": "paddleocr",
                "language": self.language
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return {
                "text": "",
                "blocks": [],
                "confidence": 0.0,
                "engine": "paddleocr",
                "error": str(e)
            }
    
    def extract_text_with_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text with enhanced layout detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with text blocks organized by layout
        """
        # First get basic OCR results
        ocr_result = self.extract_text(image)
        
        if not ocr_result["blocks"]:
            return ocr_result
        
        # Group blocks by layout
        layout_blocks = self._group_blocks_by_layout(ocr_result["blocks"])
        
        # Detect tables
        table_blocks = self._detect_table_blocks(ocr_result["blocks"])
        
        # Detect coordinate blocks
        coordinate_blocks = self._detect_coordinate_blocks(ocr_result["blocks"])
        
        return {
            **ocr_result,
            "layout_blocks": layout_blocks,
            "table_blocks": table_blocks,
            "coordinate_blocks": coordinate_blocks
        }
    
    def _group_blocks_by_layout(self, blocks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group text blocks by layout type.
        
        Args:
            blocks: List of text blocks
            
        Returns:
            Dictionary grouping blocks by layout type
        """
        # Sort blocks by y-coordinate (top to bottom)
        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])
        
        # Group into paragraphs (blocks close vertically)
        paragraphs = []
        current_paragraph = []
        
        for i, block in enumerate(sorted_blocks):
            if not current_paragraph:
                current_paragraph = [block]
            else:
                # Check if block is part of current paragraph
                last_block = current_paragraph[-1]
                y_gap = block["bbox"][1] - (last_block["bbox"][1] + last_block["bbox"][3])
                
                if y_gap < 30:  # Within 30 pixels vertically
                    current_paragraph.append(block)
                else:
                    # Start new paragraph
                    if current_paragraph:
                        paragraphs.append(current_paragraph)
                    current_paragraph = [block]
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return {
            "paragraphs": paragraphs,
            "total_blocks": len(blocks)
        }
    
    def _detect_table_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Detect blocks that are likely part of tables.
        
        Args:
            blocks: List of text blocks
            
        Returns:
            List of table blocks
        """
        table_blocks = []
        
        # Group blocks by similar y-coordinates (rows)
        y_groups = {}
        for block in blocks:
            y = block["bbox"][1]
            # Round to nearest 20 pixels to group similar rows
            y_group = round(y / 20) * 20
            if y_group not in y_groups:
                y_groups[y_group] = []
            y_groups[y_group].append(block)
        
        # Find groups with multiple blocks (potential table rows)
        for y_group, row_blocks in y_groups.items():
            if len(row_blocks) >= 3:  # At least 3 columns
                # Sort by x-coordinate
                row_blocks.sort(key=lambda b: b["bbox"][0])
                table_blocks.extend(row_blocks)
        
        return table_blocks
    
    def _detect_coordinate_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Detect blocks that likely contain coordinates.
        
        Args:
            blocks: List of text blocks
            
        Returns:
            List of coordinate blocks
        """
        coordinate_blocks = []
        
        # Patterns that indicate coordinates
        coordinate_patterns = [
            r'\d+\.\d+[°\s]*[NSEW]',  # Decimal degrees with direction
            r'\d+°\s*\d+\'\s*\d+\.?\d*"[°\s]*[NSEW]',  # DMS format
            r'\d{6,7}\s+\d{7,8}',  # UTM coordinates
            r'T\d+N\s*R\d+[EW]',  # Township/Range
            r'Section\s+\d+',  # Section references
        ]
        
        import re
        
        for block in blocks:
            text = block["text"]
            
            # Check for coordinate patterns
            for pattern in coordinate_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    coordinate_blocks.append(block)
                    break
        
        return coordinate_blocks
    
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
                result = self.extract_text_with_layout(image)
                result["page_number"] = i + 1
                results.append(result)
                logger.debug(f"Processed page {i + 1}")
            except Exception as e:
                logger.error(f"Failed to process page {i + 1}: {e}")
                results.append({
                    "text": "",
                    "blocks": [],
                    "confidence": 0.0,
                    "engine": "paddleocr",
                    "page_number": i + 1,
                    "error": str(e)
                })
        
        return results
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        # PaddleOCR supports many languages
        return [
            "en", "ch", "chinese_cht", "fr", "german", "korean", "japan",
            "it", "xi", "pu", "ru", "ar", "ta", "ug", "fa", "ur", "rs",
            "oc", "rsc", "bg", "uk", "be", "te", "kn", "ml", "ne", "hi",
            "mr", "sa", "lrc", "my", "th", "lo", "ka", "fo", "tl", "en-gb"
        ]