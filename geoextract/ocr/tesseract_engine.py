"""Tesseract OCR engine for text extraction."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

from geoextract.config import settings

logger = logging.getLogger(__name__)


class TesseractEngine:
    """Tesseract OCR engine for text extraction."""
    
    def __init__(self, language: str = None, tesseract_cmd: str = None):
        """Initialize Tesseract engine.
        
        Args:
            language: Language code (e.g., 'eng', 'spa')
            tesseract_cmd: Path to tesseract executable
        """
        if pytesseract is None:
            raise ImportError("pytesseract is not installed. Install with: pip install pytesseract")
        
        self.language = language or settings.ocr_language
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Test tesseract installation
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract initialized for language: {self.language}")
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                pil_image, 
                lang=self.language, 
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            blocks = []
            full_text = []
            confidences = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:  # Only process non-empty text
                    confidence = int(data['conf'][i])
                    if confidence > 0:  # Only process confident detections
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        block = {
                            "text": text,
                            "bbox": (x, y, w, h),
                            "confidence": confidence / 100.0,  # Convert to 0-1 scale
                            "level": data['level'][i],
                            "page_num": data['page_num'][i],
                            "block_num": data['block_num'][i],
                            "par_num": data['par_num'][i],
                            "line_num": data['line_num'][i],
                            "word_num": data['word_num'][i]
                        }
                        
                        blocks.append(block)
                        full_text.append(text)
                        confidences.append(confidence / 100.0)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "text": " ".join(full_text),
                "blocks": blocks,
                "confidence": avg_confidence,
                "engine": "tesseract",
                "language": self.language
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {
                "text": "",
                "blocks": [],
                "confidence": 0.0,
                "engine": "tesseract",
                "error": str(e)
            }
    
    def extract_text_with_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text with layout information.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with text blocks organized by layout
        """
        # Get basic OCR results
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
        # Group by paragraph
        paragraphs = {}
        for block in blocks:
            par_num = block.get("par_num", 0)
            if par_num not in paragraphs:
                paragraphs[par_num] = []
            paragraphs[par_num].append(block)
        
        # Convert to list of paragraphs
        paragraph_list = []
        for par_num in sorted(paragraphs.keys()):
            par_blocks = paragraphs[par_num]
            # Sort by line number, then word number
            par_blocks.sort(key=lambda b: (b.get("line_num", 0), b.get("word_num", 0)))
            paragraph_list.append(par_blocks)
        
        return {
            "paragraphs": paragraph_list,
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
        
        # Group blocks by line number
        line_groups = {}
        for block in blocks:
            line_num = block.get("line_num", 0)
            if line_num not in line_groups:
                line_groups[line_num] = []
            line_groups[line_num].append(block)
        
        # Find lines with multiple words (potential table rows)
        for line_num, line_blocks in line_groups.items():
            if len(line_blocks) >= 3:  # At least 3 columns
                # Sort by x-coordinate
                line_blocks.sort(key=lambda b: b["bbox"][0])
                table_blocks.extend(line_blocks)
        
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
    
    def extract_tables(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract tables using Tesseract's table detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected tables
        """
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Use Tesseract's table detection
            # This is a simplified approach - in practice, you might want to use
            # more sophisticated table detection methods
            data = pytesseract.image_to_data(
                pil_image, 
                lang=self.language, 
                output_type=pytesseract.Output.DICT
            )
            
            # Group by lines and detect table-like structures
            tables = []
            line_groups = {}
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 0:
                    line_num = data['line_num'][i]
                    if line_num not in line_groups:
                        line_groups[line_num] = []
                    
                    line_groups[line_num].append({
                        "text": text,
                        "bbox": (data['left'][i], data['top'][i], data['width'][i], data['height'][i]),
                        "confidence": int(data['conf'][i]) / 100.0
                    })
            
            # Convert line groups to tables
            for line_num, line_blocks in line_groups.items():
                if len(line_blocks) >= 3:  # At least 3 columns
                    # Sort by x-coordinate
                    line_blocks.sort(key=lambda b: b["bbox"][0])
                    
                    table = {
                        "line_number": line_num,
                        "cells": line_blocks,
                        "row_count": 1,
                        "col_count": len(line_blocks)
                    }
                    tables.append(table)
            
            return tables
            
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []
    
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
                    "engine": "tesseract",
                    "page_number": i + 1,
                    "error": str(e)
                })
        
        return results
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        try:
            # Get available languages from Tesseract
            langs = pytesseract.get_languages()
            return langs
        except Exception as e:
            logger.warning(f"Could not get supported languages: {e}")
            return ["eng"]  # Default to English