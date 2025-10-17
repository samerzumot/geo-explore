"""PDF handling and conversion to images."""

import io
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

from geoextract.config import settings

logger = logging.getLogger(__name__)


class PDFHandler:
    """Handles PDF to image conversion with preprocessing."""
    
    def __init__(self, dpi: int = None):
        """Initialize PDF handler.
        
        Args:
            dpi: Resolution for PDF to image conversion
        """
        self.dpi = dpi or settings.pdf_dpi
        self.temp_dir = settings.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_images_from_pdf(self, pdf_path: Path) -> List[np.ndarray]:
        """Extract images from PDF as numpy arrays.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of images as numpy arrays
        """
        try:
            # Try pdf2image first (better for scanned PDFs)
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                output_folder=self.temp_dir,
                fmt='png',
                thread_count=2
            )
            
            # Convert PIL images to numpy arrays
            image_arrays = []
            for i, pil_image in enumerate(images):
                # Convert to RGB if necessary
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Convert to numpy array
                img_array = np.array(pil_image)
                image_arrays.append(img_array)
                
                # Save intermediate if debug mode
                if settings.save_intermediate_outputs:
                    debug_path = self.temp_dir / f"page_{i:03d}_raw.png"
                    pil_image.save(debug_path)
                    logger.debug(f"Saved raw page {i} to {debug_path}")
            
            logger.info(f"Extracted {len(image_arrays)} pages from {pdf_path}")
            return image_arrays
            
        except Exception as e:
            logger.warning(f"pdf2image failed for {pdf_path}: {e}")
            # Fallback to PyMuPDF
            return self._extract_with_pymupdf(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> List[np.ndarray]:
        """Extract images using PyMuPDF as fallback.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of images as numpy arrays
        """
        try:
            doc = fitz.open(pdf_path)
            image_arrays = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get page as image
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 is default DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Convert to numpy array
                img_array = np.array(pil_image)
                image_arrays.append(img_array)
                
                # Save intermediate if debug mode
                if settings.save_intermediate_outputs:
                    debug_path = self.temp_dir / f"page_{page_num:03d}_pymupdf.png"
                    pil_image.save(debug_path)
                    logger.debug(f"Saved PyMuPDF page {page_num} to {debug_path}")
            
            doc.close()
            logger.info(f"Extracted {len(image_arrays)} pages using PyMuPDF from {pdf_path}")
            return image_arrays
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            raise
    
    def get_pdf_metadata(self, pdf_path: Path) -> dict:
        """Extract metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            page_count = len(doc)
            doc.close()
            
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": page_count,
                "file_size": pdf_path.stat().st_size,
            }
        except Exception as e:
            logger.error(f"Failed to extract metadata from {pdf_path}: {e}")
            return {"page_count": 0, "file_size": 0}
    
    def is_scanned_pdf(self, pdf_path: Path) -> bool:
        """Determine if PDF is scanned (image-based) or text-based.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if PDF appears to be scanned
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Check first few pages for text content
            text_content = 0
            pages_to_check = min(3, len(doc))
            
            for page_num in range(pages_to_check):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_content += len(text.strip())
            
            doc.close()
            
            # If very little text, likely scanned
            return text_content < 100
            
        except Exception as e:
            logger.warning(f"Could not determine if PDF is scanned: {e}")
            return True  # Assume scanned if we can't determine
    
    def validate_pdf(self, pdf_path: Path) -> Tuple[bool, str]:
        """Validate PDF file for processing.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not pdf_path.exists():
            return False, f"File does not exist: {pdf_path}"
        
        if not pdf_path.suffix.lower() == '.pdf':
            return False, f"File is not a PDF: {pdf_path}"
        
        # Check file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            return False, f"File too large: {file_size_mb:.1f}MB > {settings.max_file_size_mb}MB"
        
        # Try to open PDF
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            if page_count == 0:
                return False, "PDF has no pages"
            
            if page_count > 1000:  # Reasonable limit
                return False, f"PDF has too many pages: {page_count}"
                
        except Exception as e:
            return False, f"PDF is corrupted or unreadable: {e}"
        
        return True, ""