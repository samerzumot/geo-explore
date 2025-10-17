"""Image preprocessing and enhancement for better OCR results."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from geoextract.config import settings

logger = logging.getLogger(__name__)


class ImageCleaner:
    """Handles image preprocessing for better OCR accuracy."""
    
    def __init__(self, save_intermediate: bool = None):
        """Initialize image cleaner.
        
        Args:
            save_intermediate: Whether to save intermediate processing steps
        """
        self.save_intermediate = save_intermediate or settings.save_intermediate_outputs
        self.temp_dir = settings.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_image(self, image: np.ndarray, page_num: int = 0) -> np.ndarray:
        """Apply comprehensive image preprocessing pipeline.
        
        Args:
            image: Input image as numpy array
            page_num: Page number for debug output
            
        Returns:
            Preprocessed image as numpy array
        """
        original = image.copy()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Save intermediate
        if self.save_intermediate:
            self._save_debug_image(gray, f"page_{page_num:03d}_01_grayscale.png")
        
        # Deskew image
        deskewed = self._deskew_image(gray)
        if self.save_intermediate:
            self._save_debug_image(deskewed, f"page_{page_num:03d}_02_deskewed.png")
        
        # Noise reduction
        denoised = self._denoise_image(deskewed)
        if self.save_intermediate:
            self._save_debug_image(denoised, f"page_{page_num:03d}_03_denoised.png")
        
        # Contrast enhancement
        enhanced = self._enhance_contrast(denoised)
        if self.save_intermediate:
            self._save_debug_image(enhanced, f"page_{page_num:03d}_04_enhanced.png")
        
        # Binarization
        binary = self._binarize_image(enhanced)
        if self.save_intermediate:
            self._save_debug_image(binary, f"page_{page_num:03d}_05_binary.png")
        
        # Morphological operations
        cleaned = self._morphological_cleanup(binary)
        if self.save_intermediate:
            self._save_debug_image(cleaned, f"page_{page_num:03d}_06_cleaned.png")
        
        logger.debug(f"Preprocessed page {page_num}")
        return cleaned
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in scanned documents.
        
        Args:
            image: Grayscale image
            
        Returns:
            Deskewed image
        """
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (likely the page)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Correct angle if it's close to 90 degrees
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90
        
        # Only rotate if angle is significant
        if abs(angle) > 0.5:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
            return deskewed
        
        return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Denoised image
        """
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast and brightness.
        
        Args:
            image: Grayscale image
            
        Returns:
            Enhanced image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        # Convert back to numpy
        return np.array(enhanced)
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image using adaptive thresholding.
        
        Args:
            image: Grayscale image
            
        Returns:
            Binary image
        """
        # Adaptive thresholding works better for varying lighting
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if text is white on black
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def _morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Clean up binary image using morphological operations.
        
        Args:
            image: Binary image
            
        Returns:
            Cleaned binary image
        """
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in image.
        
        Args:
            image: Binary image
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        min_area = 100  # Minimum area for text region
        min_aspect_ratio = 0.1  # Minimum aspect ratio
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = h / w if w > 0 else 0
            
            # Filter by area and aspect ratio
            if area > min_area and aspect_ratio > min_aspect_ratio:
                text_regions.append((x, y, w, h))
        
        # Sort by y-coordinate (top to bottom)
        text_regions.sort(key=lambda box: box[1])
        
        return text_regions
    
    def _save_debug_image(self, image: np.ndarray, filename: str) -> None:
        """Save image for debugging.
        
        Args:
            image: Image to save
            filename: Filename to save as
        """
        try:
            debug_path = self.temp_dir / filename
            cv2.imwrite(str(debug_path), image)
            logger.debug(f"Saved debug image: {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug image {filename}: {e}")
    
    def batch_preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocess a batch of images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of preprocessed images
        """
        processed_images = []
        
        for i, image in enumerate(images):
            try:
                processed = self.preprocess_image(image, i)
                processed_images.append(processed)
            except Exception as e:
                logger.error(f"Failed to preprocess image {i}: {e}")
                # Use original image as fallback
                processed_images.append(image)
        
        return processed_images