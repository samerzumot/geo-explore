"""Layout detection for tables, columns, and text blocks."""

import logging
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a detected text block."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    text_type: str  # 'table', 'paragraph', 'header', 'footer', 'caption'
    lines: List[Tuple[int, int, int, int]] = None  # Line bounding boxes


@dataclass
class TableRegion:
    """Represents a detected table region."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    rows: int
    cols: int
    cells: List[Tuple[int, int, int, int]] = None  # Cell bounding boxes


class LayoutDetector:
    """Detects layout elements in document images."""
    
    def __init__(self, min_table_area: int = 1000, min_cell_area: int = 50):
        """Initialize layout detector.
        
        Args:
            min_table_area: Minimum area for table detection
            min_cell_area: Minimum area for cell detection
        """
        self.min_table_area = min_table_area
        self.min_cell_area = min_cell_area
    
    def detect_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect layout elements in image.
        
        Args:
            image: Binary image
            
        Returns:
            Dictionary with detected layout elements
        """
        # Detect text blocks
        text_blocks = self._detect_text_blocks(image)
        
        # Detect tables
        tables = self._detect_tables(image)
        
        # Detect columns
        columns = self._detect_columns(image, text_blocks)
        
        # Classify text blocks
        classified_blocks = self._classify_text_blocks(text_blocks, tables, columns)
        
        return {
            "text_blocks": classified_blocks,
            "tables": tables,
            "columns": columns,
            "page_structure": self._analyze_page_structure(classified_blocks)
        }
    
    def _detect_text_blocks(self, image: np.ndarray) -> List[TextBlock]:
        """Detect text blocks using contour analysis.
        
        Args:
            image: Binary image
            
        Returns:
            List of detected text blocks
        """
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_blocks = []
        min_area = 200  # Minimum area for text block
        min_aspect_ratio = 0.05  # Minimum aspect ratio
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = h / w if w > 0 else 0
            
            # Filter by area and aspect ratio
            if area > min_area and aspect_ratio > min_aspect_ratio:
                # Calculate confidence based on area and aspect ratio
                confidence = min(1.0, (area / 10000) * (aspect_ratio / 0.5))
                
                text_block = TextBlock(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    text_type="unknown"
                )
                text_blocks.append(text_block)
        
        # Sort by y-coordinate (top to bottom)
        text_blocks.sort(key=lambda block: block.bbox[1])
        
        return text_blocks
    
    def _detect_tables(self, image: np.ndarray) -> List[TableRegion]:
        """Detect table regions using line detection.
        
        Args:
            image: Binary image
            
        Returns:
            List of detected table regions
        """
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_horizontal_lines(image)
        vertical_lines = self._detect_vertical_lines(image)
        
        # Find intersections to identify table cells
        table_regions = self._find_table_regions(horizontal_lines, vertical_lines, image.shape)
        
        return table_regions
    
    def _detect_horizontal_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect horizontal lines in image.
        
        Args:
            image: Binary image
            
        Returns:
            List of horizontal line bounding boxes
        """
        # Create horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find contours of horizontal lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h < 10:  # Filter for horizontal lines
                lines.append((x, y, w, h))
        
        return lines
    
    def _detect_vertical_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect vertical lines in image.
        
        Args:
            image: Binary image
            
        Returns:
            List of vertical line bounding boxes
        """
        # Create vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours of vertical lines
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 50 and w < 10:  # Filter for vertical lines
                lines.append((x, y, w, h))
        
        return lines
    
    def _find_table_regions(self, horizontal_lines: List, vertical_lines: List, image_shape: Tuple) -> List[TableRegion]:
        """Find table regions from detected lines.
        
        Args:
            horizontal_lines: List of horizontal line bounding boxes
            vertical_lines: List of vertical line bounding boxes
            image_shape: Shape of the image
            
        Returns:
            List of detected table regions
        """
        if not horizontal_lines or not vertical_lines:
            return []
        
        # Find intersections
        intersections = self._find_line_intersections(horizontal_lines, vertical_lines)
        
        if len(intersections) < 4:  # Need at least 4 intersections for a table
            return []
        
        # Group intersections into table regions
        table_regions = self._group_intersections_to_tables(intersections, image_shape)
        
        return table_regions
    
    def _find_line_intersections(self, horizontal_lines: List, vertical_lines: List) -> List[Tuple[int, int]]:
        """Find intersections between horizontal and vertical lines.
        
        Args:
            horizontal_lines: List of horizontal line bounding boxes
            vertical_lines: List of vertical line bounding boxes
            
        Returns:
            List of intersection points
        """
        intersections = []
        
        for h_line in horizontal_lines:
            hx, hy, hw, hh = h_line
            h_y_center = hy + hh // 2
            
            for v_line in vertical_lines:
                vx, vy, vw, vh = v_line
                v_x_center = vx + vw // 2
                
                # Check if lines intersect
                if (hx <= v_x_center <= hx + hw and 
                    vy <= h_y_center <= vy + vh):
                    intersections.append((v_x_center, h_y_center))
        
        return intersections
    
    def _group_intersections_to_tables(self, intersections: List, image_shape: Tuple) -> List[TableRegion]:
        """Group intersections into table regions.
        
        Args:
            intersections: List of intersection points
            image_shape: Shape of the image
            
        Returns:
            List of table regions
        """
        if len(intersections) < 4:
            return []
        
        # Simple clustering by proximity
        tables = []
        used_intersections = set()
        
        for i, (x1, y1) in enumerate(intersections):
            if (x1, y1) in used_intersections:
                continue
            
            # Find nearby intersections
            nearby = [(x1, y1)]
            used_intersections.add((x1, y1))
            
            for j, (x2, y2) in enumerate(intersections[i+1:], i+1):
                if (x2, y2) in used_intersections:
                    continue
                
                # Check if points are close enough to be in same table
                if abs(x2 - x1) < 100 and abs(y2 - y1) < 100:
                    nearby.append((x2, y2))
                    used_intersections.add((x2, y2))
            
            # If we have enough points, create table region
            if len(nearby) >= 4:
                xs = [p[0] for p in nearby]
                ys = [p[1] for p in nearby]
                
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                # Estimate rows and columns
                rows = len(set(ys))  # Unique y coordinates
                cols = len(set(xs))  # Unique x coordinates
                
                table = TableRegion(
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    confidence=0.8,  # Could be improved with better analysis
                    rows=rows,
                    cols=cols
                )
                tables.append(table)
        
        return tables
    
    def _detect_columns(self, image: np.ndarray, text_blocks: List[TextBlock]) -> List[Tuple[int, int, int, int]]:
        """Detect column layout.
        
        Args:
            image: Binary image
            text_blocks: List of detected text blocks
            
        Returns:
            List of column bounding boxes
        """
        if not text_blocks:
            return []
        
        # Group text blocks by vertical position
        columns = []
        block_groups = self._group_blocks_by_y_position(text_blocks)
        
        for group in block_groups:
            if len(group) < 2:  # Need multiple blocks to form column
                continue
            
            # Find column boundaries
            x_coords = [block.bbox[0] for block in group]
            x_min, x_max = min(x_coords), max(x_coords)
            
            y_coords = [block.bbox[1] for block in group]
            y_min = min(y_coords)
            
            heights = [block.bbox[3] for block in group]
            y_max = max(y + h for y, h in zip(y_coords, heights))
            
            columns.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return columns
    
    def _group_blocks_by_y_position(self, text_blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """Group text blocks by similar y-position.
        
        Args:
            text_blocks: List of text blocks
            
        Returns:
            List of grouped text blocks
        """
        if not text_blocks:
            return []
        
        groups = []
        current_group = [text_blocks[0]]
        
        for block in text_blocks[1:]:
            # Check if block is in same row as current group
            current_y = current_group[0].bbox[1]
            block_y = block.bbox[1]
            
            if abs(block_y - current_y) < 50:  # Within 50 pixels
                current_group.append(block)
            else:
                groups.append(current_group)
                current_group = [block]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _classify_text_blocks(self, text_blocks: List[TextBlock], tables: List[TableRegion], 
                            columns: List[Tuple]) -> List[TextBlock]:
        """Classify text blocks by type.
        
        Args:
            text_blocks: List of text blocks
            tables: List of table regions
            columns: List of column regions
            
        Returns:
            List of classified text blocks
        """
        classified_blocks = []
        
        for block in text_blocks:
            block_type = "paragraph"  # Default type
            
            # Check if block is in a table
            for table in tables:
                if self._is_block_in_region(block.bbox, table.bbox):
                    block_type = "table"
                    break
            
            # Check if block is a header (top of page)
            if block.bbox[1] < 100:  # Within 100 pixels of top
                block_type = "header"
            
            # Check if block is a footer (bottom of page)
            # This would need page height, simplified for now
            if block.bbox[1] > 1000:  # Arbitrary threshold
                block_type = "footer"
            
            # Update block type
            block.text_type = block_type
            classified_blocks.append(block)
        
        return classified_blocks
    
    def _is_block_in_region(self, block_bbox: Tuple, region_bbox: Tuple) -> bool:
        """Check if a text block is within a region.
        
        Args:
            block_bbox: Text block bounding box
            region_bbox: Region bounding box
            
        Returns:
            True if block is within region
        """
        bx, by, bw, bh = block_bbox
        rx, ry, rw, rh = region_bbox
        
        # Check if block center is within region
        block_center_x = bx + bw // 2
        block_center_y = by + bh // 2
        
        return (rx <= block_center_x <= rx + rw and 
                ry <= block_center_y <= ry + rh)
    
    def _analyze_page_structure(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze overall page structure.
        
        Args:
            text_blocks: List of classified text blocks
            
        Returns:
            Dictionary with page structure analysis
        """
        if not text_blocks:
            return {"structure_type": "unknown", "confidence": 0.0}
        
        # Count block types
        type_counts = {}
        for block in text_blocks:
            type_counts[block.text_type] = type_counts.get(block.text_type, 0) + 1
        
        # Determine structure type
        if type_counts.get("table", 0) > len(text_blocks) * 0.5:
            structure_type = "table_heavy"
        elif type_counts.get("paragraph", 0) > len(text_blocks) * 0.7:
            structure_type = "text_heavy"
        else:
            structure_type = "mixed"
        
        return {
            "structure_type": structure_type,
            "confidence": 0.8,  # Could be improved with more analysis
            "block_counts": type_counts,
            "total_blocks": len(text_blocks)
        }