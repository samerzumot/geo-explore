from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore


class PaddleOCREngine:
    def __init__(self, language: str = "en") -> None:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR not installed. Install with extras: poetry install -E ocr")
        self.ocr = PaddleOCR(lang=language, show_log=False)

    def run(self, images: List["Image.Image"]) -> List[Dict[str, Any]]:
        """Return list of OCR blocks with text and boxes per image.

        Each item: {"page": int, "blocks": [{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}]}
        """
        results: List[Dict[str, Any]] = []
        for page_index, img in enumerate(images):
            ocr_result: List[List[Tuple[List[List[float]], Tuple[str, float]]]] = self.ocr.ocr(img, cls=True)  # type: ignore
            page_blocks: List[Dict[str, Any]] = []
            for line in ocr_result:
                for box, (text, conf) in line:
                    x_coords = [pt[0] for pt in box]
                    y_coords = [pt[1] for pt in box]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    page_blocks.append({"text": text, "bbox": bbox, "confidence": float(conf)})
            results.append({"page": page_index + 1, "blocks": page_blocks})
        return results
