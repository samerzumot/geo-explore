from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from .paddle_engine import PaddleOCREngine
from .tesseract_engine import TesseractOCREngine


class OCRManager:
    def __init__(self, engine_preference: str = "both", language: str = "en") -> None:
        self.engine_preference = engine_preference
        self.language = language

    def run(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        if self.engine_preference in {"paddle", "both"}:
            try:
                paddle = PaddleOCREngine(language=self.language)
                return paddle.run(images)
            except Exception:
                if self.engine_preference == "paddle":
                    raise
        # Fallback to tesseract
        tess_lang = "eng" if self.language == "en" else self.language
        tesseract = TesseractOCREngine(language=tess_lang)
        return tesseract.run(images)
