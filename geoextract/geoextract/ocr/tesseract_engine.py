from __future__ import annotations

from typing import Any, Dict, List

import pytesseract
from PIL import Image


class TesseractOCREngine:
    def __init__(self, language: str = "eng") -> None:
        self.language = language

    def run(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for page_index, img in enumerate(images):
            data = pytesseract.image_to_data(img, lang=self.language, output_type=pytesseract.Output.DICT)
            page_blocks: List[Dict[str, Any]] = []
            n = len(data.get("text", []))
            for i in range(n):
                text = data["text"][i]
                if not text:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                conf_str = data.get("conf", ["-1"])[i]
                try:
                    conf = float(conf_str) / 100.0 if conf_str not in {"-1", ""} else 0.0
                except ValueError:
                    conf = 0.0
                page_blocks.append({"text": text, "bbox": [x, y, x + w, y + h], "confidence": conf})
            results.append({"page": page_index + 1, "blocks": page_blocks})
        return results
