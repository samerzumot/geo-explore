from __future__ import annotations

from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from PIL import Image


def convert_to_images(path: Path, dpi: int = 300) -> List[Image.Image]:
    """Convert a PDF or image file to a list of PIL Images."""
    suffix = path.suffix.lower()
    if suffix in {".pdf"}:
        images = convert_from_path(str(path), dpi=dpi)
        return images
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return [Image.open(path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported input type: {suffix}")
