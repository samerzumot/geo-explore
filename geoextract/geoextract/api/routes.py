from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, UploadFile

from ..preprocessing.pdf_handler import convert_to_images
from ..preprocessing.image_clean import preprocess_image
from ..ocr.ocr_manager import OCRManager
from ..extraction.entity_extractor import extract_entities_mvp
from ..config import settings

router = APIRouter()


@router.get("/")
async def root() -> Dict[str, str]:
    return {"message": "GeoExtract API running"}


@router.post("/extract")
async def extract(file: UploadFile) -> Dict[str, Any]:
    tmp_path = Path(f"/tmp/{file.filename}")
    content = await file.read()
    tmp_path.write_bytes(content)

    images = convert_to_images(tmp_path, dpi=settings.pdf_dpi)
    preprocessed = [preprocess_image(img) for img in images]
    ocr = OCRManager(engine_preference=settings.ocr_engine, language=settings.ocr_language)
    ocr_blocks = ocr.run(preprocessed)
    result = extract_entities_mvp(ocr_blocks, confidence_threshold=settings.confidence_threshold)
    return result
