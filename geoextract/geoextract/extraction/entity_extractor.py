from __future__ import annotations

import json
from typing import Any, Dict, List

from .llm_client import call_llm

MVP_PROMPT = (
    "Extract only coordinates and assay tables from the following OCR text blocks.\n"
    "Return JSON with keys: coordinates: [...], assays: [{element, value, unit, context}], confidence_overall.\n"
    "Keep original units and notation; include per-item confidence in [0,1].\n\n"
)


def extract_entities_mvp(ocr_pages: List[Dict[str, Any]], confidence_threshold: float = 0.5) -> Dict[str, Any]:
    # Concatenate text with simple page markers
    texts: List[str] = []
    for page in ocr_pages:
        page_num = page.get("page", 0)
        lines = [blk.get("text", "") for blk in page.get("blocks", [])]
        texts.append(f"\n\n--- PAGE {page_num} ---\n" + "\n".join(lines))
    user_prompt = MVP_PROMPT + "\n" + "\n".join(texts)

    raw = call_llm(user_prompt)
    try:
        data = json.loads(raw)
    except Exception:
        # Best effort: wrap text
        data = {"coordinates": [], "assays": [], "confidence_overall": 0.0, "raw": raw}

    # Filter by confidence threshold if available
    coords = [c for c in data.get("coordinates", []) if c.get("confidence", 1.0) >= confidence_threshold]
    assays = [a for a in data.get("assays", []) if a.get("confidence", 1.0) >= confidence_threshold]

    return {"coordinates": coords, "assays": assays, "confidence_overall": data.get("confidence_overall", 0.0)}
