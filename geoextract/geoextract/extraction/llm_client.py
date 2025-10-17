from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..config import settings

try:  # optional
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # type: ignore

try:  # optional
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
    "You are a specialized AI assistant for extracting structured geological data from OCR text of legacy mining and exploration reports. "
    "Your task is to identify and extract specific geological entities with high precision.\n\n"
    "Focus on:\n"
    "1. Spatial coordinates in any format (decimal degrees, DMS, UTM, township-range-section)\n"
    "2. Mineral assay data (element, value, unit)\n"
    "3. Drill hole information (ID, depth intervals, lithology)\n"
    "4. Geological observations (rock types, structures, mineralization)\n\n"
    "Rules:\n"
    "- Only extract information explicitly stated in the text\n"
    "- Provide confidence scores (0-1) for each extraction\n"
    "- Flag ambiguous or potentially OCR-corrupted values\n"
    "- Preserve original units and notation\n"
    "- When coordinates appear in multiple formats, extract all versions\n"
    "- For tables, maintain row-column relationships\n\n"
    "Output valid JSON for coordinates and assay tables only for MVP."
)


def call_llm(prompt: str, model: Optional[str] = None) -> str:
    provider = settings.llm_provider
    model_name = model or settings.llm_model

    if provider == "ollama":
        if ollama is None:
            raise RuntimeError("ollama package not installed. Install with extras: poetry install -E llm")
        response = ollama.chat(model=model_name, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        return response.get("message", {}).get("content", "")
    elif provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Install with extras: poetry install -E llm")
        client = OpenAI(api_key=settings.openai_api_key)
        chat = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return chat.choices[0].message.content or ""
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
