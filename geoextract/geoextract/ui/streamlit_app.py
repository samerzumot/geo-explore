from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

from ..config import settings
from ..preprocessing.pdf_handler import convert_to_images
from ..preprocessing.image_clean import preprocess_image
from ..ocr.ocr_manager import OCRManager
from ..extraction.entity_extractor import extract_entities_mvp

st.set_page_config(page_title="GeoExtract", layout="wide")
st.title("GeoExtract – Geological Report Extractor (MVP)")

llm_provider = st.sidebar.selectbox("LLM Provider", ["ollama", "openai"], index=0)
llm_model = st.sidebar.text_input("LLM Model", value=settings.llm_model)
ocr_engine = st.sidebar.selectbox("OCR Engine", ["paddle", "tesseract", "both"], index=2)
dpi = st.sidebar.number_input("PDF DPI", min_value=100, max_value=600, value=settings.pdf_dpi, step=25)
conf_thr = st.sidebar.slider("Confidence threshold", 0.0, 1.0, settings.confidence_threshold, 0.05)

uploaded = st.file_uploader("Upload PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False)

if uploaded is not None:
    tmp_path = Path(f"/tmp/{uploaded.name}")
    tmp_path.write_bytes(uploaded.getvalue())

    settings.llm_provider = llm_provider
    settings.llm_model = llm_model
    settings.ocr_engine = ocr_engine
    settings.pdf_dpi = int(dpi)
    settings.confidence_threshold = float(conf_thr)

    with st.spinner("Processing..."):
        images = convert_to_images(tmp_path, dpi=settings.pdf_dpi)
        preprocessed = [preprocess_image(img) for img in images]
        ocr = OCRManager(engine_preference=settings.ocr_engine, language=settings.ocr_language)
        ocr_blocks = ocr.run(preprocessed)
        result = extract_entities_mvp(ocr_blocks, confidence_threshold=settings.confidence_threshold)

    st.success("Done")
    st.subheader("Extraction results")
    st.json(result)
