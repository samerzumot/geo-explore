from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "ollama"  # ollama|openai|anthropic (anthropic not implemented yet)
    llm_model: str = "llama3.1:8b"
    openai_api_key: str | None = None

    ocr_engine: str = "both"  # paddle|tesseract|both
    ocr_language: str = "en"
    pdf_dpi: int = 300

    confidence_threshold: float = 0.5
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()