"""Configuration management for GeoExtract."""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Configuration
    llm_provider: Literal["ollama", "openai", "anthropic"] = Field(
        default="ollama", env="LLM_PROVIDER"
    )
    llm_model: str = Field(default="llama3.1:8b", env="LLM_MODEL")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # OCR Configuration
    ocr_engine: Literal["paddle", "tesseract", "both"] = Field(
        default="paddle", env="OCR_ENGINE"
    )
    ocr_confidence_threshold: float = Field(
        default=0.8, env="OCR_CONFIDENCE_THRESHOLD"
    )
    ocr_language: str = Field(default="en", env="OCR_LANGUAGE")
    
    # Processing Configuration
    pdf_dpi: int = Field(default=300, env="PDF_DPI")
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    batch_size: int = Field(default=10, env="BATCH_SIZE")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./geoextract.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Output Configuration
    default_output_format: str = Field(default="geojson", env="DEFAULT_OUTPUT_FORMAT")
    output_dir: Path = Field(default=Path("./output"), env="OUTPUT_DIR")
    temp_dir: Path = Field(default=Path("./temp"), env="TEMP_DIR")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(default=None, env="LOG_FILE")
    
    # Debug
    debug: bool = Field(default=False, env="DEBUG")
    save_intermediate_outputs: bool = Field(
        default=False, env="SAVE_INTERMEDIATE_OUTPUTS"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        
        # Create output and temp directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory if log file is specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()