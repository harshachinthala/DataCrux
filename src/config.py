"""
Configuration management for Excel RAG Pipeline
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (required for queries)"
    )
    
    # Embedding Configuration
    embedding_model: Literal["huggingface", "openai"] = Field(
        default="huggingface",
        description="Embedding model provider"
    )
    huggingface_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    
    # LLM Configuration
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI LLM model"
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation"
    )
    llm_max_tokens: int = Field(
        default=2000,
        ge=100,
        le=4000,
        description="Maximum tokens in LLM response"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum characters per chunk"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks"
    )
    chunking_strategy: Literal["row", "group", "sheet", "sliding"] = Field(
        default="row",
        description="Chunking strategy for Excel data"
    )
    
    # Retrieval Configuration
    top_k_retrieval: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of chunks to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.0,  # Changed from 0.5 to 0.0 to retrieve all chunks
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval (0.0 = no filtering)"
    )
    
    # Paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Base data directory"
    )
    upload_dir: Path = Field(
        default=Path("data/uploads"),
        description="Directory for uploaded files"
    )
    vector_db_dir: Path = Field(
        default=Path("data/vector_db"),
        description="Directory for FAISS index"
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )
    
    # UI Configuration
    gradio_share: bool = Field(
        default=False,
        description="Enable Gradio public sharing"
    )
    gradio_port: int = Field(
        default=7860,
        ge=1024,
        le=65535,
        description="Gradio server port"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        Path("data/samples").mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
