"""Configuration management for RAGify."""
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_local_env() -> None:
    """Load environment variables from the closest .env file if available."""
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        # Fall back to the repository root when running from installed package
        repo_root = Path(__file__).resolve().parents[2] / ".env"
        dotenv_path = str(repo_root) if repo_root.exists() else ""

    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


_load_local_env()

# Try to load Streamlit secrets (cloud deployment)
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        try:
            for key, value in st.secrets.items():
                os.environ[key] = str(value)
        except Exception:
            # st.secrets raises when no secrets.toml is configured; safe to ignore locally
            pass
except ImportError:
    st = None  # Streamlit not installed or not running


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    
    # Model Configuration
    anthropic_model: str = Field(default="claude-3-5-sonnet-latest", alias="ANTHROPIC_MODEL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-large", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=3072, alias="EMBEDDING_DIM")
    
    # Chunking Configuration
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    
    # Vector Database Configuration
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="docs", alias="QDRANT_COLLECTION")
    
    # Inngest Configuration
    inngest_app_id: str = Field(default="rag_app", alias="INNGEST_APP_ID")
    inngest_api_base: str = Field(default="http://127.0.0.1:8288/v1", alias="INNGEST_API_BASE")
    
    # Upload Configuration
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")
    
    class Config:
        env_file = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parents[2] / ".env")
        case_sensitive = False
        populate_by_name = True


# Global settings instance
settings = Settings()
