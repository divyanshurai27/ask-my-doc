"""
Configuration management for the RAG system.
Loads environment variables and provides settings throughout the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ===== LLM Configuration =====
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o", env="LLM_MODEL")
    
    # ===== Embedding Configuration =====
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    
    # ===== ChromaDB Configuration =====
    chroma_db_path: str = Field(default="./data/chroma_db", env="CHROMA_DB_PATH")
    chroma_collection: str = Field(default="documents", env="CHROMA_COLLECTION")
    
    # ===== Retrieval Configuration =====
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")
    reranker_top_k: int = Field(default=5, env="RERANKER_TOP_K")
    confidence_threshold: float = Field(default=0.6, env="CONFIDENCE_THRESHOLD")
    bm25_weight: float = Field(default=0.4, env="BM25_WEIGHT")
    vector_weight: float = Field(default=0.6, env="VECTOR_WEIGHT")
    
    # ===== Chunking Configuration =====
    chunk_size: int = Field(default=600, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks in tokens")
    
    # ===== Evaluation Configuration =====
    ragas_batch_size: int = Field(default=5, env="RAGAS_BATCH_SIZE")
    ragas_llm_model: str = Field(default="gpt-4o", env="RAGAS_LLM_MODEL")
    
    # ===== Application Configuration =====
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


# Validate required settings
def validate_settings():
    """Validate that required settings are configured."""
    if not settings.openai_api_key or settings.openai_api_key == "sk-your-api-key-here":
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Please set it in .env file or environment variable."
        )


if __name__ == "__main__":
    # Print current settings (for debugging)
    print("Current Settings:")
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print(f"  ChromaDB Path: {settings.chroma_db_path}")
    print(f"  Top-K Retrieval: {settings.top_k_retrieval}")
    print(f"  Chunk Size: {settings.chunk_size}")
    print(f"  Chunk Overlap: {settings.chunk_overlap}")
    print(f"  Debug Mode: {settings.debug}")
