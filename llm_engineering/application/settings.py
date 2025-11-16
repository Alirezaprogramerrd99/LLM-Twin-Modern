from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl
from typing import Optional

# Application settings using pydantic-settings for structured configuration

class Settings(BaseSettings):
    # --- App ---
    app_name: str = "LLM Handbook Ground-Up - LLM Twin"
    app_env: str = "development"          # e.g., development / staging / production
    debug: bool = True
    
    # Qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    # Default collection name
    qdrant_collection: str = "documents"

    # Embeddings (MiniLM-L6-v2 is 384)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Switch: use qdrant if URL is set
    use_qdrant: bool = False
    
    
    # --- MongoDB ---
    mongo_uri: str | None = None
    mongo_db_name: str = "llm_twin"
    mongo_collection_docs: str = "documents"
    mongo_collection_history: str = "interactions"
    use_mongo: bool = False


    # --- Services (we’ll wire them later) ---
    mongo_uri: Optional[str] = None       # keep simple for now
    qdrant_url: Optional[AnyUrl] = None   # URL gets validated if provided
    openai_api_key: Optional[str] = None
    
    
    # --- LLM: Ollama ---
    use_ollama: bool = True
    ollama_model: str = "qwen3:4b"
    ollama_host: str = "http://localhost:11434"

    # pydantic v2 / pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",     # auto-load from your .env
        case_sensitive=False,  # .env keys can be upper/lower
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Cache settings so we don’t re-parse .env on every request."""
    return Settings()