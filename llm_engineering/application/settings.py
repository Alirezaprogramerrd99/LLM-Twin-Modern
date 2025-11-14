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
    # -- Embedding model ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


    # --- Services (we’ll wire them later) ---
    mongo_uri: Optional[str] = None       # keep simple for now
    qdrant_url: Optional[AnyUrl] = None   # URL gets validated if provided
    openai_api_key: Optional[str] = None

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