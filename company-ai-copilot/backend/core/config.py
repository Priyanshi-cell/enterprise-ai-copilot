from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    app_name: str = "Enterprise AI Copilot"
    app_version: str = "2.0.0"
    debug: bool = False

    
    admin_username: str = "admin"
    admin_password: str = "changeme"
    jwt_secret: str = "replace-with-a-long-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    database_url: str = "sqlite:///./chat_history.db"

    chroma_path: str = "./chroma_db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = 20     
    rerank_top_n: int = 4         
    min_doc_length: int = 80      
    llm_model: str = "llama3"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    scraper_max_pages: int = 40
    scraper_timeout_ms: int = 60_000
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 300
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton — avoids re-parsing .env on every call."""
    return Settings()