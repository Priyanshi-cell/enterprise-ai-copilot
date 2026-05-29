"""
core/db.py
Single source of truth for all database clients.
Both ingest.py and rag.py import from here — no more duplicate client handles.
"""

from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from core.config import get_settings


# ── SQLAlchemy ────────────────────────────────────────────────────────────────

Base = declarative_base()


@lru_cache(maxsize=1)
def _get_engine():
    settings = get_settings()
    connect_args = (
        {"check_same_thread": False}
        if settings.database_url.startswith("sqlite")
        else {}
    )
    return create_engine(settings.database_url, connect_args=connect_args)


@lru_cache(maxsize=1)
def _get_session_factory():
    return sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables on startup (idempotent)."""
    Base.metadata.create_all(bind=_get_engine())


# ── ChromaDB ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Single persistent ChromaDB client shared across the process.
    Multiple PersistentClient instances on the same path cause write conflicts.
    """
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_path)


@lru_cache(maxsize=1)
def get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    """
    BGE-small-en-v1.5 embedding model.
    Chosen for its MTEB performance vs. latency trade-off over the default
    all-MiniLM-L6-v2. See: https://huggingface.co/spaces/mteb/leaderboard
    """
    settings = get_settings()
    return SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model,
        device="cpu",
    )


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Cross-encoder reranker loaded once at startup.
    Used in Stage 2 of the retrieval pipeline to rerank the top-k candidates
    from vector search using full query-document interaction (vs. cosine sim).
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """
    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


def get_or_create_collection(company_name: str) -> chromadb.Collection:
    """
    Return the ChromaDB collection for a company, creating it if absent.
    Collection names are sanitised to ChromaDB's allowed character set.
    """
    safe_name = (
        company_name.strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    return get_chroma_client().get_or_create_collection(
        name=safe_name,
        embedding_function=get_embedding_fn(),
    )