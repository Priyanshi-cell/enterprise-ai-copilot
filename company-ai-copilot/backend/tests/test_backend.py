"""
tests/test_backend.py
Pytest test suite for the Enterprise AI Copilot backend.

Run with:
    pytest tests/ -v

These tests use dependency injection overrides so they run without
a live Ollama instance, real ChromaDB data, or network access.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Ingest tests ──────────────────────────────────────────────────────────────

class TestSplitText:
    """Unit tests for the text chunking function."""

    def test_basic_chunking(self):
        from ingest import split_text
        text = "This is a sentence. " * 50   # 1000 chars, well above 1 chunk
        chunks = split_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_overlap_produces_continuity(self):
        from ingest import split_text
        # With overlap, consecutive chunks should share content
        text = "Word " * 200
        chunks = split_text(text, chunk_size=100, chunk_overlap=30)
        if len(chunks) >= 2:
            # The end of chunk[0] and start of chunk[1] should share tokens
            end_of_first = chunks[0][-30:]
            start_of_second = chunks[1][:60]
            assert any(word in start_of_second for word in end_of_first.split())

    def test_short_text_single_chunk(self):
        from ingest import split_text
        text = "Short sentence."
        chunks = split_text(text, chunk_size=512, chunk_overlap=64)
        # May return empty if shorter than min_doc_length, or one chunk
        assert len(chunks) <= 1

    def test_noise_chunks_filtered(self):
        from ingest import split_text
        # Chunks shorter than min_doc_length should be excluded
        text = "Hi. " * 5   # very short repeated sentences
        chunks = split_text(text, chunk_size=512, chunk_overlap=64)
        for chunk in chunks:
            assert len(chunk.strip()) >= 80, f"Short chunk leaked through: {chunk!r}"

    def test_empty_text(self):
        from ingest import split_text
        assert split_text("") == []
        assert split_text("   \n  ") == []


class TestChunkId:
    """Chunk IDs should be deterministic and content-addressable."""

    def test_same_content_same_id(self):
        from ingest import _chunk_id
        assert _chunk_id("acme", "hello world") == _chunk_id("acme", "hello world")

    def test_different_content_different_id(self):
        from ingest import _chunk_id
        assert _chunk_id("acme", "hello world") != _chunk_id("acme", "goodbye world")

    def test_different_company_different_id(self):
        from ingest import _chunk_id
        assert _chunk_id("acme", "hello") != _chunk_id("globex", "hello")


# ── RAG pipeline tests ────────────────────────────────────────────────────────

class TestCleanDocuments:
    """Unit tests for the document cleaning function."""

    def test_deduplication(self):
        from rag import _clean_docs
        docs = ["A" * 100, "A" * 100, "B" * 100]
        result = _clean_docs(docs)
        assert result.count("A" * 100) == 1

    def test_short_docs_filtered(self):
        from rag import _clean_docs
        docs = ["short", "A" * 100]
        result = _clean_docs(docs)
        assert "short" not in result
        assert "A" * 100 in result

    def test_empty_list(self):
        from rag import _clean_docs
        assert _clean_docs([]) == []


class TestQueryRewrite:
    """Query rewriting should fall back gracefully on LLM errors."""

    def test_fallback_on_error(self):
        from rag import rewrite_query
        with patch("rag.ollama.chat", side_effect=Exception("LLM offline")):
            result = rewrite_query("what does acme do?", "llama3")
        assert result == "what does acme do?"

    def test_returns_string(self):
        from rag import rewrite_query
        mock_response = {"message": {"content": "acme products and services"}}
        with patch("rag.ollama.chat", return_value=mock_response):
            result = rewrite_query("what does acme do?", "llama3")
        assert isinstance(result, str)
        assert len(result) > 0


# ── Auth tests ────────────────────────────────────────────────────────────────

class TestAuth:

    def test_create_and_decode_token(self):
        from auth import create_access_token
        from jose import jwt
        from core.config import get_settings
        settings = get_settings()
        token = create_access_token("testuser")
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        assert payload["sub"] == "testuser"

    def test_verify_user_correct(self):
        from auth import verify_user
        # Uses settings defaults
        assert verify_user("admin", "changeme") is True

    def test_verify_user_wrong_password(self):
        from auth import verify_user
        assert verify_user("admin", "wrongpassword") is False

    def test_verify_user_wrong_username(self):
        from auth import verify_user
        assert verify_user("hacker", "changeme") is False


# ── API integration tests ─────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Test client with mocked heavy dependencies."""
    from main import app
    from core.db import get_db

    # Override DB dependency with a mock session
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


class TestAPIRoutes:

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_login_success(self, client):
        r = client.post("/login", json={"username": "admin", "password": "changeme"})
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_login_failure(self, client):
        r = client.post("/login", json={"username": "admin", "password": "wrong"})
        assert r.status_code == 401

    def test_query_requires_auth(self, client):
        r = client.post("/query", json={"company": "acme", "query": "what do you do?"})
        assert r.status_code == 403

    def test_history_requires_auth(self, client):
        r = client.get("/history/acme")
        assert r.status_code == 403