"""
main.py
Enterprise AI Knowledge Copilot — FastAPI Backend
==================================================

Author  : Final-Year AI & Data Science Project
Stack   : FastAPI · ChromaDB · Ollama · BeautifulSoup · pypdf · SQLAlchemy

Endpoints
---------
GET  /              → health check
POST /login         → authenticate with username + password
POST /ingest_url    → crawl a website and store content in ChromaDB
POST /upload_pdf    → extract text from a PDF and store in ChromaDB
POST /query         → ask a question, get a RAG-powered answer

Design principles
-----------------
- Every endpoint ALWAYS returns valid JSON — never a raw 500 error
- Every endpoint has a top-level try/except as a safety net
- Pydantic models validate all incoming request data
- SQLAlchemy persists every Q&A pair to SQLite for chat history
- All activity is logged with timestamps for easy debugging
- No Playwright · No Torch · No Transformers · No async complexity
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Local modules — each lives in the same backend/ folder
import ollama

from auth import verify_user
from ingest import ingest_text
from rag import query_rag, retrieve_context
from scraper import scrape_website
from pdf_reader import extract_pdf_text

# LLM model name — must match rag.py
LLM_MODEL = "llama3"


# =============================================================================
# LOGGING
# =============================================================================
# Sets up a simple logger that prints to the terminal with timestamps.
# Every module uses logging.getLogger(__name__) — all output appears here.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE SETUP  (SQLAlchemy + SQLite)
# =============================================================================
# SQLite is a file-based database — no server needed, perfect for a project.
# The file chat_history.db is created automatically in the backend/ folder.
# To switch to PostgreSQL later, just change DATABASE_URL.

DATABASE_URL = "sqlite:///./chat_history.db"

# create_engine — creates the database connection
# check_same_thread=False is required for SQLite when used with FastAPI
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Base — all ORM model classes inherit from this
Base = declarative_base()

# SessionLocal — a factory that creates new database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# =============================================================================
# DATABASE MODEL
# =============================================================================

class ChatHistory(Base):
    """
    Stores every question and answer pair for audit and history.

    Each row represents one complete Q&A interaction, including
    which company's knowledge base was queried and when it happened.
    """
    __tablename__ = "chat_history"

    id         = Column(Integer, primary_key=True, index=True)
    company    = Column(String(255), nullable=False, index=True)
    question   = Column(Text, nullable=False)
    answer     = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# Create the table if it does not already exist
# This runs once when the server starts — safe to call multiple times
Base.metadata.create_all(bind=engine)
logger.info("Database tables ready — chat_history.db initialised")


# =============================================================================
# HELPER — save to chat history
# =============================================================================

def save_to_history(company: str, question: str, answer: str) -> None:
    """
    Persist a Q&A pair to the SQLite database.

    Called after every successful /query response.
    Wrapped in its own try/except so a DB error never breaks the API response.

    Args:
        company  : The company namespace that was queried.
        question : The user's original question.
        answer   : The generated answer from Ollama.
    """
    db = SessionLocal()
    try:
        record = ChatHistory(
            company=company.strip().lower().replace(" ", "_"),
            question=question,
            answer=answer,
        )
        db.add(record)
        db.commit()
        logger.info("Chat history saved | company=%s", company)

    except Exception as e:
        db.rollback()
        # Log the error but do NOT raise — history failure must not block the response
        logger.error("Failed to save chat history: %s", str(e))

    finally:
        db.close()


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Enterprise AI Knowledge Copilot",
    version="1.0.0",
    description=(
        "RAG-powered knowledge assistant. "
        "Ingest websites and PDFs, then ask questions answered from your own data."
    ),
)

# CORS — allows the Streamlit frontend (port 8501) to call this API.
# In production, replace allow_origins=["*"] with your exact frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST MODELS  (Pydantic)
# =============================================================================
# Pydantic automatically validates incoming JSON and returns clear error
# messages if required fields are missing or the wrong type.

class LoginRequest(BaseModel):
    """Request body for POST /login"""
    username: str
    password: str

    @field_validator("username", "password")
    @classmethod
    def must_not_be_empty(cls, value: str, info) -> str:
        """Reject blank strings — prevents accidental empty logins."""
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip()


class URLRequest(BaseModel):
    """Request body for POST /ingest_url"""
    company: str
    url: str

    @field_validator("url")
    @classmethod
    def url_must_start_with_http(cls, value: str) -> str:
        """Ensure a valid URL is provided — catches common mistakes."""
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value

    @field_validator("company")
    @classmethod
    def company_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("company must not be empty")
        return value.strip()


class QueryRequest(BaseModel):
    """Request body for POST /query"""
    company: str
    query: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value.strip()

    @field_validator("company")
    @classmethod
    def company_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("company must not be empty")
        return value.strip()


# =============================================================================
# GLOBAL EXCEPTION HANDLER
# =============================================================================
# Catches any unhandled exception that slips past a route's own try/except.
# Ensures the client always receives JSON instead of FastAPI's default HTML 500.

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected server error occurred. Please try again.",
        },
    )


# =============================================================================
# ENDPOINT 1 — Health check
# =============================================================================

@app.get("/", summary="Health check")
def root():
    """
    Confirms the API server is running.

    Use this to verify the backend is reachable before making other calls.

    Returns:
        JSON with status, project name, and a link to the interactive API docs.
    """
    logger.info("Health check called")
    return {
        "status": "ok",
        "project": "Enterprise AI Knowledge Copilot",
        "version": "1.0.0",
        "docs": "/docs",
        "message": "API is running — visit /docs to explore all endpoints",
    }


# =============================================================================
# ENDPOINT 2 — Login
# =============================================================================

@app.post("/login", summary="Authenticate with username and password")
def login(data: LoginRequest):
    """
    Verify a username and password against the configured credentials.

    Credentials are set in auth.py via environment variables:
        ADMIN_USERNAME (default: admin)
        ADMIN_PASSWORD (default: admin123)

    Request body:
        { "username": "admin", "password": "admin123" }

    Returns:
        { "success": true,  "message": "Login successful"          }  — correct
        { "success": false, "message": "Invalid username or..."    }  — wrong
        { "success": false, "message": "Login failed due to..."    }  — server error
    """
    logger.info("Login attempt | user=%s", data.username)

    try:
        is_valid = verify_user(data.username, data.password)

        if is_valid:
            logger.info("Login successful | user=%s", data.username)
            return {
                "success": True,
                "message": "Login successful",
            }
        else:
            logger.warning("Login failed — wrong credentials | user=%s", data.username)
            return {
                "success": False,
                "message": "Invalid username or password. Please try again.",
            }

    except Exception as e:
        logger.error("Login endpoint error: %s", str(e))
        return {
            "success": False,
            "message": "Login failed due to a server error. Please try again.",
        }


# =============================================================================
# ENDPOINT 3 — Ingest URL
# =============================================================================

@app.post("/ingest_url", summary="Crawl a website and index its content")
def ingest_url(data: URLRequest):
    """
    Scrape a website using BFS crawling, then chunk and store the
    extracted text in ChromaDB under the specified company namespace.

    Each company gets its own isolated collection in ChromaDB —
    "acme" and "globex" never share or mix knowledge.

    Re-ingesting the same URL is safe — duplicate chunks are skipped
    automatically (content-addressable IDs with SHA-256).

    Request body:
        { "company": "mycompany", "url": "https://example.com" }

    Returns (success):
        {
            "success": true,
            "chunks": 42,
            "chunks_total": 42,
            "pages_scraped": 5,
            "message": "Successfully indexed 42 chunks from 5 pages"
        }

    Returns (failure):
        { "success": false, "chunks": 0, "pages_scraped": 0, "message": "..." }
    """
    logger.info("Ingest URL | company=%s | url=%s", data.company, data.url)

    # ── Step 1: Scrape the website ────────────────────────────────────────────
    try:
        scraped = scrape_website(data.url)

    except Exception as e:
        # scrape_website raises an Exception when the site blocks all requests
        # or when fewer than MIN_TOTAL_LINES lines were extracted
        logger.error("Scraping failed | url=%s | error=%s", data.url, str(e))
        return {
            "success": False,
            "chunks": 0,
            "pages_scraped": 0,
            "message": str(e),
        }

    # ── Step 2: Validate scraped content ─────────────────────────────────────
    text = scraped.get("text", "").strip()
    pages_scraped = scraped.get("pages_scraped", 0)

    if not text:
        logger.warning("Empty scrape result for URL: %s", data.url)
        return {
            "success": False,
            "chunks": 0,
            "pages_scraped": pages_scraped,
            "message": (
                "No readable text could be extracted from this URL. "
                "The site may require JavaScript. Try uploading a PDF instead."
            ),
        }

    # ── Step 3: Chunk and store in ChromaDB ───────────────────────────────────
    try:
        result = ingest_text(company_name=data.company, text=text)

    except Exception as e:
        logger.error("Ingestion failed | company=%s | error=%s", data.company, str(e))
        return {
            "success": False,
            "chunks": 0,
            "pages_scraped": pages_scraped,
            "message": f"Text was scraped but storage failed: {str(e)}",
        }

    # ── Step 4: Return success summary ────────────────────────────────────────
    chunks_added = result.get("chunks_added", 0)
    chunks_total = result.get("chunks_total", 0)

    logger.info(
        "Ingest URL complete | company=%s | new_chunks=%d | total=%d | pages=%d",
        data.company, chunks_added, chunks_total, pages_scraped,
    )

    return {
        "success": True,
        "chunks": chunks_added,
        "chunks_total": chunks_total,
        "pages_scraped": pages_scraped,
        "message": (
            f"Successfully indexed {chunks_added} new chunks "
            f"from {pages_scraped} pages "
            f"({chunks_total} total chunks in knowledge base)"
        ),
    }


# =============================================================================
# ENDPOINT 4 — Upload PDF
# =============================================================================

@app.post("/upload_pdf", summary="Extract text from a PDF and index its content")
async def upload_pdf(company: str, file: UploadFile = File(...)):
    """
    Accept a PDF file upload, extract all text using pypdf,
    then chunk and store the content in ChromaDB.

    The company query parameter identifies which knowledge base to store into:
        POST /upload_pdf?company=mycompany

    Works with text-based PDFs (reports, documentation, articles).
    Does NOT work with scanned/image PDFs — those contain no extractable text.

    Returns (success):
        {
            "success": true,
            "chunks": 18,
            "chunks_total": 60,
            "pages": 4,
            "filename": "report.pdf",
            "message": "PDF indexed — 18 chunks from 4 pages"
        }

    Returns (failure):
        { "success": false, "chunks": 0, "pages": 0, "message": "..." }
    """
    logger.info("PDF upload | company=%s | filename=%s", company, file.filename)

    # ── Validate company name ─────────────────────────────────────────────────
    if not company or not company.strip():
        return {
            "success": False,
            "chunks": 0,
            "pages": 0,
            "message": "Company name must not be empty. Add ?company=yourname to the URL.",
        }

    # ── Validate file type ────────────────────────────────────────────────────
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        logger.warning("Non-PDF file rejected: %s", filename)
        return {
            "success": False,
            "chunks": 0,
            "pages": 0,
            "message": f"Only PDF files are accepted. Received: {filename}",
        }

    # ── Step 1: Read the uploaded file bytes ──────────────────────────────────
    try:
        pdf_bytes = await file.read()

    except Exception as e:
        logger.error("Failed to read uploaded file: %s", str(e))
        return {
            "success": False,
            "chunks": 0,
            "pages": 0,
            "message": "Could not read the uploaded file. Please try again.",
        }

    if not pdf_bytes:
        return {
            "success": False,
            "chunks": 0,
            "pages": 0,
            "message": "The uploaded file is empty.",
        }

    # ── Step 2: Extract text from PDF ─────────────────────────────────────────
    try:
        extraction = extract_pdf_text(pdf_bytes)

    except Exception as e:
        logger.error("PDF extraction error: %s", str(e))
        return {
            "success": False,
            "chunks": 0,
            "pages": 0,
            "message": f"Failed to read the PDF: {str(e)}",
        }

    text = extraction.get("text", "").strip()
    page_count = extraction.get("pages", 0)

    if not text:
        logger.warning(
            "PDF had no extractable text | file=%s | pages=%d",
            filename, page_count,
        )
        return {
            "success": False,
            "chunks": 0,
            "pages": page_count,
            "message": (
                "No text could be extracted from this PDF. "
                "It may be a scanned document (image-only). "
                "Try copying text from the PDF manually and uploading as .txt."
            ),
        }

    # ── Step 3: Chunk and store in ChromaDB ───────────────────────────────────
    try:
        result = ingest_text(company_name=company, text=text)

    except Exception as e:
        logger.error("Ingestion failed after PDF extraction: %s", str(e))
        return {
            "success": False,
            "chunks": 0,
            "pages": page_count,
            "message": f"Text extracted but storage failed: {str(e)}",
        }

    # ── Step 4: Return success summary ────────────────────────────────────────
    chunks_added = result.get("chunks_added", 0)
    chunks_total = result.get("chunks_total", 0)

    logger.info(
        "PDF ingested | company=%s | file=%s | pages=%d | new_chunks=%d | total=%d",
        company, filename, page_count, chunks_added, chunks_total,
    )

    return {
        "success": True,
        "chunks": chunks_added,
        "chunks_total": chunks_total,
        "pages": page_count,
        "filename": filename,
        "message": (
            f"PDF indexed — {chunks_added} new chunks from {page_count} pages "
            f"({chunks_total} total chunks in knowledge base)"
        ),
    }


# =============================================================================
# ENDPOINT 5 — Query
# =============================================================================

@app.post("/query", summary="Ask a question answered from the company knowledge base")
def query(data: QueryRequest):
    """
    Run the full RAG (Retrieval-Augmented Generation) pipeline:

        1. Embed the user's question using ChromaDB's embedding model
        2. Retrieve the most relevant document chunks from the collection
        3. Inject those chunks as context into a prompt
        4. Send the prompt to Ollama (Llama 3) and stream back the answer
        5. Save the Q&A pair to SQLite chat history

    The answer is grounded in the company's ingested documents.
    If no relevant context is found, the model says so instead of hallucinating.

    Request body:
        { "company": "mycompany", "query": "What products do you offer?" }

    Returns (success):
        {
            "success": true,
            "answer": "Based on the knowledge base, the company offers...",
            "sources": ["chunk text 1", "chunk text 2", ...]
        }

    Returns (failure):
        { "success": false, "answer": "...", "sources": [] }
    """
    logger.info("Query | company=%s | query=%.80s", data.company, data.query)

    try:
        # Run the RAG pipeline — defined in rag.py
        result = query_rag(
            company_name=data.company,
            user_query=data.query,
        )

        answer  = result.get("answer", "No answer was generated.")
        sources = result.get("sources", [])

        # ── Persist to chat history (non-blocking) ────────────────────────────
        # save_to_history catches its own exceptions, so a DB error here
        # will never cause this endpoint to return a failure response.
        save_to_history(
            company=data.company,
            question=data.query,
            answer=answer,
        )

        logger.info(
            "Query answered | company=%s | sources=%d | answer_len=%d",
            data.company, len(sources), len(answer),
        )

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
        }

    except Exception as e:
        logger.error("Query endpoint error | company=%s | error=%s", data.company, str(e))
        return {
            "success": False,
            "answer": (
                "Sorry, I could not process your query. "
                "Please check that Ollama is running and try again."
            ),
            "sources": [],
        }


# =============================================================================
# ENDPOINT 6 — Query Stream
# =============================================================================

@app.post("/query/stream", summary="Stream a RAG answer token by token")
async def query_stream(data: QueryRequest):
    """
    Streaming version of /query using Server-Sent Events (SSE).

    How it works:
        1. retrieve_context() runs ChromaDB retrieval synchronously (fast)
        2. If retrieval fails, stream a single error token then done
        3. Call ollama.chat() with stream=True to get a token generator
        4. Yield each token as:  data: {"token": "word"}\n\n
        5. After all tokens, yield:  data: {"done": true, "sources": [...]}\n\n
        6. Save the full answer to SQLite chat history

    The Streamlit frontend reads these lines with response.iter_lines()
    and appends each token to the displayed text in real time.

    No WebSockets. No special libraries. Plain HTTP chunked transfer.
    """
    logger.info(
        "Stream query | company=%s | query=%.80s",
        data.company, data.query,
    )

    def generate():
        # ── Retrieve context (ChromaDB — fast) ────────────────────────────────
        ctx = retrieve_context(
            company_name=data.company,
            user_query=data.query,
        )

        # ── Handle retrieval failure ───────────────────────────────────────────
        if ctx["error"]:
            logger.warning(
                "Stream retrieval failed | company=%s | error=%s",
                data.company, ctx["error"],
            )
            yield f'data: {json.dumps({"token": ctx["error"]})}\n\n'
            yield f'data: {json.dumps({"done": True, "sources": []})}\n\n'
            return

        prompt  = ctx["prompt"]
        sources = ctx["sources"]
        full_answer = []

        # ── Stream tokens from Ollama ──────────────────────────────────────────
        try:
            # stream=True makes ollama.chat() return an iterator of chunks
            # instead of blocking until the full response is ready
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            for chunk in stream:
                # Each chunk: {"message": {"content": "token"}, "done": False}
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_answer.append(token)
                    # Yield this token immediately to the client
                    yield f'data: {json.dumps({"token": token})}\n\n'

        except Exception as e:
            err = str(e).lower()
            if "connection" in err or "refused" in err:
                msg = "Ollama is not running. Please run: ollama serve"
            elif "not found" in err:
                msg = f"Model not installed. Run: ollama pull {LLM_MODEL}"
            else:
                msg = f"Generation error: {str(e)}"

            logger.error("Ollama stream error: %s", str(e))
            yield f'data: {json.dumps({"token": msg})}\n\n'
            full_answer = [msg]

        # ── Send done signal with sources ──────────────────────────────────────
        yield f'data: {json.dumps({"done": True, "sources": sources})}\n\n'

        # ── Persist full answer to chat history ────────────────────────────────
        complete_answer = "".join(full_answer)
        save_to_history(
            company=data.company,
            question=data.query,
            answer=complete_answer,
        )

        logger.info(
            "Stream complete | company=%s | sources=%d",
            data.company, len(sources),
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if deployed
        },
    )


# =============================================================================
# ENDPOINT 7 — Chat history by company
# =============================================================================

@app.get("/history/{company}", summary="Get saved chat history for a company")
def get_history(company: str):
    """Return the last 50 saved Q&A pairs for a company from SQLite."""
    try:
        safe_company = company.strip().lower().replace(" ", "_")
        db      = SessionLocal()
        records = (
            db.query(ChatHistory)
            .filter(ChatHistory.company == safe_company)
            .order_by(ChatHistory.created_at.desc())
            .limit(50)
            .all()
        )
        db.close()

        return {
            "success": True,
            "company": safe_company,
            "total":   len(records),
            "history": [
                {
                    "id":       r.id,
                    "question": r.question,
                    "answer":   r.answer,
                    "time":     str(r.created_at),
                }
                for r in records
            ],
        }

    except Exception as e:
        logger.error("History endpoint error: %s", str(e))
        return {"success": False, "message": str(e), "history": []}


# =============================================================================
# ENHANCEMENT 1 — Knowledge Base Stats
# GET /stats/{company}
# =============================================================================
# Returns how many chunks are stored for a company, what model is being used,
# and basic system info. Shows up in the frontend dashboard panel.
# Purely additive — does not touch any existing endpoint.
# =============================================================================

@app.get("/stats/{company}", summary="Knowledge base statistics for a company")
def get_stats(company: str):
    """
    Return chunk count, model info, and DB size for a company.

    Used by the frontend to show a live knowledge base health panel.
    Calls ChromaDB directly — same client as ingest.py and rag.py.

    Returns:
        {
            "success": true,
            "company": "openai",
            "chunks":  42,
            "model":   "llama3",
            "db_path": "./chroma_db"
        }
    """
    try:
        import re
        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        # Sanitise name — identical logic to ingest.py and rag.py
        safe = company.strip().lower()
        safe = safe.replace(" ", "_").replace("-", "_")
        safe = re.sub(r"[^a-z0-9_]", "", safe)
        if len(safe) < 3:
            safe = safe + "_kb"
        safe = safe[:63]

        client = chromadb.PersistentClient(path="./chroma_db")
        emb_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        collection = client.get_or_create_collection(
            name=safe, embedding_function=emb_fn
        )
        chunk_count = collection.count()

        logger.info(
            "Stats | company=%s | chunks=%d", safe, chunk_count
        )

        return {
            "success":  True,
            "company":  safe,
            "chunks":   chunk_count,
            "model":    LLM_MODEL,
            "embedding_model": "all-MiniLM-L6-v2",
            "db_path":  "./chroma_db",
            "has_data": chunk_count > 0,
        }

    except Exception as e:
        logger.error("Stats error | company=%s | %s", company, str(e))
        return {
            "success": False,
            "company": company,
            "chunks":  0,
            "message": str(e),
        }


# =============================================================================
# ENHANCEMENT 2 — List All Companies
# GET /companies
# =============================================================================
# Lists every company namespace currently stored in ChromaDB.
# Useful for the frontend dropdown and for demos — shows the system
# is multi-tenant at a glance.
# =============================================================================

@app.get("/companies", summary="List all company knowledge bases")
def list_companies():
    """
    Return all ChromaDB collection names (= all company namespaces).

    Each collection = one company's isolated knowledge base.
    This endpoint makes the multi-tenant architecture visible in the UI.

    Returns:
        {
            "success":   true,
            "companies": ["openai", "tesla", "google"],
            "total":     3
        }
    """
    try:
        import chromadb

        client      = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()

        # list_collections() returns Collection objects — extract names
        names = sorted([c.name for c in collections])

        logger.info("Companies listed | total=%d", len(names))

        return {
            "success":   True,
            "companies": names,
            "total":     len(names),
        }

    except Exception as e:
        logger.error("List companies error: %s", str(e))
        return {
            "success":   False,
            "companies": [],
            "total":     0,
            "message":   str(e),
        }


# =============================================================================
# ENHANCEMENT 3 — Delete Company Knowledge Base
# DELETE /company/{company}
# =============================================================================
# Deletes the ChromaDB collection for a company — all vectors gone.
# Keeps the SQLite chat history rows (audit trail preserved).
# Safe to call multiple times — returns success even if collection
# did not exist.
# =============================================================================

@app.delete("/company/{company}", summary="Delete a company knowledge base")
def delete_company(company: str):
    """
    Delete all stored vectors for a company from ChromaDB.

    The SQLite chat history rows are NOT deleted — they form an audit
    trail and are kept for the project report / demo.

    After deletion, the company namespace can be re-ingested fresh.

    Returns:
        { "success": true, "message": "Knowledge base deleted for: openai" }
    """
    try:
        import re
        import chromadb

        safe = company.strip().lower()
        safe = safe.replace(" ", "_").replace("-", "_")
        safe = re.sub(r"[^a-z0-9_]", "", safe)
        if len(safe) < 3:
            safe = safe + "_kb"
        safe = safe[:63]

        client = chromadb.PersistentClient(path="./chroma_db")

        # get_collection raises if it does not exist — handle gracefully
        try:
            client.delete_collection(name=safe)
            logger.info("Deleted collection | company=%s", safe)
            message = f"Knowledge base deleted for: {safe}"
        except Exception:
            message = f"No knowledge base found for: {safe} (nothing to delete)"

        return {
            "success": True,
            "company": safe,
            "message": message,
        }

    except Exception as e:
        logger.error("Delete error | company=%s | %s", company, str(e))
        return {
            "success": False,
            "company": company,
            "message": str(e),
        }


# =============================================================================
# ENHANCEMENT 4 — Ollama Health Check
# GET /ollama/status
# =============================================================================
# Checks whether Ollama is running and whether llama3 is available.
# The frontend uses this to show a separate "LLM status" badge.
# Purely a GET — no side effects.
# =============================================================================

@app.get("/ollama/status", summary="Check Ollama server and model availability")
def ollama_status():
    """
    Ping Ollama and verify the configured model is installed.

    Returns separate flags for server status and model status so the
    frontend can show accurate error messages to the user.

    Returns:
        {
            "success":        true,
            "server_running": true,
            "model_available": true,
            "model":          "llama3",
            "message":        "Ollama is ready"
        }
    """
    try:
        # list() fetches available models — if this works, server is running
        models_response = ollama.list()

        # models_response is a dict with a "models" key
        available_models = models_response.get("models", [])
        model_names = [m.get("name", "") for m in available_models]

        # Check if our model is present (name may be "llama3" or "llama3:latest")
        model_available = any(
            LLM_MODEL in name for name in model_names
        )

        logger.info(
            "Ollama status | running=True | model=%s | available=%s",
            LLM_MODEL, model_available,
        )

        return {
            "success":         True,
            "server_running":  True,
            "model_available": model_available,
            "model":           LLM_MODEL,
            "available_models": model_names,
            "message": (
                "Ollama is ready"
                if model_available
                else f"Server running but '{LLM_MODEL}' not found — run: ollama pull {LLM_MODEL}"
            ),
        }

    except Exception as e:
        err = str(e).lower()
        if "connection" in err or "refused" in err:
            msg = "Ollama server is not running — run: ollama serve"
        else:
            msg = f"Ollama error: {str(e)}"

        logger.warning("Ollama not reachable: %s", str(e))

        return {
            "success":         False,
            "server_running":  False,
            "model_available": False,
            "model":           LLM_MODEL,
            "message":         msg,
        }


# =============================================================================
# ENHANCEMENT 5 — System Dashboard
# GET /dashboard
# =============================================================================
# Single endpoint that aggregates everything the frontend dashboard
# panel needs in one call: backend status, DB record count,
# ChromaDB collection list, and Ollama status.
# Reduces the number of round trips from the frontend.
# =============================================================================

@app.get("/dashboard", summary="Full system status for the dashboard panel")
def dashboard():
    """
    Aggregate health + stats for the frontend dashboard in one call.

    Returns:
        {
            "backend":       { "status": "ok", "version": "1.0.0" },
            "database":      { "total_queries": 12, "companies_tracked": 2 },
            "vector_store":  { "total_collections": 2, "companies": ["openai", "tesla"] },
            "llm":           { "server_running": true, "model_available": true }
        }
    """
    result = {}

    # ── Backend ───────────────────────────────────────────────────────────────
    result["backend"] = {
        "status":  "ok",
        "version": "1.0.0",
        "model":   LLM_MODEL,
    }

    # ── SQLite database ───────────────────────────────────────────────────────
    try:
        db    = SessionLocal()
        total = db.query(ChatHistory).count()
        companies_tracked = (
            db.query(ChatHistory.company).distinct().count()
        )
        db.close()
        result["database"] = {
            "total_queries":      total,
            "companies_tracked":  companies_tracked,
            "status":             "ok",
        }
    except Exception as e:
        result["database"] = {"status": "error", "message": str(e)}

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        cols   = client.list_collections()
        names  = sorted([c.name for c in cols])
        result["vector_store"] = {
            "total_collections": len(names),
            "companies":         names,
            "status":            "ok",
        }
    except Exception as e:
        result["vector_store"] = {"status": "error", "message": str(e)}

    # ── Ollama ────────────────────────────────────────────────────────────────
    try:
        models_response = ollama.list()
        available       = models_response.get("models", [])
        model_names     = [m.get("name", "") for m in available]
        model_ok        = any(LLM_MODEL in n for n in model_names)
        result["llm"] = {
            "server_running":  True,
            "model_available": model_ok,
            "model":           LLM_MODEL,
            "status":          "ok" if model_ok else "model_missing",
        }
    except Exception:
        result["llm"] = {
            "server_running":  False,
            "model_available": False,
            "model":           LLM_MODEL,
            "status":          "offline",
        }

    logger.info("Dashboard called | db_queries=%s | collections=%s",
                result.get("database", {}).get("total_queries", "?"),
                result.get("vector_store", {}).get("total_collections", "?"))

    return result