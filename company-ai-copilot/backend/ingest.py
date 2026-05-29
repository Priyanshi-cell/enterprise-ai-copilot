import hashlib
import logging
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chroma_db"
# Embedding model from sentence-transformers.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Chunk size — how many characters go into each chunk.
# 700 chars ≈ 100–140 words, which fits well within the model's 256-token limit.
CHUNK_SIZE = 700
# Overlap — how many characters are shared between consecutive chunks.
# Overlap ensures that sentences split across chunk boundaries
# are still findable from either side.
CHUNK_OVERLAP = 120
# Minimum chunk length — chunks shorter than this are discarded.
# Very short chunks are usually noise (single words, page numbers, etc.)
MIN_CHUNK_LENGTH = 60
# CHROMADB + EMBEDDING SETUP

_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# SentenceTransformerEmbeddingFunction wraps sentence-transformers
# so ChromaDB can embed text automatically during upsert and query.
_embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

logger.info("ChromaDB client ready | path=%s", CHROMA_DB_PATH)
logger.info("Embedding model loaded | model=%s", EMBEDDING_MODEL)


def _safe_name(company_name: str) -> str:
    """
    Convert any company name into a valid ChromaDB collection name.

    ChromaDB requires collection names to be:
    - lowercase
    - 3–63 characters
    - only letters, numbers, underscores, hyphens
    - no spaces or special characters

    Examples:
        "Acme Corp"   → "acme_corp"
        "My Company!" → "my_company"
        "OpenAI"      → "openai"
    """
    # Lowercase
    name = company_name.strip().lower()

    # Replace spaces and hyphens with underscores
    name = name.replace(" ", "_").replace("-", "_")

    # Remove any character that isn't a letter, number, or underscore
    name = re.sub(r"[^a-z0-9_]", "", name)

    # ChromaDB requires at least 3 characters
    if len(name) < 3:
        name = name + "_kb"  # e.g. "ab" → "ab_kb"

    # Truncate to 63 characters (ChromaDB maximum)
    name = name[:63]

    return name

def _make_id(company: str, chunk: str) -> str:
    """
    Create a unique, stable ID for each chunk using its content.

    Uses SHA-256 so that:
    - The same chunk always gets the same ID
    - Re-ingesting the same content is safe (upsert won't duplicate)
    - Different chunks always get different IDs

    Returns the first 20 characters of the hex digest,
    prefixed with the company name.

    Example: "acme_corp_3f7a2b9c1d4e5f6a7b8c"
    """
    content = f"{company}::{chunk}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{company}_{digest[:20]}"


def split_text(text: str) -> list[str]:
    """
    Split a long string into overlapping chunks of fixed character size.

    Why overlapping chunks?
    If a sentence is split across two chunks, overlap ensures it appears
    in at least one complete chunk — making it retrievable by search.

    Example (chunk_size=30, overlap=10):
        "The quick brown fox jumps over the lazy dog"
        Chunk 1: "The quick brown fox jumps over"   (chars 0–29)
        Chunk 2: "jumps over the lazy dog"           (chars 20–42)
        Notice "jumps over" appears in both — that is the overlap.

    The function also:
    - Normalises whitespace (collapses multiple spaces/newlines)
    - Removes duplicate chunks
    - Filters out chunks that are too short to be useful

    Args:
        text: Any string of text to be chunked.

    Returns:
        A list of clean, deduplicated text chunks.
        Returns an empty list if the input is empty or produces no valid chunks.
    """

    
    if not text or not text.strip():
        logger.warning("split_text received empty text — returning no chunks")
        return []

    cleaned = re.sub(r"\s+", " ", text).strip()

    if len(cleaned) < MIN_CHUNK_LENGTH:
        logger.warning(
            "Text too short to chunk (%d chars) — returning as single chunk",
            len(cleaned),
        )
        return [cleaned]

    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP  # how far to advance each iteration

    while start < len(cleaned):
        end = start + CHUNK_SIZE
        chunk = cleaned[start:end].strip()

        # Only keep chunks with enough content
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)

        start += step

    logger.debug("Raw chunks before dedup: %d", len(chunks))
    unique_chunks = list(dict.fromkeys(chunks))

    removed = len(chunks) - len(unique_chunks)
    if removed > 0:
        logger.debug("Removed %d duplicate chunks", removed)

    logger.info(
        "split_text complete | input_chars=%d | chunks=%d",
        len(cleaned), len(unique_chunks),
    )

    return unique_chunks

def ingest_text(company_name: str, text: str) -> dict:
    """
    Chunk, embed, and store text in ChromaDB under the company's collection.

    This is the main function called by main.py after scraping or PDF extraction.

    How it works:
        1. Sanitise the company name into a valid collection name
        2. Get or create the ChromaDB collection for that company
        3. Split the text into overlapping chunks
        4. Generate a stable ID for each chunk (SHA-256 hash)
        5. Upsert all chunks — safe to call multiple times, no duplicates

    "Upsert" means: insert if new, update if already exists (by ID).
    So re-ingesting the same URL or PDF is completely safe.

    Args:
        company_name: The company whose knowledge base to update.
                      Used as the ChromaDB collection name.
        text:         Raw text from a scraped website or extracted PDF.

    Returns:
        {
            "chunks_added": int,   # new chunks added in this ingestion run
            "chunks_total": int,   # total chunks now in the collection
            "company": str,        # sanitised collection name actually used
        }
    """

    safe_name = _safe_name(company_name)
    logger.info(
        "Starting ingestion | company=%s | collection=%s | text_chars=%d",
        company_name, safe_name, len(text),
    )

    try:
        collection = _chroma_client.get_or_create_collection(
            name=safe_name,
            embedding_function=_embedding_fn,
            metadata={"company": company_name},
        )
        count_before = collection.count()
        logger.info(
            "Collection ready | name=%s | existing_chunks=%d",
            safe_name, count_before,
        )

    except Exception as e:
        logger.error("Failed to get/create ChromaDB collection: %s", str(e))
        raise Exception(
            f"Could not connect to the vector database: {str(e)}"
        )

    

    chunks = split_text(text)

    if not chunks:
        logger.warning(
            "No valid chunks produced for company=%s — ingestion skipped",
            safe_name,
        )
        return {
            "chunks_added": 0,
            "chunks_total": count_before,
            "company": safe_name,
        }

    logger.info("Chunks to upsert: %d", len(chunks))



    ids = [_make_id(safe_name, chunk) for chunk in chunks]


    try:
        BATCH_SIZE = 100

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch_chunks = chunks[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]

            collection.upsert(
                documents=batch_chunks,
                ids=batch_ids,
            )

            logger.debug(
                "Upserted batch %d–%d of %d",
                batch_start + 1, min(batch_end, len(chunks)), len(chunks),
            )

    except Exception as e:
        logger.error("ChromaDB upsert failed: %s", str(e))
        raise Exception(
            f"Failed to store chunks in the vector database: {str(e)}"
        )



    count_after = collection.count()
    chunks_added = count_after - count_before

    logger.info(
        "Ingestion complete | company=%s | new=%d | total=%d",
        safe_name, chunks_added, count_after,
    )

    return {
        "chunks_added": chunks_added,
        "chunks_total": count_after,
        "company": safe_name,
    }