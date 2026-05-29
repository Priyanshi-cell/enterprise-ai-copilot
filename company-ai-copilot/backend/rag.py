"""
rag.py
Retrieval-Augmented Generation (RAG) pipeline.

How RAG works in plain English:
    1. The user asks a question
    2. We convert that question into a vector (a list of numbers)
       that represents its meaning
    3. We search ChromaDB for the stored chunks most similar to that vector
    4. We give those chunks to Ollama as "context"
    5. Ollama reads the context and writes a grounded answer
    6. We return the answer + the source chunks used

This approach prevents hallucination because the model is told to answer
ONLY from the provided context — not from its own training data.

No PyTorch. No Transformers. No reranker.
ChromaDB handles embeddings. Ollama handles generation.
"""

import logging

import chromadb
import ollama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)


# =============================================================================
# SETTINGS
# =============================================================================

# Must match the path used in ingest.py exactly —
# both files read from the same ChromaDB folder on disk
CHROMA_DB_PATH = "./chroma_db"

# Must match the embedding model used in ingest.py exactly —
# queries and stored chunks must be embedded by the same model
# or similarity search produces meaningless results
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# The Ollama model to use for answer generation
# Run: ollama pull llama3   before starting the server
LLM_MODEL = "llama3"

# How many chunks to retrieve from ChromaDB per query
# 4 is a good balance — enough context without overloading the prompt
TOP_K = 4

# Minimum characters a chunk must have to be used as context
# Very short chunks are usually noisy boilerplate
MIN_CHUNK_LENGTH = 60


# =============================================================================
# CHROMADB + EMBEDDING SETUP
# =============================================================================
# Created once when the module is imported — not on every query.
# Using the same embedding model as ingest.py ensures that query vectors
# and document vectors live in the same semantic space.

_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

_embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

logger.info("RAG module ready | db=%s | model=%s | llm=%s", CHROMA_DB_PATH, EMBEDDING_MODEL, LLM_MODEL)


# =============================================================================
# HELPER — safe collection name
# =============================================================================

def _safe_name(company_name: str) -> str:
    """
    Convert a company name into a valid ChromaDB collection name.
    Must produce the exact same result as ingest.py's version
    so queries hit the correct collection.
    """
    import re
    name = company_name.strip().lower()
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    if len(name) < 3:
        name = name + "_kb"
    return name[:63]


# =============================================================================
# HELPER — clean and deduplicate retrieved chunks
# =============================================================================

def _clean_chunks(raw_chunks: list[str]) -> list[str]:
    """
    Filter and deduplicate the chunks returned by ChromaDB.

    ChromaDB returns chunks sorted by similarity score, but they may:
    - Contain very short/noisy strings that slipped through ingestion
    - Have near-duplicate content from overlapping chunks

    This function:
    1. Strips leading/trailing whitespace from each chunk
    2. Drops chunks shorter than MIN_CHUNK_LENGTH
    3. Removes exact duplicates while preserving order

    Args:
        raw_chunks: The list of document strings returned by ChromaDB.

    Returns:
        A cleaned, deduplicated list ready to use as LLM context.
    """
    seen = set()
    clean = []

    for chunk in raw_chunks:
        # Normalise whitespace
        chunk = chunk.strip()

        # Skip very short chunks — they add noise, not value
        if len(chunk) < MIN_CHUNK_LENGTH:
            logger.debug("Skipping short chunk (%d chars)", len(chunk))
            continue

        # Skip exact duplicates — preserves the first (highest-scored) occurrence
        if chunk in seen:
            logger.debug("Skipping duplicate chunk")
            continue

        clean.append(chunk)
        seen.add(chunk)

    return clean


# =============================================================================
# HELPER — build the LLM prompt
# =============================================================================

def _build_prompt(context: str, question: str) -> str:
    """
    Construct the full prompt sent to Ollama.

    The prompt has three parts:
    1. Role definition — tells the model what it is
    2. Strict rules — prevents hallucination and off-topic answers
    3. Context + question — the actual information and query

    The rules section is critical:
    - "ONLY the context below" prevents the model using its training data
    - The exact fallback phrase is specified so the frontend can detect it
    - "Do not guess" reinforces the grounding requirement

    Args:
        context:  The retrieved chunks joined into a single string.
        question: The user's original question.

    Returns:
        A complete prompt string ready to send to ollama.chat().
    """
    return f"""You are an Enterprise AI Knowledge Copilot.

Your job is to answer the user's question using ONLY the context provided below.

RULES:
- Read the context carefully before answering
- Answer in a clear, professional, and concise tone
- Use ONLY information found in the context — never your own knowledge
- If the context does not contain enough information to answer, respond with exactly:
  "I could not find this information in the company knowledge base."
- Do not guess, speculate, or add information not present in the context
- Do not mention that you are using context or chunks — just answer naturally

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


# =============================================================================
# MAIN FUNCTION — query_rag
# =============================================================================

def query_rag(company_name: str, user_query: str) -> dict:
    """
    Run the full RAG pipeline for a user query.

    Pipeline:
        Step 1 — Validate inputs
        Step 2 — Connect to the company's ChromaDB collection
        Step 3 — Check the collection has data
        Step 4 — Retrieve top-K semantically similar chunks
        Step 5 — Clean and deduplicate the retrieved chunks
        Step 6 — Build the grounded prompt
        Step 7 — Generate the answer with Ollama
        Step 8 — Return answer + sources

    Every step has its own error handling. If something fails, the function
    returns a safe fallback dict — never raises an unhandled exception.

    Args:
        company_name: Which company's knowledge base to search.
                      Must match the name used during ingestion.
        user_query:   The question from the user (plain text).

    Returns:
        {
            "answer":  str,        # The generated answer from Llama 3
            "sources": list[str],  # The chunks used as context
        }
    """

    # ── Step 1: Validate inputs ───────────────────────────────────────────────

    if not company_name or not company_name.strip():
        logger.warning("query_rag called with empty company_name")
        return {
            "answer": "No company name was provided. Please specify a company.",
            "sources": [],
        }

    if not user_query or not user_query.strip():
        logger.warning("query_rag called with empty user_query")
        return {
            "answer": "No question was provided. Please enter a question.",
            "sources": [],
        }

    safe_name = _safe_name(company_name)
    query = user_query.strip()

    logger.info("RAG query | company=%s | query=%.80s", safe_name, query)

    # ── Step 2: Connect to ChromaDB collection ────────────────────────────────
    # get_or_create_collection is used (not get_collection) so the app
    # doesn't crash if the collection doesn't exist yet — it returns a
    # clear "no data" message instead.

    try:
        collection = _chroma_client.get_or_create_collection(
            name=safe_name,
            embedding_function=_embedding_fn,
        )

    except Exception as e:
        logger.error("Failed to connect to ChromaDB | company=%s | error=%s", safe_name, str(e))
        return {
            "answer": (
                "Could not connect to the knowledge base. "
                "Please try again in a moment."
            ),
            "sources": [],
        }

    # ── Step 3: Check the collection has data ─────────────────────────────────
    # Querying an empty collection raises an error in ChromaDB.
    # This guard catches that before it happens.

    total_chunks = collection.count()
    logger.info("Collection size | company=%s | chunks=%d", safe_name, total_chunks)

    if total_chunks == 0:
        logger.warning("Empty collection for company: %s", safe_name)
        return {
            "answer": (
                "No knowledge base found for this company. "
                "Please ingest a website or upload a PDF first, "
                "then ask your question."
            ),
            "sources": [],
        }

    # ── Step 4: Retrieve top-K semantically similar chunks ────────────────────
    # ChromaDB embeds the query using the same model used during ingestion,
    # then returns the TOP_K closest chunks by cosine similarity.
    #
    # n_results cannot exceed the total number of stored chunks —
    # min() prevents a ChromaDB error when the collection is small.

    try:
        n_results = min(TOP_K, total_chunks)

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        # results["documents"] is a list of lists — one list per query.
        # We only ever send one query at a time, so we take index [0].
        raw_chunks = results.get("documents", [[]])[0]

        logger.info("Retrieved %d raw chunks from ChromaDB", len(raw_chunks))

    except Exception as e:
        logger.error("ChromaDB query failed | company=%s | error=%s", safe_name, str(e))
        return {
            "answer": (
                "Failed to search the knowledge base. "
                "Please try again."
            ),
            "sources": [],
        }

    # ── Step 5: Clean and deduplicate chunks ──────────────────────────────────

    chunks = _clean_chunks(raw_chunks)

    logger.info("Clean chunks after dedup: %d", len(chunks))

    # Guard — all retrieved chunks were filtered out
    if not chunks:
        logger.info("No usable chunks after cleaning for query: %.80s", query)
        return {
            "answer": "I could not find this information in the company knowledge base.",
            "sources": [],
        }

    # ── Step 6: Build grounded prompt ────────────────────────────────────────
    # Chunks are joined with a separator line so the model can clearly see
    # where one source ends and another begins.

    context = "\n\n---\n\n".join(chunks)
    prompt = _build_prompt(context=context, question=query)

    logger.debug(
        "Prompt built | context_chars=%d | chunks_used=%d",
        len(context), len(chunks),
    )

    # ── Step 7: Generate answer with Ollama ───────────────────────────────────
    # ollama.chat() sends the prompt to the local Ollama server.
    # The model reads the context and generates a grounded answer.
    #
    # If Ollama is not running, this raises a connection error —
    # caught below with a clear user-facing message.

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response["message"]["content"].strip()

        logger.info(
            "Ollama response received | company=%s | answer_chars=%d",
            safe_name, len(answer),
        )

    except Exception as e:
        error_msg = str(e).lower()

        # Give a specific, actionable message for the most common failure
        if "connection" in error_msg or "refused" in error_msg:
            logger.error("Ollama is not running: %s", str(e))
            return {
                "answer": (
                    "The AI model is not responding. "
                    "Please start Ollama by running: ollama serve"
                ),
                "sources": chunks,
            }

        if "model" in error_msg and "not found" in error_msg:
            logger.error("Ollama model not found: %s", LLM_MODEL)
            return {
                "answer": (
                    f"The model '{LLM_MODEL}' is not installed. "
                    f"Please run: ollama pull {LLM_MODEL}"
                ),
                "sources": chunks,
            }

        logger.error("Ollama generation failed: %s", str(e))
        return {
            "answer": (
                "The AI model encountered an error while generating a response. "
                "Please try again."
            ),
            "sources": chunks,
        }

    # ── Step 8: Return answer and sources ─────────────────────────────────────

    logger.info(
        "RAG complete | company=%s | chunks_used=%d",
        safe_name, len(chunks),
    )

    return {
        "answer": answer,
        "sources": chunks,
    }


# =============================================================================
# STREAMING HELPER — retrieve_context
# =============================================================================

def retrieve_context(company_name: str, user_query: str) -> dict:
    """
    Run Steps 1-6 of the RAG pipeline (everything EXCEPT generation).

    Returns the retrieved chunks and the ready-to-send prompt so the
    streaming endpoint in main.py can call Ollama with stream=True itself.

    Why split retrieval from generation?
    StreamingResponse in FastAPI needs a plain generator function that
    yields tokens. If rag.py owned the Ollama call, it could not yield
    tokens back through the function boundary to FastAPI's response.
    Splitting lets main.py own the generator loop while rag.py handles
    all the ChromaDB logic it already knows.

    Args:
        company_name : Company whose ChromaDB collection to search.
        user_query   : The user's raw question.

    Returns:
        {
            "prompt":  str,        # full grounded prompt ready for Ollama
            "sources": list[str],  # chunks used as context
            "error":   str|None,   # human-readable error, or None on success
        }
    """
    import re

    # Sanitise — must match _safe_name() in query_rag exactly
    safe_name = company_name.strip().lower()
    safe_name = safe_name.replace(" ", "_").replace("-", "_")
    safe_name = re.sub(r"[^a-z0-9_]", "", safe_name)
    if len(safe_name) < 3:
        safe_name = safe_name + "_kb"
    safe_name = safe_name[:63]

    query = user_query.strip()
    logger.info(
        "retrieve_context | company=%s | query=%.80s", safe_name, query
    )

    # Validate inputs
    if not safe_name or not query:
        return {
            "prompt": "", "sources": [],
            "error": "Company name or query is empty.",
        }

    # Connect to ChromaDB collection
    try:
        collection = _chroma_client.get_or_create_collection(
            name=safe_name,
            embedding_function=_embedding_fn,
        )
    except Exception as e:
        return {"prompt": "", "sources": [], "error": str(e)}

    # Guard — empty collection
    total_chunks = collection.count()
    if total_chunks == 0:
        return {
            "prompt": "", "sources": [],
            "error": (
                "No knowledge base found for this company. "
                "Please ingest a website or upload a PDF first."
            ),
        }

    # Retrieve top-K semantically similar chunks
    try:
        n_results  = min(TOP_K, total_chunks)
        results    = collection.query(
            query_texts=[query], n_results=n_results
        )
        raw_chunks = results.get("documents", [[]])[0]
        logger.info(
            "retrieve_context | raw_chunks=%d", len(raw_chunks)
        )
    except Exception as e:
        return {"prompt": "", "sources": [], "error": str(e)}

    # Clean and deduplicate
    chunks = _clean_chunks(raw_chunks)
    if not chunks:
        return {
            "prompt": "", "sources": [],
            "error": (
                "I could not find this information "
                "in the company knowledge base."
            ),
        }

    # Build grounded prompt
    context = "\n\n---\n\n".join(chunks)
    prompt  = _build_prompt(context=context, question=query)

    logger.info(
        "retrieve_context complete | chunks=%d", len(chunks)
    )

    return {
        "prompt":  prompt,
        "sources": chunks,
        "error":   None,
    }