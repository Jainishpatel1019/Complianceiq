"""Backend embedding pipeline — chunk, embed, index.

Called by the embed_and_index Airflow DAG. All I/O-bound operations
use async; CPU-bound chunking runs synchronously (Python GIL is fine for
tiktoken tokenisation).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import chromadb
import httpx
from sqlalchemy import select, update

from db import get_session_factory
from db.models import Regulation, RegulationVersion

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "chromadb")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))
COLLECTION_NAME = os.environ.get("CHROMADB_COLLECTION", "regulations")

# Lazy-loaded tokenizer — avoids download at import time (important for tests
# and cold starts where tiktoken BPE files may not be cached yet).
_TOKENIZER = None


def _get_tokenizer():
    """Return tiktoken cl100k_base encoder, loading it on first call.

    Why lazy load: tiktoken downloads the BPE vocabulary file (~1MB) from
    OpenAI on first use. Import-time loading would fail in offline environments
    and slow down test collection. Lazy loading isolates the download to
    first actual chunking call.

    Returns:
        tiktoken.Encoding instance.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


# ── ChromaDB client (module-level singleton) ─────────────────────────────────

def _get_chroma_client() -> chromadb.HttpClient:
    """Return a ChromaDB HTTP client.

    Why HttpClient over EphemeralClient: ChromaDB runs as a separate Docker
    service. EphemeralClient stores data in-process (lost on restart). HttpClient
    persists data in the ChromaDB container volume.

    Returns:
        Configured ChromaDB HttpClient.
    """
    return chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)


def _get_collection() -> chromadb.Collection:
    """Get or create the regulations ChromaDB collection.

    Returns:
        ChromaDB collection with cosine distance metric.
    """
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Why cosine: regulatory text embeddings are unit-normalised by
        # nomic-embed-text. Cosine distance = 1 - cosine_similarity, which
        # maps naturally to our drift score formula.
        metadata={"hnsw:space": "cosine"},
    )


# ── Core functions ───────────────────────────────────────────────────────────

def get_unembedded_ids() -> list[str]:
    """Query PostgreSQL for regulation IDs not yet in ChromaDB.

    We track embedding status via raw_metadata['embedded'] = True to avoid
    querying ChromaDB for existence (which requires one round-trip per doc).
    An absent or False 'embedded' key means the doc needs embedding.

    Returns:
        List of UUID strings for unembedded regulations.
    """
    return asyncio.run(_async_get_unembedded_ids())


async def _async_get_unembedded_ids() -> list[str]:
    """Async implementation of get_unembedded_ids."""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Regulation.id).where(
                # JSONB containment check: doc not yet marked as embedded
                ~Regulation.raw_metadata.contains({"embedded": True})
            ).limit(500)  # Process up to 500 per DAG run
        )
        return [str(row[0]) for row in result.fetchall()]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping token chunks.

    Why token-based chunking over character-based: embedding models have a
    token limit (8192 for nomic-embed-text). Character splitting can produce
    chunks that exceed the token limit on dense regulatory text.

    Why 64-token overlap: regulatory sentences are typically 20-40 tokens.
    64-token overlap ensures a sentence split at a chunk boundary appears
    complete in at least one chunk, preserving retrieval quality.

    Args:
        text: Full document text to chunk.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Number of tokens to overlap between consecutive chunks.

    Returns:
        List of text chunks (decoded from tokens).
    """
    if not text:
        return []

    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        if end == len(tokens):
            break
        # Slide window: advance by (chunk_size - overlap)
        start += chunk_size - chunk_overlap

    return chunks


def embed_chunks(chunks: list[str], model: str, batch_size: int = 20) -> list[list[float]]:
    """Embed a list of text chunks using Ollama.

    Calls the Ollama /api/embeddings endpoint. Batches requests to avoid
    overwhelming the CPU-only Ollama instance. Each embedding is 768-dim
    (nomic-embed-text default).

    Args:
        chunks: List of text strings to embed.
        model: Ollama model name (e.g. 'nomic-embed-text').
        batch_size: Number of chunks to embed per API call.

    Returns:
        List of embedding vectors (list[float]), one per chunk.

    Raises:
        httpx.ConnectError: If Ollama service is not running.
        httpx.HTTPStatusError: On non-2xx Ollama response.
    """
    embeddings: list[list[float]] = []

    with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=60.0) as client:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            for chunk in batch:
                response = client.post(
                    "/api/embeddings",
                    json={"model": model, "prompt": chunk},
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])

    return embeddings


def embed_and_index_regulation(
    regulation_id: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
) -> dict[str, Any]:
    """Full pipeline: fetch text → chunk → embed → upsert ChromaDB.

    This is the main function called by the embed_and_index DAG via
    dynamic task mapping. One call per regulation.

    Args:
        regulation_id: UUID string of the regulation.
        chunk_size: Token chunk size (default 512).
        chunk_overlap: Token overlap between chunks (default 64).
        embed_model: Ollama model for embeddings.

    Returns:
        Dict: {'regulation_id': str, 'chunks': int, 'model': str}
        If an error occurs: {'regulation_id': str, 'error': str}
    """
    return asyncio.run(
        _async_embed_and_index(regulation_id, chunk_size, chunk_overlap, embed_model)
    )


async def _async_embed_and_index(
    regulation_id: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
) -> dict[str, Any]:
    """Async implementation of embed_and_index_regulation."""
    import uuid as uuid_module

    factory = get_session_factory()

    try:
        # 1. Fetch regulation + latest version text
        async with factory() as session:
            reg_result = await session.execute(
                select(Regulation).where(Regulation.id == uuid_module.UUID(regulation_id))
            )
            regulation = reg_result.scalar_one_or_none()
            if not regulation:
                return {"regulation_id": regulation_id, "error": "Regulation not found"}

            version_result = await session.execute(
                select(RegulationVersion)
                .where(RegulationVersion.regulation_id == regulation.id)
                .order_by(RegulationVersion.version_number.desc())
                .limit(1)
            )
            latest_version = version_result.scalar_one_or_none()

        full_text = latest_version.full_text if latest_version else (regulation.abstract or "")
        if not full_text:
            log.warning("No text content for regulation %s — skipping", regulation_id)
            return {"regulation_id": regulation_id, "chunks": 0, "model": embed_model}

        # 2. Chunk the text
        chunks = chunk_text(full_text, chunk_size, chunk_overlap)
        if not chunks:
            return {"regulation_id": regulation_id, "chunks": 0, "model": embed_model}

        # 3. Embed chunks
        embeddings = embed_chunks(chunks, embed_model)

        # 4. Upsert into ChromaDB
        collection = _get_collection()
        ids = [f"{regulation_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "regulation_id": regulation_id,
                "chunk_index": i,
                "document_number": regulation.document_number,
                "agency": regulation.agency,
                "publication_date": regulation.publication_date.isoformat()
                if regulation.publication_date else "",
                "regulation_type": regulation.regulation_type or "",
            }
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        # 5. Mark as embedded in DB
        async with factory() as session:
            metadata_copy = dict(regulation.raw_metadata or {})
            metadata_copy["embedded"] = True
            metadata_copy["chunk_count"] = len(chunks)
            await session.execute(
                update(Regulation)
                .where(Regulation.id == uuid_module.UUID(regulation_id))
                .values(raw_metadata=metadata_copy)
            )
            await session.commit()

        return {"regulation_id": regulation_id, "chunks": len(chunks), "model": embed_model}

    except Exception as exc:
        log.exception("Failed to embed regulation %s: %s", regulation_id, exc)
        return {"regulation_id": regulation_id, "error": str(exc)}
