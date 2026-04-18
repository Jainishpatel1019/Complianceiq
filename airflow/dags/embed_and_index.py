"""Airflow DAG: embed_and_index — chunk documents, embed with Ollama, upsert ChromaDB.

Triggered by ingest_sources after new documents are written to PostgreSQL.
Also runs on a daily schedule at 08:00 UTC as a catch-up.

Design decisions:
- Why chunk before embedding: embedding models have a max token limit (8192 for
  nomic-embed-text). Regulatory documents are 10,000-100,000+ words. Chunking
  at 512 tokens with 64-token overlap preserves context at chunk boundaries.
  Chunk size of 512 balances retrieval granularity vs embedding quality.
- Why nomic-embed-text over OpenAI ada-002: zero cost, 768-dim embeddings,
  comparable retrieval quality on domain text. Runs in Ollama alongside our
  other models — no extra service to manage.
- Why ChromaDB for dev + pgvector for production queries: ChromaDB has a
  simpler Python API for batch upserts. pgvector enables hybrid SQL + vector
  search in a single query (no round-trip). We maintain both.
- Dynamic task mapping: one task per document. On a day with 200 new documents,
  200 embed tasks run in parallel (limited by Airflow worker pool).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
OLLAMA_EMBED_MODEL = "nomic-embed-text"

default_args: dict[str, Any] = {
    "owner": "complianceiq",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


@dag(
    dag_id="embed_and_index",
    description="Chunk new regulations, embed with Ollama nomic-embed-text, upsert ChromaDB + pgvector",
    schedule="0 8 * * *",
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["embedding", "phase-1"],
    doc_md=__doc__,
)
def embed_and_index_dag():

    @task
    def get_unembedded_regulation_ids(**context) -> list[str]:
        """Query PostgreSQL for regulation IDs that have not yet been embedded.

        A regulation is 'unembedded' if it has no corresponding entry in the
        ChromaDB collection for its current version. We track this via a
        boolean flag in the regulations table to avoid querying ChromaDB
        (which is slower than a simple DB flag check).

        Args:
            context: Airflow task context. Used to read 'new_doc_count' from
                dag_run.conf if this was triggered by ingest_sources.

        Returns:
            List of regulation ID strings (UUID format).
        """
        from backend.pipelines.embedding import get_unembedded_ids

        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run and dag_run.conf else {}
        triggered_count = conf.get("new_doc_count", None)

        ids = get_unembedded_ids()
        log.info(
            "Found %d unembedded regulations (triggered with count: %s)",
            len(ids), triggered_count,
        )
        return ids

    @task
    def embed_and_upsert_regulation(regulation_id: str) -> dict[str, Any]:
        """Chunk, embed, and upsert a single regulation document.

        Steps:
        1. Fetch full_text from PostgreSQL (latest version).
        2. Split into overlapping chunks of CHUNK_SIZE_TOKENS tokens.
           Why overlap: without overlap, a sentence split across chunk boundary
           loses context in both chunks. 64-token overlap ensures critical
           sentences are fully captured in at least one chunk.
        3. Embed each chunk with nomic-embed-text via Ollama REST API.
           Batch size 20 to balance throughput vs memory on CPU inference.
        4. Upsert chunks into ChromaDB with metadata: regulation_id,
           chunk_index, document_number, agency, publication_date.
        5. Upsert embeddings into pgvector column in regulations table for
           hybrid SQL queries.
        6. Mark regulation as embedded in the DB flag.

        Args:
            regulation_id: UUID string of the regulation to process.

        Returns:
            Dict with: {'regulation_id': str, 'chunks': int, 'model': str}

        Raises:
            httpx.ConnectError: If Ollama is not reachable.
            chromadb.errors.ChromaError: On ChromaDB upsert failure.
        """
        from backend.pipelines.embedding import embed_and_index_regulation

        result = embed_and_index_regulation(
            regulation_id=regulation_id,
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            embed_model=OLLAMA_EMBED_MODEL,
        )
        log.info(
            "Embedded regulation %s — %d chunks with %s",
            regulation_id, result["chunks"], result["model"],
        )
        return result

    @task
    def log_embedding_summary(results: list[dict[str, Any]]) -> None:
        """Log a summary of the embedding run and push metrics to MLflow.

        Args:
            results: List of result dicts from embed_and_upsert_regulation.
        """
        import mlflow
        import os

        total_docs = len(results)
        total_chunks = sum(r.get("chunks", 0) for r in results)
        failed = sum(1 for r in results if r.get("error"))

        log.info(
            "Embedding run complete — docs: %d, chunks: %d, failed: %d",
            total_docs, total_chunks, failed,
        )

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        with mlflow.start_run(run_name=f"embed_run_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_metrics({
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "failed_documents": failed,
                "avg_chunks_per_doc": total_chunks / max(total_docs, 1),
            })

    # ── DAG wiring ──────────────────────────────────────────────────────────
    regulation_ids = get_unembedded_regulation_ids()
    # Dynamic task mapping: one embed task per regulation ID
    results = embed_and_upsert_regulation.expand(regulation_id=regulation_ids)
    log_embedding_summary(results)


embed_and_index_dag()
