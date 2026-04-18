"""Regulations REST API routes.

Exposes:
  GET  /api/v1/regulations          — paginated list with filters
  GET  /api/v1/regulations/{id}     — single regulation with latest change scores
  GET  /api/v1/regulations/search   — hybrid keyword + vector search
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db_session
from db.models import ChangeScore, Regulation

router = APIRouter()
log = structlog.get_logger()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChangeScoreSchema(BaseModel):
    """Change score between two consecutive versions of a regulation."""
    drift_score: float | None
    drift_ci_low: float | None
    drift_ci_high: float | None
    jsd_score: float | None
    jsd_p_value: float | None
    wasserstein_score: float | None
    is_significant: bool
    flagged_for_analysis: bool
    computed_at: str

    model_config = {"from_attributes": True}


class RegulationSchema(BaseModel):
    """Full regulation record returned by the API."""
    id: str
    document_number: str
    source: str
    agency: str
    title: str
    abstract: str | None
    publication_date: str
    effective_date: str | None
    regulation_type: str | None
    cfr_references: list[str] | None
    latest_change_score: ChangeScoreSchema | None = None

    model_config = {"from_attributes": True}


class RegulationListResponse(BaseModel):
    """Paginated list of regulations."""
    items: list[RegulationSchema]
    total: int
    page: int
    page_size: int


class SearchResult(BaseModel):
    """A single RAG search result chunk."""
    regulation_id: str
    document_number: str
    agency: str
    chunk_text: str
    relevance_score: float
    chunk_index: int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=RegulationListResponse)
async def list_regulations(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    agency: str | None = Query(default=None, description="Filter by agency name (partial match)"),
    regulation_type: str | None = Query(default=None, description="Filter by type: final_rule | proposed_rule"),
    flagged_only: bool = Query(default=False, description="Return only regulations flagged for analysis"),
    db: AsyncSession = Depends(get_db_session),
) -> RegulationListResponse:
    """List regulations with optional filters and pagination.

    Args:
        page: Page number starting at 1.
        page_size: Number of results per page (max 100).
        agency: Optional agency name substring filter.
        regulation_type: Optional type filter.
        flagged_only: If True, return only regulations with flagged change scores.
        db: Injected DB session.

    Returns:
        Paginated list of regulations with latest change scores.
    """
    query = select(Regulation).order_by(desc(Regulation.publication_date))

    if agency:
        query = query.where(Regulation.agency.ilike(f"%{agency}%"))
    if regulation_type:
        query = query.where(Regulation.regulation_type == regulation_type)
    if flagged_only:
        query = query.join(ChangeScore).where(ChangeScore.flagged_for_analysis == True)

    # Count total before pagination
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    regulations = result.scalars().all()

    items = []
    for reg in regulations:
        # Fetch latest change score for each regulation
        cs_result = await db.execute(
            select(ChangeScore)
            .where(ChangeScore.regulation_id == reg.id)
            .order_by(desc(ChangeScore.computed_at))
            .limit(1)
        )
        latest_cs = cs_result.scalar_one_or_none()

        items.append(
            RegulationSchema(
                id=str(reg.id),
                document_number=reg.document_number,
                source=reg.source,
                agency=reg.agency,
                title=reg.title,
                abstract=reg.abstract,
                publication_date=reg.publication_date.isoformat() if reg.publication_date else "",
                effective_date=reg.effective_date.isoformat() if reg.effective_date else None,
                regulation_type=reg.regulation_type,
                cfr_references=reg.cfr_references,
                latest_change_score=ChangeScoreSchema(
                    drift_score=latest_cs.drift_score,
                    drift_ci_low=latest_cs.drift_ci_low,
                    drift_ci_high=latest_cs.drift_ci_high,
                    jsd_score=latest_cs.jsd_score,
                    jsd_p_value=latest_cs.jsd_p_value,
                    wasserstein_score=latest_cs.wasserstein_score,
                    is_significant=latest_cs.is_significant,
                    flagged_for_analysis=latest_cs.flagged_for_analysis,
                    computed_at=latest_cs.computed_at.isoformat(),
                ) if latest_cs else None,
            )
        )

    return RegulationListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/search", response_model=list[SearchResult])
async def search_regulations(
    q: str = Query(..., min_length=3, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results"),
    agency: str | None = Query(default=None, description="Filter results by agency"),
) -> list[SearchResult]:
    """Hybrid semantic search over regulation chunks using ChromaDB.

    Uses nomic-embed-text to embed the query, then retrieves the top_k
    most relevant chunks from ChromaDB. Results are filtered by agency
    if provided.

    Args:
        q: Search query string.
        top_k: Number of chunks to return.
        agency: Optional agency filter applied post-retrieval.

    Returns:
        List of SearchResult with chunk text and relevance scores.
    """
    import os
    import httpx
    import chromadb

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
    chroma_host = os.environ.get("CHROMADB_HOST", "chromadb")
    chroma_port = int(os.environ.get("CHROMADB_PORT", "8000"))

    # Embed the query
    try:
        async with httpx.AsyncClient(base_url=ollama_url, timeout=30.0) as client:
            embed_response = await client.post(
                "/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": q},
            )
            embed_response.raise_for_status()
            query_embedding = embed_response.json()["embedding"]
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Embedding service (Ollama) is unavailable")

    # Query ChromaDB
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = chroma_client.get_or_create_collection(
        "regulations", metadata={"hnsw:space": "cosine"}
    )

    where_filter: dict[str, Any] | None = {"agency": agency} if agency else None

    chroma_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    docs = chroma_results.get("documents", [[]])[0]
    metas = chroma_results.get("metadatas", [[]])[0]
    distances = chroma_results.get("distances", [[]])[0]

    for doc, meta, distance in zip(docs, metas, distances):
        results.append(
            SearchResult(
                regulation_id=meta.get("regulation_id", ""),
                document_number=meta.get("document_number", ""),
                agency=meta.get("agency", ""),
                chunk_text=doc,
                # ChromaDB returns cosine distance (0=identical, 2=opposite).
                # Convert to similarity score in [0, 1].
                relevance_score=round(1.0 - (distance / 2.0), 4),
                chunk_index=meta.get("chunk_index", 0),
            )
        )

    log.info("search_regulations", query=q, results_returned=len(results))
    return results


@router.get("/{regulation_id}", response_model=RegulationSchema)
async def get_regulation(
    regulation_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> RegulationSchema:
    """Get a single regulation by ID with its latest change score.

    Args:
        regulation_id: UUID string.
        db: Injected DB session.

    Returns:
        RegulationSchema with latest change score.

    Raises:
        HTTPException 404 if not found.
        HTTPException 422 if regulation_id is not a valid UUID.
    """
    try:
        reg_uuid = uuid.UUID(regulation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid regulation ID format")

    result = await db.execute(
        select(Regulation).where(Regulation.id == reg_uuid)
    )
    regulation = result.scalar_one_or_none()

    if not regulation:
        raise HTTPException(status_code=404, detail=f"Regulation {regulation_id} not found")

    cs_result = await db.execute(
        select(ChangeScore)
        .where(ChangeScore.regulation_id == regulation.id)
        .order_by(desc(ChangeScore.computed_at))
        .limit(1)
    )
    latest_cs = cs_result.scalar_one_or_none()

    return RegulationSchema(
        id=str(regulation.id),
        document_number=regulation.document_number,
        source=regulation.source,
        agency=regulation.agency,
        title=regulation.title,
        abstract=regulation.abstract,
        publication_date=regulation.publication_date.isoformat(),
        effective_date=regulation.effective_date.isoformat() if regulation.effective_date else None,
        regulation_type=regulation.regulation_type,
        cfr_references=regulation.cfr_references,
        latest_change_score=ChangeScoreSchema(
            drift_score=latest_cs.drift_score,
            drift_ci_low=latest_cs.drift_ci_low,
            drift_ci_high=latest_cs.drift_ci_high,
            jsd_score=latest_cs.jsd_score,
            jsd_p_value=latest_cs.jsd_p_value,
            wasserstein_score=latest_cs.wasserstein_score,
            is_significant=latest_cs.is_significant,
            flagged_for_analysis=latest_cs.flagged_for_analysis,
            computed_at=latest_cs.computed_at.isoformat(),
        ) if latest_cs else None,
    )
