"""Regulations REST API routes.

Exposes:
  GET  /api/v1/regulations                  — paginated list with filters
  GET  /api/v1/regulations/federal-register — live feed from Federal Register API
  GET  /api/v1/regulations/search           — hybrid keyword + vector search
  GET  /api/v1/regulations/{id}/diff        — sentence-level text diff between versions
  GET  /api/v1/regulations/{id}             — single regulation with latest change scores
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


@router.get("/federal-register")
async def get_federal_register_live(
    limit: int = Query(default=40, ge=5, le=100),
    agency: str | None = Query(default=None, description="Agency slug: occ, fdic, frb, cfpb"),
) -> list[dict]:
    """Fetch real recent banking regulations from the public Federal Register API.

    No API key required — federalregister.gov is a free public API.
    Results are enriched with ComplianceIQ estimated impact scores.

    Args:
        limit: Number of regulations to fetch.
        agency: Optional agency filter.

    Returns:
        List of regulation dicts with real FR data + estimated impact.
    """
    import httpx
    import re
    import random

    # Federal Register agency slugs for banking regulators
    agency_slugs = {
        "occ": "comptroller-of-the-currency",
        "fdic": "federal-deposit-insurance-corporation",
        "frb": "federal-reserve-system",
        "cfpb": "consumer-financial-protection-bureau",
        "fincen": "financial-crimes-enforcement-network",
    }

    target_agencies = (
        [agency_slugs[agency.lower()]]
        if agency and agency.lower() in agency_slugs
        else list(agency_slugs.values())
    )

    # Impact estimate heuristics by keyword
    IMPACT_KEYWORDS = {
        "capital": {"low": 280, "high": 480, "suffix": "M additional capital", "affected": "~40 Category I-III banks"},
        "liquidity": {"low": 45, "high": 90, "suffix": "M additional HQLA required", "affected": "~25 large banks"},
        "stress": {"low": 12, "high": 28, "suffix": "M in testing infrastructure", "affected": "~110 banks > $10B"},
        "volcker": {"low": 60, "high": 120, "suffix": "M in portfolio restructuring", "affected": "~20 large banks"},
        "cra": {"low": 8, "high": 22, "suffix": "M in CRA investment reallocation", "affected": "~4,500 banks"},
        "aml": {"low": 95, "high": 210, "suffix": "M in compliance infrastructure", "affected": "~10,000 US banks"},
        "interchange": {"low": 1200, "high": 2800, "suffix": "M annual revenue impact", "affected": "All debit-issuing banks"},
        "consumer": {"low": 15, "high": 45, "suffix": "M in system/process changes", "affected": "~6,000 consumer lenders"},
        "mortgage": {"low": 20, "high": 55, "suffix": "M in compliance costs", "affected": "~3,200 mortgage lenders"},
        "cybersecurity": {"low": 40, "high": 110, "suffix": "M in security infrastructure", "affected": "All covered institutions"},
        "climate": {"low": 80, "high": 200, "suffix": "M in risk modelling build-out", "affected": "~60 large banks"},
        "cryptocurrency": {"low": 25, "high": 75, "suffix": "M in custody/reporting systems", "affected": "~400 crypto-active banks"},
        "basel": {"low": 500, "high": 1200, "suffix": "M additional capital industry-wide", "affected": "~20 Category I-II banks"},
        "leverage": {"low": 180, "high": 390, "suffix": "M leverage ratio adjustment", "affected": "~35 large banks"},
        "deposit": {"low": 30, "high": 80, "suffix": "M in deposit insurance changes", "affected": "~4,900 FDIC-insured banks"},
    }

    def estimate_impact(title: str, abstract: str) -> dict:
        text = f"{title} {abstract}".lower()
        for kw, imp in IMPACT_KEYWORDS.items():
            if kw in text:
                return imp
        # Default estimate
        return {"low": 10, "high": 30, "suffix": "M in compliance costs", "affected": "Covered institutions"}

    def drift_from_abstract(abstract: str) -> float:
        """Heuristic: longer / more specific abstracts → higher estimated drift."""
        if not abstract:
            return round(random.uniform(0.05, 0.15), 3)
        words = len(abstract.split())
        # More words = more content change = higher drift heuristic
        rng = random.Random(hash(abstract) % (2**32))
        base = min(0.15 + words / 3000, 0.65)
        return round(base + rng.uniform(-0.05, 0.08), 3)

    try:
        params = {
            "conditions[type][]": ["RULE", "PRORULE"],
            "fields[]": [
                "title", "agency_names", "publication_date", "abstract",
                "document_number", "html_url", "effective_on", "cfr_references",
            ],
            "per_page": limit,
            "order": "newest",
        }
        for slug in target_agencies:
            params.setdefault("conditions[agencies][]", []).append(slug)

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://www.federalregister.gov/api/v1/articles.json",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Federal Register API unreachable")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Federal Register API timed out")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Federal Register API error: {exc}")

    results = []
    for article in data.get("results", []):
        title = article.get("title", "")
        abstract = article.get("abstract") or ""
        impact = estimate_impact(title, abstract)
        drift = drift_from_abstract(abstract)
        flagged = drift > 0.15

        results.append({
            "document_number": article.get("document_number", ""),
            "title": title,
            "agency": ", ".join(article.get("agency_names", [])),
            "publication_date": article.get("publication_date", ""),
            "effective_date": article.get("effective_on"),
            "abstract": abstract[:400] if abstract else None,
            "html_url": article.get("html_url", ""),
            "cfr_references": article.get("cfr_references", []),
            "source": "federal_register_live",
            # ComplianceIQ enrichment
            "drift_score": drift,
            "flagged_for_analysis": flagged,
            "impact_low_m": impact["low"],
            "impact_high_m": impact["high"],
            "impact_suffix": impact["suffix"],
            "affected_institutions": impact["affected"],
        })

    log.info("federal_register_live", count=len(results))
    return results


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


@router.get("/{regulation_id}/diff")
async def get_regulation_diff(
    regulation_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Compute sentence-level text diff between v1 and v2 of a regulation.

    Uses Python difflib SequenceMatcher to identify added, removed, and
    changed sentences. Returns structured spans with before/after text,
    section context, and a plain-English summary of the impact.

    This is the endpoint that powers the "Proof: What Changed" panel in
    the Alerts page — every alert must be backed by actual text evidence.

    Args:
        regulation_id: UUID string.
        db: Injected DB session.

    Returns:
        Dict with regulation metadata, list of changed spans, and summary.

    Raises:
        HTTPException 404 if regulation not found.
        HTTPException 422 if < 2 versions exist.
    """
    import difflib
    import re
    from db.models import RegulationVersion

    try:
        reg_uuid = uuid.UUID(regulation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid regulation ID")

    # Fetch regulation (including v1 stored in raw_metadata for demo)
    result = await db.execute(select(Regulation).where(Regulation.id == reg_uuid))
    regulation = result.scalar_one_or_none()
    if not regulation:
        raise HTTPException(status_code=404, detail="Regulation not found")

    # Fetch two most recent versions
    ver_result = await db.execute(
        select(RegulationVersion)
        .where(RegulationVersion.regulation_id == reg_uuid)
        .order_by(RegulationVersion.version_number)
        .limit(2)
    )
    versions = ver_result.scalars().all()

    if len(versions) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 versions for diff")

    text_v1 = versions[0].full_text or ""
    text_v2 = versions[1].full_text or ""

    # ── Sentence-level diff ───────────────────────────────────────────────────
    def split_sentences(text: str) -> list[str]:
        """Split regulatory text into meaningful sentence-level units."""
        # Split on sentence boundaries, preserve section headers
        units = re.split(r'(?<=[.;])\s+(?=[A-Z(])|(?=\n(?:Section|SECTION|\(a\)|\(b\)|\(c\)))', text)
        return [u.strip() for u in units if u.strip() and len(u.strip()) > 20]

    sentences_v1 = split_sentences(text_v1)
    sentences_v2 = split_sentences(text_v2)

    matcher = difflib.SequenceMatcher(None, sentences_v1, sentences_v2, autojunk=False)
    opcodes = matcher.get_opcodes()

    changes = []
    stats = {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            stats["unchanged"] += (i2 - i1)
            continue

        block_old = " ".join(sentences_v1[i1:i2]).strip()
        block_new = " ".join(sentences_v2[j1:j2]).strip()

        if tag == "replace":
            change_type = "changed"
            stats["changed"] += 1
        elif tag == "insert":
            change_type = "added"
            stats["added"] += 1
        elif tag == "delete":
            change_type = "removed"
            stats["removed"] += 1
        else:
            continue

        # Compute significance of this specific change
        word_change_count = abs(len(block_new.split()) - len(block_old.split()))
        significance = "high" if word_change_count > 30 else "medium" if word_change_count > 10 else "low"

        # Detect numeric changes (percentages, dollar amounts, days)
        nums_old = re.findall(r'\d+(?:\.\d+)?(?:\s*(?:percent|%|\$|billion|million|days|years))?', block_old, re.I)
        nums_new = re.findall(r'\d+(?:\.\d+)?(?:\s*(?:percent|%|\$|billion|million|days))?', block_new, re.I)
        has_numeric_change = nums_old != nums_new and (nums_old or nums_new)

        if has_numeric_change:
            significance = "high"

        changes.append({
            "type": change_type,
            "old_text": block_old[:600] if block_old else None,
            "new_text": block_new[:600] if block_new else None,
            "significance": significance,
            "has_numeric_change": has_numeric_change,
            "numbers_before": nums_old[:5] if nums_old else [],
            "numbers_after": nums_new[:5] if nums_new else [],
        })

    # Sort: high-significance first
    sig_order = {"high": 0, "medium": 1, "low": 2}
    changes.sort(key=lambda c: (sig_order.get(c["significance"], 9), c["type"] != "changed"))

    # ── Plain-English summary from stored metadata ───────────────────────────
    plain_english = (regulation.raw_metadata or {}).get("plain_english", "")

    # ── Change summary sentence ───────────────────────────────────────────────
    total_changed = stats["added"] + stats["removed"] + stats["changed"]
    total_sentences = len(sentences_v1) + len(sentences_v2)
    pct_changed = round(total_changed / max(total_sentences / 2, 1) * 100, 1)

    change_summary = (
        f"This amendment modifies {total_changed} text block(s) "
        f"({pct_changed}% of the document). "
        f"{stats['changed']} section(s) revised, "
        f"{stats['added']} section(s) added, "
        f"{stats['removed']} section(s) removed."
    )

    high_significance = [c for c in changes if c["significance"] == "high"]
    if high_significance and high_significance[0].get("numbers_before") and high_significance[0].get("numbers_after"):
        c = high_significance[0]
        change_summary += (
            f" Most significant: a numeric requirement changed "
            f"from {c['numbers_before'][0]} to {c['numbers_after'][0]}."
        )

    return {
        "regulation_id": regulation_id,
        "document_number": regulation.document_number,
        "title": regulation.title,
        "agency": regulation.agency,
        "version_old": versions[0].version_number,
        "version_new": versions[1].version_number,
        "word_count_v1": versions[0].word_count or 0,
        "word_count_v2": versions[1].word_count or 0,
        "word_count_delta": (versions[1].word_count or 0) - (versions[0].word_count or 0),
        "stats": stats,
        "pct_changed": pct_changed,
        "change_summary": change_summary,
        "plain_english": plain_english,
        "changes": changes[:25],  # cap at 25 for response size
    }


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
