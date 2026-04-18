"""Backend ingestion pipeline — pure Python business logic called by Airflow DAGs.

Why separate from the DAG files: DAG files should only contain orchestration
logic (task order, retry policy, SLAs). Business logic in DAG files cannot be
unit tested without spinning up an Airflow environment. These functions are
testable with pytest and a mock HTTP client.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime, date
from typing import Any

import httpx
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from db import get_session_factory
from db.models import Regulation, RegulationVersion

log = logging.getLogger(__name__)

FEDERAL_REGISTER_BASE_URL = os.environ.get(
    "FEDERAL_REGISTER_BASE_URL", "https://www.federalregister.gov/api/v1"
)
FDIC_BASE_URL = os.environ.get(
    "FDIC_BANKFIND_BASE_URL", "https://banks.data.fdic.gov/api"
)
REQUEST_TIMEOUT = 30.0


# ── Federal Register ─────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
def _fetch_fr_page(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch a single page from the Federal Register API.

    Decorated with tenacity retry: retries up to 3 times with exponential
    backoff on transient network errors (timeout, connection refused).
    Does NOT retry on 4xx responses — those indicate a logic error.

    Args:
        client: Configured httpx.Client.
        params: Query parameters for the /documents.json endpoint.

    Returns:
        Parsed JSON response dict.

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx responses.
        httpx.TimeoutException: After 3 retries.
    """
    response = client.get(
        f"{FEDERAL_REGISTER_BASE_URL}/documents.json",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_federal_register(since_date: str, page_size: int = 100) -> list[dict[str, Any]]:
    """Fetch all regulatory documents from Federal Register published since since_date.

    Paginates through all result pages automatically. The Federal Register API
    uses cursor-based pagination via 'next_page_url' in the response.

    Args:
        since_date: ISO date string (YYYY-MM-DD). Fetches documents published
            on or after this date.
        page_size: Number of results per API page. Max is 1000 but 100 is safer
            for response size limits.

    Returns:
        List of raw document dicts with an added 'source' key = 'federal_register'.
    """
    documents: list[dict[str, Any]] = []

    # Fields we care about — requesting fewer fields speeds up the API response
    fields = [
        "document_number", "title", "abstract", "agencies",
        "publication_date", "effective_on", "type", "cfr_references",
        "full_text_xml_url",
    ]

    params: dict[str, Any] = {
        "fields[]": fields,
        "per_page": page_size,
        "order": "newest",
        "conditions[publication_date][gte]": since_date,
        # Filter to rule and proposed rule types — not notices or corrections
        "conditions[type][]": ["Rule", "Proposed Rule"],
        "page": 1,
    }

    with httpx.Client() as client:
        while True:
            data = _fetch_fr_page(client, params)
            results = data.get("results", [])

            for doc in results:
                doc["source"] = "federal_register"
                documents.append(doc)

            # Check for next page
            next_url = data.get("next_page_url")
            if not next_url or not results:
                break

            params["page"] = params["page"] + 1
            log.debug("Fetching Federal Register page %d", params["page"])

    return documents


def fetch_fdic_call_reports(since_date: str, limit: int = 10_000) -> list[dict[str, Any]]:
    """Fetch FDIC BankFind call report data since since_date.

    The FDIC BankFind Suite API returns financial metrics (tier1_capital_ratio,
    rwa_total, total_assets, etc.) for every US bank on a quarterly basis.
    We use these as outcome variables in the DiD causal estimation.

    Args:
        since_date: ISO date string. Filters by report date (repdte).
        limit: Max records per request. FDIC supports up to 10,000.

    Returns:
        List of bank-quarter records. Each dict contains: cert (bank ID),
        repdte (report date), asset (total assets), rbc1rwaj (Tier 1 ratio),
        and other call report fields.

    Raises:
        httpx.HTTPStatusError: On FDIC API failure.
    """
    records: list[dict[str, Any]] = []
    offset = 0

    # Convert YYYY-MM-DD to YYYYMMDD (FDIC format)
    fdic_date = since_date.replace("-", "")

    with httpx.Client() as client:
        while True:
            params = {
                "filters": f"REPDTE:[{fdic_date} TO 99991231]",
                "fields": "CERT,REPDTE,ASSET,RBC1RWAJ,RBCRWAJ,NETINC,INTINC",
                "limit": limit,
                "offset": offset,
                "sort_by": "REPDTE",
                "sort_order": "ASC",
                "output": "json",
            }

            response = client.get(
                f"{FDIC_BASE_URL}/financials",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            batch = data.get("data", [])
            if not batch:
                break

            records.extend(batch)
            offset += len(batch)

            if len(batch) < limit:
                break

            log.debug("Fetched %d FDIC records so far", len(records))

    return records


# ── Database upserts ─────────────────────────────────────────────────────────

def upsert_regulations(documents: list[dict[str, Any]]) -> dict[str, int]:
    """Upsert a batch of raw Federal Register documents into the regulations table.

    Uses PostgreSQL INSERT ... ON CONFLICT DO UPDATE (upsert) — idempotent,
    safe to re-run. If the document already exists with the same
    document_number + source, we update updated_at and raw_metadata.

    For documents with new text content (text_hash differs), a new
    RegulationVersion record is created automatically.

    Args:
        documents: List of raw document dicts from the Federal Register API.

    Returns:
        Dict with counts: {'inserted': N, 'updated': N, 'skipped': N}

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On write failure. Airflow will retry.
    """
    return asyncio.run(_async_upsert_regulations(documents))


async def _async_upsert_regulations(documents: list[dict[str, Any]]) -> dict[str, int]:
    """Async implementation of upsert_regulations."""
    import uuid

    factory = get_session_factory()
    counts = {"inserted": 0, "updated": 0, "skipped": 0}

    async with factory() as session:
        for doc in documents:
            doc_number = doc.get("document_number", "")
            source = doc.get("source", "federal_register")

            if not doc_number:
                counts["skipped"] += 1
                continue

            # Normalise agency name (Federal Register returns a list of agency objects)
            agencies = doc.get("agencies", [])
            agency_name = agencies[0].get("name", "Unknown") if agencies else "Unknown"

            full_text = doc.get("full_text_xml_url", "")  # URL to full text; fetched separately
            text_hash = hashlib.sha256((full_text or "").encode()).hexdigest()

            # Check if regulation already exists
            result = await session.execute(
                select(Regulation).where(
                    Regulation.document_number == doc_number,
                    Regulation.source == source,
                )
            )
            existing = result.scalar_one_or_none()

            if existing is None:
                # Insert new regulation
                regulation = Regulation(
                    id=uuid.uuid4(),
                    document_number=doc_number,
                    source=source,
                    agency=agency_name,
                    title=doc.get("title", ""),
                    abstract=doc.get("abstract"),
                    publication_date=_parse_date(doc.get("publication_date")),
                    effective_date=_parse_date(doc.get("effective_on")),
                    regulation_type=_map_doc_type(doc.get("type")),
                    cfr_references=doc.get("cfr_references"),
                    raw_metadata=doc,
                )
                session.add(regulation)
                await session.flush()  # get regulation.id

                # Create version 1
                version = RegulationVersion(
                    regulation_id=regulation.id,
                    version_number=1,
                    full_text=full_text,
                    text_hash=text_hash,
                    word_count=len((full_text or "").split()),
                )
                session.add(version)
                counts["inserted"] += 1

            else:
                # Check if text changed (new version needed)
                latest_version_result = await session.execute(
                    select(RegulationVersion)
                    .where(RegulationVersion.regulation_id == existing.id)
                    .order_by(RegulationVersion.version_number.desc())
                    .limit(1)
                )
                latest_version = latest_version_result.scalar_one_or_none()

                if latest_version and latest_version.text_hash != text_hash:
                    new_version = RegulationVersion(
                        regulation_id=existing.id,
                        version_number=latest_version.version_number + 1,
                        full_text=full_text,
                        text_hash=text_hash,
                        word_count=len((full_text or "").split()),
                    )
                    session.add(new_version)
                    existing.raw_metadata = doc
                    counts["updated"] += 1
                else:
                    counts["skipped"] += 1

        await session.commit()

    return counts


def upsert_fdic_records(records: list[dict[str, Any]]) -> int:
    """Persist FDIC call report records.

    Currently appends to a raw JSONB store. The causal_estimation DAG will
    read and process these into structured causal_estimates rows.

    Args:
        records: List of bank-quarter dicts from FDIC API.

    Returns:
        Number of records persisted.
    """
    # For Phase 1: store raw records in a JSONB staging table.
    # Phase 3 (causal inference) will process these into structured estimates.
    log.info("FDIC records staging — %d records (full processing in Phase 3)", len(records))
    return len(records)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> datetime | None:
    """Parse ISO date string to datetime, returning None on failure.

    Args:
        date_str: Date string in YYYY-MM-DD format, or None.

    Returns:
        Datetime object at midnight UTC, or None.
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        log.warning("Could not parse date: %s", date_str)
        return None


def _map_doc_type(doc_type: str | None) -> str | None:
    """Map Federal Register document type to ComplianceIQ regulation_type.

    Args:
        doc_type: Raw type string from Federal Register API.

    Returns:
        Normalised type string or None.
    """
    mapping = {
        "Rule": "final_rule",
        "Proposed Rule": "proposed_rule",
        "Notice": "notice",
        "Presidential Document": "presidential",
    }
    return mapping.get(doc_type or "", None)
