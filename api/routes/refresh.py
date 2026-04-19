"""Refresh API routes — on-demand data sync from Federal Register.

POST /api/v1/refresh          — trigger a background sync for recent regulations
GET  /api/v1/refresh/status   — poll sync progress
GET  /api/v1/refresh/schedule — show upcoming auto-sync schedule
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import uuid
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db_session

router = APIRouter()
log = structlog.get_logger()

# ── In-memory sync state (sufficient for single-worker HF Space) ──────────────
_sync_state: dict[str, Any] = {
    "status": "idle",          # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "inserted": 0,
    "skipped": 0,
    "errors": 0,
    "message": "No sync run yet",
    "last_successful_run": None,
}

FR_API_BASE = "https://www.federalregister.gov/api/v1/documents.json"
FR_AGENCY_SLUGS = [
    "federal-reserve-system",
    "comptroller-of-the-currency",
    "consumer-financial-protection-bureau",
    "federal-deposit-insurance-corporation",
    "securities-exchange-commission",
    "commodity-futures-trading-commission",
    "financial-crimes-enforcement-network",
    "national-credit-union-administration",
]


# ── Schemas ───────────────────────────────────────────────────────────────────

class RefreshRequest(BaseModel):
    """Parameters for a manual refresh."""
    lookback_days: int = 7       # how many days back to fetch (max 90)
    source: str = "auto"         # "fr_api" | "offline" | "auto"


class RefreshStatus(BaseModel):
    """Current or last sync status."""
    status: str
    started_at: str | None
    finished_at: str | None
    inserted: int
    skipped: int
    errors: int
    message: str
    last_successful_run: str | None


class SyncSchedule(BaseModel):
    """Upcoming automatic sync schedule."""
    next_daily_sync: str
    next_weekly_sync: str
    airflow_dag: str
    note: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("", response_model=RefreshStatus)
async def trigger_refresh(
    req: RefreshRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
) -> RefreshStatus:
    """Trigger an on-demand Federal Register sync.

    Runs in the background — returns immediately with status=running.
    Poll GET /api/v1/refresh/status for progress.
    """
    if _sync_state["status"] == "running":
        return RefreshStatus(**_sync_state)

    lookback = min(max(req.lookback_days, 1), 90)
    start_date = (date.today() - timedelta(days=lookback)).isoformat()
    end_date = date.today().isoformat()

    _sync_state.update({
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
        "message": f"Syncing regulations from {start_date} to {end_date}...",
    })

    background_tasks.add_task(
        _run_sync,
        start_date=start_date,
        end_date=end_date,
        source=req.source,
    )

    return RefreshStatus(**_sync_state)


@router.get("/status", response_model=RefreshStatus)
async def get_refresh_status() -> RefreshStatus:
    """Poll the current or last sync status."""
    return RefreshStatus(**_sync_state)


@router.get("/schedule", response_model=SyncSchedule)
async def get_sync_schedule() -> SyncSchedule:
    """Return the automatic sync schedule powered by Airflow."""
    now = datetime.utcnow()
    # Next daily: today at 07:00 UTC (or tomorrow if past)
    today_sync = now.replace(hour=7, minute=0, second=0, microsecond=0)
    next_daily = today_sync if now < today_sync else today_sync + timedelta(days=1)
    # Next weekly: next Monday at 07:00 UTC
    days_to_monday = (7 - now.weekday()) % 7 or 7
    next_weekly = (now + timedelta(days=days_to_monday)).replace(
        hour=7, minute=0, second=0, microsecond=0
    )
    return SyncSchedule(
        next_daily_sync=next_daily.isoformat() + "Z",
        next_weekly_sync=next_weekly.isoformat() + "Z",
        airflow_dag="fetch_federal_register",
        note=(
            "Daily sync fetches the last 2 days of FR publications. "
            "Weekly sync (Monday) fetches the last 14 days as a catch-up. "
            "Use POST /api/v1/refresh to trigger an immediate sync."
        ),
    )


# ── Background sync logic ─────────────────────────────────────────────────────

async def _run_sync(start_date: str, end_date: str, source: str) -> None:
    """Fetch regulations from FR API (or offline fallback) and upsert to DB."""
    import os
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession as ASession
    from sqlalchemy.orm import sessionmaker

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://complianceiq:complianceiq_demo@localhost/complianceiq",
    ).replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=ASession, expire_on_commit=False)

    try:
        documents = await _fetch_documents(start_date, end_date, source)
        async with async_session() as session:
            inserted, skipped, errors = await _upsert_documents(session, documents)

        _sync_state.update({
            "status": "done",
            "finished_at": datetime.utcnow().isoformat(),
            "inserted": inserted,
            "skipped": skipped,
            "errors": errors,
            "message": (
                f"Sync complete — {inserted} new regulations added, "
                f"{skipped} already existed."
            ),
            "last_successful_run": datetime.utcnow().isoformat(),
        })
        log.info("refresh_sync_complete", inserted=inserted, skipped=skipped)

    except Exception as exc:
        _sync_state.update({
            "status": "error",
            "finished_at": datetime.utcnow().isoformat(),
            "message": f"Sync failed: {exc}",
        })
        log.error("refresh_sync_failed", error=str(exc))
    finally:
        await engine.dispose()


async def _fetch_documents(start_date: str, end_date: str, source: str) -> list[dict]:
    """Try FR API first; fall back to offline generator if blocked."""
    if source == "offline":
        return _offline_generate(start_date, end_date)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            params = {
                "conditions[publication_date][gte]": start_date,
                "conditions[publication_date][lte]": end_date,
                "conditions[agencies][]": FR_AGENCY_SLUGS,
                "conditions[type][]": ["RULE", "PRORULE", "NOTICE"],
                "fields[]": [
                    "document_number", "title", "agency_names", "abstract",
                    "publication_date", "type", "html_url", "citation",
                ],
                "per_page": 100,
                "order": "newest",
            }
            resp = await client.get(FR_API_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

            documents = []
            for doc in data.get("results", []):
                agencies = doc.get("agency_names") or []
                documents.append({
                    "document_number": doc.get("document_number", ""),
                    "title": (doc.get("title") or "")[:500],
                    "agency": (agencies[0] if agencies else "Unknown")[:200],
                    "abstract": doc.get("abstract") or "",
                    "publication_date": doc.get("publication_date"),
                    "regulation_type": _map_type(doc.get("type", "")),
                    "html_url": doc.get("html_url", ""),
                    "citation": doc.get("citation", ""),
                    "source": "federal_register",
                })
            log.info("fr_api_fetch_ok", count=len(documents))
            return documents

    except Exception as exc:
        log.warning("fr_api_unavailable", reason=str(exc), fallback="offline")
        if source == "auto":
            return _offline_generate(start_date, end_date)
        raise


async def _upsert_documents(
    session: AsyncSession, documents: list[dict]
) -> tuple[int, int, int]:
    """Insert new regulations, skip existing ones. Returns (inserted, skipped, errors)."""
    inserted = skipped = errors = 0

    for doc in documents:
        if not doc.get("document_number"):
            continue
        try:
            exists = (await session.execute(
                text("SELECT 1 FROM regulations WHERE document_number = :dn"),
                {"dn": doc["document_number"]},
            )).fetchone()

            if exists:
                skipped += 1
                continue

            reg_id = uuid.uuid4()
            pub_date = _parse_date(doc.get("publication_date"))
            import json

            await session.execute(text("""
                INSERT INTO regulations
                    (id, document_number, title, agency, abstract,
                     publication_date, regulation_type, source,
                     full_text, raw_metadata, created_at, updated_at)
                VALUES
                    (:id, :doc_num, :title, :agency, :abstract,
                     :pub_date, :reg_type, :source,
                     :full_text, CAST(:meta AS jsonb), NOW(), NOW())
            """), {
                "id": str(reg_id),
                "doc_num": doc["document_number"],
                "title": doc["title"],
                "agency": doc["agency"],
                "abstract": doc.get("abstract", "")[:2000],
                "pub_date": pub_date,
                "reg_type": doc.get("regulation_type", "rule"),
                "source": doc.get("source", "federal_register"),
                "full_text": doc.get("abstract", ""),
                "meta": json.dumps({
                    "html_url": doc.get("html_url", ""),
                    "citation": doc.get("citation", ""),
                    "fetched_at": datetime.utcnow().isoformat(),
                }),
            })

            await session.execute(text("""
                INSERT INTO regulation_versions
                    (id, regulation_id, version_number, full_text, source_url, fetched_at)
                VALUES (:vid, :rid, 1, :text, :url, NOW())
                ON CONFLICT DO NOTHING
            """), {
                "vid": str(uuid.uuid4()),
                "rid": str(reg_id),
                "text": doc.get("abstract", ""),
                "url": doc.get("html_url", ""),
            })

            await session.commit()
            inserted += 1

        except Exception as exc:
            await session.rollback()
            log.error("upsert_error", doc=doc.get("document_number"), error=str(exc))
            errors += 1

    return inserted, skipped, errors


def _map_type(doc_type: str) -> str:
    return {"RULE": "final_rule", "PRORULE": "proposed_rule", "NOTICE": "notice"}.get(
        doc_type.upper(), "rule"
    )


def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _offline_generate(start_date: str, end_date: str) -> list[dict]:
    """Generate synthetic regulations for a date range (offline fallback)."""
    AGENCIES = ["Federal Reserve", "OCC", "CFPB", "FDIC", "SEC", "CFTC", "FinCEN", "NCUA"]
    TOPICS = [
        ("Capital Adequacy Standards Update", "capital", "12 CFR 3"),
        ("AML Transaction Monitoring Requirements", "aml", "31 CFR 1020"),
        ("Liquidity Risk Management Rules", "liquidity", "12 CFR 249"),
        ("Cybersecurity Incident Notification", "cyber", "12 CFR 748"),
        ("Climate Risk Stress Testing", "stress_test", "12 CFR 252"),
        ("Crypto Asset Custody Standards", "crypto", "12 CFR 7"),
        ("HMDA Data Reporting Expansion", "hmda", "12 CFR 1003"),
        ("Interchange Fee Cap Revision", "interchange", "12 CFR 235"),
        ("Buy Now Pay Later Oversight", "bnpl", "12 CFR 1026"),
        ("Open Banking Data Sharing Rule", "open_banking", "12 CFR 1033"),
    ]

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow()

    rng = random.Random(42)
    records = []
    current = start

    while current <= end:
        for _ in range(rng.randint(1, 3)):
            topic, slug, cfr = rng.choice(TOPICS)
            agency = rng.choice(AGENCIES)
            seq = rng.randint(10000, 99999)
            yr, mo = current.year, current.month
            doc_num = f"FR-{yr}-{mo:02d}-{seq}"

            records.append({
                "document_number": doc_num,
                "title": f"{topic} — {agency} ({yr})",
                "agency": agency,
                "abstract": (
                    f"This regulation amends {cfr} to update {topic.lower()}. "
                    f"The {agency} is revising requirements for covered institutions "
                    f"to reflect updated market conditions and supervisory guidance."
                ),
                "publication_date": current.strftime("%Y-%m-%d"),
                "regulation_type": rng.choice(["final_rule", "proposed_rule", "notice"]),
                "html_url": "",
                "citation": f"{cfr} Part {rng.randint(1, 500)}",
                "source": "offline_generated",
            })
        current += timedelta(days=1)

    return records
