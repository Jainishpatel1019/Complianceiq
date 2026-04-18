"""Agent impact reports API routes.

Exposes:
  GET  /api/v1/reports                     — paginated list of impact reports
  GET  /api/v1/reports/{regulation_id}     — latest report for a regulation
  GET  /api/v1/reports/high-impact         — regulations with P(High) > threshold
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db_session
from db.models import AgentReport, Regulation

router = APIRouter()
log = structlog.get_logger()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ImpactScoreSchema(BaseModel):
    """Bayesian posterior probabilities for impact level."""
    p_low: float | None = None
    p_medium: float | None = None
    p_high: float | None = None


class RWAEstimateSchema(BaseModel):
    """Basel III RWA delta estimate with Monte Carlo uncertainty."""
    median_million_usd: float | None = None
    ci_low_90_million_usd: float | None = None
    ci_high_90_million_usd: float | None = None

    @property
    def formatted(self) -> str:
        """Human-readable format: 'Median $340M, 90% CI [$180M, $620M]'."""
        if self.median_million_usd is None:
            return "N/A"
        return (
            f"Median ${self.median_million_usd:.0f}M, "
            f"90% CI [${self.ci_low_90_million_usd:.0f}M, ${self.ci_high_90_million_usd:.0f}M]"
        )


class AgentReportSchema(BaseModel):
    """Full structured impact report from the LangGraph agent."""
    report_id: str
    regulation_id: str
    document_number: str
    agency: str
    title: str
    summary: str | None
    impact_score: ImpactScoreSchema
    rwa_estimate: RWAEstimateSchema
    affected_business_lines: list[str] | None
    key_citations: list[str] | None
    reasoning_steps: int
    alert_dispatched: bool
    created_at: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[AgentReportSchema])
async def list_reports(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
) -> list[AgentReportSchema]:
    """List most recent impact reports across all regulations.

    Args:
        page: Page number.
        page_size: Results per page.
        db: Injected DB session.

    Returns:
        List of AgentReportSchema ordered by created_at descending.
    """
    offset = (page - 1) * page_size
    result = await db.execute(
        select(AgentReport, Regulation)
        .join(Regulation, AgentReport.regulation_id == Regulation.id)
        .order_by(desc(AgentReport.created_at))
        .offset(offset)
        .limit(page_size)
    )
    rows = result.all()
    return [_build_report_schema(report, regulation) for report, regulation in rows]


@router.get("/high-impact", response_model=list[AgentReportSchema])
async def get_high_impact_reports(
    threshold: float = Query(default=0.7, ge=0.0, le=1.0, description="P(High) threshold"),
    db: AsyncSession = Depends(get_db_session),
) -> list[AgentReportSchema]:
    """Return regulations where the latest report has P(High) >= threshold.

    This is the primary dashboard view — shows regulations requiring
    immediate compliance attention.

    Args:
        threshold: Minimum P(High) score. Default 0.7 matches the alert
            dispatch threshold in alert_dispatch.py DAG.
        db: Injected DB session.

    Returns:
        List of high-impact reports ordered by P(High) descending.
    """
    result = await db.execute(
        select(AgentReport, Regulation)
        .join(Regulation, AgentReport.regulation_id == Regulation.id)
        .where(AgentReport.impact_score_high >= threshold)
        .order_by(desc(AgentReport.impact_score_high))
        .limit(50)
    )
    rows = result.all()
    return [_build_report_schema(report, regulation) for report, regulation in rows]


@router.get("/{regulation_id}", response_model=AgentReportSchema)
async def get_report_for_regulation(
    regulation_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> AgentReportSchema:
    """Get the most recent impact report for a specific regulation.

    Args:
        regulation_id: UUID string of the regulation.
        db: Injected DB session.

    Returns:
        Latest AgentReportSchema for the regulation.

    Raises:
        HTTPException 404 if regulation or report not found.
    """
    try:
        reg_uuid = uuid.UUID(regulation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid regulation ID format")

    result = await db.execute(
        select(AgentReport, Regulation)
        .join(Regulation, AgentReport.regulation_id == Regulation.id)
        .where(AgentReport.regulation_id == reg_uuid)
        .order_by(desc(AgentReport.created_at))
        .limit(1)
    )
    row = result.first()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No impact report found for regulation {regulation_id}",
        )

    report, regulation = row
    return _build_report_schema(report, regulation)


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_report_schema(report: AgentReport, regulation: Regulation) -> AgentReportSchema:
    """Convert ORM objects to API response schema.

    Args:
        report: AgentReport ORM instance.
        regulation: Associated Regulation ORM instance.

    Returns:
        Populated AgentReportSchema.
    """
    return AgentReportSchema(
        report_id=str(report.id),
        regulation_id=str(report.regulation_id),
        document_number=regulation.document_number,
        agency=regulation.agency,
        title=regulation.title,
        summary=report.summary,
        impact_score=ImpactScoreSchema(
            p_low=report.impact_score_low,
            p_medium=report.impact_score_medium,
            p_high=report.impact_score_high,
        ),
        rwa_estimate=RWAEstimateSchema(
            median_million_usd=report.delta_rwa_median_m,
            ci_low_90_million_usd=report.delta_rwa_ci_low_m,
            ci_high_90_million_usd=report.delta_rwa_ci_high_m,
        ),
        affected_business_lines=report.affected_business_lines,
        key_citations=report.key_citations,
        reasoning_steps=len(report.agent_reasoning_trace or []),
        alert_dispatched=report.alert_dispatched,
        created_at=report.created_at.isoformat(),
    )
