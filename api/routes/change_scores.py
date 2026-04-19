"""Change scores API routes.

Exposes:
  GET /api/v1/change-scores                    — recent scores, sorted by drift
  GET /api/v1/change-scores/{regulation_id}    — score history for one regulation
  GET /api/v1/change-scores/heatmap            — section-level heatmap data
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db_session
from db.models import ChangeScore, Regulation

router = APIRouter()
log = structlog.get_logger()


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChangeScoreFull(BaseModel):
    """Full change score record for one version pair."""
    score_id: str
    regulation_id: str
    document_number: str
    agency: str
    title: str
    version_old: int
    version_new: int
    drift_score: float | None
    drift_ci_low: float | None
    drift_ci_high: float | None
    drift_display: str          # "0.31 ± 0.04"
    jsd_score: float | None
    jsd_p_value: float | None
    jsd_significant: bool
    wasserstein_score: float | None
    composite_score: float      # weighted average for dashboard ranking
    is_significant: bool
    flagged_for_analysis: bool
    computed_at: str


class HeatmapSection(BaseModel):
    """One section of a regulation with its drift score."""
    section_title: str
    drift_score: float
    word_count: int


class RegulationHeatmap(BaseModel):
    """Section-level heatmap data for one regulation."""
    regulation_id: str
    document_number: str
    title: str
    sections: list[HeatmapSection]
    total_drift: float


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db_session)):
    """Return quick summary stats — total regulations, flagged count, agencies."""
    try:
        total_regs   = (await db.execute(select(func.count(Regulation.id)))).scalar() or 0
        total_scores = (await db.execute(select(func.count(ChangeScore.id)))).scalar() or 0
        flagged      = (await db.execute(
            select(func.count(ChangeScore.id)).where(ChangeScore.flagged_for_analysis == True)
        )).scalar() or 0
        agencies     = (await db.execute(
            select(func.count(func.distinct(Regulation.agency)))
        )).scalar() or 0
        return {
            "total_regulations":  total_regs,
            "total_change_scores": total_scores,
            "flagged":            flagged,
            "agencies":           agencies,
            "seeding_complete":   total_regs >= 500,
        }
    except Exception as exc:
        return {"total_regulations": 0, "flagged": 0, "agencies": 0,
                "total_change_scores": 0, "seeding_complete": False, "error": str(exc)}


@router.get("", response_model=list[ChangeScoreFull])
async def list_change_scores(
    limit: int = Query(default=50, ge=1, le=1000),
    flagged_only: bool = Query(default=False),
    min_drift: float = Query(default=0.0, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db_session),
) -> list[ChangeScoreFull]:
    """Return most recent change scores sorted by drift descending.

    Args:
        limit: Max results.
        flagged_only: If True, only return flagged regulations.
        min_drift: Only return scores above this drift threshold.
        db: Injected DB session.

    Returns:
        List of ChangeScoreFull sorted by drift_score descending.
    """
    query = (
        select(ChangeScore, Regulation)
        .join(Regulation, ChangeScore.regulation_id == Regulation.id)
        .order_by(desc(ChangeScore.drift_score))
        .limit(limit)
    )
    if flagged_only:
        query = query.where(ChangeScore.flagged_for_analysis == True)
    if min_drift > 0:
        query = query.where(ChangeScore.drift_score >= min_drift)

    result = await db.execute(query)
    return [_build_score_schema(cs, reg) for cs, reg in result.all()]


@router.get("/heatmap/{regulation_id}", response_model=RegulationHeatmap)
async def get_regulation_heatmap(
    regulation_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> RegulationHeatmap:
    """Compute section-level drift heatmap for a regulation.

    Splits the old and new versions into sections (by heading patterns),
    computes per-section drift, returns sorted by drift descending.
    This powers the heatmap visualisation on the dashboard.

    Args:
        regulation_id: UUID string.
        db: Injected DB session.

    Returns:
        RegulationHeatmap with per-section drift scores.

    Raises:
        HTTPException 404 if regulation not found.
    """
    from db.models import RegulationVersion
    import numpy as np

    try:
        reg_uuid = uuid.UUID(regulation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid regulation ID")

    # Fetch regulation
    reg_result = await db.execute(
        select(Regulation).where(Regulation.id == reg_uuid)
    )
    regulation = reg_result.scalar_one_or_none()
    if not regulation:
        raise HTTPException(status_code=404, detail="Regulation not found")

    # Fetch two most recent versions
    versions_result = await db.execute(
        select(RegulationVersion)
        .where(RegulationVersion.regulation_id == reg_uuid)
        .order_by(desc(RegulationVersion.version_number))
        .limit(2)
    )
    versions = versions_result.scalars().all()

    if len(versions) < 2:
        raise HTTPException(
            status_code=422,
            detail="Need at least 2 versions to compute heatmap",
        )

    v_new, v_old = versions[0], versions[1]
    sections = _compute_section_heatmap(v_old.full_text or "", v_new.full_text or "")
    total_drift = float(np.mean([s.drift_score for s in sections])) if sections else 0.0

    return RegulationHeatmap(
        regulation_id=regulation_id,
        document_number=regulation.document_number,
        title=regulation.title,
        sections=sections,
        total_drift=round(total_drift, 4),
    )


@router.get("/{regulation_id}", response_model=list[ChangeScoreFull])
async def get_scores_for_regulation(
    regulation_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> list[ChangeScoreFull]:
    """Get full change score history for one regulation (all version pairs).

    Args:
        regulation_id: UUID string.
        db: Injected DB session.

    Returns:
        List of ChangeScoreFull ordered by computed_at descending.
    """
    try:
        reg_uuid = uuid.UUID(regulation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid regulation ID")

    result = await db.execute(
        select(ChangeScore, Regulation)
        .join(Regulation, ChangeScore.regulation_id == Regulation.id)
        .where(ChangeScore.regulation_id == reg_uuid)
        .order_by(desc(ChangeScore.computed_at))
    )
    return [_build_score_schema(cs, reg) for cs, reg in result.all()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_score_schema(cs: ChangeScore, reg: Regulation) -> ChangeScoreFull:
    """Build API schema from ORM objects."""
    drift = cs.drift_score or 0.0
    ci_low = cs.drift_ci_low or 0.0
    ci_high = cs.drift_ci_high or 0.0
    half_width = (ci_high - ci_low) / 2

    # Composite score: weighted average of all three measures
    # Weights reflect ablation study F1 contributions
    jsd = cs.jsd_score or 0.0
    wass = cs.wasserstein_score or 0.0
    composite = round(0.5 * drift + 0.3 * jsd + 0.2 * wass, 4)

    return ChangeScoreFull(
        score_id=str(cs.id),
        regulation_id=str(cs.regulation_id),
        document_number=reg.document_number,
        agency=reg.agency,
        title=reg.title,
        version_old=cs.version_old,
        version_new=cs.version_new,
        drift_score=cs.drift_score,
        drift_ci_low=cs.drift_ci_low,
        drift_ci_high=cs.drift_ci_high,
        drift_display=f"{drift:.2f} ± {half_width:.2f}",
        jsd_score=cs.jsd_score,
        jsd_p_value=cs.jsd_p_value,
        jsd_significant=cs.jsd_p_value is not None and cs.jsd_p_value < 0.05,
        wasserstein_score=cs.wasserstein_score,
        composite_score=composite,
        is_significant=cs.is_significant,
        flagged_for_analysis=cs.flagged_for_analysis,
        computed_at=cs.computed_at.isoformat(),
    )


def _compute_section_heatmap(
    text_old: str, text_new: str
) -> list[HeatmapSection]:
    """Split documents into sections and compute per-section TF-IDF cosine drift.

    Uses TF-IDF cosine (not neural embeddings) for per-section heatmap
    because it's fast enough to run synchronously in the API layer.
    Full neural drift is computed async by the change_detection DAG.

    Args:
        text_old: Old version full text.
        text_new: New version full text.

    Returns:
        List of HeatmapSection sorted by drift_score descending.
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Split on common regulatory heading patterns: "Section X.", "§ X", "PART X"
    heading_pattern = re.compile(
        r"(?:^|\n)(?:Section\s+\d+[\.\s]|§\s*\d+|PART\s+[IVX\d]+|[A-Z]{2,}[\.\s])",
        re.MULTILINE | re.IGNORECASE,
    )

    def split_sections(text: str) -> dict[str, str]:
        """Return {heading: body} dict."""
        parts = heading_pattern.split(text)
        headings = heading_pattern.findall(text)
        if not headings:
            return {"Full Document": text}
        sections: dict[str, str] = {}
        for i, heading in enumerate(headings):
            body = parts[i + 1] if i + 1 < len(parts) else ""
            sections[heading.strip()] = body.strip()
        return sections

    old_sections = split_sections(text_old)
    new_sections = split_sections(text_new)

    # Compare sections that exist in both versions
    all_headings = set(old_sections) | set(new_sections)
    results: list[HeatmapSection] = []

    for heading in all_headings:
        body_old = old_sections.get(heading, "")
        body_new = new_sections.get(heading, "")

        if not body_old and not body_new:
            continue

        # Section added or removed → maximum drift
        if not body_old or not body_new:
            results.append(HeatmapSection(
                section_title=heading[:80],
                drift_score=1.0,
                word_count=len((body_old or body_new).split()),
            ))
            continue

        # TF-IDF cosine drift for this section
        try:
            vec = TfidfVectorizer(max_features=500)
            matrix = vec.fit_transform([body_old, body_new])
            sim = float(cosine_similarity(matrix[0], matrix[1])[0][0])
            drift = round(1.0 - sim, 4)
        except ValueError:
            drift = 0.0

        results.append(HeatmapSection(
            section_title=heading[:80],
            drift_score=drift,
            word_count=len(body_new.split()),
        ))

    return sorted(results, key=lambda x: x.drift_score, reverse=True)
