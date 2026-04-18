"""
Causal inference REST endpoints — Phase 3.

Routes
------
GET /causal/estimates            → All stored causal estimates (DiD + SCM + RDD)
GET /causal/estimates/{reg_id}   → Estimates for one regulation
GET /causal/did/{reg_id}         → Live DiD computation (bypasses cache)
GET /causal/rdd/{reg_id}         → Live RDD computation for a threshold
GET /causal/bn/score             → Bayesian network posterior given evidence
GET /causal/summary              → Aggregated stats across all estimates
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter()


# ── Response schemas ──────────────────────────────────────────────────────────

class DiDResponse(BaseModel):
    regulation_id: str
    method: str = "did"
    att: float
    att_se: float
    ci_low_95: float
    ci_high_95: float
    pre_trend_p: float
    n_treated: int
    n_control: int
    n_periods: int
    interpretation: str = Field(default="")


class SCMResponse(BaseModel):
    regulation_id: str
    method: str = "synthetic_control"
    att: float
    placebo_p_value: float
    pre_rmspe: float
    post_rmspe: float
    rmspe_ratio: float
    donor_weights: dict[str, float]
    interpretation: str = Field(default="")


class RDDResponse(BaseModel):
    regulation_id: str
    method: str = "rdd"
    threshold_label: str
    threshold_value: float
    rd_estimate: float
    rd_se: float
    ci_low_95: float
    ci_high_95: float
    bandwidth: float
    n_left: int
    n_right: int
    interpretation: str = Field(default="")


class BNScoreRequest(BaseModel):
    drift_score: float = Field(..., ge=0.0, le=1.0)
    jsd_p_value: float | None = None
    rwa_median_million: float | None = None


class BNScoreResponse(BaseModel):
    p_low: float
    p_medium: float
    p_high: float
    p_alert: float
    most_likely_impact: str


class CausalSummary(BaseModel):
    n_did: int
    n_scm: int
    n_rdd: int
    mean_att_did: float | None
    significant_did: int     # ATT CI does not straddle zero
    significant_rdd: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interpret_did(r: dict) -> str:
    att_pct = r["att"] * 100
    sig = "not " if r["ci_low_95"] < 0 < r["ci_high_95"] else ""
    return (
        f"ATT = {att_pct:+.2f}% ({sig}significant). "
        f"95% CI [{r['ci_low_95']*100:.2f}%, {r['ci_high_95']*100:.2f}%]. "
        f"Parallel trends p={r['pre_trend_p']:.3f} "
        f"({'fails' if r['pre_trend_p'] < 0.05 else 'passes'} pre-trend test)."
    )


def _interpret_scm(r: dict) -> str:
    att_pct = r["att"] * 100
    return (
        f"Synthetic control ATT = {att_pct:+.2f}%. "
        f"Placebo p={r['placebo_p_value']:.3f}. "
        f"RMSPE ratio = {r['rmspe_ratio']:.2f} "
        f"({'strong' if r['rmspe_ratio'] > 2 else 'weak'} post-event signal)."
    )


def _interpret_rdd(r: dict) -> str:
    est_pct = r["rd_estimate"] * 100
    sig = "not " if r["ci_low_95"] < 0 < r["ci_high_95"] else ""
    return (
        f"RD estimate at ${r['threshold_value']/1000:.0f}B = {est_pct:+.2f}% "
        f"({sig}significant). "
        f"95% CI [{r['ci_low_95']*100:.2f}%, {r['ci_high_95']*100:.2f}%]. "
        f"Bandwidth = {r['bandwidth']:.3f} log-asset units."
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/estimates", response_model=list[dict[str, Any]])
async def list_causal_estimates(
    method: str | None = Query(None, description="Filter by method: did|synthetic_control|rdd"),
    regulation_id: str | None = Query(None),
    limit: int = Query(50, le=200),
) -> list[dict[str, Any]]:
    """
    Return stored causal estimates. Falls back to on-the-fly computation
    if the DB table is empty (dev mode without a running Airflow pipeline).
    """
    from backend.models.causal_inference import (
        run_all_causal_estimates, LANDMARK_REGULATIONS,
    )

    try:
        # Try DB first
        from db import get_db_session
        from db.models import CausalEstimate
        from sqlalchemy import select

        async with get_db_session() as session:
            q = select(CausalEstimate)
            if regulation_id:
                q = q.where(CausalEstimate.regulation_id == regulation_id)
            if method:
                q = q.where(CausalEstimate.method == method)
            q = q.limit(limit)
            rows = (await session.execute(q)).scalars().all()
            if rows:
                results = [r.estimate_json for r in rows]
                return results
    except Exception:
        log.warning("DB unavailable for causal estimates, computing on-the-fly")

    # Dev fallback: compute fresh
    all_results = run_all_causal_estimates()
    if method:
        all_results = [r for r in all_results if r.get("method") == method]
    if regulation_id:
        all_results = [r for r in all_results if r.get("regulation_id") == regulation_id]
    return all_results[:limit]


@router.get("/estimates/{regulation_id}", response_model=list[dict[str, Any]])
async def get_regulation_estimates(regulation_id: str) -> list[dict[str, Any]]:
    """All causal estimates for a single regulation."""
    return await list_causal_estimates(regulation_id=regulation_id)


@router.get("/did/{regulation_id}", response_model=DiDResponse)
async def compute_did_live(
    regulation_id: str,
    treatment_year: int = Query(...),
    outcome_var: str = Query("tier1_capital_ratio"),
) -> DiDResponse:
    """Live DiD computation (bypasses cache — useful for ad-hoc queries)."""
    from backend.models.causal_inference import compute_did
    try:
        r = compute_did(regulation_id, treatment_year, outcome_var)
        d = r.to_dict()
        d["interpretation"] = _interpret_did(d)
        return DiDResponse(**d)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/rdd/{regulation_id}", response_model=RDDResponse)
async def compute_rdd_live(
    regulation_id: str,
    threshold_label: str = Query("sifi_10b", description="sifi_10b | sifi_50b"),
    outcome_var: str = Query("tier1_capital_ratio"),
) -> RDDResponse:
    """Live RDD computation at the specified asset threshold."""
    from backend.models.causal_inference import compute_rdd, ASSET_THRESHOLDS
    if threshold_label not in ASSET_THRESHOLDS:
        raise HTTPException(
            status_code=422,
            detail=f"threshold_label must be one of {list(ASSET_THRESHOLDS.keys())}",
        )
    try:
        r = compute_rdd(regulation_id, threshold_label, outcome_var)
        d = r.to_dict()
        d["interpretation"] = _interpret_rdd(d)
        return RDDResponse(**d)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/bn/score", response_model=BNScoreResponse)
async def bayesian_network_score(body: BNScoreRequest) -> BNScoreResponse:
    """
    Compute Bayesian Network posterior P(ImpactLevel | evidence).

    Pass continuous scores; the BN discretises them internally using
    the calibrated thresholds (drift < 0.15/0.50, rwa < $50M/$500M, jsd p < 0.05).
    """
    from backend.models.bayesian_network import get_default_bn
    bn = get_default_bn()
    result = bn.infer_from_scores(
        drift_score=body.drift_score,
        jsd_p_value=body.jsd_p_value,
        rwa_median_million=body.rwa_median_million,
    )
    return BNScoreResponse(**result)


@router.get("/summary", response_model=CausalSummary)
async def causal_summary() -> CausalSummary:
    """Aggregate stats across all stored causal estimates."""
    results = await list_causal_estimates()

    dids = [r for r in results if r.get("method") == "did"]
    scms = [r for r in results if r.get("method") == "synthetic_control"]
    rdds = [r for r in results if r.get("method") == "rdd"]

    mean_att_did = (
        float(sum(d["att"] for d in dids) / len(dids)) if dids else None
    )
    sig_did = sum(
        1 for d in dids
        if not (d.get("ci_low_95", -1) < 0 < d.get("ci_high_95", 1))
    )
    sig_rdd = sum(
        1 for r in rdds
        if not (r.get("ci_low_95", -1) < 0 < r.get("ci_high_95", 1))
    )

    return CausalSummary(
        n_did=len(dids),
        n_scm=len(scms),
        n_rdd=len(rdds),
        mean_att_did=mean_att_did,
        significant_did=sig_did,
        significant_rdd=sig_rdd,
    )
