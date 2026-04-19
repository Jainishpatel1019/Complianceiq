"""
Knowledge Graph REST endpoints — Phase 4.

Routes
------
GET /graph/snapshot          → Full graph (nodes + edges + analytics summary)
GET /graph/nodes             → Paginated node list with PageRank + community
GET /graph/subgraph/{reg_id} → k-hop ego-graph around a regulation
GET /graph/pagerank          → Top-N regulations by PageRank score
GET /graph/communities       → Community membership list
POST /graph/agent/{reg_id}   → Run impact agent and return final report
GET /graph/agent/status/{reg_id} → Latest agent trace steps from DB
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter()

# ── Cached in-process snapshot (rebuilt by Airflow, refreshed on startup) ─────
_SNAPSHOT_CACHE: Any = None
_SNAPSHOT_BUILT_AT: float = 0.0
_CACHE_TTL_SECONDS: int = 3600  # 1 hour — balances freshness vs. rebuild cost


def _get_snapshot() -> Any:
    """Return cached snapshot or build from synthetic data (dev fallback).

    TTL behaviour: cache is rebuilt every _CACHE_TTL_SECONDS seconds so
    the knowledge graph reflects updated PageRank / community scores from
    the latest Airflow run without requiring a pod restart.
    """
    global _SNAPSHOT_CACHE, _SNAPSHOT_BUILT_AT

    age = time.time() - _SNAPSHOT_BUILT_AT
    if _SNAPSHOT_CACHE is not None and age < _CACHE_TTL_SECONDS:
        return _SNAPSHOT_CACHE

    if _SNAPSHOT_CACHE is not None:
        log.info("Graph snapshot TTL expired (age=%.0fs) — rebuilding", age)

    from backend.models.graph_model import make_synthetic_regulations, build_full_snapshot
    regs = make_synthetic_regulations(n=40)
    _SNAPSHOT_CACHE = build_full_snapshot(regs, compute_gat=False)
    _SNAPSHOT_BUILT_AT = time.time()
    return _SNAPSHOT_CACHE


def _invalidate_cache() -> None:
    """Force cache expiry — call this after an Airflow graph-build run."""
    global _SNAPSHOT_CACHE, _SNAPSHOT_BUILT_AT
    _SNAPSHOT_CACHE = None
    _SNAPSHOT_BUILT_AT = 0.0


# ── Schemas ────────────────────────────────────────────────────────────────────

class NodeResponse(BaseModel):
    regulation_id: str
    document_number: str
    agency: str
    title: str
    doc_type: str
    pagerank: float
    community: int
    in_degree: int
    out_degree: int


class EdgeResponse(BaseModel):
    source_id: str
    target_id: str
    weight: float
    edge_type: str
    valid_from: str | None = None
    valid_to: str | None = None


class GraphSummary(BaseModel):
    n_nodes: int
    n_edges: int
    n_communities: int
    modularity: float
    built_at: str
    top_pagerank: list[tuple[str, float]]


class SubgraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


class AgentReportResponse(BaseModel):
    regulation_id: str
    document_number: str = ""
    agency: str = ""
    title: str = ""
    summary: str = ""
    impact_score: dict = Field(default_factory=dict)
    rwa_estimate: dict = Field(default_factory=dict)
    drift_score: float = 0.0
    graph_pagerank: float = 0.0
    community_id: int = -1
    reasoning_steps: int = 0
    alert_dispatched: bool = False


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/cache/invalidate", tags=["graph"])
async def invalidate_graph_cache() -> dict:
    """Force immediate cache invalidation.

    Called by the Airflow graph-build DAG after each successful run so the
    next request triggers a fresh snapshot. Safe to call multiple times.
    """
    _invalidate_cache()
    return {"invalidated": True, "message": "Graph cache cleared — will rebuild on next request"}


@router.get("/snapshot", response_model=GraphSummary)
async def get_graph_snapshot() -> GraphSummary:
    """High-level summary of the current regulation knowledge graph."""
    snapshot = _get_snapshot()
    s = snapshot.to_summary()
    return GraphSummary(**s)


@router.get("/nodes", response_model=list[NodeResponse])
async def list_nodes(
    agency: str | None = Query(None),
    community: int | None = Query(None),
    min_pagerank: float = Query(0.0),
    limit: int = Query(50, le=500),
    offset: int = Query(0),
) -> list[NodeResponse]:
    """List regulation nodes with filtering and pagination."""
    snapshot = _get_snapshot()
    nodes = snapshot.nodes

    if agency:
        nodes = [n for n in nodes if n.agency.upper() == agency.upper()]
    if community is not None:
        nodes = [n for n in nodes if n.community == community]
    if min_pagerank > 0:
        nodes = [n for n in nodes if n.pagerank >= min_pagerank]

    # Sort by PageRank desc
    nodes = sorted(nodes, key=lambda n: n.pagerank, reverse=True)
    page = nodes[offset : offset + limit]

    return [
        NodeResponse(
            regulation_id=n.regulation_id,
            document_number=n.document_number,
            agency=n.agency,
            title=n.title,
            doc_type=n.doc_type,
            pagerank=n.pagerank,
            community=n.community,
            in_degree=n.in_degree,
            out_degree=n.out_degree,
        )
        for n in page
    ]


@router.get("/edges", response_model=list[EdgeResponse])
async def list_edges(
    edge_type: str | None = Query(None),
    min_weight: float = Query(0.0),
    limit: int = Query(100, le=1000),
) -> list[EdgeResponse]:
    """List graph edges."""
    snapshot = _get_snapshot()
    edges = snapshot.edges

    if edge_type:
        edges = [e for e in edges if e.edge_type == edge_type]
    if min_weight > 0:
        edges = [e for e in edges if e.weight >= min_weight]

    edges = sorted(edges, key=lambda e: e.weight, reverse=True)[:limit]
    return [EdgeResponse(**e.to_dict()) for e in edges]


@router.get("/subgraph/{regulation_id}", response_model=SubgraphResponse)
async def get_subgraph(
    regulation_id: str,
    hops: int = Query(2, ge=1, le=3),
) -> SubgraphResponse:
    """k-hop ego-graph around a regulation node."""
    snapshot = _get_snapshot()
    from backend.models.graph_model import get_subgraph
    result = get_subgraph(snapshot, regulation_id, hops=hops)
    if not result["nodes"]:
        raise HTTPException(
            status_code=404,
            detail=f"Regulation '{regulation_id}' not found in graph",
        )
    return SubgraphResponse(**result)


@router.get("/pagerank", response_model=list[dict[str, Any]])
async def top_pagerank(
    n: int = Query(20, le=100),
) -> list[dict[str, Any]]:
    """Top-N regulations by PageRank score."""
    snapshot = _get_snapshot()
    ranked = sorted(
        snapshot.pagerank.items(), key=lambda x: x[1], reverse=True
    )[:n]

    node_map = {nd.regulation_id: nd for nd in snapshot.nodes}
    return [
        {
            "regulation_id": rid,
            "pagerank": pr,
            "agency": node_map.get(rid, type("x", (), {"agency": ""})()).agency,
            "title": (node_map.get(rid, type("x", (), {"title": ""})()).title)[:80],
            "community": node_map.get(rid, type("x", (), {"community": -1})()).community,
        }
        for rid, pr in ranked
    ]


@router.get("/communities", response_model=list[dict[str, Any]])
async def list_communities() -> list[dict[str, Any]]:
    """Community membership grouped by community id."""
    snapshot = _get_snapshot()
    community_map: dict[int, list[str]] = {}
    for nid, cid in snapshot.communities.items():
        community_map.setdefault(cid, []).append(nid)

    node_map = {nd.regulation_id: nd for nd in snapshot.nodes}
    result = []
    for cid, members in sorted(community_map.items()):
        agencies = list({node_map[m].agency for m in members if m in node_map})
        avg_pr = float(
            sum(snapshot.pagerank.get(m, 0) for m in members) / max(len(members), 1)
        )
        result.append({
            "community_id": cid,
            "n_members": len(members),
            "agencies": sorted(agencies),
            "avg_pagerank": round(avg_pr, 6),
            "members": members[:10],   # preview
        })
    return result


@router.post("/agent/{regulation_id}", response_model=AgentReportResponse)
async def run_agent(regulation_id: str) -> AgentReportResponse:
    """
    Run the LangGraph impact agent for a regulation.

    Runs synchronously (background task version is in api/websockets.py).
    Returns the assembled impact report.
    """
    try:
        from backend.agents.impact_agent import run_impact_agent
        report = run_impact_agent(regulation_id)
        if not report:
            raise HTTPException(status_code=500, detail="Agent returned empty report")
        return AgentReportResponse(**{
            k: report.get(k, AgentReportResponse.model_fields[k].default)
            for k in AgentReportResponse.model_fields
        })
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Agent run failed", regulation_id=regulation_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/agent/steps/{regulation_id}", response_model=list[str])
async def get_agent_steps(regulation_id: str) -> list[str]:
    """
    Return the reasoning steps from the most recent agent run for this regulation.
    Polled by the WebSocket client.
    """
    try:
        from db import get_db_session
        from db.models import AgentReport
        from sqlalchemy import select, desc

        async with get_db_session() as session:
            q = (
                select(AgentReport)
                .where(AgentReport.regulation_id == regulation_id)
                .order_by(desc(AgentReport.created_at))
                .limit(1)
            )
            row = (await session.execute(q)).scalar_one_or_none()
            if row:
                return row.agent_reasoning_trace or []
    except Exception as exc:
        log.warning("Failed to fetch agent trace for %s: %s", regulation_id, exc)
    return []
