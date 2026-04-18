"""Airflow DAG: graph_update — rebuild knowledge graph, run GAT re-embedding, Louvain.

Runs weekly (Sunday 02:00 UTC) — graph structure changes slowly; daily updates
would waste compute rebuilding PageRank on a near-identical graph.

Pillar 3 of the mathematical framework.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

default_args: dict[str, Any] = {"owner": "complianceiq", "retries": 1}


@dag(
    dag_id="graph_update",
    description="Rebuild regulation knowledge graph, recompute PageRank + Louvain, re-embed with GAT",
    schedule="0 2 * * 0",  # Weekly, Sunday 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["graph", "phase-4"],
)
def graph_update_dag():

    @task
    def parse_cross_references() -> int:
        """Parse CFR cross-references from regulation text, upsert graph edges.

        Returns:
            Number of edges upserted.
        """
        from backend.models.graph_model import parse_and_upsert_edges
        return parse_and_upsert_edges()

    @task
    def compute_pagerank(edges_updated: int) -> dict[str, float]:
        """Run NetworkX PageRank (d=0.85) on the full graph.

        Returns:
            Dict mapping node_id -> pagerank_score.
        """
        from backend.models.graph_model import compute_pagerank
        return compute_pagerank()

    @task
    def run_louvain(pagerank_scores: dict[str, float]) -> dict[str, int]:
        """Detect communities with Louvain algorithm.

        Returns:
            Dict mapping node_id -> community_id.
        """
        from backend.models.graph_model import run_louvain_communities
        return run_louvain_communities()

    @task
    def persist_graph_metrics(
        pagerank_scores: dict[str, float],
        communities: dict[str, int],
    ) -> None:
        """Write PageRank + community labels back to graph_nodes table."""
        from backend.models.graph_model import persist_metrics
        persist_metrics(pagerank_scores, communities)

    edges = parse_cross_references()
    pr = compute_pagerank(edges)
    communities = run_louvain(pr)
    persist_graph_metrics(pr, communities)


graph_update_dag()
