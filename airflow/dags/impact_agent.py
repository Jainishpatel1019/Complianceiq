"""Airflow DAG: impact_agent — trigger LangGraph agent for flagged regulations.

Runs daily at 09:00 UTC. Queries for regulations flagged by change_detection
(drift > 0.15 OR JSD p-value < 0.05). Each flagged regulation becomes one
LangGraph agent invocation.

This is the orchestration layer only. The agent logic lives in
backend/agents/impact_agent.py.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

default_args: dict[str, Any] = {
    "owner": "complianceiq",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}


@dag(
    dag_id="impact_agent",
    description="Run LangGraph impact assessment agent on flagged regulations",
    schedule="0 9 * * *",
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["agent", "phase-4"],
)
def impact_agent_dag():

    @task
    def get_flagged_regulation_ids() -> list[str]:
        """Query for regulation IDs flagged by change detection today.

        Returns:
            List of UUID strings for flagged regulations.
        """
        from backend.agents.impact_agent import get_flagged_ids
        ids = get_flagged_ids()
        log.info("Found %d flagged regulations for agent analysis", len(ids))
        return ids

    @task
    def run_impact_agent(regulation_id: str) -> dict[str, Any]:
        """Run the full LangGraph StateGraph agent for one regulation.

        The agent executes: fetch_classify → rag_query → graph_reasoning
        → causal_math → report_generation (5 nodes, some parallel).

        Args:
            regulation_id: UUID string.

        Returns:
            Structured report dict written to agent_reports table.
        """
        from backend.agents.impact_agent import run_agent
        return run_agent(regulation_id)

    @task
    def check_alert_threshold(reports: list[dict[str, Any]]) -> list[str]:
        """Return regulation IDs where P(High) > 0.8 — need alert dispatch.

        Args:
            reports: List of report dicts from run_impact_agent.

        Returns:
            List of regulation IDs requiring alerts.
        """
        high_impact = [
            r["regulation_id"]
            for r in reports
            if r.get("impact_score_high", 0) > 0.8
        ]
        log.info("%d regulations exceed P(High) > 0.8 alert threshold", len(high_impact))
        return high_impact

    flagged_ids = get_flagged_regulation_ids()
    reports = run_impact_agent.expand(regulation_id=flagged_ids)
    check_alert_threshold(reports)


impact_agent_dag()
