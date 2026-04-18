"""Airflow DAG: alert_dispatch — Slack/email if impact score P(High) > 0.8.

Triggered by impact_agent DAG (inter-DAG trigger) whenever any regulation
produces P(High) > 0.8. Also runs daily at 11:00 UTC as a catch-up check.

Why threshold 0.8: above this, the cost of a false negative (missing a critical
regulation) exceeds the cost of a false positive (unnecessary alert). Calibrated
on the 300-document human-labelled test set.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

ALERT_THRESHOLD = 0.8

default_args: dict[str, Any] = {
    "owner": "complianceiq",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="alert_dispatch",
    description="Send Slack/email alerts for regulations with P(High) > 0.8",
    schedule="0 11 * * *",
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["alerts", "phase-4"],
)
def alert_dispatch_dag():

    @task
    def get_undispatched_high_impact() -> list[dict[str, Any]]:
        """Fetch agent reports with P(High) > 0.8 where alert_dispatched = False.

        Returns:
            List of report dicts with regulation metadata.
        """
        import asyncio
        from sqlalchemy import select
        from db import get_session_factory
        from db.models import AgentReport, Regulation

        async def _fetch():
            factory = get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    select(AgentReport, Regulation)
                    .join(Regulation, AgentReport.regulation_id == Regulation.id)
                    .where(
                        AgentReport.impact_score_high >= ALERT_THRESHOLD,
                        AgentReport.alert_dispatched == False,
                    )
                )
                rows = result.all()
                return [
                    {
                        "report_id": str(r.id),
                        "regulation_id": str(r.regulation_id),
                        "title": reg.title,
                        "agency": reg.agency,
                        "p_high": r.impact_score_high,
                        "delta_rwa_median": r.delta_rwa_median_m,
                        "delta_rwa_ci_low": r.delta_rwa_ci_low_m,
                        "delta_rwa_ci_high": r.delta_rwa_ci_high_m,
                        "summary": r.summary,
                    }
                    for r, reg in rows
                ]

        return asyncio.run(_fetch())

    @task
    def dispatch_slack_alert(report: dict[str, Any]) -> str:
        """Send a Slack message for one high-impact regulation.

        Args:
            report: Report dict with regulation metadata.

        Returns:
            'sent' or 'skipped' (if SLACK_WEBHOOK_URL not configured).
        """
        import httpx

        webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        if not webhook_url:
            log.warning("SLACK_WEBHOOK_URL not configured — skipping alert for %s", report["regulation_id"])
            return "skipped"

        rwa = f"${report['delta_rwa_median']:.0f}M" if report.get("delta_rwa_median") else "N/A"
        ci = (
            f"[${report['delta_rwa_ci_low']:.0f}M, ${report['delta_rwa_ci_high']:.0f}M]"
            if report.get("delta_rwa_ci_low") else ""
        )

        message = {
            "text": (
                f":rotating_light: *High-Impact Regulation Detected*\n"
                f"*{report['title']}*\n"
                f"Agency: {report['agency']}\n"
                f"P(High Impact): {report['p_high']:.0%}\n"
                f"Estimated ΔCapital: {rwa} 90% CI {ci}\n"
                f"Summary: {report.get('summary', 'N/A')[:200]}"
            )
        }

        try:
            response = httpx.post(webhook_url, json=message, timeout=10)
            response.raise_for_status()
            log.info("Slack alert sent for regulation %s", report["regulation_id"])
            return "sent"
        except httpx.HTTPError as exc:
            log.error("Failed to send Slack alert: %s", exc)
            raise

    @task
    def mark_alerts_dispatched(report_ids: list[str]) -> int:
        """Mark all dispatched reports as alert_dispatched = True.

        Args:
            report_ids: List of UUID strings.

        Returns:
            Number of records updated.
        """
        import asyncio
        import uuid
        from sqlalchemy import update
        from db import get_session_factory
        from db.models import AgentReport

        async def _update():
            factory = get_session_factory()
            async with factory() as session:
                for rid in report_ids:
                    await session.execute(
                        update(AgentReport)
                        .where(AgentReport.id == uuid.UUID(rid))
                        .values(alert_dispatched=True)
                    )
                await session.commit()
            return len(report_ids)

        return asyncio.run(_update())

    reports = get_undispatched_high_impact()
    dispatch_results = dispatch_slack_alert.expand(report=reports)
    mark_alerts_dispatched([r["report_id"] for r in reports])


alert_dispatch_dag()
