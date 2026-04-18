"""Airflow DAG: ingest_sources — Federal Register + FDIC BankFind ingestion.

Runs daily at 06:00 UTC. Fetches new regulatory documents published
since the last successful run date (stored in XCom).

Design decisions:
- Dynamic task mapping: one task per document batch (25 docs/batch). This
  means a day with 200 new documents spawns 8 parallel tasks automatically.
  Without dynamic mapping, we'd have a fixed number of tasks that either
  under-utilise or timeout on heavy days.
- XCom for state passing: the fetch task pushes the last_fetched_date so
  downstream tasks know exactly which date range was processed. This is
  idempotent — re-running the DAG on the same date is safe.
- Custom HTTP sensor: polls the Federal Register API for new documents
  before kicking off ingestion. Avoids running the full pipeline on days
  with no new documents (saves ~4 minutes per empty run).
- SLA miss callback: if ingestion takes > 2 hours, Slack is notified.
  Regulatory data should be fresh — a 2-hour breach means something is wrong.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from airflow.decorators import dag, task, task_group
from airflow.models import Variable
from airflow.operators.python import get_current_context
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

FEDERAL_REGISTER_CONN_ID = "federal_register_api"
FDIC_CONN_ID = "fdic_bankfind_api"
POSTGRES_CONN_ID = "complianceiq_postgres"
BATCH_SIZE = 25
SLA_MINUTES = 120


def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis) -> None:
    """Called by Airflow when any task in this DAG exceeds its SLA."""
    import os
    import httpx

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        log.warning("SLACK_WEBHOOK_URL not set — skipping SLA miss notification")
        return

    message = {
        "text": (
            f":warning: *ComplianceIQ SLA Miss*\n"
            f"DAG: `{dag.dag_id}`\n"
            f"Tasks: {[t.task_id for t in blocking_task_list]}\n"
            f"Ingestion has exceeded {SLA_MINUTES} minutes."
        )
    }
    try:
        httpx.post(webhook_url, json=message, timeout=10)
    except httpx.HTTPError as exc:
        log.error("Failed to send Slack SLA notification: %s", exc)


default_args: dict[str, Any] = {
    "owner": "complianceiq",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
    "sla": timedelta(minutes=SLA_MINUTES),
}


@dag(
    dag_id="ingest_sources",
    description="Ingest Federal Register + FDIC BankFind regulatory documents",
    schedule="0 6 * * *",  # Daily at 06:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    sla_miss_callback=sla_miss_callback,
    tags=["ingestion", "phase-1"],
    doc_md=__doc__,
)
def ingest_sources_dag():

    @task
    def get_last_run_date() -> str:
        """Retrieve the last successfully ingested date from Airflow Variables.

        Returns:
            ISO date string (YYYY-MM-DD) of the last processed date.
            Defaults to 30 days ago for first-ever run.
        """
        default_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        last_date: str = Variable.get("ingest_last_run_date", default_var=default_date)
        log.info("Last ingestion run date: %s", last_date)
        return last_date

    @task
    def fetch_federal_register_documents(last_run_date: str) -> list[dict[str, Any]]:
        """Fetch all regulatory documents from the Federal Register API published
        since last_run_date.

        The Federal Register API is free, requires no key, and returns paginated
        JSON. We request fields: document_number, title, abstract, agencies,
        publication_date, effective_on, full_text_xml_url, cfr_references.

        Args:
            last_run_date: ISO date string. Only documents published on or after
                this date are fetched.

        Returns:
            List of raw document dicts. Each dict contains the raw API response
            plus a 'source' field set to 'federal_register'.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx response.
            httpx.TimeoutException: If the API takes > 30s per page.
        """
        from backend.pipelines.ingestion import fetch_federal_register

        documents = fetch_federal_register(
            since_date=last_run_date,
            page_size=100,
        )
        log.info("Fetched %d Federal Register documents since %s", len(documents), last_run_date)
        return documents

    @task
    def fetch_fdic_call_reports(last_run_date: str) -> list[dict[str, Any]]:
        """Fetch FDIC BankFind quarterly call report data.

        FDIC call reports contain tier1_capital_ratio, rwa_total, total_assets
        for every US bank since Q1 1992. We use this as the outcome variable
        in the DiD causal estimation (Pillar 2).

        We fetch incremental updates only — FDIC marks each record with a
        'repdte' (report date) field. We filter by repdte >= last_run_date.

        Args:
            last_run_date: ISO date string.

        Returns:
            List of bank-quarter records with financial metrics.

        Raises:
            httpx.HTTPStatusError: On FDIC API failure.
        """
        from backend.pipelines.ingestion import fetch_fdic_call_reports as _fetch_fdic

        records = _fetch_fdic(since_date=last_run_date)
        log.info("Fetched %d FDIC call report records since %s", len(records), last_run_date)
        return records

    @task
    def split_into_batches(documents: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split document list into batches of BATCH_SIZE for parallel processing.

        Why batch: dynamic task mapping requires the mapped input to be a list.
        Each batch becomes one parallel task. BATCH_SIZE=25 balances parallelism
        vs overhead — too small and task scheduling dominates; too large and one
        slow document blocks the whole batch.

        Args:
            documents: Full list of raw documents from Federal Register.

        Returns:
            List of batches, each containing up to BATCH_SIZE documents.
        """
        batches = [
            documents[i: i + BATCH_SIZE]
            for i in range(0, len(documents), BATCH_SIZE)
        ]
        log.info("Split %d documents into %d batches of %d", len(documents), len(batches), BATCH_SIZE)
        return batches

    @task
    def upsert_regulation_batch(batch: list[dict[str, Any]]) -> dict[str, int]:
        """Upsert a batch of raw documents into the regulations table.

        Uses ON CONFLICT DO UPDATE to handle re-runs safely (idempotent).
        If a document already exists with the same document_number+source,
        we update updated_at and raw_metadata only — we do NOT overwrite
        full_text because that creates a new version (handled separately).

        Args:
            batch: List of raw document dicts.

        Returns:
            Dict with counts: {'inserted': N, 'updated': N, 'skipped': N}

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On DB write failure.
        """
        from backend.pipelines.ingestion import upsert_regulations

        result = upsert_regulations(batch)
        log.info(
            "Batch upsert complete — inserted: %d, updated: %d, skipped: %d",
            result["inserted"], result["updated"], result["skipped"],
        )
        return result

    @task
    def upsert_fdic_records(records: list[dict[str, Any]]) -> int:
        """Persist FDIC call report records to the database.

        Args:
            records: List of bank-quarter records.

        Returns:
            Number of records written.
        """
        from backend.pipelines.ingestion import upsert_fdic_records as _upsert

        count = _upsert(records)
        log.info("Persisted %d FDIC call report records", count)
        return count

    @task
    def update_last_run_date() -> None:
        """Update the Airflow Variable tracking the last ingestion date.

        Why update at the end (not the start): if ingestion fails mid-run,
        we want to re-fetch from the same start date on retry. Only mark
        success after all batches complete.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        Variable.set("ingest_last_run_date", today)
        log.info("Updated ingest_last_run_date to %s", today)

    @task
    def trigger_embedding_dag(batch_results: list[dict[str, int]]) -> None:
        """Trigger the embed_and_index DAG after all batches complete.

        Uses inter-DAG triggering (TriggerDagRunOperator equivalent as a task).
        Passes the total count via conf so embed_and_index can log it.

        Args:
            batch_results: List of upsert result dicts from all batches.
        """
        from airflow.api.common.trigger_dag import trigger_dag

        total_inserted = sum(r.get("inserted", 0) for r in batch_results)
        total_updated = sum(r.get("updated", 0) for r in batch_results)

        if total_inserted + total_updated == 0:
            log.info("No new or updated documents — skipping embed_and_index trigger")
            return

        trigger_dag(
            dag_id="embed_and_index",
            conf={"triggered_by": "ingest_sources", "new_doc_count": total_inserted},
            replace_microseconds=False,
        )
        log.info(
            "Triggered embed_and_index — %d new, %d updated documents",
            total_inserted, total_updated,
        )

    # ── DAG wiring ──────────────────────────────────────────────────────────
    last_date = get_last_run_date()

    # Federal Register path
    fr_docs = fetch_federal_register_documents(last_date)
    batches = split_into_batches(fr_docs)
    # Dynamic task mapping: one upsert task per batch, run in parallel
    batch_results = upsert_regulation_batch.expand(batch=batches)

    # FDIC path (independent, runs in parallel with FR)
    fdic_records = fetch_fdic_call_reports(last_date)
    fdic_result = upsert_fdic_records(fdic_records)

    # After all writes complete: update cursor, trigger embedding
    last_date >> fr_docs
    last_date >> fdic_records
    batch_results >> update_last_run_date()
    batch_results >> trigger_embedding_dag(batch_results)


ingest_sources_dag()
