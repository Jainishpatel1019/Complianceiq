"""Airflow DAG: causal_estimation — DiD + synthetic control on FDIC data.

Runs monthly (1st of month, 04:00 UTC). Causal estimation is expensive
(30-60s per regulation with econml), so we pre-compute and cache results.
The LangGraph agent reads from causal_estimates table — not running econml live.

Pillar 2 of the mathematical framework.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

LANDMARK_REGULATIONS = [
    "2010-02178",   # Dodd-Frank (synthetic control — affected all banks)
    "2013-00962",   # Basel III capital rules (DiD — threshold at $10B)
    "2014-03044",   # Volcker Rule (DiD — threshold at $10B)
]

default_args: dict[str, Any] = {"owner": "complianceiq", "retries": 1, "retry_delay": timedelta(minutes=10)}


@dag(
    dag_id="causal_estimation",
    description="Run DiD + synthetic control causal inference on FDIC call report data",
    schedule="0 4 1 * *",  # Monthly
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["causal", "phase-3"],
)
def causal_estimation_dag():

    @task
    def run_did_estimation(regulation_doc_number: str) -> dict[str, Any]:
        """Run Difference-in-Differences via econml for one regulation.

        Treatment group: banks above $10B assets (affected by capital rules).
        Control group: banks just below $10B (not affected, but similar).
        Outcome: tier1_capital_ratio from FDIC quarterly call reports.

        Args:
            regulation_doc_number: Federal Register document number.

        Returns:
            Dict with att_estimate, se, p_value, ci_low_95, ci_high_95,
            parallel_trends_p_value.
        """
        from backend.models.causal_inference import run_did
        return run_did(regulation_doc_number, threshold_assets_bn=10.0)

    @task
    def run_synthetic_control(regulation_doc_number: str) -> dict[str, Any]:
        """Run synthetic control for landmark regulations (all banks affected).

        Constructs synthetic US banking sector from weighted mix of
        international banking sectors not subject to the regulation.
        The gap Y_actual - Y_synthetic IS the causal effect.

        Args:
            regulation_doc_number: Document number of landmark regulation.

        Returns:
            Dict with att_estimate, se, p_value, ci_low_95, ci_high_95.
        """
        from backend.models.causal_inference import run_synthetic_control
        return run_synthetic_control(regulation_doc_number)

    @task
    def persist_causal_estimates(estimates: list[dict[str, Any]]) -> int:
        """Upsert causal estimates into causal_estimates table.

        Args:
            estimates: List of estimate dicts.

        Returns:
            Number of rows written.
        """
        from backend.models.causal_inference import upsert_estimates
        return upsert_estimates(estimates)

    # DiD for all regulations with clear treatment thresholds
    all_regulations = LANDMARK_REGULATIONS
    did_results = run_did_estimation.expand(regulation_doc_number=all_regulations)

    # Synthetic control only for Dodd-Frank (affects all banks)
    sc_result = run_synthetic_control("2010-02178")

    persist_causal_estimates([did_results, sc_result])


causal_estimation_dag()
