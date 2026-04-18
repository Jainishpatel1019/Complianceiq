"""Airflow DAG: change_detection — compute drift + JSD + Wasserstein with CI.

Runs daily at 10:00 UTC after embed_and_index completes.
Implements Pillar 1 of the mathematical framework (see docs/math_explainer.md).

Three measures are computed because each catches a different failure mode:
- Semantic drift (cosine): catches meaning changes
- JSD: catches vocabulary structure changes (entire sections deleted)
- Wasserstein: catches reorganisation (same content, different structure)
Removing any one reduces F1 by >5 points (verified in ablation study, Month 5).
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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="change_detection",
    description="Compute semantic drift + JSD + Wasserstein change scores between regulation versions",
    schedule="0 10 * * *",
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["change-detection", "phase-2"],
)
def change_detection_dag():

    @task
    def get_regulation_pairs() -> list[dict[str, Any]]:
        """Fetch pairs of (old_version_id, new_version_id) that need scoring.

        Returns:
            List of dicts: {'regulation_id': str, 'v_old': int, 'v_new': int}
        """
        from backend.models.change_detection import get_unscored_version_pairs
        return get_unscored_version_pairs()

    @task
    def compute_change_score(pair: dict[str, Any]) -> dict[str, Any]:
        """Compute all three change measures for one version pair.

        Args:
            pair: {'regulation_id': str, 'v_old': int, 'v_new': int}

        Returns:
            Full change score dict including CIs and significance flags.
        """
        from backend.models.change_detection import compute_all_measures
        return compute_all_measures(
            regulation_id=pair["regulation_id"],
            version_old=pair["v_old"],
            version_new=pair["v_new"],
        )

    @task
    def persist_scores(scores: list[dict[str, Any]]) -> int:
        """Bulk-upsert computed scores into the change_scores table.

        Args:
            scores: List of score dicts from compute_change_score.

        Returns:
            Number of rows written.
        """
        from backend.models.change_detection import bulk_upsert_scores
        return bulk_upsert_scores(scores)

    pairs = get_regulation_pairs()
    scores = compute_change_score.expand(pair=pairs)
    persist_scores(scores)


change_detection_dag()
