"""Airflow DAG: model_registry — MLflow tracking + auto-promote on F1 threshold.

Runs after evaluate_pipeline completes (inter-DAG trigger).
Compares current model vs champion. Auto-promotes if F1 improves by >= 0.02.

Why MLflow model registry: records every model version, its hyperparameters,
evaluation scores, and which git commit produced it. Interviewers at FAANG
expect to see experiment tracking — not just a final model file.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

F1_IMPROVEMENT_THRESHOLD = 0.02  # Promote if new F1 > champion F1 + this value

default_args: dict[str, Any] = {"owner": "complianceiq", "retries": 1}


@dag(
    dag_id="model_registry",
    description="Compare new model vs MLflow champion, auto-promote on F1 threshold",
    schedule=None,  # Triggered by evaluate_pipeline via inter-DAG call
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["mlflow", "phase-5"],
)
def model_registry_dag():

    @task
    def get_champion_metrics() -> dict[str, float]:
        """Fetch current champion model metrics from MLflow registry.

        Returns:
            Dict with f1, auroc, calibration_error for champion model.
            Returns zeros if no champion exists yet (first run).
        """
        import mlflow
        import os

        client = mlflow.tracking.MlflowClient(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        try:
            champion = client.get_model_version_by_alias("change_detector", "champion")
            run = client.get_run(champion.run_id)
            return {
                "f1": float(run.data.metrics.get("f1", 0)),
                "auroc": float(run.data.metrics.get("auroc", 0)),
            }
        except mlflow.exceptions.MlflowException:
            log.info("No champion model found — first run, will promote automatically")
            return {"f1": 0.0, "auroc": 0.0}

    @task
    def get_challenger_metrics() -> dict[str, float]:
        """Fetch the latest challenger run metrics from MLflow.

        Returns:
            Dict with f1, auroc for the latest evaluation run.
        """
        import mlflow
        import os

        client = mlflow.tracking.MlflowClient(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        runs = client.search_runs(
            experiment_ids=["0"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return {"f1": 0.0, "auroc": 0.0}

        latest = runs[0]
        return {
            "f1": float(latest.data.metrics.get("f1", 0)),
            "auroc": float(latest.data.metrics.get("auroc", 0)),
            "run_id": latest.info.run_id,
        }

    @task
    def maybe_promote(
        champion: dict[str, float],
        challenger: dict[str, float],
    ) -> str:
        """Promote challenger to champion if F1 improves by >= threshold.

        Args:
            champion: Current champion metrics.
            challenger: Latest challenger metrics.

        Returns:
            'promoted' or 'kept_champion'.
        """
        import mlflow
        import os

        improvement = challenger.get("f1", 0) - champion.get("f1", 0)
        log.info(
            "Champion F1=%.4f, Challenger F1=%.4f, improvement=%.4f",
            champion["f1"], challenger.get("f1", 0), improvement,
        )

        if improvement >= F1_IMPROVEMENT_THRESHOLD:
            client = mlflow.tracking.MlflowClient(
                tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            )
            run_id = challenger.get("run_id", "")
            client.set_registered_model_alias("change_detector", "champion", run_id)
            log.info("Promoted challenger (run %s) to champion", run_id)
            return "promoted"

        log.info("Challenger did not beat champion — keeping current champion")
        return "kept_champion"

    champion = get_champion_metrics()
    challenger = get_challenger_metrics()
    maybe_promote(champion, challenger)


model_registry_dag()
