"""Airflow DAG: evaluate_pipeline — RAGAS scores, ablation, calibration.

Runs weekly (Saturday 03:00 UTC). Computes retrieval and generation
quality metrics. Results committed to git as JSON artifacts via DVC.

Why weekly: evaluation requires human-labelled data (300 doc pairs for
change detection, 500 RAG queries for RAGAS). Running daily adds no value
since the labelled set is static between labelling sessions.
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
    dag_id="evaluate_pipeline",
    description="Compute RAGAS retrieval scores, change detection ablation, calibration curves",
    schedule="0 3 * * 6",  # Weekly, Saturday 03:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["evaluation", "phase-5"],
)
def evaluate_pipeline_dag():

    @task
    def run_ragas_evaluation() -> dict[str, float]:
        """Run RAGAS on 500 RAG queries from the labelled test set.

        Metrics: faithfulness, answer_relevancy, context_recall, context_precision.
        All four are needed — faithfulness alone misses retrieval failures.

        Returns:
            Dict of metric_name -> score.
        """
        from backend.pipelines.evaluation import run_ragas
        return run_ragas(n_queries=500)

    @task
    def run_change_detection_ablation() -> dict[str, Any]:
        """Run 16-cell ablation: all combinations of {drift, JSD, Wasserstein} on/off.

        Returns:
            Dict with F1, AUROC, calibration for each of 16 combinations.
        """
        from backend.pipelines.evaluation import run_ablation
        return run_ablation()

    @task
    def compute_calibration_curve() -> dict[str, Any]:
        """Compute calibration curve for change detection threshold.

        Plots predicted drift score vs actual label proportion to verify
        the threshold=0.15 cutoff is well-calibrated. Also runs bootstrap
        CI on the optimal threshold.

        Returns:
            Dict with calibration_error, optimal_threshold, threshold_ci.
        """
        from backend.pipelines.evaluation import compute_calibration
        return compute_calibration()

    @task
    def log_to_mlflow(
        ragas: dict[str, float],
        ablation: dict[str, Any],
        calibration: dict[str, Any],
    ) -> str:
        """Log all evaluation metrics to MLflow for tracking.

        Returns:
            MLflow run ID.
        """
        import mlflow
        import os
        from datetime import datetime

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        run_name = f"eval_{datetime.utcnow().strftime('%Y%m%d')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_metrics(ragas)
            mlflow.log_dict(ablation, "ablation_results.json")
            mlflow.log_dict(calibration, "calibration_results.json")
            log.info("Evaluation metrics logged to MLflow run %s", run.info.run_id)
            return run.info.run_id

    ragas = run_ragas_evaluation()
    ablation = run_change_detection_ablation()
    calibration = compute_calibration_curve()
    log_to_mlflow(ragas, ablation, calibration)


evaluate_pipeline_dag()
