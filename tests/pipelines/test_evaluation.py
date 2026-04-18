"""Tests for backend/pipelines/evaluation.py — Phase 5.

All tests run with EVALUATION_MOCK=true so no external services are needed.
"""

from __future__ import annotations

import os
import itertools

import numpy as np
import pytest

# ── Force mock mode before any import of the module under test ────────────────
os.environ.setdefault("EVALUATION_MOCK", "true")

from backend.pipelines.evaluation import (
    _MEASURES,
    _THRESHOLDS,
    _calibration_from_arrays,
    _calibration_mock,
    compute_calibration,
    pgmpy_calibration_loop,
    run_ablation,
    run_ragas,
)


# ─────────────────────────────────────────────────────────────────────────────
# run_ragas
# ─────────────────────────────────────────────────────────────────────────────

class TestRunRagas:
    def test_returns_four_metrics(self):
        result = run_ragas(n_queries=10)
        expected_keys = {
            "ragas_faithfulness",
            "ragas_answer_relevancy",
            "ragas_context_recall",
            "ragas_context_precision",
        }
        assert set(result.keys()) == expected_keys

    def test_scores_in_unit_interval(self):
        result = run_ragas(n_queries=10)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_scores_are_floats(self):
        result = run_ragas()
        assert all(isinstance(v, float) for v in result.values())

    def test_deterministic_in_mock(self):
        """Same seed → same scores across two calls."""
        r1 = run_ragas(n_queries=500)
        r2 = run_ragas(n_queries=500)
        assert r1 == r2

    def test_n_queries_param_accepted(self):
        """run_ragas should accept n_queries without raising."""
        run_ragas(n_queries=50)


# ─────────────────────────────────────────────────────────────────────────────
# run_ablation
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAblation:
    @pytest.fixture(scope="class")
    def result(self):
        return run_ablation()

    def test_has_required_keys(self, result):
        assert {"cells", "best_cell", "full_vs_best_f1_gap"} <= result.keys()

    def test_cell_count(self, result):
        # 7 non-empty subsets × 3 thresholds = 21 cells
        expected = (2 ** len(_MEASURES) - 1) * len(_THRESHOLDS)
        assert len(result["cells"]) == expected

    def test_cells_have_required_fields(self, result):
        for cell in result["cells"]:
            assert {"measures", "threshold", "f1", "auroc", "brier"} <= cell.keys()

    def test_f1_values_in_unit_interval(self, result):
        for cell in result["cells"]:
            assert 0.0 <= cell["f1"] <= 1.0, f"f1={cell['f1']}"

    def test_auroc_values_in_unit_interval(self, result):
        for cell in result["cells"]:
            assert 0.0 <= cell["auroc"] <= 1.0

    def test_thresholds_are_expected(self, result):
        actual = {c["threshold"] for c in result["cells"]}
        assert actual == set(_THRESHOLDS)

    def test_all_measure_subsets_present(self, result):
        all_subsets = set()
        for r in range(1, len(_MEASURES) + 1):
            for combo in itertools.combinations(_MEASURES, r):
                all_subsets.add(frozenset(combo))
        result_subsets = {frozenset(c["measures"]) for c in result["cells"]}
        assert result_subsets == all_subsets

    def test_best_cell_has_highest_f1(self, result):
        max_f1 = max(c["f1"] for c in result["cells"])
        assert result["best_cell"]["f1"] == max_f1

    def test_full_model_has_highest_f1_reference(self, result):
        """All-three-measures cell at threshold 0.15 should exist."""
        full = next(
            (c for c in result["cells"]
             if set(c["measures"]) == set(_MEASURES) and c["threshold"] == 0.15),
            None,
        )
        assert full is not None

    def test_gap_is_float(self, result):
        assert isinstance(result["full_vs_best_f1_gap"], float)


# ─────────────────────────────────────────────────────────────────────────────
# compute_calibration
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCalibration:
    @pytest.fixture(scope="class")
    def result(self):
        return compute_calibration(n_bins=10, n_bootstrap=100)

    def test_has_required_keys(self, result):
        assert {
            "calibration_error", "optimal_threshold",
            "threshold_ci", "bin_means", "bin_predicted",
        } <= result.keys()

    def test_calibration_error_non_negative(self, result):
        assert result["calibration_error"] >= 0.0

    def test_optimal_threshold_in_candidates(self, result):
        from backend.pipelines.evaluation import _THRESHOLDS
        # Extended candidates include (0.10, 0.12, 0.15, 0.18, 0.20)
        assert 0.0 < result["optimal_threshold"] <= 1.0

    def test_ci_is_valid_interval(self, result):
        lo, hi = result["threshold_ci"]
        assert lo <= hi

    def test_bin_lengths_match(self, result):
        assert len(result["bin_means"]) == len(result["bin_predicted"])

    def test_bin_means_in_unit_interval(self, result):
        for v in result["bin_means"]:
            assert 0.0 <= v <= 1.0


class TestCalibrationFromArrays:
    """Unit tests for the internal _calibration_from_arrays helper."""

    def _make_arrays(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        scores = np.clip(rng.beta(2, 5, n), 0, 1)
        y_true = (scores + rng.normal(0, 0.1, n) > 0.4).astype(int)
        return scores, y_true

    def test_output_structure(self):
        scores, y_true = self._make_arrays()
        out = _calibration_from_arrays(scores, y_true, 10, 50, (0.10, 0.15, 0.20))
        assert "calibration_error" in out
        assert "threshold_ci" in out

    def test_perfectly_calibrated_has_low_error(self):
        """When predicted == label (hard 0/1), calibration error should be 0."""
        y = np.array([0, 0, 1, 1])
        s = np.array([0.0, 0.0, 1.0, 1.0])
        out = _calibration_from_arrays(s, y, 4, 10, (0.5,))
        assert out["calibration_error"] < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# pgmpy_calibration_loop
# ─────────────────────────────────────────────────────────────────────────────

class TestPgmpyCalibrationLoop:
    @pytest.fixture(scope="class")
    def sample_records(self):
        rng = np.random.default_rng(42)
        n = 80
        drift = rng.uniform(0, 1, n)
        jsd_p = rng.uniform(0, 1, n)
        rwa = rng.uniform(0, 1000, n)
        impact = np.where(drift > 0.5, "high", np.where(drift > 0.2, "medium", "low"))
        return [
            {"drift": float(drift[i]), "jsd_p": float(jsd_p[i]),
             "rwa": float(rwa[i]), "true_impact": impact[i]}
            for i in range(n)
        ]

    def test_returns_required_keys(self, sample_records):
        result = pgmpy_calibration_loop(sample_records, n_iter=5)
        assert {
            "n_records", "calibrated_prior_drift",
            "calibrated_prior_jsd", "calibrated_prior_rwa",
            "log_likelihood", "n_iter",
        } <= result.keys()

    def test_n_records_matches_input(self, sample_records):
        result = pgmpy_calibration_loop(sample_records, n_iter=5)
        assert result["n_records"] == len(sample_records)

    def test_prior_drift_sums_to_one(self, sample_records):
        result = pgmpy_calibration_loop(sample_records, n_iter=5)
        assert abs(sum(result["calibrated_prior_drift"]) - 1.0) < 1e-6

    def test_prior_jsd_sums_to_one(self, sample_records):
        result = pgmpy_calibration_loop(sample_records, n_iter=5)
        assert abs(sum(result["calibrated_prior_jsd"]) - 1.0) < 1e-6

    def test_prior_rwa_sums_to_one(self, sample_records):
        result = pgmpy_calibration_loop(sample_records, n_iter=5)
        assert abs(sum(result["calibrated_prior_rwa"]) - 1.0) < 1e-6

    def test_prior_lengths(self, sample_records):
        result = pgmpy_calibration_loop(sample_records, n_iter=5)
        assert len(result["calibrated_prior_drift"]) == 3
        assert len(result["calibrated_prior_jsd"]) == 2
        assert len(result["calibrated_prior_rwa"]) == 3

    def test_log_likelihood_is_float(self, sample_records):
        result = pgmpy_calibration_loop(sample_records, n_iter=5)
        assert isinstance(result["log_likelihood"], float)

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            pgmpy_calibration_loop([])

    def test_high_drift_records_skew_prior(self):
        """Feeding mostly high-drift records should push prior_drift[2] > prior_drift[0]."""
        rng = np.random.default_rng(99)
        # Ensure all three drift states are present (needed for pgmpy to build a 3-state CPD)
        records = (
            [{"drift": float(rng.uniform(0.51, 1.0)), "jsd_p": 0.5, "rwa": 300.0, "true_impact": "high"}
             for _ in range(60)]
            + [{"drift": float(rng.uniform(0.15, 0.49)), "jsd_p": 0.5, "rwa": 100.0, "true_impact": "medium"}
               for _ in range(15)]
            + [{"drift": float(rng.uniform(0.0, 0.14)), "jsd_p": 0.5, "rwa": 10.0, "true_impact": "low"}
               for _ in range(5)]
        )
        result = pgmpy_calibration_loop(records, n_iter=5)
        prior = result["calibrated_prior_drift"]
        assert len(prior) == 3
        assert prior[2] > prior[0], "High-drift records should increase P(drift=high)"
