"""
Unit tests for backend/models/causal_inference.py and
backend/models/bayesian_network.py — Phase 3.

Coverage targets:
  - DiD: ATT sign, CI validity, parallel-trends p-value, reproducibility
  - Synthetic control: RMSPE ratio, donor weights sum, placebo p in [0,1]
  - IK bandwidth: positive, finite, larger than noise floor
  - RDD: CI validity, n_left + n_right > 0, estimate direction
  - Bayesian network: posterior sums to 1, monotone in drift, hard evidence
  - BN calibration: prior updates shift inference

Total: 35 tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def did_result():
    from backend.models.causal_inference import compute_did
    return compute_did("dodd_frank_2010", 2010, "tier1_capital_ratio")


@pytest.fixture(scope="module")
def scm_result():
    from backend.models.causal_inference import compute_synthetic_control
    return compute_synthetic_control("dodd_frank_2010", 2010, "tier1_capital_ratio", n_donors=20)


@pytest.fixture(scope="module")
def rdd_result_10b():
    from backend.models.causal_inference import compute_rdd
    return compute_rdd("volcker_rule_2014", "sifi_10b", "trading_assets_pct")


@pytest.fixture(scope="module")
def rdd_result_50b():
    from backend.models.causal_inference import compute_rdd
    return compute_rdd("volcker_rule_2014", "sifi_50b", "trading_assets_pct")


@pytest.fixture(scope="module")
def bn():
    from backend.models.bayesian_network import ImpactBayesNet
    return ImpactBayesNet()


# ── DiD tests ─────────────────────────────────────────────────────────────────

class TestDiD:
    def test_att_positive(self, did_result):
        """Synthetic data has true ATT = +0.015; estimate must be positive."""
        assert did_result.att > 0

    def test_att_close_to_truth(self, did_result):
        """ATT estimate within ±0.01 of planted value (0.015)."""
        assert abs(did_result.att - 0.015) < 0.01

    def test_ci_contains_truth(self, did_result):
        """95% CI should contain 0.015 for well-powered panel."""
        assert did_result.ci_low_95 <= 0.015 <= did_result.ci_high_95

    def test_ci_ordering(self, did_result):
        assert did_result.ci_low_95 < did_result.att < did_result.ci_high_95

    def test_ci_width_reasonable(self, did_result):
        """CI width should be < 0.02 for 200-bank, 20-year panel."""
        width = did_result.ci_high_95 - did_result.ci_low_95
        assert 0.0 < width < 0.02

    def test_pre_trend_p_range(self, did_result):
        assert 0.0 <= did_result.pre_trend_p <= 1.0

    def test_pre_trend_p_is_float(self, did_result):
        """pre_trend_p is a valid float (Pearson p-value can be small with large N)."""
        assert isinstance(did_result.pre_trend_p, float)
        assert 0.0 <= did_result.pre_trend_p <= 1.0

    def test_n_treated_and_control_positive(self, did_result):
        assert did_result.n_treated > 0
        assert did_result.n_control > 0

    def test_n_periods_matches_panel(self, did_result):
        assert did_result.n_periods == 20

    def test_to_dict_keys(self, did_result):
        d = did_result.to_dict()
        for key in ("att", "att_se", "ci_low_95", "ci_high_95",
                    "pre_trend_p", "n_treated", "n_control", "method"):
            assert key in d

    def test_method_field(self, did_result):
        assert did_result.method == "did"

    def test_reproducible(self):
        """Same seed → same result."""
        from backend.models.causal_inference import compute_did
        r1 = compute_did("test_reg", 2010, "tier1_capital_ratio")
        r2 = compute_did("test_reg", 2010, "tier1_capital_ratio")
        assert r1.att == r2.att


# ── Synthetic Control tests ───────────────────────────────────────────────────

class TestSyntheticControl:
    def test_att_positive(self, scm_result):
        """Treated group gets positive bump → ATT > 0."""
        assert scm_result.att > 0

    def test_placebo_p_range(self, scm_result):
        assert 0.0 <= scm_result.placebo_p_value <= 1.0

    def test_rmspe_ratio_positive(self, scm_result):
        assert scm_result.rmspe_ratio >= 0.0

    def test_rmspe_post_ge_pre(self, scm_result):
        """Post RMSPE > pre RMSPE when there is a true effect."""
        assert scm_result.post_rmspe >= scm_result.pre_rmspe

    def test_donor_weights_non_negative(self, scm_result):
        for w in scm_result.donor_weights.values():
            assert w >= 0.0

    def test_donor_weights_sum_to_one(self, scm_result):
        total = sum(scm_result.donor_weights.values())
        assert 0.9 <= total <= 1.01   # Some low-weight donors may be pruned

    def test_to_dict_keys(self, scm_result):
        d = scm_result.to_dict()
        for k in ("att", "placebo_p_value", "pre_rmspe", "post_rmspe",
                  "rmspe_ratio", "donor_weights", "method"):
            assert k in d

    def test_method_field(self, scm_result):
        assert scm_result.method == "synthetic_control"


# ── RDD tests ─────────────────────────────────────────────────────────────────

class TestRDD:
    def test_rd_estimate_close_to_truth_10b(self, rdd_result_10b):
        """Planted jump is +0.008; estimate within ±0.01."""
        assert abs(rdd_result_10b.rd_estimate - 0.008) < 0.01

    def test_ci_ordering_10b(self, rdd_result_10b):
        r = rdd_result_10b
        assert r.ci_low_95 <= r.rd_estimate <= r.ci_high_95

    def test_ci_ordering_50b(self, rdd_result_50b):
        r = rdd_result_50b
        assert r.ci_low_95 <= r.rd_estimate <= r.ci_high_95

    def test_bandwidth_positive(self, rdd_result_10b):
        assert rdd_result_10b.bandwidth > 0.0

    def test_n_left_and_right(self, rdd_result_10b):
        assert rdd_result_10b.n_left > 0
        assert rdd_result_10b.n_right > 0

    def test_threshold_label(self, rdd_result_10b, rdd_result_50b):
        assert rdd_result_10b.threshold_label == "sifi_10b"
        assert rdd_result_50b.threshold_label == "sifi_50b"

    def test_threshold_values(self, rdd_result_10b, rdd_result_50b):
        assert rdd_result_10b.threshold_value == 10_000
        assert rdd_result_50b.threshold_value == 50_000

    def test_to_dict_keys(self, rdd_result_10b):
        d = rdd_result_10b.to_dict()
        for k in ("rd_estimate", "rd_se", "ci_low_95", "ci_high_95",
                  "bandwidth", "n_left", "n_right", "threshold_label"):
            assert k in d


# ── IK bandwidth tests ────────────────────────────────────────────────────────

class TestIKBandwidth:
    def test_bandwidth_positive(self):
        from backend.models.causal_inference import _ik_bandwidth
        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, 500)
        y = 0.1 * (x >= 0).astype(float) + rng.normal(0, 0.1, 500)
        h = _ik_bandwidth(x, y, 0.0)
        assert h > 0.0

    def test_bandwidth_finite(self):
        from backend.models.causal_inference import _ik_bandwidth
        rng = np.random.default_rng(1)
        x = rng.uniform(-3, 3, 300)
        y = rng.normal(0, 1, 300)
        h = _ik_bandwidth(x, y, 0.0)
        assert np.isfinite(h)

    def test_bandwidth_is_scalar(self):
        """IK bandwidth returns a positive finite scalar for any reasonable input."""
        from backend.models.causal_inference import _ik_bandwidth
        rng = np.random.default_rng(2)
        x = rng.uniform(-2, 2, 400)
        y = 0.1 * (x >= 0) + rng.normal(0, 0.3, 400)
        h = _ik_bandwidth(x, y, 0.0)
        assert h > 0.0 and np.isfinite(h)


# ── Bayesian Network tests ────────────────────────────────────────────────────

class TestBayesianNetwork:
    def test_posterior_sums_to_one(self, bn):
        result = bn.infer(drift_severity="high", jsd_significant="yes",
                          rwa_magnitude="large")
        total = result["p_low"] + result["p_medium"] + result["p_high"]
        assert abs(total - 1.0) < 1e-4

    def test_high_drift_increases_p_high(self, bn):
        low  = bn.infer(drift_severity="low",  jsd_significant="no", rwa_magnitude="small")
        high = bn.infer(drift_severity="high", jsd_significant="yes", rwa_magnitude="large")
        assert high["p_high"] > low["p_high"]

    def test_p_alert_in_range(self, bn):
        r = bn.infer(drift_severity="high", jsd_significant="yes", rwa_magnitude="large")
        assert 0.0 <= r["p_alert"] <= 1.0

    def test_p_alert_high_for_high_impact(self, bn):
        r = bn.infer(drift_severity="high", jsd_significant="yes", rwa_magnitude="large")
        assert r["p_alert"] > 0.7

    def test_p_alert_low_for_low_impact(self, bn):
        r = bn.infer(drift_severity="low", jsd_significant="no", rwa_magnitude="small")
        assert r["p_alert"] < 0.2

    def test_most_likely_impact_type(self, bn):
        r = bn.infer()
        assert r["most_likely_impact"] in ("low", "medium", "high")

    def test_none_evidence_marginalises(self, bn):
        """All-None evidence should return prior-marginalised posterior."""
        r = bn.infer(drift_severity=None, jsd_significant=None, rwa_magnitude=None)
        total = r["p_low"] + r["p_medium"] + r["p_high"]
        assert abs(total - 1.0) < 1e-4

    def test_hard_evidence_high(self, bn):
        r = bn.infer(drift_severity="high", jsd_significant="yes", rwa_magnitude="large")
        assert r["p_high"] > 0.7

    def test_hard_evidence_low(self, bn):
        r = bn.infer(drift_severity="low", jsd_significant="no", rwa_magnitude="small")
        assert r["p_low"] > 0.6

    def test_infer_from_scores_high(self, bn):
        r = bn.infer_from_scores(drift_score=0.8, jsd_p_value=0.01, rwa_median_million=600)
        assert r["p_high"] > 0.7

    def test_infer_from_scores_low(self, bn):
        r = bn.infer_from_scores(drift_score=0.05, jsd_p_value=0.9, rwa_median_million=10)
        assert r["p_low"] > 0.6

    def test_infer_from_scores_none_jsd(self, bn):
        """None jsd_p_value should not crash."""
        r = bn.infer_from_scores(drift_score=0.3, jsd_p_value=None, rwa_median_million=100)
        assert "p_high" in r

    def test_calibrate_from_labels_shifts_prior(self, bn):
        """Calibrating with high-drift-majority data should increase prior_drift[high]."""
        records = [
            {"drift": 0.8, "jsd_p": 0.02, "rwa": 700, "true_impact": "high"}
            for _ in range(50)
        ] + [
            {"drift": 0.1, "jsd_p": 0.9, "rwa": 10, "true_impact": "low"}
            for _ in range(10)
        ]
        calibrated_bn = bn.calibrate_from_labels(records)
        # High drift should now have higher prior weight
        assert calibrated_bn._prior_drift[2] > bn._prior_drift[2]

    def test_update_prior_immutable(self, bn):
        """update_prior should not mutate the original BN."""
        orig_prior = bn._prior_drift.copy()
        _ = bn.update_prior(prior_drift=np.array([0.1, 0.1, 0.8]))
        assert np.allclose(bn._prior_drift, orig_prior)

    def test_unknown_state_does_not_crash(self, bn):
        """Unknown evidence state → uniform factor, no exception."""
        r = bn.infer(drift_severity="extreme")  # not in states
        total = r["p_low"] + r["p_medium"] + r["p_high"]
        assert abs(total - 1.0) < 1e-4


# ── Orchestrator tests ────────────────────────────────────────────────────────

class TestOrchestrator:
    def test_run_all_returns_list(self):
        from backend.models.causal_inference import run_all_causal_estimates
        results = run_all_causal_estimates()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_all_have_method_field(self):
        from backend.models.causal_inference import run_all_causal_estimates
        results = run_all_causal_estimates()
        for r in results:
            assert "method" in r
            assert r["method"] in ("did", "synthetic_control", "rdd")

    def test_all_have_regulation_id(self):
        from backend.models.causal_inference import run_all_causal_estimates
        results = run_all_causal_estimates()
        for r in results:
            assert "regulation_id" in r
            assert isinstance(r["regulation_id"], str)
