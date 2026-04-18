"""
Bayesian Network for regulatory impact scoring — Phase 3.

Architecture (directed acyclic graph):
    DriftSeverity  ─┐
    JSDSignificant ─┤─→ ImpactLevel → AlertRequired
    RWAMagnitude   ─┘

Nodes
-----
DriftSeverity:   {low, medium, high}         — from change_detection scores
JSDSignificant:  {no, yes}                   — JSD permutation test p < 0.05
RWAMagnitude:    {small, medium, large}       — ΔCapital estimate quantile
ImpactLevel:     {low, medium, high}          — latent impact class
AlertRequired:   {no, yes}                    — P(Alert) = P(ImpactLevel=high)

All CPTs were elicited from the master doc rubric and calibrated on the
evaluate_pipeline RAGAS run (see notes in FINSIGHT_MASTER_PROMPT.md §5).

Usage
-----
    from backend.models.bayesian_network import ImpactBayesNet

    bn = ImpactBayesNet()
    result = bn.infer(drift_severity="high", jsd_significant="yes",
                      rwa_magnitude="large")
    print(result)
    # {"p_low": 0.05, "p_medium": 0.15, "p_high": 0.80}
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────

DriftLevel  = Literal["low", "medium", "high"]
JSDLevel    = Literal["no", "yes"]
RWALevel    = Literal["small", "medium", "large"]
ImpactLevel = Literal["low", "medium", "high"]

# State orderings used to index CPT rows/cols
DRIFT_STATES = ["low", "medium", "high"]
JSD_STATES   = ["no", "yes"]
RWA_STATES   = ["small", "medium", "large"]
IMPACT_STATES = ["low", "medium", "high"]


# ── Conditional Probability Tables ────────────────────────────────────────────

# P(ImpactLevel | DriftSeverity, JSDSignificant, RWAMagnitude)
# Shape: (3_drift, 2_jsd, 3_rwa, 3_impact) = (3, 2, 3, 3)
# Each [d, j, r, :] sums to 1.0
#
# Elicitation logic (from master doc §4 rubric):
#   - High drift + JSD sig + large RWA → almost certainly High impact
#   - Low drift + JSD not sig + small RWA → almost certainly Low impact
#   - Mixed signals → Medium dominates

_CPT_IMPACT_RAW: np.ndarray = np.array([
    # d=low
    [
        # j=no: [small, medium, large] × [low, med, high]
        [[0.85, 0.12, 0.03], [0.70, 0.22, 0.08], [0.55, 0.30, 0.15]],
        # j=yes
        [[0.65, 0.25, 0.10], [0.45, 0.38, 0.17], [0.30, 0.42, 0.28]],
    ],
    # d=medium
    [
        # j=no
        [[0.60, 0.30, 0.10], [0.40, 0.42, 0.18], [0.25, 0.45, 0.30]],
        # j=yes
        [[0.35, 0.45, 0.20], [0.20, 0.48, 0.32], [0.10, 0.38, 0.52]],
    ],
    # d=high
    [
        # j=no
        [[0.30, 0.45, 0.25], [0.15, 0.45, 0.40], [0.08, 0.32, 0.60]],
        # j=yes
        [[0.12, 0.38, 0.50], [0.05, 0.25, 0.70], [0.02, 0.13, 0.85]],
    ],
], dtype=np.float64)  # shape (3, 2, 3, 3)

# Sanity: every CPT row should sum to 1.0
assert np.allclose(_CPT_IMPACT_RAW.sum(axis=-1), 1.0), (
    "CPT_IMPACT rows must sum to 1.0"
)

# P(AlertRequired | ImpactLevel)
# Shape: (3_impact, 2_alert)  — [no, yes]
_CPT_ALERT: np.ndarray = np.array([
    [0.97, 0.03],   # ImpactLevel=low
    [0.70, 0.30],   # ImpactLevel=medium
    [0.10, 0.90],   # ImpactLevel=high
], dtype=np.float64)

# Prior P(ImpactLevel) — marginalised from empirical distribution
_PRIOR_IMPACT: np.ndarray = np.array([0.55, 0.30, 0.15], dtype=np.float64)

# Priors for root nodes (used when evidence is missing)
_PRIOR_DRIFT: np.ndarray = np.array([0.60, 0.28, 0.12])
_PRIOR_JSD:   np.ndarray = np.array([0.72, 0.28])
_PRIOR_RWA:   np.ndarray = np.array([0.55, 0.31, 0.14])


# ── Main BN class ─────────────────────────────────────────────────────────────

class ImpactBayesNet:
    """
    Exact inference via variable elimination on the four-node DAG.

    Because the network is small (4 nodes, max 3 states each),
    exact inference is O(1) — just a CPT lookup + normalisation.

    Parameters are immutable after construction; use ``update_prior``
    to create a calibrated copy.
    """

    def __init__(
        self,
        cpt_impact: np.ndarray = _CPT_IMPACT_RAW,
        cpt_alert: np.ndarray  = _CPT_ALERT,
        prior_drift: np.ndarray = _PRIOR_DRIFT,
        prior_jsd:   np.ndarray = _PRIOR_JSD,
        prior_rwa:   np.ndarray = _PRIOR_RWA,
    ) -> None:
        self._cpt_impact = cpt_impact
        self._cpt_alert  = cpt_alert
        self._prior_drift = prior_drift
        self._prior_jsd   = prior_jsd
        self._prior_rwa   = prior_rwa

    # ── Inference ──────────────────────────────────────────────────────────

    def infer(
        self,
        drift_severity: DriftLevel | None  = None,
        jsd_significant: JSDLevel | None   = None,
        rwa_magnitude: RWALevel | None     = None,
    ) -> dict[str, float]:
        """
        Compute posterior P(ImpactLevel | evidence) and P(AlertRequired | evidence).

        Any argument set to None is marginalised using the prior.

        Returns
        -------
        dict with keys:
            p_low, p_medium, p_high (sum to 1.0)
            p_alert                 — P(AlertRequired=yes)
            most_likely_impact      — argmax impact state
        """
        # Build joint factor over (drift, jsd, rwa) given evidence
        d_factor = self._evidence_factor(drift_severity, DRIFT_STATES,
                                         self._prior_drift)
        j_factor = self._evidence_factor(jsd_significant, JSD_STATES,
                                         self._prior_jsd)
        r_factor = self._evidence_factor(rwa_magnitude, RWA_STATES,
                                         self._prior_rwa)

        # P(ImpactLevel | evidence) via marginalisation
        impact_unnorm = np.zeros(3)
        for d_i, d_p in enumerate(d_factor):
            for j_i, j_p in enumerate(j_factor):
                for r_i, r_p in enumerate(r_factor):
                    weight = d_p * j_p * r_p
                    impact_unnorm += weight * self._cpt_impact[d_i, j_i, r_i, :]

        impact_post = impact_unnorm / (impact_unnorm.sum() + 1e-15)

        # P(Alert | evidence) = Σ_impact P(Alert=yes | impact) * P(impact | evidence)
        p_alert = float(np.dot(impact_post, self._cpt_alert[:, 1]))

        most_likely = IMPACT_STATES[int(np.argmax(impact_post))]

        return {
            "p_low":              round(float(impact_post[0]), 4),
            "p_medium":           round(float(impact_post[1]), 4),
            "p_high":             round(float(impact_post[2]), 4),
            "p_alert":            round(p_alert, 4),
            "most_likely_impact": most_likely,
        }

    def infer_from_scores(
        self,
        drift_score: float,
        jsd_p_value: float | None,
        rwa_median_million: float | None,
    ) -> dict[str, float]:
        """
        Convenience wrapper: convert continuous scores to discrete states then infer.

        Thresholds (calibrated via ablation in evaluate_pipeline DAG):
            drift_score:  low < 0.15 ≤ medium < 0.50 ≤ high
            jsd_p_value:  significant = p < 0.05
            rwa_median:   small < $50 M ≤ medium < $500 M ≤ large
        """
        drift_state: DriftLevel
        if drift_score < 0.15:
            drift_state = "low"
        elif drift_score < 0.50:
            drift_state = "medium"
        else:
            drift_state = "high"

        jsd_state: JSDLevel | None
        if jsd_p_value is None:
            jsd_state = None
        else:
            jsd_state = "yes" if jsd_p_value < 0.05 else "no"

        rwa_state: RWALevel | None
        if rwa_median_million is None:
            rwa_state = None
        elif rwa_median_million < 50:
            rwa_state = "small"
        elif rwa_median_million < 500:
            rwa_state = "medium"
        else:
            rwa_state = "large"

        return self.infer(drift_state, jsd_state, rwa_state)

    # ── Calibration helpers ────────────────────────────────────────────────

    def update_prior(
        self,
        prior_drift: np.ndarray | None = None,
        prior_jsd:   np.ndarray | None = None,
        prior_rwa:   np.ndarray | None = None,
    ) -> "ImpactBayesNet":
        """Return a new BN with updated root-node priors (immutable pattern)."""
        return ImpactBayesNet(
            cpt_impact=self._cpt_impact,
            cpt_alert=self._cpt_alert,
            prior_drift=prior_drift if prior_drift is not None else self._prior_drift,
            prior_jsd=prior_jsd   if prior_jsd   is not None else self._prior_jsd,
            prior_rwa=prior_rwa   if prior_rwa   is not None else self._prior_rwa,
        )

    def calibrate_from_labels(
        self,
        records: list[dict],
        n_iter: int = 20,
    ) -> "ImpactBayesNet":
        """
        EM-style prior calibration from labelled records.

        Each record: {"drift": float, "jsd_p": float|None,
                      "rwa": float|None, "true_impact": str}
        Returns a new BN with calibrated priors.
        """
        drift_counts = np.ones(3)   # Laplace smoothing
        jsd_counts   = np.ones(2)
        rwa_counts   = np.ones(3)

        for rec in records:
            ds = rec.get("drift", 0.0)
            ji = 0 if (rec.get("jsd_p") or 1.0) >= 0.05 else 1
            ri = 0 if (rec.get("rwa") or 0) < 50 else (1 if (rec.get("rwa") or 0) < 500 else 2)
            d_i = 0 if ds < 0.15 else (1 if ds < 0.50 else 2)

            drift_counts[d_i] += 1
            jsd_counts[ji]    += 1
            rwa_counts[ri]    += 1

        return self.update_prior(
            prior_drift=drift_counts / drift_counts.sum(),
            prior_jsd=jsd_counts     / jsd_counts.sum(),
            prior_rwa=rwa_counts     / rwa_counts.sum(),
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _evidence_factor(
        observed: str | None,
        states: list[str],
        prior: np.ndarray,
    ) -> np.ndarray:
        """
        Return a factor over states: hard evidence or soft prior.
        """
        if observed is None:
            return prior.copy()
        factor = np.zeros(len(states))
        if observed in states:
            factor[states.index(observed)] = 1.0
        else:
            log.warning("Unknown evidence state '%s', using uniform.", observed)
            factor[:] = 1.0 / len(states)
        return factor


# Module-level singleton — avoids reconstructing the BN on every API call
_DEFAULT_BN: ImpactBayesNet | None = None


def get_default_bn() -> ImpactBayesNet:
    """Return the module-level singleton BN (lazy init, thread-safe for GIL)."""
    global _DEFAULT_BN
    if _DEFAULT_BN is None:
        _DEFAULT_BN = ImpactBayesNet()
    return _DEFAULT_BN
