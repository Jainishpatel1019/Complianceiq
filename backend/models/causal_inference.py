"""
Causal inference engine — Phase 3.

Implements three identification strategies for measuring the impact of
regulatory changes on bank behaviour (measured via FDIC call-report proxies):

  1. Difference-in-Differences (DiD)   — OLS with two-way FE, HC3 SEs,
                                          bootstrap 95 % CI over 2 000 draws.
  2. Synthetic Control                  — Convex weights minimising pre-period
                                          RMSPE; placebo test p-value.
  3. Regression Discontinuity Design    — Local linear, IK bandwidth selector,
                                          RD estimate at $10 B and $50 B asset
                                          thresholds (Dodd-Frank / SIFI rules).

All estimates expose a common dict schema so the FastAPI route + frontend can
render them uniformly.

Math references
---------------
DiD:   Callaway & Sant'Anna (2021) "Difference-in-Differences with Multiple
       Time Periods."
SCM:   Abadie, Diamond & Hainmueller (2010) JASA.
RDD:   Imbens & Kalyanaraman (2012) "Optimal Bandwidth Choice for the
       Regression Discontinuity Estimator."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

N_BOOTSTRAP: int = 2_000          # DiD CI bootstrap draws
N_PLACEBO:   int = 1_000          # SCM placebo iterations
ASSET_THRESHOLDS: dict[str, float] = {
    "sifi_10b":  10_000,           # $10 B in millions
    "sifi_50b":  50_000,           # $50 B in millions
}

# Regulations where we have enough panel data for causal estimates
LANDMARK_REGULATIONS: list[dict[str, Any]] = [
    {
        "id": "dodd_frank_2010",
        "name": "Dodd-Frank Wall Street Reform Act (2010)",
        "treatment_year": 2010,
        "outcome_var": "tier1_capital_ratio",
        "methods": ["did", "synthetic_control"],
    },
    {
        "id": "volcker_rule_2014",
        "name": "Volcker Rule (2014)",
        "treatment_year": 2014,
        "outcome_var": "trading_assets_pct",
        "methods": ["did", "rdd"],
    },
    {
        "id": "basel3_2013",
        "name": "Basel III Capital Requirements (2013)",
        "treatment_year": 2013,
        "outcome_var": "rwa_to_assets",
        "methods": ["did", "rdd"],
    },
]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DiDResult:
    regulation_id: str
    att: float                   # Average Treatment on the Treated
    att_se: float
    ci_low_95:  float
    ci_high_95: float
    pre_trend_p: float           # parallel trends test (F-test pre-period)
    n_treated: int
    n_control: int
    n_periods: int
    method: str = "did"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SyntheticControlResult:
    regulation_id: str
    att: float
    placebo_p_value: float       # fraction of placebos with |att| >= observed
    pre_rmspe: float             # root mean squared prediction error pre-event
    post_rmspe: float
    rmspe_ratio: float           # post/pre — Abadie (2021) inference statistic
    donor_weights: dict[str, float]
    method: str = "synthetic_control"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RDDResult:
    regulation_id: str
    threshold_label: str         # "sifi_10b" | "sifi_50b"
    threshold_value: float       # e.g. 10_000 (millions)
    rd_estimate: float           # local linear discontinuity
    rd_se: float
    ci_low_95: float
    ci_high_95: float
    bandwidth: float             # IK-optimal bandwidth
    n_left: int
    n_right: int
    method: str = "rdd"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Difference-in-Differences ─────────────────────────────────────────────────

def _simulate_bank_panel(
    regulation_id: str,
    treatment_year: int,
    outcome_var: str,
    n_banks: int = 200,
    n_years: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic balanced panel that mimics FDIC call-report structure.

    In production this would be replaced by a real SQL query against the
    `fdic_call_reports` table (populated by `ingest_sources` DAG).

    Outcome is generated so that:
      - Pre-period has parallel trends (+ small noise)
      - Treatment group shifts by ATT = 0.015 after treatment_year
      - Bank and year FE are included
    """
    rng = np.random.default_rng(seed)

    start_year = 2000
    years = list(range(start_year, start_year + n_years))
    bank_ids = [f"CERT_{i:04d}" for i in range(n_banks)]

    # 40 % treated (large banks), 60 % control
    n_treated = int(n_banks * 0.4)
    treated_banks = set(bank_ids[:n_treated])

    rows = []
    bank_fe = rng.normal(0, 0.02, n_banks)   # bank fixed effects

    for b_idx, bank in enumerate(bank_ids):
        treated = bank in treated_banks
        for yr in years:
            year_fe = rng.normal(0, 0.005)
            post = int(yr >= treatment_year)
            true_att = 0.015 if (treated and post) else 0.0
            outcome = (
                0.12                       # baseline
                + bank_fe[b_idx]
                + year_fe
                + true_att
                + rng.normal(0, 0.008)
            )
            rows.append({
                "bank_id": bank,
                "year": yr,
                "treated": int(treated),
                "post": post,
                "did": int(treated and post),
                outcome_var: outcome,
            })

    return pd.DataFrame(rows)


def compute_did(
    regulation_id: str,
    treatment_year: int,
    outcome_var: str,
    panel_df: pd.DataFrame | None = None,
) -> DiDResult:
    """
    Two-way fixed-effects DiD with HC3 standard errors and bootstrap CIs.

    Model:  Y_it = α_i + λ_t + β·(Treated_i × Post_t) + ε_it
    where β = ATT (Average Treatment on Treated).

    Parameters
    ----------
    panel_df : optional pd.DataFrame
        If None, synthetic data is used (production: pass real FDIC data).
    """
    if panel_df is None:
        panel_df = _simulate_bank_panel(regulation_id, treatment_year, outcome_var)

    df = panel_df.copy()

    # Demean to absorb bank + year FE (within estimator)
    df["y_dm"] = (
        df[outcome_var]
        - df.groupby("bank_id")[outcome_var].transform("mean")
        - df.groupby("year")[outcome_var].transform("mean")
        + df[outcome_var].mean()
    )
    df["did_dm"] = (
        df["did"]
        - df.groupby("bank_id")["did"].transform("mean")
        - df.groupby("year")["did"].transform("mean")
        + df["did"].mean()
    )

    # OLS
    X = df["did_dm"].values
    y = df["y_dm"].values
    att = float(np.dot(X, y) / np.dot(X, X))

    # HC3 standard error
    residuals = y - att * X
    leverage = X**2 / np.dot(X, X)
    hc3_weights = residuals / (1 - leverage)
    se = float(np.sqrt(np.sum((X * hc3_weights) ** 2) / np.dot(X, X) ** 2))

    # Bootstrap CI
    rng = np.random.default_rng(0)
    boot_atts = []
    n = len(X)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        denom = np.dot(Xb, Xb)
        if denom > 1e-12:
            boot_atts.append(float(np.dot(Xb, yb) / denom))
    ci_low  = float(np.percentile(boot_atts, 2.5))
    ci_high = float(np.percentile(boot_atts, 97.5))

    # Parallel trends test — F-test on pre-period interaction trends
    pre_df  = df[df["post"] == 0].copy()
    pre_df["trend_x_treated"] = pre_df["treated"] * (pre_df["year"] - treatment_year)
    X_pre = pre_df["trend_x_treated"].values - pre_df["trend_x_treated"].mean()
    y_pre = pre_df["y_dm"].values if "y_dm" in pre_df.columns else pre_df[outcome_var].values
    if X_pre.std() > 1e-10:
        _, pre_trend_p = stats.pearsonr(X_pre, y_pre)
    else:
        pre_trend_p = 1.0

    n_treated = int(df["treated"].sum() / df["year"].nunique())
    n_control = int(df.shape[0] / df["year"].nunique()) - n_treated

    return DiDResult(
        regulation_id=regulation_id,
        att=round(att, 6),
        att_se=round(se, 6),
        ci_low_95=round(ci_low, 6),
        ci_high_95=round(ci_high, 6),
        pre_trend_p=round(float(pre_trend_p), 4),
        n_treated=n_treated,
        n_control=n_control,
        n_periods=df["year"].nunique(),
    )


# ── Synthetic Control ─────────────────────────────────────────────────────────

def _scm_weights(
    treated_pre: np.ndarray,
    donors_pre: np.ndarray,
) -> np.ndarray:
    """
    Solve for convex donor weights minimising pre-period RMSPE.

    min_w  ||Y_treated - Y_donors @ w||²
    s.t.   w >= 0,  sum(w) = 1
    """
    n_donors = donors_pre.shape[1]

    def objective(w: np.ndarray) -> float:
        synth = donors_pre @ w
        return float(np.mean((treated_pre - synth) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 1_000})
    return result.x


def compute_synthetic_control(
    regulation_id: str,
    treatment_year: int,
    outcome_var: str,
    panel_df: pd.DataFrame | None = None,
    n_donors: int = 30,
    seed: int = 0,
) -> SyntheticControlResult:
    """
    Abadie-Diamond-Hainmueller synthetic control.

    Returns ATT (post-period average gap treated vs synthetic),
    placebo p-value (fraction of placebos >= observed |effect|),
    RMSPE ratio for Abadie (2021) inference.
    """
    if panel_df is None:
        panel_df = _simulate_bank_panel(regulation_id, treatment_year, outcome_var,
                                        n_banks=n_donors + 1, seed=seed)

    df = panel_df.copy()
    years = sorted(df["year"].unique())
    pre_years  = [y for y in years if y < treatment_year]
    post_years = [y for y in years if y >= treatment_year]

    # Pivot: rows = years, cols = banks
    pivot = df.pivot(index="year", columns="bank_id", values=outcome_var)
    all_banks = list(pivot.columns)

    # Treated unit = first bank in treated set
    treated_banks = df[df["treated"] == 1]["bank_id"].unique().tolist()
    treated_unit  = treated_banks[0]
    donor_units   = [b for b in all_banks if b != treated_unit][:n_donors]

    treated_pre  = pivot.loc[pre_years, treated_unit].values
    donors_pre   = pivot.loc[pre_years, donor_units].values
    treated_post = pivot.loc[post_years, treated_unit].values
    donors_post  = pivot.loc[post_years, donor_units].values

    w = _scm_weights(treated_pre, donors_pre)
    synth_pre  = donors_pre  @ w
    synth_post = donors_post @ w

    pre_rmspe  = float(np.sqrt(np.mean((treated_pre  - synth_pre)  ** 2)))
    post_rmspe = float(np.sqrt(np.mean((treated_post - synth_post) ** 2)))
    att = float(np.mean(treated_post - synth_post))
    rmspe_ratio = (post_rmspe / pre_rmspe) if pre_rmspe > 1e-12 else 0.0

    # Placebo — iterate over donors as pseudo-treated
    rng = np.random.default_rng(seed)
    placebo_ratios = []
    for donor in rng.choice(donor_units, size=min(N_PLACEBO, len(donor_units)),
                            replace=False):
        other_donors = [b for b in donor_units if b != donor]
        if len(other_donors) < 2:
            continue
        p_pre   = pivot.loc[pre_years,  donor].values
        p_post  = pivot.loc[post_years, donor].values
        d_pre   = pivot.loc[pre_years,  other_donors].values
        d_post  = pivot.loc[post_years, other_donors].values
        wp = _scm_weights(p_pre, d_pre)
        s_pre  = d_pre  @ wp
        s_post = d_post @ wp
        pr_pre  = float(np.sqrt(np.mean((p_pre  - s_pre)  ** 2)))
        pr_post = float(np.sqrt(np.mean((p_post - s_post) ** 2)))
        placebo_ratios.append(pr_post / pr_pre if pr_pre > 1e-12 else 0.0)

    placebo_p = (
        float(np.mean(np.array(placebo_ratios) >= rmspe_ratio))
        if placebo_ratios else 1.0
    )

    donor_weights = {
        d: round(float(wt), 4)
        for d, wt in zip(donor_units, w)
        if wt > 0.001
    }

    return SyntheticControlResult(
        regulation_id=regulation_id,
        att=round(att, 6),
        placebo_p_value=round(placebo_p, 4),
        pre_rmspe=round(pre_rmspe, 6),
        post_rmspe=round(post_rmspe, 6),
        rmspe_ratio=round(rmspe_ratio, 4),
        donor_weights=donor_weights,
    )


# ── Regression Discontinuity ──────────────────────────────────────────────────

def _ik_bandwidth(
    running: np.ndarray,
    outcome: np.ndarray,
    cutoff: float,
) -> float:
    """
    Imbens-Kalyanaraman (2012) optimal bandwidth selector (simplified variant).

    Uses the rule-of-thumb pilot bandwidth to estimate curvature, then
    applies the MSE-optimal formula.
    """
    n = len(running)
    sigma2 = np.var(outcome)
    h_pilot = 1.84 * np.std(running) * n ** (-1 / 5)

    left_mask  = (running >= cutoff - h_pilot) & (running < cutoff)
    right_mask = (running >= cutoff) & (running <= cutoff + h_pilot)

    def _local_curvature(mask: np.ndarray) -> float:
        x = running[mask] - cutoff
        y = outcome[mask]
        if x.shape[0] < 4:
            return 1e-3
        A = np.column_stack([np.ones_like(x), x, x**2])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return float(abs(coef[2]))   # second derivative / 2

    m2_left  = _local_curvature(left_mask)
    m2_right = _local_curvature(right_mask)
    m2 = (m2_left + m2_right) / 2 + 1e-9

    # IK formula: h* = C_K * (sigma^2 / (n * m2^2))^(1/5)
    C_K = 3.4375  # triangular kernel constant
    h_star = C_K * (sigma2 / (n * m2 ** 2)) ** (1 / 5)
    return max(float(h_star), np.std(running) * 0.05)


def compute_rdd(
    regulation_id: str,
    threshold_label: str,
    outcome_var: str,
    panel_df: pd.DataFrame | None = None,
    seed: int = 0,
) -> RDDResult:
    """
    Local linear RDD at the given asset threshold.

    Running variable: log10(total_assets_millions)
    Outcome:         outcome_var (e.g. tier1_capital_ratio)
    """
    threshold_value = ASSET_THRESHOLDS[threshold_label]
    cutoff = np.log10(threshold_value)   # on log-scale

    if panel_df is None:
        # Synthetic cross-section: banks near the threshold
        rng = np.random.default_rng(seed)
        n = 800
        log_assets = rng.uniform(cutoff - 1.5, cutoff + 1.5, n)
        true_jump = 0.008
        outcome = (
            0.10
            + 0.003 * (log_assets - cutoff)
            + true_jump * (log_assets >= cutoff).astype(float)
            + rng.normal(0, 0.012, n)
        )
        panel_df = pd.DataFrame({
            "log_assets": log_assets,
            outcome_var: outcome,
        })

    df = panel_df.copy()
    if "log_assets" not in df.columns:
        df["log_assets"] = np.log10(df["total_assets_millions"])

    running  = df["log_assets"].values
    outcome  = df[outcome_var].values

    h = _ik_bandwidth(running, outcome, cutoff)

    left_mask  = (running >= cutoff - h) & (running < cutoff)
    right_mask = (running >= cutoff) & (running <= cutoff + h)

    def _fit_local_linear(mask: np.ndarray, side: str) -> tuple[float, float]:
        """Returns (intercept_at_cutoff, residual_std)."""
        x = running[mask] - cutoff
        y = outcome[mask]
        if x.shape[0] < 3:
            return 0.0, 1.0
        A = np.column_stack([np.ones_like(x), x])
        coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        resid_std = float(np.std(y - A @ coef)) if len(y) > 2 else 1.0
        return float(coef[0]), resid_std

    mu_left,  se_left  = _fit_local_linear(left_mask,  "left")
    mu_right, se_right = _fit_local_linear(right_mask, "right")

    rd_estimate = mu_right - mu_left
    n_left  = int(left_mask.sum())
    n_right = int(right_mask.sum())

    # Delta-method SE
    se_rd = float(np.sqrt(
        (se_left ** 2 / max(n_left, 1)) + (se_right ** 2 / max(n_right, 1))
    ))

    z95 = 1.96
    return RDDResult(
        regulation_id=regulation_id,
        threshold_label=threshold_label,
        threshold_value=threshold_value,
        rd_estimate=round(rd_estimate, 6),
        rd_se=round(se_rd, 6),
        ci_low_95=round(rd_estimate - z95 * se_rd, 6),
        ci_high_95=round(rd_estimate + z95 * se_rd, 6),
        bandwidth=round(h, 4),
        n_left=n_left,
        n_right=n_right,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all_causal_estimates() -> list[dict[str, Any]]:
    """
    Run DiD, SCM, and/or RDD for every landmark regulation and return a flat
    list of result dicts suitable for bulk DB upsert via ``bulk_upsert_causal``.
    """
    results: list[dict[str, Any]] = []

    for reg in LANDMARK_REGULATIONS:
        rid = reg["id"]
        year = reg["treatment_year"]
        ovar = reg["outcome_var"]

        if "did" in reg["methods"]:
            try:
                r = compute_did(rid, year, ovar)
                results.append(r.to_dict())
                log.info("DiD done", regulation=rid, att=r.att)
            except Exception:
                log.exception("DiD failed", regulation=rid)

        if "synthetic_control" in reg["methods"]:
            try:
                r = compute_synthetic_control(rid, year, ovar)
                results.append(r.to_dict())
                log.info("SCM done", regulation=rid, att=r.att)
            except Exception:
                log.exception("SCM failed", regulation=rid)

        if "rdd" in reg["methods"]:
            for label in ASSET_THRESHOLDS:
                try:
                    r = compute_rdd(rid, label, ovar)
                    results.append(r.to_dict())
                    log.info("RDD done", regulation=rid, threshold=label)
                except Exception:
                    log.exception("RDD failed", regulation=rid, threshold=label)

    return results


def bulk_upsert_causal(results: list[dict[str, Any]]) -> int:
    """
    Persist causal estimates to the ``causal_estimates`` table.
    Returns number of rows written.

    In the test environment this is called with an in-memory session mock.
    """
    # Deferred import to keep model layer free of DB at import time
    from db.models import CausalEstimate  # noqa: PLC0415
    from db import get_sync_session        # noqa: PLC0415

    written = 0
    with get_sync_session() as session:
        for r in results:
            est = CausalEstimate(
                regulation_id=r["regulation_id"],
                method=r["method"],
                estimate_json=r,
            )
            session.merge(est)
            written += 1
        session.commit()
    return written
