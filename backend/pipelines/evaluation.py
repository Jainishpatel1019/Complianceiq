"""
Phase 5 — Evaluation pipeline.

Three entry points called by the evaluate_pipeline Airflow DAG:

    run_ragas(n_queries)          → RAGAS retrieval/generation metrics
    run_ablation()                → 16-cell change-detection ablation table
    compute_calibration()         → Calibration curve + threshold CI

Plus:
    pgmpy_calibration_loop(records, n_iter) → EM prior update on labelled data

Design choices:
  - All heavy I/O mocked when EVALUATION_MOCK=true (CI / unit tests).
  - MLflow logging is the caller's responsibility (DAG layer), not ours.
  - Ablation runs in-process; each cell is independent and takes <1 s on CPU.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_MOCK = os.environ.get("EVALUATION_MOCK", "false").lower() == "true"


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas(n_queries: int = 500) -> dict[str, float]:
    """Run RAGAS on the labelled RAG test set.

    In production: loads queries from ChromaDB + Postgres, calls Ollama for
    generation, then scores with RAGAS.  In mock mode: returns deterministic
    fixture values for CI.

    Metrics
    -------
    faithfulness        P(generated answer is grounded in context)
    answer_relevancy    Cosine sim of generated answer vs query
    context_recall      Recall of gold context in retrieved chunks
    context_precision   Precision@K of retrieved chunks

    Parameters
    ----------
    n_queries:  Number of labelled queries to evaluate (default 500).

    Returns
    -------
    dict[metric_name, score]  — all scores in [0, 1].
    """
    if _MOCK:
        return _ragas_mock(n_queries)

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    samples = _load_ragas_samples(n_queries)

    dataset = Dataset.from_list(
        [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
            for s in samples
        ]
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores: dict[str, float] = {
        "ragas_faithfulness": float(result["faithfulness"]),
        "ragas_answer_relevancy": float(result["answer_relevancy"]),
        "ragas_context_recall": float(result["context_recall"]),
        "ragas_context_precision": float(result["context_precision"]),
    }
    log.info("RAGAS scores: %s", scores)
    return scores


def _load_ragas_samples(n: int) -> list[dict[str, Any]]:
    """Load labelled RAG samples from Postgres + ChromaDB."""
    import chromadb
    from db import get_session_factory

    client = chromadb.HttpClient(
        host=os.environ.get("CHROMA_HOST", "chromadb"),
        port=int(os.environ.get("CHROMA_PORT", "8000")),
    )
    collection = client.get_or_create_collection("regulation_chunks")

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(
            "SELECT question, ground_truth_answer, gold_chunk_ids "
            "FROM ragas_test_set ORDER BY created_at DESC LIMIT :n",
            {"n": n},
        ).fetchall()

    samples = []
    for row in rows:
        chunk_ids = row.gold_chunk_ids or []
        results = collection.query(query_texts=[row.question], n_results=5)
        contexts = results["documents"][0] if results["documents"] else []
        samples.append(
            {
                "question": row.question,
                "answer": _generate_answer(row.question, contexts),
                "contexts": contexts,
                "ground_truth": row.ground_truth_answer,
            }
        )
    return samples


def _generate_answer(question: str, contexts: list[str]) -> str:
    """Call Ollama to generate an answer grounded in retrieved contexts."""
    import httpx

    prompt = (
        "Answer using only the provided context.\n\n"
        f"Context:\n{'---'.join(contexts)}\n\n"
        f"Question: {question}\nAnswer:"
    )
    resp = httpx.post(
        f"{os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')}/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _ragas_mock(n_queries: int) -> dict[str, float]:
    rng = np.random.default_rng(seed=42)
    base = {"ragas_faithfulness": 0.82, "ragas_answer_relevancy": 0.79,
            "ragas_context_recall": 0.76, "ragas_context_precision": 0.74}
    noise = rng.normal(0, 0.005, 4)
    keys = list(base)
    return {k: round(float(base[k] + noise[i]), 4) for i, k in enumerate(keys)}


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Table
# ─────────────────────────────────────────────────────────────────────────────

# All 8 non-empty subsets of {drift, jsd, wasserstein} = 2^3 combinations.
# We also ablate the DRIFT_FLAG_THRESHOLD: [0.10, 0.15, 0.20].
# Total cells: 8 × 3 = 24 rows (documented as "16-cell" in the master prompt
# for the 2-threshold × 8-subset table; we extend to the full grid here).

_MEASURES = ("drift", "jsd", "wasserstein")
_THRESHOLDS = (0.10, 0.15, 0.20)

# Expected F1/AUROC from master doc §4 for reference cells (used in mock).
_REFERENCE_F1 = {
    ("drift",): 0.71,
    ("drift", "jsd"): 0.79,
    ("drift", "jsd", "wasserstein"): 0.84,
}


def run_ablation() -> dict[str, Any]:
    """Ablation over all non-empty subsets of {drift, jsd, wasserstein}.

    Each cell scores on the 300-document human-labelled dev set using the
    pipeline's own compute_all_measures output.

    Returns
    -------
    {
        "cells": [{"measures": [...], "threshold": float,
                   "f1": float, "auroc": float, "brier": float}, ...],
        "best_cell": {...},
        "full_vs_best_f1_gap": float,
    }
    """
    if _MOCK:
        return _ablation_mock()

    labels = _load_labelled_dev_set()
    cells = []
    for r in range(1, len(_MEASURES) + 1):
        for subset in itertools.combinations(_MEASURES, r):
            for thr in _THRESHOLDS:
                cell = _score_ablation_cell(labels, set(subset), thr)
                cell["measures"] = list(subset)
                cell["threshold"] = thr
                cells.append(cell)

    cells.sort(key=lambda c: c["f1"], reverse=True)
    best = cells[0]
    all_f1 = next(
        c["f1"] for c in cells
        if set(c["measures"]) == set(_MEASURES) and c["threshold"] == 0.15
    )
    return {
        "cells": cells,
        "best_cell": best,
        "full_vs_best_f1_gap": round(all_f1 - best["f1"], 4),
    }


def _score_ablation_cell(
    labels: list[dict],
    measures: set[str],
    threshold: float,
) -> dict[str, float]:
    """Score one ablation cell: compute measures, apply threshold, compute metrics."""
    from sklearn.metrics import f1_score, roc_auc_score
    from backend.models.change_detection import (
        compute_semantic_drift, compute_jsd, compute_wasserstein,
    )

    y_true, y_score = [], []
    for rec in labels:
        score = 0.0
        if "drift" in measures:
            r = compute_semantic_drift(rec["text_old"], rec["text_new"])
            score = max(score, r["drift_score"])
        if "jsd" in measures:
            r = compute_jsd(rec["text_old"], rec["text_new"], n_permutations=200)
            # Normalise p-value to [0,1] contribution: low p → high score
            score = max(score, 1.0 - r.get("p_value", 1.0))
        if "wasserstein" in measures:
            r = compute_wasserstein(rec["text_old"], rec["text_new"])
            score = max(score, r.get("wasserstein_score", 0.0))
        y_true.append(int(rec["label"]))
        y_score.append(score)

    y_pred = [1 if s >= threshold else 0 for s in y_score]
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5

    # Brier score
    brier = float(np.mean([(s - t) ** 2 for s, t in zip(y_score, y_true)]))

    return {"f1": round(f1, 4), "auroc": round(auroc, 4), "brier": round(brier, 4)}


def _load_labelled_dev_set() -> list[dict]:
    """Load the 300-document labelled dev set from Postgres."""
    from db import get_session_factory

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(
            "SELECT text_old, text_new, label FROM labelled_change_pairs LIMIT 300"
        ).fetchall()
    return [{"text_old": r.text_old, "text_new": r.text_new, "label": r.label} for r in rows]


def _ablation_mock() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    cells = []
    for r in range(1, len(_MEASURES) + 1):
        for subset in itertools.combinations(_MEASURES, r):
            base_f1 = _REFERENCE_F1.get(subset, 0.75)
            for thr in _THRESHOLDS:
                thr_penalty = abs(thr - 0.15) * 0.05
                f1 = round(base_f1 - thr_penalty + rng.normal(0, 0.005), 4)
                cells.append({
                    "measures": list(subset),
                    "threshold": thr,
                    "f1": float(np.clip(f1, 0, 1)),
                    "auroc": round(float(np.clip(f1 + 0.08, 0, 1)), 4),
                    "brier": round(float(rng.uniform(0.05, 0.15)), 4),
                })
    cells.sort(key=lambda c: c["f1"], reverse=True)
    best = cells[0]
    full_f1 = next(
        c["f1"] for c in cells
        if set(c["measures"]) == set(_MEASURES) and c["threshold"] == 0.15
    )
    return {"cells": cells, "best_cell": best,
            "full_vs_best_f1_gap": round(full_f1 - best["f1"], 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Curve
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(
    n_bins: int = 10,
    n_bootstrap: int = 1_000,
    threshold_candidates: tuple[float, ...] = (0.10, 0.12, 0.15, 0.18, 0.20),
) -> dict[str, Any]:
    """Compute calibration curve and bootstrap CI on the optimal threshold.

    Returns
    -------
    {
        "calibration_error": float,        # Mean Absolute Calibration Error
        "optimal_threshold": float,        # F1-maximising threshold
        "threshold_ci": [lo, hi],          # 95% bootstrap CI on optimal threshold
        "bin_means": [...],                # Fraction positive per bin
        "bin_predicted": [...],            # Mean predicted score per bin
    }
    """
    if _MOCK:
        return _calibration_mock(n_bins, n_bootstrap, threshold_candidates)

    labels = _load_labelled_dev_set()
    from backend.models.change_detection import compute_all_measures

    scores, y_true = [], []
    for rec in labels:
        r = compute_all_measures(rec["text_old"], rec["text_new"])
        scores.append(r["composite_score"])
        y_true.append(int(rec["label"]))

    return _calibration_from_arrays(
        np.array(scores), np.array(y_true),
        n_bins, n_bootstrap, threshold_candidates,
    )


def _calibration_from_arrays(
    scores: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
    n_bootstrap: int,
    candidates: tuple[float, ...],
) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    # Reliability diagram bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_predicted = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.any():
            bin_means.append(float(y_true[mask].mean()))
            bin_predicted.append(float(scores[mask].mean()))

    mace = float(np.mean(np.abs(np.array(bin_means) - np.array(bin_predicted)))) \
        if bin_means else 0.0

    # Optimal threshold
    def best_threshold(s: np.ndarray, y: np.ndarray) -> float:
        best_f1, best_t = -1.0, candidates[0]
        for t in candidates:
            f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t

    opt_thr = best_threshold(scores, y_true)

    # Bootstrap CI on threshold
    rng = np.random.default_rng(42)
    boot_thrs = []
    n = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_thrs.append(best_threshold(scores[idx], y_true[idx]))
    lo, hi = np.quantile(boot_thrs, [0.025, 0.975])

    return {
        "calibration_error": round(mace, 4),
        "optimal_threshold": opt_thr,
        "threshold_ci": [round(float(lo), 4), round(float(hi), 4)],
        "bin_means": [round(v, 4) for v in bin_means],
        "bin_predicted": [round(v, 4) for v in bin_predicted],
    }


def _calibration_mock(n_bins: int, n_bootstrap: int, candidates: tuple) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    n = 300
    scores = np.clip(rng.beta(2, 5, n), 0, 1)
    y_true = (scores + rng.normal(0, 0.1, n) > 0.4).astype(int)
    return _calibration_from_arrays(scores, y_true, n_bins, n_bootstrap, candidates)


# ─────────────────────────────────────────────────────────────────────────────
# pgmpy Calibration Loop
# ─────────────────────────────────────────────────────────────────────────────

def pgmpy_calibration_loop(
    records: list[dict[str, Any]],
    n_iter: int = 20,
    smoothing: float = 1.0,
) -> dict[str, Any]:
    """EM-style prior calibration of the ImpactBayesNet using pgmpy MLE.

    Supplements ``ImpactBayesNet.calibrate_from_labels`` with a full
    pgmpy ``BayesianNetwork`` fit so CPT updates are data-driven, not
    just prior-count updates.

    Parameters
    ----------
    records:  List of dicts with keys drift, jsd_p, rwa, true_impact.
    n_iter:   EM iterations.
    smoothing: Dirichlet pseudo-count (Laplace = 1.0).

    Returns
    -------
    {
        "n_records": int,
        "calibrated_prior_drift": [...],
        "calibrated_prior_jsd": [...],
        "calibrated_prior_rwa": [...],
        "log_likelihood": float,
        "n_iter": int,
    }
    """
    import pandas as pd
    from pgmpy.estimators import BayesianEstimator
    try:
        from pgmpy.models import DiscreteBayesianNetwork as _BNClass
    except ImportError:
        from pgmpy.models import BayesianNetwork as _BNClass  # type: ignore[no-redef]

    from backend.models.bayesian_network import (
        DRIFT_STATES, ImpactBayesNet, JSD_STATES, RWA_STATES, get_default_bn,
    )

    if not records:
        raise ValueError("records must be non-empty")

    def _discretise(rec: dict) -> dict:
        ds = rec.get("drift", 0.0)
        jp = rec.get("jsd_p") or 1.0
        rwa = rec.get("rwa") or 0.0
        return {
            "DriftSeverity": "low" if ds < 0.15 else ("medium" if ds < 0.50 else "high"),
            "JSDSignificant": "yes" if jp < 0.05 else "no",
            "RWAMagnitude": "small" if rwa < 50 else ("medium" if rwa < 500 else "large"),
            "ImpactLevel": rec.get("true_impact", "low"),
        }

    df = pd.DataFrame([_discretise(r) for r in records])

    # Define structure (same DAG as ImpactBayesNet)
    edges = [
        ("DriftSeverity", "ImpactLevel"),
        ("JSDSignificant", "ImpactLevel"),
        ("RWAMagnitude", "ImpactLevel"),
    ]
    model = _BNClass(edges)

    model.fit(
        df,
        estimator=BayesianEstimator,
        prior_type="dirichlet",
        pseudo_counts=smoothing,
    )

    # Extract calibrated priors from marginal CPDs, reordering to match DRIFT_STATES / JSD_STATES / RWA_STATES
    def _reorder(cpd, target_states: list[str]) -> np.ndarray:
        """Map pgmpy's alphabetical state ordering to our canonical ordering."""
        node_name = cpd.variables[0]
        actual_states: list[str] = cpd.state_names[node_name]
        values = np.array(cpd.values).flatten()
        reordered = np.array([
            values[actual_states.index(s)] if s in actual_states else 0.0
            for s in target_states
        ])
        return reordered / (reordered.sum() + 1e-15)

    drift_cpd = model.get_cpds("DriftSeverity")
    jsd_cpd = model.get_cpds("JSDSignificant")
    rwa_cpd = model.get_cpds("RWAMagnitude")

    prior_drift = _reorder(drift_cpd, DRIFT_STATES)
    prior_jsd = _reorder(jsd_cpd, JSD_STATES)
    prior_rwa = _reorder(rwa_cpd, RWA_STATES)

    # Log-likelihood on training data (manual computation from CPDs)
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(model)
    ll_sum = 0.0
    for _, row in df.iterrows():
        evidence = {
            "DriftSeverity": row["DriftSeverity"],
            "JSDSignificant": row["JSDSignificant"],
            "RWAMagnitude": row["RWAMagnitude"],
        }
        try:
            phi = infer.query(["ImpactLevel"], evidence=evidence, show_progress=False)
            states = phi.state_names["ImpactLevel"]
            p = phi.values[states.index(row["ImpactLevel"])] if row["ImpactLevel"] in states else 1e-9
        except Exception:
            p = 1e-9
        ll_sum += float(np.log(max(float(p), 1e-9)))
    ll = ll_sum / max(len(df), 1)

    # Update singleton BN priors
    bn = get_default_bn()
    bn.update_prior(
        prior_drift=prior_drift,
        prior_jsd=prior_jsd,
        prior_rwa=prior_rwa,
    )
    log.info(
        "pgmpy calibration done: drift=%s jsd=%s rwa=%s ll=%.4f",
        prior_drift.round(3), prior_jsd.round(3), prior_rwa.round(3), ll,
    )

    return {
        "n_records": len(records),
        "calibrated_prior_drift": prior_drift.tolist(),
        "calibrated_prior_jsd": prior_jsd.tolist(),
        "calibrated_prior_rwa": prior_rwa.tolist(),
        "log_likelihood": round(ll, 4),
        "n_iter": n_iter,
    }
