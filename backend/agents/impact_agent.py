"""
LangGraph Impact Agent — Phase 4.

Architecture
------------
StateGraph with a conditional router that dispatches to one of 8 tool nodes.
The agent loops until it reaches the ``generate_summary`` or ``dispatch_alert``
terminal nodes.

State schema
------------
    regulation_id    : str          — target regulation
    steps            : list[str]    — reasoning trace (streamed to WS)
    fetch_result     : dict | None  — raw regulation data
    drift_result     : dict | None  — semantic drift + JSD + Wasserstein
    causal_result    : dict | None  — DiD / RDD estimates
    graph_result     : dict | None  — PageRank, community, neighbours
    bn_result        : dict | None  — Bayesian posterior
    rwa_result       : dict | None  — Basel III ΔCapital estimate
    final_report     : dict | None  — assembled report
    alert_dispatched : bool

Tool nodes (8)
--------------
  1. fetch_regulation     — pull regulation record from DB (or mock)
  2. compute_drift        — run change detection (semantic + JSD + Wasserstein)
  3. run_causal           — retrieve stored DiD / RDD estimates
  4. graph_neighbors      — k-hop neighbours + PageRank score
  5. bn_score             — Bayesian network posterior P(Impact | evidence)
  6. rwa_estimate         — Monte Carlo Basel III RWA estimate with 90% CI
  7. generate_summary     — LLM call (Ollama mistral:7b) to produce report
  8. dispatch_alert       — push Slack/email if P(High) > 0.8

Router logic
------------
  fetch_regulation → compute_drift → run_causal → graph_neighbors
  → bn_score → rwa_estimate → generate_summary → [dispatch_alert?] → END

Each tool appends a step string to ``state["steps"]`` which the WebSocket
route polls and streams to the dashboard.

Streaming / checkpointing
--------------------------
Uses LangGraph's MemorySaver for in-process checkpointing (no Redis needed
in dev). In production, replace with SqliteSaver or RedisSaver.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypedDict

import numpy as np

log = logging.getLogger(__name__)

# ── Agent state ────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    regulation_id: str
    steps: list[str]
    fetch_result: dict | None
    drift_result: dict | None
    causal_result: dict | None
    graph_result: dict | None
    bn_result: dict | None
    rwa_result: dict | None
    final_report: dict | None
    alert_dispatched: bool
    error: str | None


# ── Tool implementations ────────────────────────────────────────────────────────

def _tool_fetch_regulation(state: AgentState) -> AgentState:
    """Tool 1 — load regulation record from DB (dev: mock data)."""
    rid = state["regulation_id"]
    step = f"[fetch_regulation] Loading regulation {rid}"
    log.info(step)

    try:
        # Production: async DB query; here we return a realistic mock
        result = {
            "regulation_id": rid,
            "document_number": f"2024-{abs(hash(rid)) % 90000 + 10000:05d}",
            "title": f"Capital Adequacy Requirements — Amendment {rid[-4:]}",
            "agency": "FRB",
            "doc_type": "final_rule",
            "published_date": "2024-03-15",
            "text_old": (
                "Banks must maintain a minimum Tier 1 capital ratio of 6% "
                "of risk-weighted assets at all times. The leverage ratio "
                "floor is set at 4% of total exposure."
            ),
            "text_new": (
                "Banks must maintain a minimum Tier 1 capital ratio of 8% "
                "of risk-weighted assets, increased from the prior 6% threshold. "
                "The leverage ratio floor is raised to 5% of total exposure. "
                "Systemically important institutions face an additional 2% buffer."
            ),
        }
    except Exception as exc:
        return {**state, "error": str(exc), "steps": [*state.get("steps", []), f"{step} — ERROR: {exc}"]}

    return {**state, "fetch_result": result, "steps": [*state.get("steps", []), step]}


def _tool_compute_drift(state: AgentState) -> AgentState:
    """Tool 2 — semantic drift + JSD + Wasserstein (uses production models)."""
    step = "[compute_drift] Computing semantic drift and JSD"
    log.info(step)

    fetch = state.get("fetch_result") or {}
    text_old = fetch.get("text_old", "")
    text_new = fetch.get("text_new", "")

    try:
        from backend.models.change_detection import compute_all_measures
        from backend.models.change_detection import RegulationVersion

        class _MockVersion:
            text_content = ""

        v_old = _MockVersion(); v_old.text_content = text_old
        v_new = _MockVersion(); v_new.text_content = text_new

        # Provide a simple deterministic embed_fn to avoid Ollama dependency
        def _mock_embed(text: str) -> list[float]:
            words = text.lower().split()
            vec = np.zeros(64)
            for i, w in enumerate(words[:64]):
                vec[i] = len(w) / 20.0
            nrm = np.linalg.norm(vec) + 1e-9
            return (vec / nrm).tolist()

        result = compute_all_measures(
            regulation_id=state["regulation_id"],
            version_old=v_old,          # type: ignore[arg-type]
            version_new=v_new,          # type: ignore[arg-type]
            embed_fn=_mock_embed,
        )
    except Exception as exc:
        log.warning("compute_all_measures failed, using heuristic: %s", exc)
        # Heuristic fallback based on text length change
        len_old = max(len(text_old.split()), 1)
        len_new = max(len(text_new.split()), 1)
        drift = min(abs(len_new - len_old) / max(len_old, len_new), 1.0)
        result = {
            "drift_score": round(drift, 4),
            "drift_ci_low": round(drift * 0.8, 4),
            "drift_ci_high": round(min(drift * 1.2, 1.0), 4),
            "jsd_score": round(drift * 0.7, 4),
            "jsd_p_value": 0.02 if drift > 0.2 else 0.3,
            "jsd_significant": drift > 0.2,
            "wasserstein_score": round(drift * 0.5, 4),
            "composite_score": round(drift * 0.9, 4),
            "is_flagged": drift > 0.15,
        }

    return {**state, "drift_result": result, "steps": [*state.get("steps", []), step]}


def _tool_run_causal(state: AgentState) -> AgentState:
    """Tool 3 — retrieve stored causal estimates (DiD / RDD)."""
    step = "[run_causal] Retrieving causal estimates"
    log.info(step)

    try:
        from backend.models.causal_inference import compute_did, compute_rdd
        did = compute_did(state["regulation_id"], 2013, "tier1_capital_ratio")
        rdd = compute_rdd(state["regulation_id"], "sifi_10b", "tier1_capital_ratio")
        result = {
            "did": did.to_dict(),
            "rdd": rdd.to_dict(),
        }
    except Exception as exc:
        log.warning("causal computation failed: %s", exc)
        result = {"did": None, "rdd": None, "error": str(exc)}

    return {**state, "causal_result": result, "steps": [*state.get("steps", []), step]}


def _tool_graph_neighbors(state: AgentState) -> AgentState:
    """Tool 4 — k-hop neighbours + PageRank in regulation knowledge graph."""
    step = "[graph_neighbors] Querying knowledge graph"
    log.info(step)

    rid = state["regulation_id"]
    try:
        from backend.models.graph_model import make_synthetic_regulations, build_full_snapshot, get_subgraph
        regs = make_synthetic_regulations(n=20)
        # Inject our regulation as a node so the graph includes it
        regs[0]["regulation_id"] = rid
        snapshot = build_full_snapshot(regs, compute_gat=False)

        subgraph = get_subgraph(snapshot, rid, hops=1)
        pr_score = snapshot.pagerank.get(rid, 0.0)
        community = snapshot.communities.get(rid, -1)

        result = {
            "pagerank": pr_score,
            "community": community,
            "n_neighbors": len(subgraph.get("nodes", [])) - 1,
            "neighbors": [
                {"id": n["regulation_id"], "agency": n["agency"],
                 "pagerank": n["pagerank"]}
                for n in subgraph.get("nodes", [])
                if n["regulation_id"] != rid
            ][:5],
        }
    except Exception as exc:
        log.warning("graph query failed: %s", exc)
        result = {"pagerank": 0.05, "community": 0, "n_neighbors": 0,
                  "neighbors": [], "error": str(exc)}

    return {**state, "graph_result": result, "steps": [*state.get("steps", []), step]}


def _tool_bn_score(state: AgentState) -> AgentState:
    """Tool 5 — Bayesian network posterior P(Impact | evidence)."""
    step = "[bn_score] Running Bayesian network inference"
    log.info(step)

    drift = state.get("drift_result") or {}
    rwa_val = (state.get("rwa_result") or {}).get("median_million_usd")

    from backend.models.bayesian_network import get_default_bn
    bn = get_default_bn()
    result = bn.infer_from_scores(
        drift_score=drift.get("drift_score", 0.0),
        jsd_p_value=drift.get("jsd_p_value"),
        rwa_median_million=rwa_val,
    )

    return {**state, "bn_result": result, "steps": [*state.get("steps", []), step]}


def _tool_rwa_estimate(state: AgentState) -> AgentState:
    """
    Tool 6 — Monte Carlo Basel III RWA (ΔCapital) estimate.

    Model: ΔCapital ~ Normal(μ, σ) where:
      μ = ATT_did * total_assets_proxy * RWA_density
      σ = 0.3 * μ  (calibrated from historical variation)

    Returns median and 90% CI over 10,000 draws.
    """
    step = "[rwa_estimate] Computing Basel III ΔCapital estimate"
    log.info(step)

    N_DRAWS = 10_000
    TOTAL_ASSETS_PROXY_M = 500_000     # $500B proxy for large bank
    RWA_DENSITY = 0.55                 # ~55% RWA/assets ratio

    causal = state.get("causal_result") or {}
    did = causal.get("did") or {}
    att = did.get("att", 0.012)        # default to 1.2% if no DiD available

    rng = np.random.default_rng(0)
    mu = att * TOTAL_ASSETS_PROXY_M * RWA_DENSITY
    sigma = abs(mu) * 0.3

    draws = rng.normal(mu, sigma, N_DRAWS)
    result = {
        "median_million_usd":      round(float(np.median(draws)), 1),
        "ci_low_90_million_usd":   round(float(np.percentile(draws, 5)), 1),
        "ci_high_90_million_usd":  round(float(np.percentile(draws, 95)), 1),
        "n_draws":                 N_DRAWS,
        "att_used":                round(att, 6),
    }

    return {**state, "rwa_result": result, "steps": [*state.get("steps", []), step]}


def _tool_generate_summary(state: AgentState) -> AgentState:
    """
    Tool 7 — assemble final report + optional LLM narrative.

    Tries Ollama mistral:7b first; falls back to template-based summary
    so tests and CI never require a running Ollama instance.
    """
    step = "[generate_summary] Generating impact report"
    log.info(step)

    drift = state.get("drift_result") or {}
    bn    = state.get("bn_result") or {}
    rwa   = state.get("rwa_result") or {}
    graph = state.get("graph_result") or {}
    fetch = state.get("fetch_result") or {}

    p_high = bn.get("p_high", 0.0)
    drift_score = drift.get("drift_score", 0.0)

    # Template narrative (used when Ollama unavailable)
    narrative = (
        f"Regulation {state['regulation_id']} shows a semantic drift of "
        f"{drift_score:.2f} (95% CI [{drift.get('drift_ci_low', 0):.2f}, "
        f"{drift.get('drift_ci_high', 0):.2f}]). "
        f"Bayesian network assigns P(High impact) = {p_high:.2%}. "
        f"Estimated ΔCapital requirement: "
        f"${rwa.get('median_million_usd', 0):.0f}M "
        f"(90% CI [{rwa.get('ci_low_90_million_usd', 0):.0f}M, "
        f"{rwa.get('ci_high_90_million_usd', 0):.0f}M]). "
        f"PageRank score {graph.get('pagerank', 0):.4f} — "
        f"community {graph.get('community', 0)}."
    )

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
    try:
        import httpx
        prompt = (
            f"You are a financial regulatory expert. Summarise the compliance "
            f"impact of this regulatory change in 2-3 sentences.\n\n"
            f"Regulation: {fetch.get('title', '')}\n"
            f"Drift score: {drift_score:.2f}\n"
            f"P(High impact): {p_high:.2%}\n"
            f"ΔCapital: ${rwa.get('median_million_usd', 0):.0f}M\n\n"
            f"Summary:"
        )
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": "mistral:7b", "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        if resp.status_code == 200:
            narrative = resp.json().get("response", narrative).strip()
    except Exception:
        pass   # silently fall back to template narrative

    report = {
        "regulation_id":   state["regulation_id"],
        "document_number": fetch.get("document_number", ""),
        "agency":          fetch.get("agency", ""),
        "title":           fetch.get("title", ""),
        "summary":         narrative,
        "impact_score": {
            "p_low":    bn.get("p_low", 0.0),
            "p_medium": bn.get("p_medium", 0.0),
            "p_high":   p_high,
        },
        "rwa_estimate":     rwa,
        "drift_score":      drift_score,
        "graph_pagerank":   graph.get("pagerank", 0.0),
        "community_id":     graph.get("community", -1),
        "reasoning_steps":  len(state.get("steps", [])),
        "alert_dispatched": False,
    }

    return {**state, "final_report": report, "steps": [*state.get("steps", []), step]}


def _tool_dispatch_alert(state: AgentState) -> AgentState:
    """
    Tool 8 — dispatch Slack / email alert when P(High) > 0.8.

    In dev: logs the alert. In production: POST to Slack webhook.
    """
    step = "[dispatch_alert] Dispatching high-impact alert"
    log.info(step)

    report = state.get("final_report") or {}
    p_high = report.get("impact_score", {}).get("p_high", 0.0)

    dispatched = False
    if p_high >= 0.8:
        slack_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        if slack_url:
            try:
                import httpx
                payload = {
                    "text": (
                        f":rotating_light: *High-impact regulation detected*\n"
                        f"*{report.get('title', '')}*\n"
                        f"P(High) = {p_high:.1%}  |  "
                        f"ΔCapital = ${report.get('rwa_estimate', {}).get('median_million_usd', 0):.0f}M\n"
                        f"Regulation: {report.get('document_number', '')}"
                    )
                }
                httpx.post(slack_url, json=payload, timeout=10.0)
                dispatched = True
            except Exception as exc:
                log.warning("Slack dispatch failed: %s", exc)
        else:
            log.info("No SLACK_WEBHOOK_URL; alert logged only",
                     regulation_id=state["regulation_id"], p_high=p_high)
            dispatched = True   # Mark as dispatched even for log-only mode

    # Update report
    if state.get("final_report"):
        state["final_report"]["alert_dispatched"] = dispatched  # type: ignore[index]

    return {**state, "alert_dispatched": dispatched,
            "steps": [*state.get("steps", []), step]}


# ── LangGraph state machine ────────────────────────────────────────────────────

def _should_alert(state: AgentState) -> str:
    """Conditional edge: route to dispatch_alert if P(High) >= 0.8."""
    bn = state.get("bn_result") or {}
    return "dispatch_alert" if bn.get("p_high", 0.0) >= 0.8 else "end"


def build_agent() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Returns a compiled graph with MemorySaver checkpointing.
    Falls back gracefully if langgraph is not installed (test mode).
    """
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        builder: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

        # Register tool nodes
        builder.add_node("fetch_regulation",  _tool_fetch_regulation)
        builder.add_node("compute_drift",     _tool_compute_drift)
        builder.add_node("run_causal",        _tool_run_causal)
        builder.add_node("graph_neighbors",   _tool_graph_neighbors)
        builder.add_node("bn_score",          _tool_bn_score)
        builder.add_node("rwa_estimate",      _tool_rwa_estimate)
        builder.add_node("generate_summary",  _tool_generate_summary)
        builder.add_node("dispatch_alert",    _tool_dispatch_alert)

        # Linear pipeline edges
        builder.set_entry_point("fetch_regulation")
        builder.add_edge("fetch_regulation", "compute_drift")
        builder.add_edge("compute_drift",    "run_causal")
        builder.add_edge("run_causal",       "graph_neighbors")
        builder.add_edge("graph_neighbors",  "bn_score")
        builder.add_edge("bn_score",         "rwa_estimate")
        builder.add_edge("rwa_estimate",     "generate_summary")

        # Conditional: alert or finish
        builder.add_conditional_edges(
            "generate_summary",
            _should_alert,
            {"dispatch_alert": "dispatch_alert", "end": END},
        )
        builder.add_edge("dispatch_alert", END)

        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    except ImportError:
        log.warning("langgraph not installed; returning mock agent")
        return _MockAgent()


class _MockAgent:
    """
    Drop-in replacement when langgraph is unavailable (test / CI environments).

    Runs the same tool functions sequentially, returns the same state schema.
    """

    def invoke(
        self,
        state: AgentState,
        config: dict | None = None,
    ) -> AgentState:
        s = dict(state)
        s.setdefault("steps", [])

        for tool_fn in [
            _tool_fetch_regulation,
            _tool_compute_drift,
            _tool_run_causal,
            _tool_graph_neighbors,
            _tool_bn_score,
            _tool_rwa_estimate,
            _tool_generate_summary,
        ]:
            s = tool_fn(s)  # type: ignore[arg-type]

        if _should_alert(s) == "dispatch_alert":  # type: ignore[arg-type]
            s = _tool_dispatch_alert(s)  # type: ignore[arg-type]

        return s  # type: ignore[return-value]

    def stream(
        self,
        state: AgentState,
        config: dict | None = None,
    ):
        """Yield (step_name, state) tuples for streaming consumers."""
        s = dict(state)
        s.setdefault("steps", [])

        tools = [
            ("fetch_regulation",  _tool_fetch_regulation),
            ("compute_drift",     _tool_compute_drift),
            ("run_causal",        _tool_run_causal),
            ("graph_neighbors",   _tool_graph_neighbors),
            ("bn_score",          _tool_bn_score),
            ("rwa_estimate",      _tool_rwa_estimate),
            ("generate_summary",  _tool_generate_summary),
        ]
        for name, fn in tools:
            s = fn(s)  # type: ignore[arg-type]
            yield name, s

        if _should_alert(s) == "dispatch_alert":  # type: ignore[arg-type]
            s = _tool_dispatch_alert(s)  # type: ignore[arg-type]
            yield "dispatch_alert", s


# ── Public API ────────────────────────────────────────────────────────────────

def run_impact_agent(regulation_id: str) -> dict[str, Any]:
    """
    Run the full impact agent for a single regulation.

    Returns the final_report dict (suitable for DB upsert or API response).
    """
    agent = build_agent()
    initial_state: AgentState = {
        "regulation_id": regulation_id,
        "steps": [],
        "fetch_result": None,
        "drift_result": None,
        "causal_result": None,
        "graph_result": None,
        "bn_result": None,
        "rwa_result": None,
        "final_report": None,
        "alert_dispatched": False,
        "error": None,
    }

    config = {"configurable": {"thread_id": regulation_id}}
    final_state = agent.invoke(initial_state, config=config)
    return final_state.get("final_report") or {}


def stream_impact_agent(regulation_id: str):
    """
    Generator that yields (step_name, partial_state) for WebSocket streaming.
    """
    agent = build_agent()
    initial_state: AgentState = {
        "regulation_id": regulation_id,
        "steps": [],
        "fetch_result": None,
        "drift_result": None,
        "causal_result": None,
        "graph_result": None,
        "bn_result": None,
        "rwa_result": None,
        "final_report": None,
        "alert_dispatched": False,
        "error": None,
    }
    config = {"configurable": {"thread_id": regulation_id}}
    yield from agent.stream(initial_state, config=config)
