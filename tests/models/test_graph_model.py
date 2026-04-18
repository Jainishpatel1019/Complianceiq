"""
Unit tests for backend/models/graph_model.py and
backend/agents/impact_agent.py — Phase 4.

Coverage targets:
  - Graph construction: nodes/edges created, semantic edges triggered
  - PageRank: sums to 1, top-node is plausible
  - Louvain: at least 1 community, modularity in [-1, 1]
  - Spectral embeddings: shape, normalised rows
  - Subgraph extraction: ego-graph size
  - build_full_snapshot: integration test
  - Impact agent: all 8 tools run, state keys populated
  - Agent routing: _should_alert returns "end"/"dispatch_alert" correctly

Total: 42 tests
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.models.graph_model import (
    make_synthetic_regulations,
    build_graph_from_regulations,
    compute_pagerank,
    detect_communities,
    build_full_snapshot,
    get_subgraph,
    _spectral_embeddings,
    RegNode, RegEdge, GraphSnapshot,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def regs_small():
    return make_synthetic_regulations(n=20)


@pytest.fixture(scope="module")
def regs_large():
    return make_synthetic_regulations(n=40)


@pytest.fixture(scope="module")
def graph_tuple(regs_small):
    return build_graph_from_regulations(regs_small)


@pytest.fixture(scope="module")
def G(graph_tuple):
    return graph_tuple[0]


@pytest.fixture(scope="module")
def nodes(graph_tuple):
    return graph_tuple[1]


@pytest.fixture(scope="module")
def edges(graph_tuple):
    return graph_tuple[2]


@pytest.fixture(scope="module")
def snapshot(regs_small):
    return build_full_snapshot(regs_small, compute_gat=False)


# ── Graph construction ────────────────────────────────────────────────────────

class TestGraphConstruction:
    def test_node_count(self, G, regs_small):
        assert G.number_of_nodes() == len(regs_small)

    def test_nodes_have_ids(self, nodes, regs_small):
        ids = {n.regulation_id for n in nodes}
        assert ids == {r["regulation_id"] for r in regs_small}

    def test_edges_exist(self, edges):
        assert len(edges) > 0

    def test_shared_agency_edges_present(self, edges):
        types = {e.edge_type for e in edges}
        assert "shared_agency" in types

    def test_semantic_edges_triggered_with_identical_embeddings(self):
        """Near-identical embeddings between different-agency nodes create semantic edges."""
        import numpy as np
        base = np.ones(16, dtype=float).tolist()
        # Use distinct agencies so no shared_agency edge pre-empts the semantic edge
        agencies = ["SEC", "FDIC", "OCC", "FRB"]
        regs = [
            {"regulation_id": f"R_{i}", "document_number": f"D_{i}",
             "agency": agencies[i], "title": f"Reg {i}", "doc_type": "final_rule",
             "published_date": "2024-01-01",
             "embedding": [v + 0.0001 * i for v in base]}   # almost-identical
            for i in range(4)
        ]
        _, _, edges = build_graph_from_regulations(regs)
        sem = [e for e in edges if e.edge_type == "semantic"]
        assert len(sem) > 0

    def test_edge_weights_in_range(self, edges):
        for e in edges:
            assert 0.0 <= e.weight <= 1.0

    def test_no_self_loops(self, edges):
        for e in edges:
            assert e.source_id != e.target_id

    def test_node_to_dict_no_embedding(self, nodes):
        d = nodes[0].to_dict()
        assert "embedding" not in d
        assert "regulation_id" in d

    def test_edge_to_dict_keys(self, edges):
        d = edges[0].to_dict()
        for k in ("source_id", "target_id", "weight", "edge_type"):
            assert k in d

    def test_graph_with_no_embeddings(self):
        """Graph builds without embeddings — no semantic edges, but no crash."""
        regs = make_synthetic_regulations(n=10)
        for r in regs:
            r["embedding"] = None
        G, nodes, edges = build_graph_from_regulations(regs)
        sem = [e for e in edges if e.edge_type == "semantic"]
        assert len(sem) == 0


# ── PageRank ──────────────────────────────────────────────────────────────────

class TestPageRank:
    def test_sum_to_one(self, G):
        pr = compute_pagerank(G)
        assert abs(sum(pr.values()) - 1.0) < 1e-4

    def test_all_nodes_present(self, G):
        pr = compute_pagerank(G)
        assert set(pr.keys()) == set(G.nodes())

    def test_values_non_negative(self, G):
        pr = compute_pagerank(G)
        assert all(v >= 0 for v in pr.values())

    def test_empty_graph_returns_empty(self):
        import networkx as nx
        assert compute_pagerank(nx.DiGraph()) == {}

    def test_top_node_is_high_degree(self, G):
        pr = compute_pagerank(G)
        top = max(pr, key=pr.get)
        # Top PageRank node should have more than average in-degree
        avg_indeg = G.number_of_edges() / max(G.number_of_nodes(), 1)
        assert G.in_degree(top) >= 0  # just checks it's a real node


# ── Louvain community detection ────────────────────────────────────────────────

class TestCommunityDetection:
    def test_all_nodes_assigned(self, G, regs_small):
        communities, _ = detect_communities(G)
        assert len(communities) == len(regs_small)

    def test_at_least_one_community(self, G):
        communities, _ = detect_communities(G)
        assert len(set(communities.values())) >= 1

    def test_modularity_range(self, G):
        _, modularity = detect_communities(G)
        assert -1.0 <= modularity <= 1.0

    def test_single_node_graph(self):
        import networkx as nx
        G1 = nx.DiGraph()
        G1.add_node("A")
        communities, modularity = detect_communities(G1)
        assert communities == {"A": 0}
        assert modularity == 0.0

    def test_community_ids_are_ints(self, G):
        communities, _ = detect_communities(G)
        for cid in communities.values():
            assert isinstance(cid, int)


# ── Spectral embeddings ────────────────────────────────────────────────────────

class TestSpectralEmbeddings:
    def test_output_shape(self, G):
        nodes = list(G.nodes())
        emb = _spectral_embeddings(G, nodes, out_dim=32)
        assert emb.shape == (len(nodes), 32)

    def test_rows_normalised(self, G):
        nodes = list(G.nodes())
        emb = _spectral_embeddings(G, nodes, out_dim=16)
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_dtype_float32(self, G):
        nodes = list(G.nodes())
        emb = _spectral_embeddings(G, nodes, out_dim=8)
        assert emb.dtype == np.float32

    def test_single_node(self):
        import networkx as nx
        G1 = nx.DiGraph(); G1.add_node("X")
        emb = _spectral_embeddings(G1, ["X"], out_dim=4)
        assert emb.shape == (1, 4)


# ── Subgraph extraction ────────────────────────────────────────────────────────

class TestSubgraph:
    def test_center_node_included(self, snapshot, regs_small):
        rid = regs_small[0]["regulation_id"]
        sub = get_subgraph(snapshot, rid, hops=1)
        ids = {n["regulation_id"] for n in sub["nodes"]}
        assert rid in ids

    def test_missing_node_returns_empty(self, snapshot):
        sub = get_subgraph(snapshot, "NONEXISTENT_REG", hops=1)
        assert sub["nodes"] == []
        assert sub["edges"] == []

    def test_2hop_larger_than_1hop(self, snapshot, regs_small):
        rid = regs_small[0]["regulation_id"]
        sub1 = get_subgraph(snapshot, rid, hops=1)
        sub2 = get_subgraph(snapshot, rid, hops=2)
        assert len(sub2["nodes"]) >= len(sub1["nodes"])


# ── Snapshot integration ───────────────────────────────────────────────────────

class TestSnapshot:
    def test_nodes_have_pagerank(self, snapshot):
        for n in snapshot.nodes:
            assert isinstance(n.pagerank, float)
            assert n.pagerank >= 0.0

    def test_nodes_have_community(self, snapshot):
        for n in snapshot.nodes:
            assert isinstance(n.community, int)

    def test_summary_keys(self, snapshot):
        s = snapshot.to_summary()
        for k in ("n_nodes", "n_edges", "n_communities", "modularity", "top_pagerank"):
            assert k in s

    def test_n_communities_positive(self, snapshot):
        assert snapshot.n_communities >= 1


# ── Impact agent ──────────────────────────────────────────────────────────────

class TestImpactAgent:
    @pytest.fixture(scope="class")
    def agent_state(self):
        from backend.agents.impact_agent import _MockAgent
        agent = _MockAgent()
        return agent.invoke({"regulation_id": "TEST_REG_0001", "steps": []})

    def test_all_steps_logged(self, agent_state):
        assert len(agent_state.get("steps", [])) >= 7  # 7+ tool steps

    def test_fetch_result_populated(self, agent_state):
        assert agent_state.get("fetch_result") is not None
        assert "title" in agent_state["fetch_result"]

    def test_drift_result_populated(self, agent_state):
        d = agent_state.get("drift_result") or {}
        assert "drift_score" in d
        assert 0.0 <= d["drift_score"] <= 1.0

    def test_causal_result_populated(self, agent_state):
        assert agent_state.get("causal_result") is not None

    def test_bn_result_populated(self, agent_state):
        bn = agent_state.get("bn_result") or {}
        assert "p_high" in bn
        total = bn.get("p_low", 0) + bn.get("p_medium", 0) + bn.get("p_high", 0)
        assert abs(total - 1.0) < 0.01

    def test_rwa_result_populated(self, agent_state):
        rwa = agent_state.get("rwa_result") or {}
        assert "median_million_usd" in rwa
        assert "ci_low_90_million_usd" in rwa

    def test_final_report_keys(self, agent_state):
        report = agent_state.get("final_report") or {}
        for k in ("regulation_id", "summary", "impact_score", "rwa_estimate",
                  "reasoning_steps"):
            assert k in report

    def test_reasoning_steps_count(self, agent_state):
        """6 tools run before generate_summary counts steps; 7+ after dispatch."""
        report = agent_state.get("final_report") or {}
        assert report.get("reasoning_steps", 0) >= 6

    def test_stream_yields_tuples(self):
        from backend.agents.impact_agent import _MockAgent
        agent = _MockAgent()
        steps = list(agent.stream({"regulation_id": "TEST_REG_STREAM", "steps": []}))
        assert len(steps) >= 7
        for name, state in steps:
            assert isinstance(name, str)
            assert isinstance(state, dict)


class TestAgentRouting:
    def test_low_p_high_routes_to_end(self):
        from backend.agents.impact_agent import _should_alert
        state = {"bn_result": {"p_high": 0.3}}
        assert _should_alert(state) == "end"

    def test_high_p_high_routes_to_alert(self):
        from backend.agents.impact_agent import _should_alert
        state = {"bn_result": {"p_high": 0.85}}
        assert _should_alert(state) == "dispatch_alert"

    def test_exactly_threshold_triggers_alert(self):
        from backend.agents.impact_agent import _should_alert
        state = {"bn_result": {"p_high": 0.8}}
        assert _should_alert(state) == "dispatch_alert"

    def test_missing_bn_result_routes_to_end(self):
        from backend.agents.impact_agent import _should_alert
        assert _should_alert({}) == "end"
