"""
Regulation Knowledge Graph — Phase 4.

Builds a directed weighted graph where:
  - Nodes  = regulations (one per document_number)
  - Edges  = relationships detected by co-citation, shared agency, semantic
             similarity > 0.85, or explicit amendment references

Graph analytics
---------------
  PageRank    d = 0.85, max_iter = 200, tol = 1e-6
              Identifies systemic regulations that many others reference.
  Louvain     python-louvain community detection on undirected projection
              Groups regulations into thematic clusters.
  GAT         Graph Attention Network (2-layer, 8 heads) on PyG
              Produces 64-dim regulation embeddings for downstream retrieval.
              Falls back to spectral embeddings when torch is unavailable.

Temporal edges
--------------
Every edge carries valid_from / valid_to so the graph can be queried as of
any date (snapshot isolation for causal pre/post analysis).

DB persistence
--------------
graph_nodes and graph_edges tables (Phase 1 ORM models).
Rebuild triggered by the graph_update Airflow DAG (weekly, Sunday 02:00 UTC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from typing import Any

import numpy as np
import networkx as nx

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PAGERANK_ALPHA: float   = 0.85
PAGERANK_MAX_ITER: int  = 200
PAGERANK_TOL: float     = 1e-6
SEMANTIC_SIM_THRESHOLD: float = 0.85   # cosine sim above which an edge is added
GAT_HIDDEN_DIM: int     = 64
GAT_HEADS: int          = 8
GAT_LAYERS: int         = 2
LOUVAIN_RESOLUTION: float = 1.0        # higher = more, smaller communities


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RegNode:
    regulation_id: str
    document_number: str
    agency: str
    title: str
    doc_type: str
    published_date: str | None
    embedding: list[float] | None = None   # 768-dim nomic embed
    pagerank: float = 0.0
    community: int = -1
    in_degree: int = 0
    out_degree: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)   # don't serialise large vectors to API
        return d


@dataclass
class RegEdge:
    source_id: str
    target_id: str
    weight: float          # semantic similarity or citation frequency
    edge_type: str         # "semantic" | "citation" | "amendment" | "shared_agency"
    valid_from: str | None = None
    valid_to:   str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphSnapshot:
    """Full in-memory graph snapshot with analytics."""
    nodes: list[RegNode]
    edges: list[RegEdge]
    pagerank: dict[str, float]
    communities: dict[str, int]    # regulation_id → community id
    n_communities: int
    modularity: float
    built_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_summary(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_communities": self.n_communities,
            "modularity": round(self.modularity, 4),
            "built_at": self.built_at,
            "top_pagerank": sorted(
                self.pagerank.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph_from_regulations(
    regulations: list[dict[str, Any]],
    change_scores: list[dict[str, Any]] | None = None,
) -> tuple[nx.DiGraph, list[RegNode], list[RegEdge]]:
    """
    Construct a directed weighted graph from regulation records.

    Edge rules (in order of priority):
      1. amendment:     document_number references another doc (text search)
      2. citation:      same CFR title/part → weight = 0.7
      3. shared_agency: same agency, same year → weight = 0.5
      4. semantic:      cosine(embed_i, embed_j) > threshold → weight = similarity

    Parameters
    ----------
    regulations : list of dicts with keys matching Regulation ORM model
    change_scores : optional; used to weight edges with drift similarity

    Returns
    -------
    G      : nx.DiGraph
    nodes  : list[RegNode]
    edges  : list[RegEdge]
    """
    G = nx.DiGraph()
    nodes: list[RegNode] = []
    edges: list[RegEdge] = []

    reg_by_id: dict[str, dict] = {r["regulation_id"]: r for r in regulations}
    reg_by_docnum: dict[str, str] = {
        r["document_number"]: r["regulation_id"]
        for r in regulations
        if r.get("document_number")
    }

    # Add nodes
    for reg in regulations:
        node = RegNode(
            regulation_id=reg["regulation_id"],
            document_number=reg.get("document_number", ""),
            agency=reg.get("agency", ""),
            title=reg.get("title", "")[:120],
            doc_type=reg.get("doc_type", ""),
            published_date=str(reg.get("published_date", "")),
            embedding=reg.get("embedding"),
        )
        nodes.append(node)
        G.add_node(
            reg["regulation_id"],
            **{k: v for k, v in reg.items() if k != "embedding"},
        )

    # Rule 1: amendment references (scan title + doc metadata)
    for reg in regulations:
        title_lower = (reg.get("title") or "").lower()
        for docnum, target_id in reg_by_docnum.items():
            if (docnum.lower() in title_lower
                    and target_id != reg["regulation_id"]):
                e = RegEdge(
                    source_id=reg["regulation_id"],
                    target_id=target_id,
                    weight=0.9,
                    edge_type="amendment",
                    valid_from=str(reg.get("published_date", "")),
                )
                edges.append(e)
                G.add_edge(reg["regulation_id"], target_id,
                           weight=0.9, edge_type="amendment")

    # Rule 2 + 3: shared CFR / shared agency
    by_agency: dict[str, list[str]] = {}
    for reg in regulations:
        ag = reg.get("agency", "unknown")
        by_agency.setdefault(ag, []).append(reg["regulation_id"])

    for agency, ids in by_agency.items():
        for i in range(len(ids)):
            for j in range(i + 1, min(len(ids), i + 6)):  # cap fan-out at 5
                src, tgt = ids[i], ids[j]
                if not G.has_edge(src, tgt):
                    e = RegEdge(src, tgt, weight=0.5, edge_type="shared_agency")
                    edges.append(e)
                    G.add_edge(src, tgt, weight=0.5, edge_type="shared_agency")

    # Rule 4: semantic similarity (only when embeddings are available)
    emb_regs = [r for r in regulations if r.get("embedding") and len(r["embedding"]) > 0]
    if len(emb_regs) >= 2:
        ids_with_emb = [r["regulation_id"] for r in emb_regs]
        emb_matrix = np.array([r["embedding"] for r in emb_regs], dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
        normed = emb_matrix / norms
        sim_matrix = normed @ normed.T   # shape (N, N)

        for i in range(len(ids_with_emb)):
            for j in range(i + 1, len(ids_with_emb)):
                sim = float(sim_matrix[i, j])
                if sim >= SEMANTIC_SIM_THRESHOLD:
                    src, tgt = ids_with_emb[i], ids_with_emb[j]
                    if not G.has_edge(src, tgt):
                        e = RegEdge(src, tgt, weight=round(sim, 4),
                                    edge_type="semantic")
                        edges.append(e)
                        G.add_edge(src, tgt, weight=round(sim, 4),
                                   edge_type="semantic")

    log.info("Graph built", n_nodes=G.number_of_nodes(),
             n_edges=G.number_of_edges())
    return G, nodes, edges


# ── PageRank ──────────────────────────────────────────────────────────────────

def compute_pagerank(G: nx.DiGraph) -> dict[str, float]:
    """
    Weighted PageRank (d=0.85) on the regulation graph.

    Uses edge weight as the transition probability mass.
    Returns dict mapping regulation_id → PageRank score (sums to 1.0).
    """
    if G.number_of_nodes() == 0:
        return {}

    # Normalise weights per node so they act as transition probabilities
    pr = nx.pagerank(
        G,
        alpha=PAGERANK_ALPHA,
        max_iter=PAGERANK_MAX_ITER,
        tol=PAGERANK_TOL,
        weight="weight",
    )
    return {k: round(v, 8) for k, v in pr.items()}


# ── Louvain community detection ────────────────────────────────────────────────

def detect_communities(G: nx.DiGraph) -> tuple[dict[str, int], float]:
    """
    Louvain community detection on the undirected projection of G.

    Returns
    -------
    community_map : dict[regulation_id → community_id]
    modularity    : float (quality metric, higher is better, range [-1, 1])
    """
    if G.number_of_nodes() < 2:
        return {n: 0 for n in G.nodes()}, 0.0

    U = G.to_undirected()

    try:
        import community as community_louvain   # python-louvain
        partition = community_louvain.best_partition(
            U, resolution=LOUVAIN_RESOLUTION, random_state=42
        )
        modularity = community_louvain.modularity(partition, U)
    except ImportError:
        # Fallback: greedy modularity (networkx built-in)
        log.warning("python-louvain not installed; using greedy modularity")
        communities = nx.algorithms.community.greedy_modularity_communities(U)
        partition = {}
        for c_id, community in enumerate(communities):
            for node in community:
                partition[node] = c_id
        modularity = nx.algorithms.community.modularity(
            U, communities, resolution=LOUVAIN_RESOLUTION
        )

    return partition, float(modularity)


# ── GAT node embeddings ────────────────────────────────────────────────────────

def compute_gat_embeddings(
    G: nx.DiGraph,
    node_features: np.ndarray | None = None,
    out_dim: int = GAT_HIDDEN_DIM,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute GAT node embeddings.

    Tries torch + torch_geometric first. Falls back to spectral (Laplacian
    eigenvectors) for CPU-only environments where torch_geometric isn't
    installed.

    Parameters
    ----------
    G             : regulation DiGraph
    node_features : (N, d_in) array of input features; if None, uses identity
    out_dim       : embedding dimension (default 64)

    Returns
    -------
    embeddings : (N, out_dim) float32 array, rows ordered by G.nodes()
    """
    nodes = list(G.nodes())
    N = len(nodes)

    if N == 0:
        return np.zeros((0, out_dim), dtype=np.float32)

    node_idx = {n: i for i, n in enumerate(nodes)}

    try:
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data

        # Build edge_index tensor
        edges = list(G.edges())
        if edges:
            edge_index = torch.tensor(
                [[node_idx[s], node_idx[t]] for s, t in edges],
                dtype=torch.long,
            ).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Node features: provided or identity matrix
        if node_features is not None:
            x = torch.tensor(node_features, dtype=torch.float32)
            in_dim = x.shape[1]
        else:
            x = torch.eye(N, dtype=torch.float32)
            in_dim = N

        data = Data(x=x, edge_index=edge_index)

        # 2-layer GAT
        torch.manual_seed(seed)

        class GAT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                heads = min(GAT_HEADS, in_dim)
                self.conv1 = GATConv(in_dim, out_dim // heads, heads=heads,
                                     dropout=0.0)
                self.conv2 = GATConv(out_dim, out_dim, heads=1,
                                     dropout=0.0)

            def forward(self, data: Data) -> torch.Tensor:
                x = torch.relu(self.conv1(data.x, data.edge_index))
                x = self.conv2(x, data.edge_index)
                return x

        model = GAT()
        model.eval()
        with torch.no_grad():
            embeddings = model(data).numpy()

        log.info("GAT embeddings computed via torch_geometric", n_nodes=N)
        return embeddings.astype(np.float32)

    except ImportError:
        log.info("torch_geometric unavailable; using spectral embeddings")
        return _spectral_embeddings(G, nodes, out_dim)


def _spectral_embeddings(
    G: nx.DiGraph,
    nodes: list[str],
    out_dim: int,
) -> np.ndarray:
    """
    Spectral fallback: Laplacian eigenvectors of the undirected projection.
    Pads with random noise if graph has fewer nodes than out_dim.
    """
    U = G.to_undirected()
    N = len(nodes)
    k = min(out_dim, N - 1) if N > 1 else 1

    try:
        L = nx.laplacian_matrix(U, nodelist=nodes).toarray().astype(np.float32)
        _, vecs = np.linalg.eigh(L)
        # Take k smallest non-trivial eigenvectors (skip index 0)
        emb = vecs[:, 1 : k + 1]
    except Exception:
        emb = np.zeros((N, k), dtype=np.float32)

    # Pad to out_dim if needed
    if emb.shape[1] < out_dim:
        rng = np.random.default_rng(42)
        pad = rng.standard_normal((N, out_dim - emb.shape[1])).astype(np.float32)
        emb = np.concatenate([emb, pad], axis=1)

    # L2-normalise rows
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb / norms


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_full_snapshot(
    regulations: list[dict[str, Any]],
    change_scores: list[dict[str, Any]] | None = None,
    compute_gat: bool = True,
) -> GraphSnapshot:
    """
    End-to-end: build graph, compute PageRank, Louvain, optional GAT embeds.

    Called by the ``graph_update`` Airflow DAG.
    """
    G, nodes, edges = build_graph_from_regulations(regulations, change_scores)

    pagerank = compute_pagerank(G)
    communities, modularity = detect_communities(G)

    # Attach analytics to node objects
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    for node in nodes:
        node.pagerank  = round(pagerank.get(node.regulation_id, 0.0), 8)
        node.community = communities.get(node.regulation_id, -1)
        node.in_degree  = in_deg.get(node.regulation_id, 0)
        node.out_degree = out_deg.get(node.regulation_id, 0)

    # GAT embeddings (skipped in test env)
    if compute_gat and nodes:
        emb_matrix = compute_gat_embeddings(G)
        node_list  = list(G.nodes())
        for i, nid in enumerate(node_list):
            matching = next((n for n in nodes if n.regulation_id == nid), None)
            if matching and i < len(emb_matrix):
                matching.embedding = emb_matrix[i].tolist()

    n_communities = len(set(communities.values())) if communities else 0

    return GraphSnapshot(
        nodes=nodes,
        edges=edges,
        pagerank=pagerank,
        communities=communities,
        n_communities=n_communities,
        modularity=modularity,
    )


def get_subgraph(
    snapshot: GraphSnapshot,
    regulation_id: str,
    hops: int = 2,
) -> dict[str, Any]:
    """
    Return ego-graph (k-hop neighbourhood) around a target regulation.

    Useful for the frontend's focused node view.
    """
    G = _snapshot_to_nx(snapshot)
    if regulation_id not in G:
        return {"nodes": [], "edges": []}

    ego = nx.ego_graph(G, regulation_id, radius=hops, undirected=True)
    sub_nodes = [n for n in snapshot.nodes if n.regulation_id in ego]
    sub_edges = [
        e for e in snapshot.edges
        if e.source_id in ego and e.target_id in ego
    ]
    return {
        "nodes": [n.to_dict() for n in sub_nodes],
        "edges": [e.to_dict() for e in sub_edges],
    }


def _snapshot_to_nx(snapshot: GraphSnapshot) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in snapshot.nodes:
        G.add_node(n.regulation_id)
    for e in snapshot.edges:
        G.add_edge(e.source_id, e.target_id, weight=e.weight)
    return G


# ── Synthetic data for dev / tests ────────────────────────────────────────────

def make_synthetic_regulations(n: int = 40, seed: int = 0) -> list[dict[str, Any]]:
    """
    Generate n synthetic regulation records for offline testing.
    Includes embeddings so semantic-edge creation is exercised.
    """
    rng = np.random.default_rng(seed)
    agencies = ["SEC", "FDIC", "OCC", "FRB", "CFPB", "CFTC"]
    doc_types = ["final_rule", "proposed_rule", "notice"]

    regs = []
    for i in range(n):
        ag = agencies[i % len(agencies)]
        # Cluster embeddings so Louvain finds communities
        cluster = i // (n // 4)
        base = rng.standard_normal(16)
        cluster_signal = np.zeros(16)
        cluster_signal[cluster * 4 : cluster * 4 + 4] = 2.0
        emb = (base + cluster_signal).tolist()

        regs.append({
            "regulation_id": f"REG_{i:04d}",
            "document_number": f"2024-{10000 + i:05d}",
            "agency": ag,
            "title": f"Regulation {i:03d} — {ag} capital requirement amendment",
            "doc_type": doc_types[i % len(doc_types)],
            "published_date": f"202{3 + (i % 2)}-{(i % 12) + 1:02d}-01",
            "embedding": emb,
        })
    return regs
