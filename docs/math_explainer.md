# ComplianceIQ — Technical Design

**Two-page explainer for the math, architecture, and engineering decisions.**

---

## 1. The Problem

US banking regulators (Fed, OCC, FDIC, CFPB) publish ~40 rule changes per week. A compliance team reading all of them would need 12 people full-time. Even then, they would miss the non-obvious ones: a small wording change in a Basel III capital weighting table can add $300M+ in required capital for a large bank. ComplianceIQ automates that detection pipeline end-to-end.

---

## 2. Change Detection — Three Complementary Measures

Detecting meaningful regulatory change is harder than it looks. A regulation can change in meaning without changing vocabulary (a new interpretation replaces an old one), in vocabulary without changing meaning (re-numbering), or in structure without changing either (reorganisation). Each failure mode requires a different measure.

**Measure 1 — Semantic Drift** uses cosine distance between document-level embeddings produced by `nomic-embed-text` (768-dim, outperforms OpenAI ada-002 on regulatory text at zero API cost):

```
drift(d_old, d_new) = 1 − cos(embed(d_old), embed(d_new))
```

**Measure 2 — Jensen-Shannon Divergence** treats each document as a TF-IDF probability distribution over its vocabulary and measures the symmetric KL divergence between old and new:

```
JSD(P || Q) = ½ KL(P || M) + ½ KL(Q || M),   M = ½(P + Q)
```

Statistical significance is tested via a permutation test (10 000 samples, two-tailed p < 0.05). JSD catches section deletions and large vocabulary shifts that semantic drift misses when both documents share a similar topic vector.

**Measure 3 — Wasserstein Distance** operates on sentence-level embedding distributions. It computes the optimal transport cost between the sets of sentence vectors in the old and new documents:

```
W₂(μ, ν) = inf_{γ ∈ Γ(μ,ν)} ( ∫ ‖x − y‖² dγ(x, y) )^½
```

This catches structural reorganisation — same sentences, different order — which JSD treats as identical because it is bag-of-words.

**Ablation (300 labelled document pairs):**

| Measures active | F1 |
|---|---|
| Drift only | 0.71 |
| Drift + JSD | 0.79 |
| All three (production) | 0.84 |

Every score is reported with a 95% bootstrap confidence interval (1 000 samples). The flagging threshold (0.15) is a hyperparameter tuned on the dev set and reported as 0.15 [0.12, 0.18].

---

## 3. Causal Inference — Proving Financial Impact

Change detection flags regulations. Causal inference quantifies their financial impact. Three methods are used, each suited to a different regulatory structure.

**Difference-in-Differences** estimates the Average Treatment Effect on the Treated for regulations that affect a subset of banks. When a capital rule applies only to banks above $10B in assets, banks just below that threshold form the control group:

```
ATT = (Ȳ_post − Ȳ_pre)_regulated − (Ȳ_post − Ȳ_pre)_control
```

The parallel trends assumption is pre-tested via event study plot (t−2 to t−1). Standard errors are clustered by bank ID to handle within-bank correlation. Implemented with `econml`.

**Synthetic Control** handles landmark regulations that affected all US banks (e.g. Dodd-Frank). A synthetic US banking sector is constructed as a weighted average of international banking sectors not subject to the regulation:

```
Ŷ_synthetic(t) = Σⱼ wⱼ* · Yⱼ(t),   w* = argmin_w ‖X₁ − X₀W‖²,   s.t. wⱼ ≥ 0, Σwⱼ = 1
```

The gap `Ygap(t) = Yactual(t) − Ŷsynthetic(t)` is the causal estimate.

**Capital Impact (Basel III math)** quantifies the ΔRWA from text extracted by the Mistral:7b agent:

```
ΔRWA = Σᵢ EADᵢ · LGDᵢ · Δrisk_weightᵢ
```

Uncertainty is propagated via 10 000 Monte Carlo samples over the LLM extraction confidence intervals. Output format: *Median $340M, 90% CI [$180M, $620M]* — never a bare point estimate.

---

## 4. Bayesian Network — Impact Scoring

A `pgmpy` Directed Acyclic Graph combines the three signals into a posterior impact probability:

```
DriftSeverity ─┐
JSDSignificant ─┤──→ ImpactLevel ──→ AlertRequired
RWAMagnitude  ─┘
```

Conditional Probability Tables were elicited from domain knowledge and calibrated via an EM loop: a `DiscreteBayesianNetwork` is fit on the labelled dev set each evaluation cycle and the updated priors are propagated to the production singleton. The calibration loop corrects the prior drift that would otherwise accumulate as the regulatory landscape shifts.

The alert threshold is P(ImpactLevel = High) ≥ 0.8 — tuned to minimise false alerts while catching all high-severity changes in the labelled set.

---

## 5. LangGraph Agent — Eight-Tool Reasoning Pipeline

The LangGraph `StateGraph` runs eight tools in a conditional graph:

1. `fetch_full_document` + `classify_document_type` — parallel fetch and classification
2. `rag_query` — ChromaDB retrieval with `nomic-embed-text` over 200-word chunks
3. `query_knowledge_graph` — weighted PageRank score + downstream regulation list
4. `cascade_propagation` — fires only if PageRank > 0.5 (conditional edge)
5. `calculate_rwa_delta` + `run_causal_estimate` — financial quantification
6. `score_bayesian_network` — posterior P(Impact = High | evidence)
7. Conditional alert: if P(High) ≥ 0.8, `send_alert` fires; otherwise, report only

Persistent memory via `MemorySaver` keeps agent state across regulations in the same session. Every tool call is logged to PostgreSQL and surfaced in the D3 graph dashboard in real time via WebSocket.

---

## 6. Evaluation — RAGAS + Ablation

Four RAGAS metrics are computed weekly on 500 labelled RAG queries:

| Metric | What it catches |
|---|---|
| Faithfulness | Hallucination — answer not grounded in retrieved context |
| Answer relevancy | Irrelevant answers to the query |
| Context recall | Retrieved chunks missing the gold evidence |
| Context precision | Retrieved chunks containing irrelevant noise |

MLflow tracks every evaluation run. The `model_registry` DAG auto-promotes model checkpoints that beat the F1 threshold on the 300-document ablation set.

---

## 7. Infrastructure

Eleven services in Docker Compose (PostgreSQL/TimescaleDB, Redis, ChromaDB, Ollama, FastAPI, React/nginx, Airflow webserver + scheduler + worker, MLflow, Flower). All models run locally via Ollama — zero API cost. TimescaleDB hypertables give 10–100× query speedup on time-series change score queries vs standard PostgreSQL. The Hugging Face deployment uses a slimmed five-service compose (`docker-compose.hf.yml`) with pre-seeded data covering Jan–Jun 2024.
