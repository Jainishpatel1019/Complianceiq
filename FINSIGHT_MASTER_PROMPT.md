# FinSight — Regulatory Intelligence Platform
## Master Prompt Reference (converted from PDF)

---

## SECTION 1 — WHAT WE ARE BUILDING

### The project in plain English
FinSight reads new financial regulations automatically every day, figures out which ones actually matter and why, calculates how much they would cost a bank to comply with, and explains all of this in plain English — with mathematical proof behind every claim.

Think of it as a Bloomberg Terminal crossed with a research analyst, but fully automated, completely free to run, and rigorous enough to publish as an academic paper.

### Why this project stands out
- **Generic ML projects:** "My model got 94% accuracy" — but on what? With what baseline?
- **This project's answer:** Every result has a confidence interval. Every method is compared against 4 baselines. Tested on data from a different time period than it was trained on. Every mathematical choice is justified.
- **The technology choices:** Airflow, PostgreSQL, Ollama, LangGraph, and Docker are not decoration. Each one solves a specific problem.
- **The business story:** Banks paid $14 billion in compliance fines in 2023. Compliance teams read regulations manually — it takes days per document. This system does it in 4 minutes, with a confidence score.

### Data sources — all completely free
| Source | What it contains | Scale |
|--------|-----------------|-------|
| Federal Register API | All US regulatory documents since 1994, full text | 90,000+ documents |
| SEC EDGAR full-text | All SEC rule releases and enforcement actions | 10M+ filings |
| FDIC BankFind Suite | Quarterly financial data for every US bank since 1992 | 5,000+ banks x 120 quarters |
| CFPB public API | Consumer finance rules + 2M+ complaint records | 2M+ rows |

---

## SECTION 2 — HOW EACH TECHNOLOGY IS USED (DEEPLY)

### Apache Airflow — the system's nervous system
9 production DAGs with real dependencies. Uses dynamic task mapping, XCom to pass state between tasks, custom HTTP sensors to detect new regulatory documents, SLA miss callbacks, inter-DAG triggers.
- TaskGroups | Dynamic task mapping | XCom state passing | Custom sensors | SLA callbacks | Inter-DAG dependencies | CeleryExecutor | Flower monitoring

### PostgreSQL — the system's long-term memory
TimescaleDB extension for time-series regulatory data. pgvector extension for hybrid keyword + semantic search. Materialized views that refresh daily. Partitioned tables. JSONB columns for flexible metadata storage.
- TimescaleDB extension | pgvector | Materialized views | Table partitioning | Full-text search indexes | JSONB columns | SQLAlchemy ORM | Alembic migrations

### Ollama — fully local AI, zero API cost forever
Three models: mistral:7b (reasoning + report generation), nomic-embed-text (document embeddings for RAG), llama3.2:3b (fast classification). Ollama REST API called from Airflow DAGs and LangGraph agent.
- mistral:7b — reasoning | nomic-embed-text — RAG embeddings | llama3.2:3b — classification | Ollama REST API in Docker | Streaming responses | Model health checks

### LangGraph — the agent's decision-making brain
StateGraph with conditional edges routing to different tools based on document type. Parallel tool execution. Human-in-the-loop checkpoints. Persistent memory across regulatory documents in the same session. Every step logged to PostgreSQL and shown on dashboard.
- StateGraph with conditional edges | Parallel tool nodes | Checkpointing & persistence | Streaming token output | Tool: search_federal_register | Tool: calculate_rwa_delta | Tool: query_knowledge_graph | Tool: run_causal_estimate

### Docker Compose — run everything in 2 commands
11 services: airflow-webserver | airflow-scheduler | airflow-worker | postgres | redis | chromadb | ollama | api (FastAPI) | frontend (nginx) | mlflow | flower
- Separate docker-compose.prod.yml for Hugging Face deployment with slimmed service set and pre-seeded data.

---

## SECTION 3 — FOLDER STRUCTURE & GITHUB STRATEGY

### Repository structure
```
finsight/
  airflow/
    dags/
      ingest_sources.py        # Federal Register + FDIC ingestion
      embed_and_index.py       # Chunk docs, call Ollama, upsert ChromaDB
      change_detection.py      # Drift + JSD + Wasserstein with CI
      graph_update.py          # NetworkX graph, GAT re-embed, Louvain
      causal_estimation.py     # DiD on FDIC data, synthetic control
      impact_agent.py          # Trigger LangGraph agent for flagged docs
      evaluate_pipeline.py     # RAGAS scores, ablation, calibration
      model_registry.py        # MLflow tracking, auto-promote on threshold
      alert_dispatch.py        # Slack/email if impact score P(High) > 0.8
  backend/
    agents/
      impact_agent.py          # LangGraph StateGraph definition
      tools.py                 # All 8 agent tools
      prompts.py               # System + tool prompts
    models/
      bayesian_network.py      # pgmpy BN, structure learning, inference
      causal_inference.py      # econml DiD, synthetic control, RDD
      change_detection.py      # Drift, JSD, Wasserstein, bootstrap CI
      graph_model.py           # PyTorch Geometric GAT implementation
    pipelines/
      ingestion.py
      embedding.py
      evaluation.py
  api/
    main.py                    # FastAPI app
    routes/                    # One file per domain area
    websockets.py              # Live agent trace streaming
  db/
    models.py                  # SQLAlchemy ORM models
    migrations/                # Alembic migration files
  frontend/
    src/
      pages/                   # One file per dashboard page
      components/              # Reusable UI components
      hooks/                   # Custom React hooks (useWebSocket, etc.)
    package.json
    vite.config.js
  docs/
    architecture.md
    math_explainer.md          # Your 2-page technical write-up
    diagrams/
  docker-compose.yml           # Full local dev environment
  docker-compose.prod.yml      # Hugging Face slim deployment
  Makefile                     # make setup / make dev / make test
  .env.example                 # Template — SAFE to commit
  .env                         # Real secrets — NEVER commit
  .cursorrules                 # Cursor AI coding standards
  .gitignore
  dvc.yaml                     # DVC pipeline definitions
  dvc.lock                     # Reproducibility lock file
  pyproject.toml
  README.md                    # Written in your own voice
```

### GitHub security — what goes where
**NEVER COMMIT:**
- .env (all secrets)
- chromadb/ (vector data)
- mlruns/ (MLflow artifacts)
- airflow/logs/
- .ollama/ (model weights)
- data/raw/ and data/processed/
- __pycache__/ and *.pyc
- frontend/node_modules/
- .cursor/ (IDE settings)
- .dvc/cache/

**ALWAYS COMMIT:**
- .env.example (template only)
- All Python source code
- docker-compose.yml
- All 9 Airflow DAG files
- requirements.txt + pyproject.toml
- React frontend source (not built files)
- dvc.yaml + dvc.lock
- Evaluation results as JSON/CSV
- Makefile
- README.md + docs/

---

## SECTION 4 — THE MATHEMATICS (DO NOT SIMPLIFY)

### Three scientific pillars that make this FAANG-level

#### Pillar 1 — Detecting how much a regulation changed

**MEASURE 1: SEMANTIC DRIFT (catches meaning changes)**
```
drift(d_old, d_new) = 1 - cosine_similarity( embed(d_old), embed(d_new) )
Score range: 0 = identical meaning, 1 = completely different
Threshold: drift > 0.15 triggers full analysis
Uncertainty: 95% CI via bootstrap resampling over 1,000 chunk pairs
```
embed() uses nomic-embed-text running in Ollama. Every score is reported as a range (e.g. 0.31 +/- 0.04) not a single number.

**MEASURE 2: JENSEN-SHANNON DIVERGENCE (catches vocabulary structure changes)**
```
JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
where M = 0.5 * (P + Q), and KL = Kullback-Leibler divergence
P = TF-IDF term distribution of old document
Q = TF-IDF term distribution of new document
p-value from permutation test with n=10,000 shuffles
```
JSD catches structural removal — if an entire section is deleted, the vocabulary distribution shifts dramatically even if the remaining text is semantically similar.

**MEASURE 3: WASSERSTEIN DISTANCE (catches reorganisation)**
```
W2(mu, nu) = inf over all joint distributions gamma of:
( integral of ||x - y||^2 d_gamma(x,y) ) ^ 0.5
Applied to: sentence-level embedding distributions of each document
```
Captures optimal transport distance between the full distributions of sentence embeddings. Sensitive to geometry of the embedding space — catches when a regulation is reorganised (same content, different structure) even when vocabulary and meaning are unchanged. Use scipy.stats.wasserstein_distance.

#### Pillar 2 — Proving the financial impact with causal maths

**METHOD 1: DIFFERENCE-IN-DIFFERENCES (for regulations that affect some banks but not others)**
```
ATT = E[ Y(1) - Y(0) | D=1 ]
DiD estimate = (Y_post - Y_pre)_regulated - (Y_post - Y_pre)_control
Standard error via heteroskedasticity-robust clustering on bank ID
Pre-test: parallel trends assumption via event study plot (t-2 to t-1)
```
Data: FDIC quarterly call reports. When a capital rule affects banks above $10B in assets, banks just below that threshold serve as the control group. Use econml library.

**METHOD 2: SYNTHETIC CONTROL (for landmark regulations affecting everyone)**
```
Y_synthetic(t) = sum_j ( w_j* x Y_j(t) )
w* = argmin over w of ||X_1 - X_0 * W||^2
subject to: w_j >= 0 and sum(w_j) = 1
Gap = Y_actual(t) - Y_synthetic(t) (this IS the causal estimate)
```
For regulations like Dodd-Frank (2010) where ALL banks were affected, construct a synthetic US banking sector from a weighted mix of international banking sectors that were NOT affected.

**METHOD 3: CAPITAL IMPACT FORMULA (BASEL III MATH)**
```
Delta_RWA = sum_i ( EAD_i x LGD_i x Delta_risk_weight_i )
EAD = Exposure at Default, LGD = Loss Given Default
Delta_risk_weight extracted from regulation text by Ollama agent
Uncertainty: 10,000 MC samples over LLM extraction confidence intervals
Output: 'Median $340M, 90% CI [$180M, $620M]' — not a point estimate
```

#### Pillar 3 — The regulation knowledge graph

**GRAPH STRUCTURE (HETEROGENEOUS, TEMPORAL)**
```
Nodes: Regulation | Agency | Institution | CFR_Section | Concept
Edges: amends(t) | supersedes(t) | references | applies_to | enforces
Every edge carries a timestamp — the graph is queryable at any point in time
Example query: 'What did the regulatory graph look like in 2008?'
```
Build with NetworkX first. Move to PyTorch Geometric for GAT training.

**PAGERANK — FINDING THE MOST CRITICAL REGULATIONS**
```
PR(r) = (1-d) + d x sum over j in In(r) of: PR(j) / |Out(j)|
d = damping factor (0.85), In(r) = regulations that reference r
High PageRank = structurally critical node
```

**GRAPH ATTENTION NETWORK — LEARNED NODE REPRESENTATIONS**
```
h'_i = sigma( sum over j in N(i) of: alpha_ij x W x h_j )
alpha_ij = softmax( LeakyReLU( a^T [ W*h_i || W*h_j ] ) )
Trained task: classify each regulation into impact category
Library: PyTorch Geometric (PyG), free and well-documented
```

---

## SECTION 5 — THE LANGGRAPH AGENT

### How the AI reasoning system works

| Step | What happens |
|------|-------------|
| trigger | Airflow fires the agent — impact_agent DAG runs at 09:00 UTC. Queries PostgreSQL for documents flagged by change detection (drift > 0.15 OR JSD p-value < 0.05) |
| node 1 | Fetch & classify (parallel) — fetch_full_document() + classify_document_type() simultaneously |
| node 2 | RAG query over document chunks — rag_query() with targeted questions. ChromaDB returns most relevant chunks. nomic-embed-text handles embeddings |
| node 3 | Graph reasoning — query_knowledge_graph() to find PageRank score, downstream regulations. If PageRank > 0.5, also runs cascade_propagation() |
| node 4 | Causal + financial math — calculate_rwa_delta() + run_causal_estimate() using pre-computed DiD results. Calls Bayesian network to compute posterior P(Impact = High \| evidence) |
| output | Structured report generation — mistral:7b synthesises all tool outputs into structured JSON: { summary, impact_score, impact_ci, delta_rwa_median, delta_rwa_ci, affected_business_lines, key_citations, agent_reasoning_trace } |

### The 8 agent tools
| Tool name | What it does | Returns |
|-----------|-------------|---------|
| fetch_full_document(doc_id) | Fetches complete regulation text from PostgreSQL + metadata | dict with text, metadata, version history |
| rag_query(question, doc_id) | Searches ChromaDB for relevant chunks from a specific document | list of chunks with relevance scores |
| classify_document_type(text) | Uses llama3.2:3b to classify into regulatory category | category string + confidence float |
| calculate_rwa_delta(risk_weights) | Runs Basel III RWA formula with MC uncertainty | dict: median, ci_low, ci_high in $M |
| query_knowledge_graph(doc_id) | Returns node neighbours, PageRank, Louvain community | dict: pagerank, community, dependents list |
| cascade_propagation(doc_id) | BFS from node, sums cascade impact scores downstream | list of (doc_id, cascade_score) tuples |
| run_causal_estimate(regulation_id) | Looks up pre-computed DiD estimate from causal DAG output | dict: att, se, p_value, ci_95 |
| compute_bayesian_posterior(evidence) | Runs pgmpy BN inference given extracted evidence dict | dict: P(Low), P(Medium), P(High) |

---

## SECTION 6 — HOW TO USE CURSOR AI PROPERLY

### Building fast without losing ownership

**FIRST: Create .cursorrules before anything else**
```
You are building FinSight. Stack: Python 3.11, FastAPI, SQLAlchemy 2.0, Airflow 2.9, LangGraph 0.2, Ollama REST API, PyTorch Geometric, pgmpy, econml. Rules: Always write type hints on all functions. Always write Google-style docstrings. Never use print() — use Python logging module. All DB queries via SQLAlchemy ORM. All functions must have at least one unit test. Assume all services run in Docker. Error handling: never use bare except. Always catch specific exceptions.
```

**RULE 1:** One function at a time — never generate whole modules
**RULE 2:** Ask to justify every decision
**RULE 3:** Commit after every function — write real commit messages
**RULE 4:** Use Composer for multi-file features only
**RULE 5:** Never let Cursor write your README or documentation

---

## SECTION 7 — BUILD PHASES

### Six months, in the right order

| Month | Phase | Goal |
|-------|-------|------|
| 1 | Data foundation | Docker-compose with all 11 services. Federal Register + FDIC ingestion DAGs. PostgreSQL schema with TimescaleDB. Embedding pipeline with nomic-embed-text. Goal: data flows daily, you can query 'show me all capital rules from 2023' |
| 2 | Change detection math + first dashboard | JSD + drift + Wasserstein with bootstrap CI. Section heatmap. First React dashboard page live — the regulatory feed with change scores |
| 3 | Causal inference engine | DiD on FDIC data for 3 real regulations (Dodd-Frank, Volcker Rule, Basel III). Synthetic control for Dodd-Frank. RDD at $10B and $50B asset thresholds. Causal dashboard page with confidence bands |
| 4 | Knowledge graph + GAT + LangGraph agent | Heterogeneous graph in NetworkX. PyTorch Geometric for GAT. Louvain community detection. LangGraph agent with all 8 tools. D3 force-directed graph visualisation |
| 5 | Bayesian network + evaluation | pgmpy BN structure learning on FDIC enforcement data. Monte Carlo uncertainty for ΔRWA. Human-labelled test set of 300 document pairs. 16-cell ablation table. RAGAS evaluation on 500 RAG queries |
| 6 | Polish + Hugging Face deployment + write-up | Final UI polish. Hugging Face Docker Space with pre-seeded data. 2-page technical write-up. Makefile: 'make dev' brings up entire system in 2 commands. 3-minute demo video |

---

## SECTION 8 — DEPLOYING TO HUGGING FACE

Hugging Face Spaces free tier: 16GB RAM, 2 vCPUs. Enough for React frontend, FastAPI backend, PostgreSQL with pre-seeded data, ChromaDB with pre-built index, Ollama with models pre-pulled. Airflow pipelines stay local.

| Step | Action |
|------|--------|
| 1 | Pre-seed the data locally — run full pipeline on 6 months of Federal Register data. Export pg_dump + ChromaDB snapshot. Commit to data/seed/ in HF Space repo |
| 2 | Create docker-compose.hf.yml — slimmed version: frontend (nginx), api (FastAPI), postgres (loads seed dump), chromadb (loads pre-built index), ollama (with mistral:7b pre-pulled). Remove: airflow, redis, mlflow, flower |
| 3 | Use HF Secrets for environment variables |
| 4 | Add a 'Live Demo' banner — 'Demo data: Jan 2024 - Jun 2024. Full pipeline runs locally — see GitHub for setup.' |

---

## SECTION 9 — INTERVIEW QUESTIONS YOU MUST BE ABLE TO ANSWER

| Difficulty | Question | Answer |
|------------|----------|--------|
| HARD | Why DiD and not a simple before/after regression? | DiD controls for time-invariant confounders. A bank's size, risk appetite, and existing capital buffer all affect both regulatory exposure and outcomes. DiD differences those away by design. |
| HARD | Your Bayesian network has a Cost and RWA node. Are they conditionally independent given Impact? | No. The model includes a direct Cost to RWA edge because compliance costs affect capital ratios directly. Tested the structure using BIC-based hill climbing on the FDIC dataset. |
| HARD | How do you know your change detection threshold (drift > 0.15) is the right value? | I treat it as a hyperparameter and optimise it on a labelled development set using F1 score as the objective. I report the threshold alongside its confidence interval from bootstrap resampling. |
| MEDIUM | How do you handle LLM extraction errors in the RWA calculation? | Three ways: run each extraction 3 times with temperature 0.3 and take the median. Model extraction uncertainty as a CI fed into Monte Carlo sampling. Human-labelled test set of 200 documents to measure extraction accuracy independently. |
| MEDIUM | What is your baseline for change detection? | Four baselines: exact string diff (Levenshtein ratio), TF-IDF cosine similarity, BM25, and fine-tuned FinBERT. I report F1, AUROC, and calibration for all five methods on a held-out test set of 300 manually labelled document pairs. |
| MEDIUM | Why three change detection measures instead of just one? | Each catches a different type of change. Ablation experiments show removing any one of the three reduces F1 by more than 5 points. |
| SYSTEM | Why Airflow and not Prefect or Dagster? | Airflow has the largest production footprint at financial institutions — appears in over 70% of data engineering job descriptions at banks and fintechs. |
| SYSTEM | How would this scale to 100x the current document volume? | Embedding pipeline parallelises trivially — Airflow dynamic task mapping. Graph updates are the bottleneck — would move from NetworkX to Neo4j. Bayesian inference — switch from variable elimination to variational inference. |
| DESIGN | Why use a GAT instead of a simpler GCN? | GCN aggregates neighbour embeddings with equal weights. GAT learns attention weights over neighbours. In ablation experiments, GAT outperforms GCN by 4.2 F1 points on regulation category classification. |

---

## SECTION 10 — SETUP QUESTIONS (ask before writing any code)

Q1. **Machine specs** — How much RAM does your machine have, and do you have a GPU? (< 16GB RAM → use llama3.2:3b for everything instead of mistral:7b; GPU → enable GPU passthrough in Docker for Ollama)

Q2. **Python experience** — On a scale of 1-5, how comfortable are you with Python type hints, SQLAlchemy ORM, and async/await?

Q3. **Starting point** — Is this a brand new empty folder, or do you have any existing code?

Q4. **GitHub username** — What is your GitHub username?

Q5. **Project name preference** — Do you want to keep the name 'FinSight'?

Q6. **Phase priority** — Which phase do you want to start with? (Recommended: Month 1 — Docker + Airflow + data ingestion)

Q7. **Deployment target** — Do you already have a Hugging Face account? Public or private until polished?

Q8. **Test coverage** — Write unit tests alongside every function (recommended) or write tests yourself after?
